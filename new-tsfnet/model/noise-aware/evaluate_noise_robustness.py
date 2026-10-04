"""
evaluate_noise_robustness.py -- test-time EEG noise sweep on ALREADY-TRAINED
checkpoints. No GPU training, no re-running run_all_subjects.py.

WHAT THIS IMPLEMENTS
MGFNet's (Zhang et al., Sensors 2026, 26, 3402, Sec. 4.3) clean-training /
noisy-testing protocol: take a model trained on clean data, inject AWGN into
the EEG stream of the held-out test set ONLY (fNIRS untouched), and measure
how accuracy decays as SNR drops. In that paper the adaptive-fusion model and
the static-fusion variant were indistinguishable on clean data and at 10 dB;
the gap only opened up at 5 dB and below. So a null result at high SNR is
EXPECTED and is not evidence against gating.

WHY THIS IS THE CHEAP EXPERIMENT
It reuses checkpoints you already have. Run it once on the accepted
`paper_mean` checkpoints and once on the `gated` checkpoints, then compare
the two curves. If gating buys robustness, the gated curve decays more
slowly. That is a publishable finding even if clean-data accuracy is a tie.

HONEST LIMITS -- read before putting numbers in the thesis:
  * AWGN is not a real EEG artifact. MGFNet says this too: it does not
    represent EOG, EMG, electrode drift or motion. It is a controlled,
    reproducible probe of the fusion layer, not a simulation of bad data.
  * SNR here is relative to the PREPROCESSED signal power, not the raw
    recording, because that is what the dataloader hands us.
  * This is clean-training only. MGFNet's second experiment (noise-AWARE
    training) needs a real training run -- that is what
    --noise-aware-prob in run_all_subjects.py is for.

USAGE
    python evaluate_noise_robustness.py \
        --results-dir "E:\\...\\results_<hash>_seed42" \
        --data-root  "E:\\...\\TSFNET_PAPER_EXACT_nback" \
        --out robustness_seed42.csv
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tsfnet_model_notebook import (TSFNet, NBackDataset, _unpack_batch,
                                       inject_eeg_awgn, cohen_kappa)
except ImportError:
    print("Run this in the same folder/kernel as tsfnet_model_notebook.py "
          "(or %run tsfnet_model_notebook.py first).")
    sys.exit(1)

DEFAULT_SNRS = [None, 10.0, 5.0, 0.0, -5.0, -10.0]   # None == clean


def evaluate_at_snr(model, loader, device, snr_db, seed=0):
    """One full pass over the test set with EEG corrupted at a fixed SNR.

    snr_db=None means no corruption at all (the clean reference point).
    A fresh generator seeded identically for every SNR level keeps the
    *sample-selection* randomness comparable across levels.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    all_true, all_pred = [], []
    with torch.no_grad():
        for _batch in loader:
            x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(_batch)
            x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            if snr_db is not None:
                # corrupt_prob=1.0: every test sample at this exact SNR, so
                # the reported number is "accuracy AT this SNR", not an
                # average over a mixture of clean and noisy samples.
                x_eeg = inject_eeg_awgn(x_eeg, [snr_db], 1.0, generator=gen)
            _, _, _, y_pred, _ = model(x_eeg, x_fnirs, x_conn, x_chan)
            all_true.append(y.numpy())
            all_pred.append(y_pred.argmax(dim=-1).cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return float((y_true == y_pred).mean()), float(cohen_kappa(y_true, y_pred))


def infer_fusion_combination(state_dict):
    """A 'gated' checkpoint carries gate_fc.*; the others do not.

    *** VERIFIED LIMITATION -- READ THIS ***
    This can only separate 'gated' from 'not gated'. "paper_mean" and "wsum"
    have BYTE-IDENTICAL state_dict key sets (both use just the three scalars
    w_eeg/w_fusion/w_fnirs; they differ only in how forward() combines them:
    mean/3 vs. renormalize-by-sum). So a "wsum" checkpoint is detected here
    as "paper_mean" and would be evaluated with the WRONG formula, silently
    and with no error -- producing plausible but incorrect accuracies.

    Pass --fusion-combination explicitly whenever the checkpoints came from
    a `--fusion-combination wsum` run. The accepted configuration for this
    project uses paper_mean or gated, so the default is safe there, but do
    not rely on auto-detection for a wsum comparison run.
    """
    return "gated" if any(k.startswith("gate_fc.") for k in state_dict) else "paper_mean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="Folder holding the *_best.pt checkpoints for ONE seed/variant.")
    ap.add_argument("--data-root", required=True,
                    help="Folder holding the per-subject preprocessed .npy files.")
    ap.add_argument("--out", default="robustness.csv")
    ap.add_argument("--snr-db", type=float, nargs="*", default=None,
                    help="SNR levels in dB. Clean is always included.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fnirs-num-segments", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fusion-combination", choices=["paper_mean", "wsum", "gated"],
                    default=None,
                    help="Override auto-detection. REQUIRED for 'wsum' checkpoints: "
                         "wsum and paper_mean have identical state_dict keys, so "
                         "auto-detection reports both as paper_mean and would "
                         "evaluate a wsum model with the wrong formula.")
    ap.add_argument("--fnirs-temporal-align", choices=["none", "egta"], default="egta",
                    help="MUST match the training run's flag exactly, or load_state_dict "
                         "fails (egta adds cafe.egta_q/egta_k submodules). Accepted project "
                         "config uses egta, so that's the default here.")
    ap.add_argument("--efgf-gate-mode", choices=["clamp", "sigmoid"], default="sigmoid",
                    help="MUST match the training run's flag exactly, or load_state_dict "
                         "fails (sigmoid stores gamma_raw/alpha_raw instead of gamma/alpha). "
                         "Accepted project config uses sigmoid, so that's the default here.")
    args = ap.parse_args()

    snrs = [None] + list(args.snr_db) if args.snr_db else DEFAULT_SNRS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(glob.glob(os.path.join(args.results_dir, "*_best.pt")))
    if not ckpts:
        print(f"No *_best.pt found in {args.results_dir}")
        sys.exit(1)
    print(f"Found {len(ckpts)} checkpoints. Device={device}. SNR levels={snrs}\n")

    rows = []
    for ck_path in ckpts:
        name = os.path.basename(ck_path)
        # Expected: cross_session_VPxxx_fold<k>_best.pt -- adapt if your
        # naming differs; the subject id is what matters.
        subject_id = next((p for p in name.split("_") if p.startswith("VP")), None)
        if subject_id is None:
            print(f"  [skip] cannot parse subject id from {name}")
            continue

        ckpt = torch.load(ck_path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        detected = infer_fusion_combination(sd)
        fusion = args.fusion_combination or detected
        # The one contradiction we CAN detect: asking for a non-gated mode on
        # a checkpoint that has gate_fc.*, or vice versa. (paper_mean vs wsum
        # remains undetectable -- see infer_fusion_combination's docstring.)
        if (fusion == "gated") != (detected == "gated"):
            print(f"  [ERROR] {name}: --fusion-combination {fusion} contradicts the "
                  f"checkpoint (gate_fc.* {'present' if detected == 'gated' else 'absent'}). "
                  f"Skipping rather than loading a mismatched model.")
            continue

        subject_dir = os.path.join(args.data_root, subject_id)
        if not os.path.isdir(subject_dir):
            print(f"  [skip] no data dir for {subject_id}")
            continue

        ds = NBackDataset(subject_dir, subject_id)
        # BUGFIX: tsfnet_model_notebook.py's train_one_fold (called from
        # run_cross_session loop) does torch.save(model.state_dict(), ckpt_path)
        # -- it NEVER wraps the state_dict in {"model_state_dict":..., "test_session":...}.
        # So ckpt.get("test_session") always returned None here, silently evaluating
        # every checkpoint on ALL sessions (train+test mixed) instead of just the
        # held-out one. The held-out session is only recorded in the FILENAME itself:
        # ckpt_path = f"{subject_id}_session{test_session}_best.pt" (notebook line ~3655).
        # Parse it from there instead.
        m = re.search(r"_session(\d+)_best\.pt$", name)
        test_session = int(m.group(1)) if m else None
        if test_session is not None:
            idx = np.where(ds.session_ids == test_session)[0]
            ds = NBackDataset(subject_dir, subject_id, indices=idx)
        else:
            print(f"  [warn] {name}: filename doesn't match '..._session<N>_best.pt'; "
                  f"cannot determine the held-out session, evaluating on ALL sessions "
                  f"(includes training data -- treat these numbers as diagnostic only)")

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        model = TSFNet(num_classes=3, fusion_combination=fusion,
                       fnirs_num_segments=args.fnirs_num_segments,
                       fnirs_temporal_align=args.fnirs_temporal_align,
                       efgf_gate_mode=args.efgf_gate_mode).to(device)
        model.load_state_dict(sd, strict=True)

        for snr in snrs:
            acc, kap = evaluate_at_snr(model, loader, device, snr, seed=args.seed)
            rows.append({"checkpoint": name, "subject_id": subject_id,
                         "fusion_combination": fusion,
                         "snr_db": "clean" if snr is None else snr,
                         "accuracy": round(acc, 4), "kappa": round(kap, 4)})
            print(f"  {subject_id} [{fusion}] SNR={'clean' if snr is None else f'{snr:+g}dB':>8} "
                  f"acc={acc:.4f} kappa={kap:.4f}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    # Mean curve across subjects, which is the thing to plot.
    print("\nMean accuracy across subjects:")
    for snr in snrs:
        key = "clean" if snr is None else snr
        vals = [r["accuracy"] for r in rows if r["snr_db"] == key]
        if vals:
            print(f"  SNR={str(key):>8}: {np.mean(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
