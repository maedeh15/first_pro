"""
representation_collapse_check.py -- is the effective-rank collapse seen in
full_architecture_diagnostics.py's Section 3 (effective_rank=1.6 on CAFE's
512-dim fusion embedding, batch_size=16) a real, systematic property of the
trained model, or just an artifact of that one 16-window batch from one
subject?

WHY THIS EXISTS
Section 3's effective-rank number is capped by min(batch_size, embedding_dim)
-- with batch_size=16 the ceiling is 16, so "1.6" could either mean genuine
collapse (the model routes almost all 16 DIFFERENT real windows to nearly
the same point in feature space) or just be noisy with so few samples. This
script removes that ambiguity two ways: (1) pools ALL of a fold's held-out
windows (typically 250-320, not 16) before computing effective rank, so the
ceiling is no longer the bottleneck, and (2) repeats this across MANY
subjects and folds, so a systematic pattern can be told apart from one
subject's quirk.

WHAT IS MEASURED (same CKA / effective-rank definitions as
full_architecture_diagnostics.py's Section 3, applied via forward hooks --
no gradients, no training, pure inference)
  cafe.fusion_out_proj   (B, 512) -- CAFE's output feature for the fusion branch
  clf_fusion.net.0       (B, 128) -- first dense layer of the fusion classifier
  effective_rank(X) = (sum(eig))^2 / sum(eig^2)  where eig = singular values^2
                       of the centered embedding matrix (participation ratio;
                       1 = all variance on one direction, dim = perfectly
                       isotropic/no collapse)
  linear_cka(X, Y)   = how much of layer Y is just a linear transform of
                       layer X (near 1 = redundant, near 0 = genuinely new)

OUTPUT
  Per (subject, session) row: n_windows actually pooled, effective rank of
  both layers, and their CKA -- plus a group summary (mean/SD effective rank
  across all folds, % of embedding dim it represents, and how consistent
  the collapse is across subjects).

USAGE
=====
    python representation_collapse_check.py --results-dir "E:\\...\\results_40b7d8ea53_seed42" ^
        --data-root "E:\\...\\TSFNET_PAPER_EXACT_nback"

No GPU required, much faster with one -- this is pure inference (no
training, no backward passes), one forward pass per fold.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--results-dir", required=True)
p.add_argument("--data-root", required=True)
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-dir", default=None, help="default: <results-dir>/representation_collapse_check")
p.add_argument("--forward-batch-size", type=int, default=64,
              help="batch size used only for chunking the forward pass (memory), "
                   "NOT the pooling unit -- all of a fold's windows are pooled regardless.")
p.add_argument("--adabn", action="store_true", default=True)
p.add_argument("--no-adabn", dest="adabn", action="store_false")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ARGS = p.parse_args()

LAYERS = ["cafe.fusion_out_proj", "clf_fusion.net.0"]


def parse_subjects(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [f"VP{i:03d}" for i in out]


def build_model(n_segments):
    return TSFNet(num_classes=3, fusion_combination="paper_mean",
                  fnirs_num_segments=n_segments, fnirs_temporal_align="egta",
                  efgf_gate_mode="sigmoid", efgf_gate_init=-2.0)


def effective_rank(X):
    """X: (N, D) numpy array. Participation ratio of its covariance eigenvalues."""
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    eig = s ** 2
    if eig.sum() <= 0:
        return float("nan")
    return float((eig.sum() ** 2) / (eig ** 2).sum())


def linear_cka(X, Y):
    """Dual/kernel form (see full_architecture_diagnostics.py's _linear_cka) --
    safe for any embedding dimension since it only ever builds (N, N) Gram
    matrices, N = number of pooled windows here (can be in the hundreds)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    hsic = (K * L).sum()
    norm_x = np.sqrt((K * K).sum())
    norm_y = np.sqrt((L * L).sum())
    return float(hsic / (norm_x * norm_y + 1e-12))


def collect_fold_embeddings(full, test_idx, ckpt_path, device, forward_batch_size, adabn):
    n_segments = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    model = build_model(n_segments).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    acts = {name: [] for name in LAYERS}
    handles = []

    def hook(name):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            acts[name].append(t.detach().reshape(t.shape[0], -1).cpu().numpy())
        return fn
    for name, m in model.named_modules():
        if name in LAYERS:
            handles.append(m.register_forward_hook(hook(name)))

    test_loader = DataLoader(Subset(full, test_idx), batch_size=forward_batch_size, shuffle=False)
    if adabn:
        adapt_batchnorm_to_target(model, test_loader, device, log_fn=lambda *a, **k: None)
        for name in LAYERS:
            acts[name].clear()   # AdaBN's own forward passes don't count

    with torch.no_grad():
        for batch in test_loader:
            x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(batch)
            x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            model(x_eeg, x_fnirs, x_conn, x_chan)

    for h in handles:
        h.remove()
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {name: np.concatenate(v, axis=0) for name, v in acts.items()}


def main():
    device = ARGS.device
    out_dir = ARGS.out_dir or os.path.join(ARGS.results_dir, "representation_collapse_check")
    os.makedirs(out_dir, exist_ok=True)
    subjects = parse_subjects(ARGS.subjects)

    rows = []
    for sid in subjects:
        subject_dir = os.path.join(ARGS.data_root, sid)
        try:
            full = NBackDataset(subject_dir, sid)
        except FileNotFoundError as e:
            print(f"[{sid}] SKIP -- {e}")
            continue
        for test_session in (1, 2, 3):
            ckpt_path = os.path.join(ARGS.results_dir, f"{sid}_session{test_session}_best.pt")
            if not os.path.isfile(ckpt_path):
                print(f"[{sid} s{test_session}] SKIP -- checkpoint not found")
                continue
            test_idx = np.flatnonzero(full.session_ids == test_session)
            try:
                acts = collect_fold_embeddings(full, test_idx, ckpt_path, device,
                                               ARGS.forward_batch_size, ARGS.adabn)
            except Exception as e:
                print(f"[{sid} s{test_session}] SKIP -- {type(e).__name__}: {e}")
                continue
            n = acts[LAYERS[0]].shape[0]
            er = {name: effective_rank(acts[name]) for name in LAYERS}
            cka = linear_cka(acts[LAYERS[0]], acts[LAYERS[1]])
            rows.append({"subject_id": sid, "session": test_session, "n_windows": n,
                        f"eff_rank_{LAYERS[0]}": er[LAYERS[0]],
                        f"eff_rank_{LAYERS[1]}": er[LAYERS[1]],
                        "cka": cka})
            print(f"[{sid} s{test_session}] n={n:4d}  "
                  f"eff_rank(fusion_out_proj, dim=512)={er[LAYERS[0]]:6.2f}  "
                  f"eff_rank(clf.net0, dim=128)={er[LAYERS[1]]:6.2f}  cka={cka:.3f}")

    if len(rows) < 6:
        sys.exit("ERROR: fewer than 6 (subject, session) folds processed -- check paths.")

    hline = "=" * 100
    print("\n" + hline)
    print(f"REPRESENTATION COLLAPSE SUMMARY -- {len(rows)} (subject, session) folds, "
          f"{len({r['subject_id'] for r in rows})} subjects")
    print(hline)
    for name, dim in zip(LAYERS, (512, 128)):
        er = np.array([r[f"eff_rank_{name}"] for r in rows])
        er = er[~np.isnan(er)]
        pct = 100 * er / dim
        print(f"\n{name} (dim={dim}):")
        print(f"  effective rank: mean={er.mean():.2f}  SD={er.std(ddof=1):.2f}  "
              f"median={np.median(er):.2f}  min={er.min():.2f}  max={er.max():.2f}")
        print(f"  as % of dim:    mean={pct.mean():.2f}%  (a HEALTHY, non-collapsed embedding "
              f"would typically use at least 10-30% of its dimensions; well under that across "
              f"many diverse real windows, not just one small batch, is a strong, systematic "
              f"collapse signal)")
        n_low = int((pct < 10).sum())
        print(f"  folds below 10% of dim: {n_low}/{len(pct)} ({100*n_low/len(pct):.0f}%)")

    ckas = np.array([r["cka"] for r in rows])
    print(f"\nCKA(fusion_out_proj, clf.net0): mean={ckas.mean():.3f}  SD={ckas.std(ddof=1):.3f}  "
          f"(near 1 = classifier's first layer is ~redundant with CAFE's output across many "
          f"real windows, not just one batch)")

    print("\nREADING")
    pct512 = 100 * np.array([r[f"eff_rank_{LAYERS[0]}"] for r in rows]) / 512
    if np.nanmean(pct512) < 5 and (pct512 < 10).sum() / len(pct512) > 0.8:
        print("  The collapse is SYSTEMATIC: across most subjects and folds, CAFE's 512-dim fusion "
              "embedding uses well under 5% of its available dimensions even when pooled over "
              "hundreds of genuinely different real windows. This is not a small-batch artifact -- "
              "it is a real property of the trained architecture, and a plausible contributor to "
              "the project's documented 2-back/3-back confusability (less room in feature space to "
              "keep those classes apart). Worth a dedicated ablation (e.g. does reducing CAFE's "
              "output dimension, or adding a decorrelation/whitening regularizer, change this).")
    elif np.nanmean(pct512) < 5:
        print("  Low average effective rank but with some high-variance folds/subjects -- the "
              "collapse is real on average but not perfectly uniform; check the per-fold table "
              "above for which subjects/sessions are the exceptions before generalizing.")
    else:
        print("  With full-fold pooling, effective rank recovers to a more reasonable fraction of "
              "the embedding dimension -- the earlier 1.6/16 number WAS mostly a small-batch "
              "artifact, not a systematic architectural collapse. Good news; no action needed here.")

    with open(os.path.join(out_dir, "representation_collapse_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, default=float)
    with open(os.path.join(out_dir, "representation_collapse_per_fold.csv"), "w",
             newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["subject_id", "session", "n_windows", f"eff_rank_{LAYERS[0]}",
                  f"eff_rank_{LAYERS[1]}", "cka"]
        w.writerow(header)
        for r in rows:
            w.writerow([r[h] for h in header])
    print(f"\n[written] {out_dir}: representation_collapse_summary.json, "
          f"representation_collapse_per_fold.csv")


if __name__ == "__main__":
    main()
