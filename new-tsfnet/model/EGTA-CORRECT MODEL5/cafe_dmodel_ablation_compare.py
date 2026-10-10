r"""
cafe_dmodel_ablation_compare.py -- does CAFE's internal bottleneck dimension
(--cafe-d-model, already an existing, previously-ACCURACY-tested-and-rejected
flag in run_all_subjects.py) also control the representation-collapse found
by representation_collapse_check.py (effective_rank ~1.8 out of 512,
systematic across all 78 folds, but NOT correlated with per-subject
accuracy)?

WHY THIS EXISTS / WHAT IT DOES NOT DO
This script does NOT train anything new -- it only COMPARES checkpoints you
have already trained with different --cafe-d-model values, using the exact
same effective-rank/CKA machinery as representation_collapse_check.py, side
by side with each variant's own accuracy (from its baseline_all_subjects*.csv).
Training itself is just the existing flag, run as many times as you want to
compare:

    python run_all_subjects.py --cafe-d-model 16  --seed 42 ...(your usual flags)
    python run_all_subjects.py --cafe-d-model 32  --seed 42 ...
    python run_all_subjects.py --cafe-d-model 64  --seed 42 ...   (= the accepted default)

Each run gets its own results_<hash>_seed<N> folder (cafe_d_model enters the
variant hash/tag automatically, same as any other flag) and its own
baseline_all_subjects_*.csv. Point this script at all of them together.

IMPORTANT CAVEAT (read before interpreting the numbers)
--cafe-d-model controls the internal attention bottleneck (Q/K/V dimension),
NOT the final output dimension: CAFE's fusion_out_proj is always
Linear(cafe_d_model, 512) -- it still projects UP to 512 regardless of
cafe_d_model's value. A SMALLER cafe_d_model could, if anything, make the
upstream bottleneck narrower and push MORE collapse into the final 512-dim
output, not less. If this ablation shows no effect, that is an expected,
informative negative result pinpointing that the 512-dim collapse is NOT
caused by this particular bottleneck, and the next thing to isolate would be
the gated residual combination in CAFE.forward (g_fusion/g_fnirs mixing) or
fusion_out_proj's own weight structure, not cafe_d_model.

USAGE
=====
    python cafe_dmodel_ablation_compare.py ^
        --variant "d16=E:\...\results_<hash16>_seed42" --acc-csv-16 "...\baseline_all_subjects_..._<hash16>.csv" ^
        --variant "d32=E:\...\results_<hash32>_seed42" --acc-csv-32 "...\baseline_all_subjects_..._<hash32>.csv" ^
        --variant "d64=E:\...\results_40b7d8ea53_seed42" --acc-csv-64 "...\baseline_all_subjects_...40b7d8ea53.csv" ^
        --data-root "E:\...\TSFNET_PAPER_EXACT_nback"

Simpler form (recommended): pass each variant as "label=results_dir=acc_csv"
(three parts, same separator) so you don't need separate --acc-csv-* flags:
    --variant "d16=E:\...\results_<hash16>_seed42=E:\...\baseline_all_subjects_..._<hash16>.csv"

No GPU required for this script itself (pure inference on existing
checkpoints), much faster with one.
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
p.add_argument("--variant", action="append", required=True,
              help="label=results_dir=accuracy_csv (3 parts, '=' separated). Repeat once per "
                   "cafe_d_model value to compare, e.g. --variant d16=...=... --variant d64=...=...")
p.add_argument("--data-root", required=True)
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-dir", default=None, help="default: ./cafe_dmodel_ablation_check")
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


def parse_variants(variant_args):
    variants = []
    for v in variant_args:
        parts = v.split("=")
        if len(parts) != 3:
            sys.exit(f"ERROR: --variant must be 'label=results_dir=accuracy_csv', got: {v!r}")
        label, results_dir, acc_csv = parts
        variants.append({"label": label, "results_dir": results_dir, "acc_csv": acc_csv})
    return variants


def infer_cafe_d_model(ckpt_path, device):
    """Reads cafe_d_model directly off the checkpoint's own tensor shapes --
    no need to trust a filename or ask the user to re-type it."""
    state = torch.load(ckpt_path, map_location=device)
    w = state.get("cafe.fusion_proj.weight")
    if w is None:
        return None
    return int(w.shape[0])   # fusion_proj: Linear(fusion_dim, d_model) -> weight (d_model, fusion_dim)


def build_model(n_segments, cafe_d_model):
    return TSFNet(num_classes=3, fusion_combination="paper_mean",
                  fnirs_num_segments=n_segments, fnirs_temporal_align="egta",
                  efgf_gate_mode="sigmoid", efgf_gate_init=-2.0,
                  cafe_d_model=cafe_d_model)


def effective_rank(X):
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    eig = s ** 2
    if eig.sum() <= 0:
        return float("nan")
    return float((eig.sum() ** 2) / (eig ** 2).sum())


def collect_fold_embeddings(full, test_idx, ckpt_path, device, cafe_d_model, adabn):
    n_segments = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    model = build_model(n_segments, cafe_d_model).to(device)
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

    test_loader = DataLoader(Subset(full, test_idx), batch_size=64, shuffle=False)
    if adabn:
        adapt_batchnorm_to_target(model, test_loader, device, log_fn=lambda *a, **k: None)
        for name in LAYERS:
            acts[name].clear()

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


def load_acc_csv(path):
    d = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("subject_id") and row.get("mean_acc"):
                d[row["subject_id"]] = float(row["mean_acc"])
    return d


def run_variant(variant, data_root, subjects, device, adabn):
    results_dir = variant["results_dir"]
    acc = load_acc_csv(variant["acc_csv"])
    rows = []
    cafe_d_model = None
    for sid in subjects:
        subject_dir = os.path.join(data_root, sid)
        try:
            full = NBackDataset(subject_dir, sid)
        except FileNotFoundError:
            continue
        for test_session in (1, 2, 3):
            ckpt_path = os.path.join(results_dir, f"{sid}_session{test_session}_best.pt")
            if not os.path.isfile(ckpt_path):
                continue
            if cafe_d_model is None:
                cafe_d_model = infer_cafe_d_model(ckpt_path, device)
                print(f"  [{variant['label']}] inferred cafe_d_model={cafe_d_model} from checkpoint shapes")
            test_idx = np.flatnonzero(full.session_ids == test_session)
            try:
                acts = collect_fold_embeddings(full, test_idx, ckpt_path, device, cafe_d_model, adabn)
            except Exception as e:
                print(f"  [{sid} s{test_session}] SKIP -- {type(e).__name__}: {e}")
                continue
            er = effective_rank(acts[LAYERS[0]])
            rows.append({"subject_id": sid, "session": test_session, "effective_rank": er})
    return cafe_d_model, rows, acc


def main():
    device = ARGS.device
    out_dir = ARGS.out_dir or "./cafe_dmodel_ablation_check"
    os.makedirs(out_dir, exist_ok=True)
    subjects = parse_subjects(ARGS.subjects)
    variants = parse_variants(ARGS.variant)

    hline = "=" * 100
    print(hline)
    print(f"CAFE d_model ABLATION vs REPRESENTATION COLLAPSE -- {len(variants)} variant(s)")
    print(hline)

    summary = {}
    for variant in variants:
        print(f"\n[{variant['label']}] {variant['results_dir']}")
        cafe_d_model, rows, acc = run_variant(variant, ARGS.data_root, subjects, device, ARGS.adabn)
        if not rows:
            print(f"  SKIP -- no folds found.")
            continue
        er = np.array([r["effective_rank"] for r in rows])
        acc_vals = np.array(list(acc.values()))
        summary[variant["label"]] = {
            "cafe_d_model": cafe_d_model, "n_folds": len(rows),
            "effective_rank_mean": float(er.mean()), "effective_rank_sd": float(er.std(ddof=1)),
            "effective_rank_pct_of_512": float(100 * er.mean() / 512),
            "accuracy_mean": float(acc_vals.mean()) if len(acc_vals) else float("nan"),
            "accuracy_sd": float(acc_vals.std(ddof=1)) if len(acc_vals) > 1 else float("nan"),
            "n_subjects_acc": len(acc_vals),
        }

    print("\n" + hline)
    print("COMPARISON TABLE")
    print(hline)
    print(f"{'variant':10s} {'cafe_d_model':>12s} {'n_folds':>8s} {'eff_rank':>10s} {'% of 512':>9s} "
          f"{'grand acc':>10s} {'acc SD':>8s}")
    for label, s in summary.items():
        print(f"{label:10s} {s['cafe_d_model']:12d} {s['n_folds']:8d} {s['effective_rank_mean']:10.2f} "
              f"{s['effective_rank_pct_of_512']:8.2f}% {s['accuracy_mean']:10.4f} {s['accuracy_sd']:8.4f}")

    print("\nREADING")
    if len(summary) >= 2:
        labels = list(summary.keys())
        ers = [summary[l]["effective_rank_mean"] for l in labels]
        accs = [summary[l]["accuracy_mean"] for l in labels]
        er_range = max(ers) - min(ers)
        acc_range = max(accs) - min(accs)
        print(f"  effective_rank range across variants: {er_range:.2f}  |  accuracy range: {acc_range:.4f}")
        if er_range < 0.5:
            print("  cafe_d_model does NOT meaningfully change the collapse (effective rank barely moves "
                  "across variants). This is an informative NEGATIVE result: the 512-dim collapse is NOT "
                  "caused by this bottleneck -- look instead at CAFE's gated residual combination "
                  "(g_fusion/g_fnirs) or fusion_out_proj's own learned weight structure.")
        else:
            print("  cafe_d_model DOES move the effective rank meaningfully -- worth checking whether "
                  "smaller/larger values also move accuracy in the same or opposite direction (see table).")
        if acc_range < 0.01:
            print("  accuracy is essentially unchanged across cafe_d_model values -- consistent with this "
                  "project's existing priority-4 finding ('result: rejected').")
    else:
        print("  Only one variant given -- nothing to compare yet. Add more --variant entries.")

    with open(os.path.join(out_dir, "cafe_dmodel_ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[written] {out_dir}/cafe_dmodel_ablation_summary.json")


if __name__ == "__main__":
    main()
