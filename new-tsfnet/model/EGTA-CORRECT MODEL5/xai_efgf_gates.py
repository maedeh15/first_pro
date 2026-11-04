"""
xai_efgf_gates.py -- what did the EFGF scalars (gamma, alpha: the residual/gating parameters TSFNet's authors call hard to interpret) become after training?
Checkpoint-only (no data, seconds). For every checkpoint of a results folder, for both EFGF layers:
  gamma = sigmoid(efgf{1,2}.gamma_raw), alpha = sigmoid(efgf{1,2}.alpha_raw)   (gate_mode='sigmoid'; with 'clamp' the raw scalar is clamped to [0, 1])
  ||W|| of the two attention convolutions (eeg_attn_conv, fnirs_attn_conv), descriptive only
Pre-stated reading (before running): a gate has MOVED if its median differs from its initial value sigmoid(gate_init) by more than 0.05 AND the change is the same in >= 80% of the
checkpoints (sign test); a gate is SATURATED if it is below 0.01 or above 0.99 in >= 80% of the checkpoints. A gate that has not moved carries no learned, subject-specific information:
reading it as a "weight of the fusion" is not supported. A gate that has moved by the same amount everywhere is a global constant, not an explanation of single windows.
LIMITS: values are scalars shared by all windows of a model; they say nothing about WHICH input mattered; this is not a faithfulness test of the fusion itself.
  python xai_efgf_gates.py --results-dir "...\\results_cf64a71127_seed42" --gate-init -2.0 --out-dir "...\\efgf_gates"
  python xai_efgf_gates.py --self-test
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sig(x): return 1.0 / (1.0 + np.exp(-x))


def checkpoint_row(path):
    import torch
    from xai_trunk import trunk_state, load_checkpoint
    sd = trunk_state(load_checkpoint(path, map_location="cpu")); row = {}
    for L in (1, 2):
        for g in ("gamma", "alpha"):
            if f"efgf{L}.{g}_raw" in sd: row[f"efgf{L}_{g}"] = float(sig(sd[f"efgf{L}.{g}_raw"].double().item()))
            elif f"efgf{L}.{g}" in sd: row[f"efgf{L}_{g}"] = float(min(max(sd[f"efgf{L}.{g}"].double().item(), 0.0), 1.0))
            else: raise KeyError(f"{path}: no efgf{L}.{g}(_raw) key")
        for nm in ("eeg_attn_conv", "fnirs_attn_conv"):
            w = sd.get(f"efgf{L}.{nm}.conv.weight"); row[f"efgf{L}_{nm}_norm"] = float(w.double().norm()) if w is not None else float("nan")
    return row


def summarise(rows, init):
    out = {}
    for k in [k for k in rows[0] if not k.endswith("_norm")]:
        v = np.array([r[k] for r in rows]); med = float(np.median(v)); diff = v - init
        n_up, n_dn = int((diff > 0.05).sum()), int((diff < -0.05).sum()); n = len(v); same = max(n_up, n_dn) / n
        sat = float(np.mean((v < 0.01) | (v > 0.99)))
        p = float(binomtest(max(n_up, n_dn), n, 0.5).pvalue) if max(n_up, n_dn) > 0 else 1.0
        moved = bool(abs(med - init) > 0.05 and same >= 0.8)
        out[k] = dict(median=med, mean=float(v.mean()), sd=float(v.std(ddof=1)) if n > 1 else float("nan"), min=float(v.min()), max=float(v.max()), init=init, n=n, up_gt_005=n_up, down_gt_005=n_dn,
                      same_direction_share=same, sign_test_p=p, saturated_share=sat, verdict=("MOVED" if moved else "NOT MOVED"))
    return out


def figure(rows, init, out_dir):
    import fig_style as fs; fs.apply(); import matplotlib.pyplot as plt
    keys = [k for k in rows[0] if not k.endswith("_norm")]; fig, ax = plt.subplots(1, 2, figsize=(fs.W2, 2.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for j, k in enumerate(keys):
        v = np.array([r[k] for r in rows]); fs.strip(ax[0], j, v, color=fs.C["blue"], width=0.18); fs.mean_ci(ax[0], j, v, color="k")
    ax[0].axhline(init, ls="--", color="k", lw=0.6); ax[0].text(len(keys) - 0.5, init, f" init {init:.3f}", fontsize=6, va="bottom", ha="right")
    ax[0].set_xticks(range(len(keys))); ax[0].set_xticklabels([k.replace("efgf", "L").replace("_", "\n") for k in keys], fontsize=6.5); ax[0].set_ylabel("sigmoid(gate)"); ax[0].set_ylim(0, max(0.5, max(r[k] for r in rows for k in keys) * 1.2)); fs.panel(ax[0], "a")
    nk = [k for k in rows[0] if k.endswith("_norm")]
    for j, k in enumerate(nk):
        v = np.array([r[k] for r in rows]); fs.strip(ax[1], j, v, color=fs.C["orange"], width=0.18); fs.mean_ci(ax[1], j, v, color="k")
    ax[1].set_xticks(range(len(nk))); ax[1].set_xticklabels([k.replace("efgf", "L").replace("_attn_conv_norm", "\n" + "attn").replace("_", "\n") for k in nk], fontsize=6); ax[1].set_ylabel("Frobenius norm of the attention conv"); fs.panel(ax[1], "b")
    fig.suptitle(f"EFGF gates after training ({len(rows)} checkpoints)", fontsize=8); fig.tight_layout(); return fs.save(fig, out_dir, "fig_efgf_gates")


def self_test():
    import tempfile, torch
    import tsfnet_model_notebook as M
    d = tempfile.mkdtemp(); rng = np.random.RandomState(0); init = float(sig(-2.0)); paths = []
    for i in range(12):
        m = M.TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=21, fnirs_temporal_align="egta", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0)
        with torch.no_grad():
            m.efgf1.gamma_raw.fill_(-2.0 + 0.02 * rng.randn())                 # unchanged gate
            m.efgf1.alpha_raw.fill_(1.5 + 0.1 * rng.randn())                  # moved up everywhere
            m.efgf2.gamma_raw.fill_(-8.0)                                      # saturated low
            m.efgf2.alpha_raw.fill_(float(rng.choice([-3.0, 0.5])))            # moved in both directions
        p = os.path.join(d, f"VP{i:03d}_session1_best.pt"); torch.save(m.state_dict(), p); paths.append(p)
    rows = [checkpoint_row(p) for p in paths]; s = summarise(rows, init)
    assert s["efgf1_gamma"]["verdict"] == "NOT MOVED" and s["efgf1_alpha"]["verdict"] == "MOVED" and s["efgf2_gamma"]["saturated_share"] == 1.0
    assert s["efgf2_gamma"]["verdict"] == "MOVED" and s["efgf2_alpha"]["verdict"] == "NOT MOVED", s["efgf2_alpha"]
    print("[1] unchanged gate -> NOT MOVED; shifted gate -> MOVED; saturated gate detected; gate that goes both ways -> NOT MOVED (not a global constant)")
    print("[2] figure:", figure(rows, init, d)); print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir"); ap.add_argument("--gate-init", type=float, default=-2.0, help="efgf gate_init used in training (the final model: -2.0)"); ap.add_argument("--out-dir", default=None); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    if not a.results_dir: sys.exit("need --results-dir")
    paths = sorted(glob.glob(os.path.join(a.results_dir, "VP*_session*_best.pt"))); rows = []; names = []
    for p in paths:
        try: rows.append(checkpoint_row(p)); names.append(os.path.basename(p))
        except Exception as e: print(f"SKIP {os.path.basename(p)} -- {type(e).__name__}: {e}")
    if len(rows) < 6: sys.exit(f"{len(rows)} usable checkpoints")
    init = float(sig(a.gate_init)); s = summarise(rows, init)
    print(f"\nEFGF GATES after training, {len(rows)} checkpoints; initial value sigmoid({a.gate_init}) = {init:.4f}")
    print(f"  {'gate':16s} {'median':>8s} {'mean±sd':>14s} {'min..max':>14s} {'up/down >0.05':>14s} {'saturated':>10s}  verdict")
    for k, v in s.items(): print(f"  {k:16s} {v['median']:8.4f} {v['mean']:6.4f}±{v['sd']:.4f} {v['min']:6.4f}..{v['max']:.4f} {v['up_gt_005']:>6d}/{v['down_gt_005']:<6d} {100 * v['saturated_share']:9.0f}%  {v['verdict']}")
    print("  reading: MOVED = same shift in >= 80% of checkpoints (a global constant, still not a per-window explanation); NOT MOVED = no learned information in the scalar.")
    od = a.out_dir or os.path.join(a.results_dir, "efgf_gates"); os.makedirs(od, exist_ok=True); json.dump(s, open(os.path.join(od, "efgf_gates_summary.json"), "w"), indent=2)
    with open(os.path.join(od, "efgf_gates_per_checkpoint.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["checkpoint"] + list(rows[0])); [w.writerow([n] + [r[k] for k in rows[0]]) for n, r in zip(names, rows)]
    print("figure:", figure(rows, init, od))


if __name__ == "__main__":
    main()
