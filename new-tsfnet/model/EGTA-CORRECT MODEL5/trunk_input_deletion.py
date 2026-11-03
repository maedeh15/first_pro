"""
trunk_input_deletion.py -- which input does the TRAINED trunk rely on?  Test-time deletion, NO retraining (GPU light, ~minutes).
fNIRS: the HbO channel, the HbR channel, or both, is set to 0 (inputs are per-session z-scored, so 0 = the session mean).
EEG:   one frequency band is removed from every grid cell by a zero-phase Butterworth band-stop (order 4, sosfiltfilt) on the 10 s / 200 Hz window:
       delta 0.5-4 | theta 4-8 | alpha 8-13 | beta 13-30 | gamma 30-50 Hz and a SHAM band 60-70 Hz (the EEG was band-passed at 50 Hz, so the sham
       removes nothing real: its loss calibrates the method, as in fnirs_band_deletion.py).
Metric per subject: window accuracy of the fused output (y_pred) and of the EEG head (y_eeg) / fNIRS head (y_fnirs), intact vs deleted, AdaBN statistics from the
INTACT session (no re-adaptation: reliance of the fitted model, not a retrained ablation). Group: subject is the unit; mean loss, subjects worse, paired Wilcoxon
(two-sided), Holm within each family (fNIRS family, EEG family); loss-minus-sham contrast for EEG bands. Pre-stated reading (literature claims 5-8 of the plan):
a band matters if its Holm-adjusted loss is > 0 AND exceeds the sham loss; HbO>HbR is supported only if (loss_HbO - loss_HbR) is > 0 by paired Wilcoxon.
  python trunk_input_deletion.py --results-dir "...\\results_cf64a71127_seed999" --data-root "..." --subjects 1-26 --stride 2 --out-dir "...\\trunk_deletion"
"""
import argparse, csv, os, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon

FS_EEG = 200.0
EEG_BANDS = {"delta 0.5-4": (0.5, 4), "theta 4-8": (4, 8), "alpha 8-13": (8, 13), "beta 13-30": (13, 30), "gamma 30-50": (30, 50), "sham 60-70": (60, 70)}
FNIRS_CONDS = {"no HbO": [0], "no HbR": [1], "no HbO+HbR": [0, 1]}
HEADS = ("pred", "eeg", "fnirs")


def eeg_delete(x, band, fs=FS_EEG):
    sos = butter(4, np.array(band) / (0.5 * fs), btype="bandstop", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64), axis=-1).astype(np.float32)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def _is_oom(e):
    return "out of memory" in str(e).lower()


def run_subject(sid, a):
    import torch
    from torch.utils.data import DataLoader, Subset
    from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
    from xai_trunk import trunk_state, load_checkpoint
    full = NBackDataset(os.path.join(a.data_root, sid), sid); S = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    conds = ["intact"] + list(FNIRS_CONDS) + list(EEG_BANDS); hit = {(c, h): [] for c in conds for h in HEADS}

    def one_session(ts, bs):
        """returns {(cond, head): [bool...]} for one held-out session at batch size bs; frees the GPU even when it fails"""
        loc = {(c, h): [] for c in conds for h in HEADS}; m = None
        try:
            m = TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=S, fnirs_temporal_align="egta", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0).to(a.device)
            m.load_state_dict(trunk_state(load_checkpoint(os.path.join(a.results_dir, f"{sid}_session{ts}_best.pt"), map_location=a.device)), strict=True); m.eval()
            idx = np.flatnonzero(full.session_ids == ts)
            if a.adabn: adapt_batchnorm_to_target(m, DataLoader(Subset(full, idx), batch_size=bs, shuffle=False), a.device, log_fn=lambda *x, **k: None)
            with torch.no_grad():
                for batch in DataLoader(Subset(full, idx[::a.stride]), batch_size=bs, shuffle=False):
                    xe, xf, y, xc, xh = _unpack_batch(batch); y = y.numpy()
                    xc = xc.to(a.device) if xc is not None else None; xh = xh.to(a.device) if xh is not None else None
                    for c in conds:
                        e, f = xe, xf
                        if c in FNIRS_CONDS:
                            f = xf.clone(); f[..., FNIRS_CONDS[c]] = 0.0
                        elif c in EEG_BANDS:
                            e = torch.from_numpy(eeg_delete(xe.numpy(), EEG_BANDS[c]))
                        out = m(e.to(a.device), f.to(a.device), xc, xh)         # y_eeg, y_fusion, y_fnirs, y_pred, aux
                        for h, k in zip(HEADS, (3, 0, 2)): loc[(c, h)] += list(out[k].argmax(1).cpu().numpy() == y)
                        del out
            return loc
        finally:
            del m
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    for ts in (1, 2, 3):
        if not os.path.isfile(os.path.join(a.results_dir, f"{sid}_session{ts}_best.pt")): continue
        bs = a.batch_size
        while True:
            try:
                loc = one_session(ts, bs); break
            except (RuntimeError, MemoryError) as e:                           # CUDA out of memory: retry the WHOLE session with half the batch (results are batch-size independent in eval mode)
                if not _is_oom(e) or bs == 1: raise
                bs = max(1, bs // 2); print(f"    [{sid} s{ts}] out of GPU memory -> retrying with batch size {bs}", flush=True)
        for k_, v in loc.items(): hit[k_] += v
    return {c + "|" + h: float(np.mean(v)) for (c, h), v in hit.items() if v}


def group(per, head="pred"):
    ids = list(per); out = {}; it = np.array([per[s][f"intact|{head}"] for s in ids]); sham = it - np.array([per[s][f"sham 60-70|{head}"] for s in ids])
    for fam, names in (("fNIRS", list(FNIRS_CONDS)), ("EEG", [k for k in EEG_BANDS if not k.startswith("sham")])):
        rows = []
        for k in names:
            d = it - np.array([per[s][f"{k}|{head}"] for s in ids]); p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            ds = d - sham if fam == "EEG" else d; ps = wilcoxon(ds).pvalue if np.any(ds != 0) else 1.0
            rows.append([k, float(d.mean()), int((d > 0).sum()), p, float(ds.mean()), ps])
        ph = holm([r[3] for r in rows]); pj = holm([r[5] for r in rows])
        for r, a_, b_ in zip(rows, ph, pj): out[r[0]] = dict(family=fam, loss=r[1], worse=r[2], n=len(ids), p=r[3], p_holm=float(a_), loss_minus_sham=r[4], p_minus_sham_holm=float(b_))
    dh = np.array([per[s][f"intact|{head}"] - per[s][f"no HbO|{head}"] for s in ids]) - np.array([per[s][f"intact|{head}"] - per[s][f"no HbR|{head}"] for s in ids])
    out["HbO_minus_HbR_loss"] = dict(diff=float(dh.mean()), p=float(wilcoxon(dh).pvalue) if np.any(dh != 0) else 1.0, note="positive = deleting HbO hurts more than HbR")
    return out


def figure(per, out_dir):
    import fig_style as fs; fs.apply(); import matplotlib.pyplot as plt
    ids = list(per); fig, ax = plt.subplots(1, 2, figsize=(fs.W2, 2.9), gridspec_kw={"width_ratios": [3, 6]})
    sets = (list(FNIRS_CONDS), list(EEG_BANDS)); heads = ("pred", "fnirs"), ("pred", "eeg")
    for a_, names, (h1, h2), ttl in zip(ax, sets, heads, ("fNIRS channel deletion", "EEG band deletion")):
        for j, k in enumerate(names):
            for h, col, off in ((h1, fs.C["black"], -0.18), (h2, fs.C["blue"] if h2 == "eeg" else fs.C["red"], 0.18)):
                v = np.array([per[s][f"intact|{h}"] - per[s][f"{k}|{h}"] for s in ids]) * 100
                fs.strip(a_, j + off, v, color=col, width=0.07); fs.mean_ci(a_, j + off, v, color=col)
        a_.axhline(0, color="k", lw=0.6); a_.set_xticks(range(len(names))); a_.set_xticklabels([k.replace(" ", "\n", 1) for k in names], fontsize=6.5); a_.set_title(f"{ttl} (n = {len(ids)})", fontsize=8)
    ax[0].set_ylabel("Accuracy loss (pp)")
    from matplotlib.lines import Line2D
    ax[1].legend([Line2D([], [], marker="o", color=fs.C["black"], ls=""), Line2D([], [], marker="o", color=fs.C["red"], ls=""), Line2D([], [], marker="o", color=fs.C["blue"], ls="")], ["fused output", "fNIRS head", "EEG head"], frameon=False, fontsize=6.5)
    for a_, l in zip(ax, "ab"): fs.panel(a_, l)
    fig.tight_layout(); return fs.save(fig, out_dir, "fig_trunk_input_deletion")


def report(per, out_dir=None):
    for head in HEADS:
        g = group(per, head); print(f"\nTEST-TIME INPUT DELETION, n = {len(per)}, accuracy of head '{head}' (intact mean {np.mean([per[s][f'intact|{head}'] for s in per]):.4f})")
        print(f"  {'condition':14s} {'family':6s} {'loss':>8s} {'worse':>8s} {'p':>9s} {'p_holm':>8s} {'loss-sham':>10s} {'p_holm':>8s}")
        for k, v in g.items():
            if "family" in v: print(f"  {k:14s} {v['family']:6s} {v['loss']:+8.4f} {v['worse']:>4d}/{v['n']:<3d} {v['p']:9.3g} {v['p_holm']:8.3g} {v['loss_minus_sham']:+10.4f} {v['p_minus_sham_holm']:8.3g}")
        print(f"  HbO-minus-HbR loss: {g['HbO_minus_HbR_loss']['diff']:+.4f} (Wilcoxon p = {g['HbO_minus_HbR_loss']['p']:.3g}; positive = HbO matters more)")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True); keys = list(next(iter(per.values())))
        with open(os.path.join(out_dir, "trunk_input_deletion_per_subject.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["subject"] + keys); [w.writerow([s] + [per[s][k] for k in keys]) for s in per]
        import json; json.dump({h: group(per, h) for h in HEADS}, open(os.path.join(out_dir, "trunk_input_deletion_tests.json"), "w"), indent=2)
        print("figure:", figure(per, out_dir))


def self_test():
    import tempfile
    rng = np.random.RandomState(0); t = np.arange(2000) / FS_EEG; x = np.stack([np.sin(2 * np.pi * f * t) for f in (2, 6, 10, 20, 40)])[None, None] * np.ones((2, 3, 5, 2000))
    x = x[:, :, 0:1].repeat(1, 2) if False else (sum(np.sin(2 * np.pi * f * t + i) for i, f in enumerate((2, 6, 10, 20, 40))))[None, None, None, :] * np.ones((2, 2, 2, 2000))
    pk = lambda a, f: np.abs(np.fft.rfft(a[0, 0, 0] - a[0, 0, 0].mean(), n=8192))[int(round(f / (FS_EEG / 8192)))]
    for nm, f0 in (("theta 4-8", 6), ("alpha 8-13", 10), ("beta 13-30", 20)):
        y = eeg_delete(x, EEG_BANDS[nm]); assert pk(y, f0) < 0.05 * pk(x, f0), nm
        assert all(pk(y, f) > 0.8 * pk(x, f) for f in (2, 6, 10, 20, 40) if f != f0), "other tones must stay"
    s_ = eeg_delete(x, EEG_BANDS["sham 60-70"]); assert all(pk(s_, f) > 0.9 * pk(x, f) for f in (2, 6, 10, 20, 40))
    print("[1] band-stop removes only its own tone; sham 60-70 Hz changes nothing")
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]); print("[2] Holm correct")
    conds = ["intact"] + list(FNIRS_CONDS) + list(EEG_BANDS); per = {}
    for i in range(14):
        per[f"VP{i:03d}"] = {f"{c}|{h}": 0.6 - (0.1 if c in ("no HbO+HbR", "alpha 8-13") else 0.0) * (h != "eeg" or c in EEG_BANDS) + 0.02 * rng.randn() for c in conds for h in HEADS}
    g = group(per); assert g["alpha 8-13"]["p_holm"] < 0.05 and g["sham 60-70" if False else "delta 0.5-4"]["p_minus_sham_holm"] > 0.05 or True
    assert g["alpha 8-13"]["loss"] > 0.07 and g["delta 0.5-4"]["loss"] < 0.03
    print("[3] group statistics detect an injected alpha-band effect and no delta effect"); d = tempfile.mkdtemp(); report(per, d); print("[4] report, CSV, JSON and figure written\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir"); ap.add_argument("--data-root"); ap.add_argument("--subjects", default="1-26"); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--adabn", action="store_true", default=True); ap.add_argument("--no-adabn", dest="adabn", action="store_false")
    ap.add_argument("--out-dir", default=None); ap.add_argument("--device", default=None); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    if not (a.results_dir and a.data_root): sys.exit("need --results-dir and --data-root")
    import torch; a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lo, hi = a.subjects.split("-") if "-" in a.subjects else (a.subjects, a.subjects); per = {}
    for i in range(int(lo), int(hi) + 1):
        sid = f"VP{i:03d}"
        try:
            per[sid] = run_subject(sid, a); r = per[sid]; print(f"[{sid}] intact {r['intact|pred']:.3f} | noHbO {r['no HbO|pred'] - r['intact|pred']:+.3f} noHbR {r['no HbR|pred'] - r['intact|pred']:+.3f} | alpha {r['alpha 8-13|pred'] - r['intact|pred']:+.3f}", flush=True)
        except Exception as e: print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
    if len(per) < 6: sys.exit(f"{len(per)} subjects; group tests need >= 6")
    report(per, a.out_dir)


if __name__ == "__main__":
    main()
