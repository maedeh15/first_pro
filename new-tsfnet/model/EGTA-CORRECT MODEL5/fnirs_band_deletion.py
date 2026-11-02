"""
fnirs_band_deletion.py -- how much does the TRAINED model rely on each fNIRS frequency band? (test-time deletion, per-band accuracy loss table)
For every held-out window the S overlapping 3 s segments are rebuilt into one continuous trace (23 s for S = 21), a zero-phase Butterworth filter
(order 2, sosfiltfilt) removes ONE band, the trace is cut back into the same segments, and the unchanged trained trunk classifies it (AdaBN statistics
from the INTACT test session, no re-adaptation: this measures reliance of the fitted model, it is not a retrained ablation).
Bands: cardiac 0.8-2.0 Hz | respiration/Mayer 0.1-0.4 Hz | 0.4-0.8 Hz | above 2.0 Hz (low-pass 2 Hz) | below 0.1 Hz (high-pass 0.1 Hz) and a SHAM band 3.5-4.5 Hz
(almost no power: its deletion should cost ~0, which calibrates the method). Per subject: accuracy intact vs deleted; group: mean loss, subjects worse,
paired Wilcoxon (two-sided, exploratory, uncorrected), plus a loss-vs-sham contrast.
LIMITS: a 23 s window resolves 0.043 Hz, so "below 0.1 Hz" here is only the part of the slow band that fits in a window; true drift (<0.01 Hz) needs the
retraining runs of the preprocessing script (--fnirs-bandpass hp001 / lp001 ...). Filter edge effects on a 23 s trace exist for the lowest bands.
  python fnirs_band_deletion.py --results-dir "...\\results_cf64a71127_seed999" --data-root "..." --subjects 1-26 --stride 2
"""
import argparse, os, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon

STEP, SEG, FS = 10, 30, 10.0
BANDS = {"cardiac 0.8-2.0": ("bandstop", (0.8, 2.0)), "resp/Mayer 0.1-0.4": ("bandstop", (0.1, 0.4)), "0.4-0.8": ("bandstop", (0.4, 0.8)),
         "above 2.0": ("lowpass", 2.0), "below 0.1": ("highpass", 0.1), "sham 3.5-4.5": ("bandstop", (3.5, 4.5))}


def to_trace(x):
    """x (B, S, H, W, T, C) -> (B, H, W, L, C)"""
    B, S, H, W, T, C = x.shape; L = (S - 1) * STEP + T
    tr = np.zeros((B, H, W, L, C), dtype=np.float64)
    for k in range(S):
        tr[:, :, :, k * STEP:k * STEP + T, :] = x[:, k]
    return tr


def to_segments(tr, S, T=SEG):
    return np.stack([tr[:, :, :, k * STEP:k * STEP + T, :] for k in range(S)], axis=1)


def delete_band(x, spec):
    kind, w = spec; S = x.shape[1]
    sos = butter(2, np.array(w) / (0.5 * FS), btype=kind, output="sos")
    return to_segments(sosfiltfilt(sos, to_trace(x), axis=3), S).astype(np.float32)


def self_test():
    rng = np.random.RandomState(0); S = 21; L = (S - 1) * STEP + SEG; t = np.arange(L) / FS
    tone = lambda f: np.sin(2 * np.pi * f * t + rng.uniform(0, 6))
    tr = np.stack([tone(0.3) + tone(1.2) + tone(4.0) for _ in range(2)])[:, None, None, :, None] * np.ones((2, 2, 2, L, 2))
    x = to_segments(tr, S).astype(np.float32)
    assert np.allclose(to_trace(x), tr, atol=1e-5), "segment <-> trace round trip"
    y = to_trace(delete_band(x, BANDS["cardiac 0.8-2.0"]))
    sp = lambda a: np.abs(np.fft.rfft(a[0, 0, 0, :, 0] - a[0, 0, 0, :, 0].mean(), n=2048)); f = np.fft.rfftfreq(2048, 1 / FS); pk = lambda s_, fr: s_[np.argmin(abs(f - fr))]
    assert pk(sp(y), 1.2) < 0.1 * pk(sp(tr), 1.2) and pk(sp(y), 0.3) > 0.6 * pk(sp(tr), 0.3), "the cardiac tone must go, the 0.3 Hz tone must stay"
    z = to_trace(delete_band(x, BANDS["sham 3.5-4.5"])); assert pk(sp(z), 1.2) > 0.7 * pk(sp(tr), 1.2) and pk(sp(z), 4.0) < 0.3 * pk(sp(tr), 4.0)
    print("[1] segment/trace round trip exact; the cardiac band removes the 1.2 Hz tone and keeps 0.3 Hz; the sham band removes only the 4 Hz tone")
    print("SELFTEST PASSED")


def run_subject(sid, a):
    import torch
    from torch.utils.data import DataLoader, Subset
    from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
    from xai_trunk import trunk_state, load_checkpoint
    full = NBackDataset(os.path.join(a.data_root, sid), sid); S = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    hit = {k: [] for k in ["intact"] + list(BANDS)}
    for ts in (1, 2, 3):
        ck = os.path.join(a.results_dir, f"{sid}_session{ts}_best.pt")
        if not os.path.isfile(ck): continue
        m = TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=S, fnirs_temporal_align="egta", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0).to(a.device)
        m.load_state_dict(trunk_state(load_checkpoint(ck, map_location=a.device)), strict=True); m.eval()
        idx = np.flatnonzero(full.session_ids == ts)
        if a.adabn: adapt_batchnorm_to_target(m, DataLoader(Subset(full, idx), batch_size=a.batch_size, shuffle=False), a.device, log_fn=lambda *x, **k: None)
        dl = DataLoader(Subset(full, idx[::a.stride]), batch_size=a.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in dl:
                xe, xf, y, xc, xh = _unpack_batch(batch); xe = xe.to(a.device); y = y.numpy()
                xc = xc.to(a.device) if xc is not None else None; xh = xh.to(a.device) if xh is not None else None
                xfn = xf.numpy().astype(np.float64)
                for name in hit:
                    xi = xf if name == "intact" else torch.from_numpy(delete_band(xfn, BANDS[name]))
                    pred = m(xe, xi.to(a.device), xc, xh)[3].argmax(1).cpu().numpy(); hit[name] += list(pred == y)
        del m
    return {k: float(np.mean(v)) for k, v in hit.items() if v}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir"); ap.add_argument("--data-root"); ap.add_argument("--subjects", default="1-26"); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--adabn", action="store_true", default=True); ap.add_argument("--no-adabn", dest="adabn", action="store_false")
    ap.add_argument("--device", default=None); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    if not (a.results_dir and a.data_root): sys.exit("need --results-dir and --data-root")
    import torch; a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2] if "-" in a.subjects else (a.subjects, a.subjects); per = {}
    for i in range(int(lo), int(hi) + 1):
        sid = f"VP{i:03d}"
        try:
            per[sid] = run_subject(sid, a); print(f"[{sid}] intact {per[sid]['intact']:.3f} | " + " ".join(f"{k.split()[0]} {per[sid][k] - per[sid]['intact']:+.3f}" for k in BANDS if k in per[sid]), flush=True)
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
    if len(per) < 6: sys.exit(f"{len(per)} subjects; group tests need >= 6")
    ids = list(per); intact = np.array([per[s]["intact"] for s in ids]); sham = np.array([per[s]["sham 3.5-4.5"] for s in ids]) - intact
    print(f"\nTEST-TIME BAND DELETION, {len(ids)} subjects (window accuracy of the trained trunk; intact mean {intact.mean():.4f})")
    print(f"  {'band':22s} {'mean loss':>10s} {'subjects worse':>15s} {'Wilcoxon p':>11s} {'loss minus sham':>16s} {'p':>8s}")
    for k in BANDS:
        d = intact - np.array([per[s][k] for s in ids]); pw = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        ds = d + sham; ps = wilcoxon(ds).pvalue if (k != "sham 3.5-4.5" and np.any(ds != 0)) else float("nan")
        print(f"  {k:22s} {d.mean():+10.4f} {int((d > 0).sum()):>9d}/{len(d):<5d} {pw:11.3g} {ds.mean():+16.4f} {ps:8.3g}")
    print("  loss = intact minus deleted accuracy (positive = the model used that band). Two-sided Wilcoxon, exploratory, uncorrected. The sham row calibrates the method.")


if __name__ == "__main__":
    main()
