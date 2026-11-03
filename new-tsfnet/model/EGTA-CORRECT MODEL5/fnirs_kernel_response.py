"""
fnirs_kernel_response.py -- frequency response of the model's OWN first-layer fNIRS temporal kernels (CPU, from checkpoints).
Tests the claim "the first convolution acts like a low-pass filter" directly. fnirs_branch.conv1 has a (4,4,6) kernel per (output channel, chromophore):
6 temporal taps at 10 Hz = 0.6 s. For a spatially uniform input the temporal kernel is k[o,c,:] = sum over the 4x4 spatial taps; its gain is |FFT(k)|.
Reported for the trained kernels and for a random-initialisation null of the same shape (N(0, 1/fan_in), 300 draws), per frequency:
mean gain relative to the gain at 1 Hz, and the share of kernels whose gain at 0.1 Hz exceeds that at 1 Hz ("low-pass-like"), plus the share of squared gain above 0.1 Hz.
A 6-tap kernel cannot be a sharp filter (frequency step 1.7 Hz without zero padding), and the layer is followed by ELU/BatchNorm/stride 2 (aliasing above 2.5 Hz),
so this is a description of the linear part only, not of the whole layer. Both outcomes (low-pass-like or not) are reportable.
  python fnirs_kernel_response.py --results-dir "...\\results_cf64a71127_seed999" --subjects 1-26
"""
import argparse, glob, os, sys
import numpy as np

FS = 10.0
FREQS = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0)


def temporal_kernels(w):
    """w (out, in, kh, kw, T) -> (out*in, T): sum over the spatial taps"""
    return w.sum(axis=(2, 3)).reshape(-1, w.shape[-1])


def gain(k, n=2048):
    f = np.fft.rfftfreq(n, d=1.0 / FS); return f, np.abs(np.fft.rfft(k, n=n, axis=-1))


def describe(k):
    f, g = gain(k); at = lambda x: g[:, np.argmin(np.abs(f - x))]
    rel = np.stack([at(x) / (at(1.0) + 1e-12) for x in FREQS], 1)                       # (kernels, freqs), gain relative to 1 Hz
    lowpass = float(np.mean(at(0.1) > at(1.0)))
    hi = float(np.mean(((g[:, f > 0.1] ** 2).sum(1)) / ((g[:, f > 0] ** 2).sum(1) + 1e-30)))
    return dict(rel=np.median(rel, 0), lowpass=lowpass, high_share=hi)


def load_kernels(results_dir, subjects):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from xai_trunk import load_checkpoint, trunk_state
    ks = []
    for sid in subjects:
        for p in sorted(glob.glob(os.path.join(results_dir, f"{sid}_session*_best.pt"))):
            sd = trunk_state(load_checkpoint(p, map_location="cpu"))
            key = next((k for k in sd if k.endswith("fnirs_branch.conv1.conv.conv.weight")), None)
            if key is None: raise SystemExit(f"{p}: no fnirs_branch.conv1 weight found")
            ks.append(sd[key].numpy().astype(np.float64))
    return ks


def figure(trained, rng, out_dir, n_null=200):
    """gain vs frequency (log x), trained kernels (median, IQR) against random-init null (median, 5-95%); inset-free, thesis-ready"""
    import fig_style as fs; fs.apply(); import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True); shp = trained[0].shape; fan = int(np.prod(shp[1:]))
    f, g = gain(temporal_kernels(np.concatenate(trained, 0))); m = (f > 0) & (f <= 5)
    gn = np.concatenate([gain(temporal_kernels(rng.randn(*shp) / np.sqrt(fan)))[1] for _ in range(n_null)], 0)
    ref = lambda G: G[:, np.argmin(np.abs(f - 1.0))][:, None] + 1e-12
    fig, ax = plt.subplots(1, 2, figsize=(fs.W2, 2.8))
    for G, col, lab in ((gn / ref(gn), fs.C["grey"], "random-init null"), (g / ref(g), fs.C["blue"], "trained")):
        lo, md, hi = np.percentile(G[:, m], [25, 50, 75], axis=0); ax[0].fill_between(f[m], lo, hi, color=col, alpha=0.2, lw=0); ax[0].plot(f[m], md, color=col, label=lab)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("Frequency (Hz)"); ax[0].set_ylabel("Gain relative to 1 Hz (median, IQR)"); ax[0].legend(frameon=False); fs.panel(ax[0], "a")
    at = lambda G, x: G[:, np.argmin(np.abs(f - x))]
    r_tr = np.log10((at(g, 0.1) + 1e-12) / (at(g, 1.0) + 1e-12)); r_nl = np.log10((at(gn, 0.1) + 1e-12) / (at(gn, 1.0) + 1e-12)); bins = np.linspace(-1.5, 1.5, 31)
    ax[1].hist(r_nl, bins, color=fs.C["grey"], alpha=0.6, density=True, label="null"); ax[1].hist(r_tr, bins, color=fs.C["blue"], alpha=0.6, density=True, label="trained")
    ax[1].axvline(0, color="k", lw=0.6); ax[1].set_xlabel("log10 [gain(0.1 Hz) / gain(1 Hz)]"); ax[1].set_ylabel("Density"); ax[1].legend(frameon=False); fs.panel(ax[1], "b")
    fig.tight_layout(); print("figure:", fs.save(fig, out_dir, "fig_kernel_response"))


def report(trained, rng, n_null=300):
    shp = trained[0].shape; fan = int(np.prod(shp[1:]))
    K = temporal_kernels(np.concatenate(trained, 0)); dt = describe(K)
    null = [describe(temporal_kernels(rng.randn(*shp) / np.sqrt(fan))) for _ in range(n_null)]
    nl = np.mean([d["lowpass"] for d in null]); nh = np.mean([d["high_share"] for d in null]); nr = np.mean([d["rel"] for d in null], 0)
    print(f"\nfirst-layer fNIRS temporal kernels: {len(trained)} checkpoints x {shp[0]*shp[1]} (out,chromophore) kernels, {shp[-1]} taps = {shp[-1]/FS:.1f} s")
    print("  median gain relative to 1 Hz   " + "  ".join(f"{x:>5g}Hz" for x in FREQS))
    print("    trained                      " + "  ".join(f"{v:7.2f}" for v in dt["rel"]))
    print("    random-init null (mean)      " + "  ".join(f"{v:7.2f}" for v in nr))
    print(f"  share of kernels with gain(0.1 Hz) > gain(1 Hz): trained {dt['lowpass']:.2f} | random null {nl:.2f}")
    print(f"  share of squared gain above 0.1 Hz:            trained {dt['high_share']:.3f} | random null {nh:.3f}")
    verdict = "low-pass-like beyond chance (share of low-pass kernels exceeds the random null by > 0.15)" if dt["lowpass"] > nl + 0.15 else "NOT clearly low-pass: the learned linear part passes the 1-2 Hz range about as the random null does"
    print("  reading: " + verdict + " (descriptive; 6 taps cannot be sharp; ELU/BN/stride-2 follow)")
    return dt, dict(lowpass=nl, high_share=nh, rel=nr)


def self_test():
    rng = np.random.RandomState(0); T = 6
    lp = np.ones((16, 2, 4, 4, T)) / (16 * T) + 0.01 * rng.randn(16, 2, 4, 4, T)          # moving average = low-pass
    d = describe(temporal_kernels(lp)); assert d["lowpass"] > 0.95 and d["rel"][2] > 1.0, d
    hp = lp * (-1.0) ** np.arange(T); d2 = describe(temporal_kernels(hp)); assert d2["lowpass"] < 0.2, d2
    print("[1] a moving-average kernel is classified low-pass, an alternating-sign kernel is not")
    report([lp], rng, 50); import tempfile; figure([lp], rng, tempfile.mkdtemp(), 20); print("[2] report and figure run\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir"); ap.add_argument("--subjects", default="1-26"); ap.add_argument("--out-dir", default=None); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    if not a.results_dir: sys.exit("need --results-dir")
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2] if "-" in a.subjects else (a.subjects, a.subjects)
    ks = load_kernels(a.results_dir, [f"VP{i:03d}" for i in range(int(lo), int(hi) + 1)])
    if not ks: sys.exit("no checkpoints found")
    report(ks, np.random.RandomState(0))
    if a.out_dir: figure(ks, np.random.RandomState(1), a.out_dir)


if __name__ == "__main__":
    main()
