"""
xai_classic_plots.py -- the standard figure formats for Grad-CAM and SHAP, on the trained models. Subject-level examples + group summaries; thesis style (fig_style.py).

  collect   (needs torch/captum + checkpoints; per subject, random windows of each held-out session, target = predicted class, AdaBN as everywhere else)
            writes classic_<sid>.npz:  y, pred, p (P of the target class), cam_seg (N,S), occ (N,S) occlusion impact, cam_map (N,S,h,w) Grad-CAM spatial map per segment,
            shap_seg (N,S) SIGNED GradientShap summed over each segment, x_lvl (N,S) mean input level of the segment, base (N,) E[f(baseline)], fx (N,) = f(x)
  plots     Grad-CAM : example window and class mean per predicted class (segment profile with occlusion impact + spatial map)             -> fig_gradcam
            GradientShap : mean|phi| per segment, beeswarm (colour = segment input level, SHAP convention), waterfall of one window       -> fig_shap_trunk
            exact member SHAP (from member_shap.npz of xai_member_shap.py) : band shares, power vs coupling share, channel x band map, deletion check -> fig_shap_member
Implementation notes (state them in the Methods; they are adaptations, not the reference code):
  * Grad-CAM follows Selvaraju et al. 2017 (alpha = spatial-temporal mean of the gradient of log P(c) w.r.t. the activation, cam = ReLU(sum_k alpha_k A_k)) applied
    to fnirs_branch.conv2 per 3 s segment; the reference 2-D implementation is e.g. github.com/jacobgil/pytorch-grad-cam. Maps are normalised per window only for display.
  * GradientShap is captum's estimator of SHAP's expected gradients (Lundberg & Lee 2017; Erion et al. 2021): baselines = other windows of the SAME held-out session,
    stdevs 0.05, few samples (it is an ESTIMATE; the attribution sums to f(x) - f(baseline) only approximately, the waterfall shows the residual). The plot formats follow
    the shap package (github.com/shap-project/shap or github.com/shap/shap: beeswarm and waterfall).
  * The spatial axes are the interpolated 16x16 grid pooled by the conv layers: NO claim of a cortical region is made (channel identity is not verified).
  python xai_classic_plots.py collect --results-dir "..." --data-root "..." --subjects 1-3 --out-dir "...\\classic"
  python xai_classic_plots.py plot --out-dir "...\\classic" --subject-id VP001 [--member-shap "...\\member_shap.npz"]
  python xai_classic_plots.py --self-test
"""
import argparse, glob, os, sys
import numpy as np
import fig_style as fs

CLS = {0: "0-back", 1: "2-back", 2: "3-back"}
CC = {0: fs.C["blue"], 1: fs.C["orange"], 2: fs.C["red"]}


# ------------------------------------------------------------------ compute (torch)
def explain_spatial(model, xe, xf, xc, xh, n_bg=6, shap_samples=6):
    import torch
    from xai_validated_maps import pc
    with torch.no_grad():
        yp0 = model(xe, xf, xc, xh)[3]
    c = yp0.argmax(1); p0 = pc(yp0, c); B, S = xf.shape[:2]
    store = {}; h = model.fnirs_branch.conv2.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    xg = xf.clone().requires_grad_(True); yp = model(xe, xg, xc, xh)[3]; h.remove()
    g, ga = torch.autograd.grad(torch.log(pc(yp, c).clamp_min(1e-8)).sum(), [xg, store["a"]])
    a = store["a"].detach(); alpha = ga.detach().mean(dim=tuple(range(2, ga.dim())), keepdim=True)
    cam = torch.relu((alpha * a).sum(dim=1))                                   # (B*S, h, w, T')
    cam_map = cam.mean(-1).reshape(B, S, *cam.shape[1:3]).cpu().numpy(); cam_seg = cam.mean(dim=tuple(range(1, cam.dim()))).reshape(B, S).cpu().numpy()
    imp = np.zeros((B, S))
    with torch.no_grad():
        for s in range(S):
            xa = xf.clone(); xa[:, s] = 0.0; imp[:, s] = (p0 - pc(model(xe, xa, xc, xh)[3], c)).cpu().numpy()
    from captum.attr import GradientShap
    class _F(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x, e, k, q): return s.m(e, x, k, q)[3]
    bg = xf[torch.randperm(B, device=xf.device)[:min(n_bg, B)]].detach()
    CHUNK = 6; parts, left = [], shap_samples                                   # chunked: captum copies the whole input (EEG too) n_samples times
    while left > 0:
        n_ = min(CHUNK, left); parts.append((n_, GradientShap(_F(model)).attribute(xf.detach(), baselines=bg, n_samples=n_, stdevs=0.05, target=c, additional_forward_args=(xe, xc, xh)).detach())); left -= n_
    at = sum(n_ * a_ for n_, a_ in parts) / shap_samples
    shap_seg = at.detach().sum(dim=tuple(range(2, at.dim()))).cpu().numpy()
    with torch.no_grad():                                                      # E[f(baseline)]: P(target) with the window's own EEG and each baseline fNIRS window
        base = torch.stack([pc(model(xe, bg[j:j + 1].expand_as(xf), xc, xh)[3], c) for j in range(len(bg))]).mean(0).cpu().numpy()
    return dict(pred=c.cpu().numpy(), p=p0.detach().cpu().numpy(), cam_seg=cam_seg, occ=imp, cam_map=cam_map, shap_seg=shap_seg, x_lvl=xf.detach().mean(dim=(2, 3, 4)).mean(-1).cpu().numpy(), base=base, fx=p0.detach().cpu().numpy())


def collect_subject(sid, a):
    import torch
    from torch.utils.data import DataLoader, Subset
    from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
    from xai_trunk import trunk_state, load_checkpoint, set_xai_output
    full = NBackDataset(os.path.join(a.data_root, sid), sid); S = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21; acc = {}
    for ts in (1, 2, 3):
        ck = os.path.join(a.results_dir, f"{sid}_session{ts}_best.pt")
        if not os.path.isfile(ck): continue
        m = TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=S, fnirs_temporal_align="egta", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0).to(a.device)
        m.load_state_dict(trunk_state(load_checkpoint(ck, map_location=a.device)), strict=True); set_xai_output(m); m.eval()
        idx = np.random.RandomState(1000 * int(sid[2:]) + ts).permutation(np.flatnonzero(full.session_ids == ts))
        dl = DataLoader(Subset(full, idx), batch_size=a.batch_size, shuffle=False)
        adapt_batchnorm_to_target(m, dl, a.device, log_fn=lambda *x, **k: None)
        for bi, batch in enumerate(dl):
            if bi >= a.max_batches: break
            xe, xf, y, xc, xh = _unpack_batch(batch); xe, xf = xe.to(a.device), xf.to(a.device)
            xc = xc.to(a.device) if xc is not None else None; xh = xh.to(a.device) if xh is not None else None
            r = explain_spatial(m, xe, xf, xc, xh, shap_samples=a.shap_samples); r["y"] = y.numpy()
            for k, v in r.items(): acc.setdefault(k, []).append(v)
        del m
    return {k: np.concatenate(v) for k, v in acc.items()}


def completeness(D):
    """GradientShap should satisfy sum_k phi_k = f(x) - E f(baseline); report how badly the estimator misses it"""
    d = D["fx"] - D["base"]; sm = D["shap_seg"].sum(1); res = d - sm
    r = float(np.corrcoef(d, sm)[0, 1]) if len(d) > 2 else float("nan")
    return f"completeness: median |f(x)-base| {np.median(np.abs(d)):.3f}, median |residual| {np.median(np.abs(res)):.3f}, corr(sum phi, f(x)-base) {r:+.2f}"


# ------------------------------------------------------------------ plots
def _up(m, n=16):
    """bilinear upsample (h,w) -> (n,n) for display"""
    h, w = m.shape; yi = np.linspace(0, h - 1, n); xi = np.linspace(0, w - 1, n)
    y0 = np.floor(yi).astype(int); y1 = np.minimum(y0 + 1, h - 1); x0 = np.floor(xi).astype(int); x1 = np.minimum(x0 + 1, w - 1); fy = (yi - y0)[:, None]; fx = (xi - x0)[None]
    return (m[y0][:, x0] * (1 - fy) * (1 - fx) + m[y1][:, x0] * fy * (1 - fx) + m[y0][:, x1] * (1 - fy) * fx + m[y1][:, x1] * fy * fx)


def plot_gradcam(D, out, sid=""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 4, figsize=(fs.W2, 6.0)); S = D["cam_seg"].shape[1]; x = np.arange(1, S + 1)
    for r, c in enumerate(CLS):
        m = D["pred"] == c; ok = np.flatnonzero(m & (D["y"] == c)); idx = ok[0] if len(ok) else (np.flatnonzero(m)[0] if m.any() else None)
        for j in (0, 1): ax[r, j].set_visible(idx is not None)
        if idx is not None:
            cs = D["cam_seg"][idx]; ax[r, 0].bar(x, cs / (cs.max() + 1e-12), color=CC[c], alpha=0.8, label="Grad-CAM"); ax[r, 0].plot(x, D["occ"][idx] / (np.abs(D["occ"][idx]).max() + 1e-12), color="k", lw=0.9, marker="o", ms=2.5, label="occlusion")
            ax[r, 1].imshow(_up(D["cam_map"][idx].mean(0)), cmap="viridis", origin="upper"); ax[r, 1].set_xticks([]); ax[r, 1].set_yticks([])
        ax[r, 2].set_visible(m.any()); ax[r, 3].set_visible(m.any())
        if m.any():
            A = D["cam_seg"][m] / (D["cam_seg"][m].max(1, keepdims=True) + 1e-12); mu = A.mean(0); lo, hi = np.array([fs.boot_ci(A[:, k], n=500) for k in range(S)]).T
            ax[r, 2].fill_between(x, lo, hi, color=CC[c], alpha=0.25, lw=0); ax[r, 2].plot(x, mu, color=CC[c])
            ax[r, 3].imshow(_up(D["cam_map"][m].mean((0, 1))), cmap="viridis", origin="upper"); ax[r, 3].set_xticks([]); ax[r, 3].set_yticks([])
            ax[r, 3].set_title(f"mean, n = {int(m.sum())}", fontsize=7)
        ax[r, 0].set_ylabel(f"predicted {CLS[c]}\nrelative importance"); ax[r, 0].set_ylim(0, 1.35)
        for j in (0, 2): ax[r, j].set_xlabel("Segment position (1 s steps)")
    ax[0, 0].set_title("Example window", fontsize=8); ax[0, 1].set_title("Spatial map", fontsize=8); ax[0, 2].set_title("Class mean, 95% CI", fontsize=8); ax[0, 0].legend(frameon=False, fontsize=5.5, loc="upper center", ncol=2, handlelength=1.2, columnspacing=0.8, borderaxespad=0.1)
    fs.panel(ax[0, 0], "a"); fs.panel(ax[0, 1], "b"); fs.panel(ax[0, 2], "c"); fs.panel(ax[0, 3], "d")
    fig.suptitle(f"Grad-CAM on fnirs_branch.conv2 {sid} (adapted from Selvaraju et al. 2017; display-normalised per window)", fontsize=8); fig.tight_layout(); return fs.save(fig, out, "fig_gradcam")


def _swarm_offsets(v, rng, w=0.35, bins=30):
    h, e = np.histogram(v, bins=bins); b = np.clip(np.digitize(v, e[1:-1]), 0, bins - 1); dens = h[b] / (h.max() + 1e-12); return rng.uniform(-1, 1, len(v)) * w * dens


def plot_shap_trunk(D, out, sid=""):
    import matplotlib.pyplot as plt
    ph, xl = D["shap_seg"], D["x_lvl"]; S = ph.shape[1]; rng = np.random.RandomState(0)
    fig, ax = plt.subplots(1, 3, figsize=(fs.W2 * 1.15, 4.2), gridspec_kw={"width_ratios": [1, 1.7, 1.5]})
    order = np.argsort(np.abs(ph).mean(0)); ax[0].barh(range(S), np.abs(ph).mean(0)[order], color=fs.C["blue"]); ax[0].set_yticks(range(S)); ax[0].set_yticklabels([f"seg {k + 1}" for k in order], fontsize=5.5); ax[0].set_xlabel("mean |SHAP| (change in P)")
    z = (xl - xl.mean(0)) / (xl.std(0) + 1e-12); sc = None
    for r, k in enumerate(order):
        sc = ax[1].scatter(ph[:, k], r + _swarm_offsets(ph[:, k], rng), c=np.clip(z[:, k], -2, 2), cmap="coolwarm", s=4, linewidths=0, vmin=-2, vmax=2)
    ax[1].axvline(0, color="k", lw=0.5); ax[1].set_yticks(range(S)); ax[1].set_yticklabels([]); ax[1].set_xlabel("SHAP value (impact on P of predicted class)")
    cb = fig.colorbar(sc, ax=ax[1], fraction=0.04, pad=0.02); cb.set_label("segment input level (z)", fontsize=6.5); cb.set_ticks([-2, 0, 2]); cb.set_ticklabels(["low", "", "high"])
    i = int(np.argmax(np.abs(ph).sum(1))); top = np.argsort(-np.abs(ph[i]))[:8]; rest = ph[i].sum() - ph[i][top].sum()
    base, fx = float(D["base"][i]), float(D["fx"][i]); resid = fx - base - ph[i].sum()                 # completeness holds only approximately for the estimator
    steps = [(f"seg {k + 1}", ph[i][k]) for k in top] + [("other segments", rest), ("residual (estimator)", resid)]; ys = list(range(len(steps)))[::-1]; cum = base
    for y_, (nm, v) in zip(ys, steps):
        ax[2].barh(y_, v, left=cum, color=fs.C["grey"] if "residual" in nm else (fs.C["red"] if v > 0 else fs.C["blue"])); ax[2].text(cum + v, y_, f" {v:+.3f}", va="center", fontsize=5.5); cum += v
    ax[2].set_yticks(ys); ax[2].set_yticklabels([s[0] for s in steps], fontsize=6); ax[2].axvline(base, color="0.5", lw=0.6, ls=":"); ax[2].axvline(fx, color="k", lw=0.6)
    ax[2].set_xlabel(f"P(class): base {base:.3f} -> f(x) {fx:.3f}"); ax[2].set_title("one window", fontsize=8)
    for a, l in zip(ax, "abc"): fs.panel(a, l)
    fig.suptitle(f"GradientShap on the trunk, segment level {sid} (n = {len(ph)} windows)\n{completeness(D)}", fontsize=7.5); fig.tight_layout(); return fs.save(fig, out, "fig_shap_trunk")


def plot_shap_member(path, out):
    import matplotlib.pyplot as plt
    Z = np.load(path, allow_pickle=True); band, dsh, ch = Z["band"], Z["diag_share"], Z["channel"]; names = ["broad", "theta", "alpha", "beta"]; n = len(band)
    fig, ax = plt.subplots(1, 4, figsize=(fs.W2 * 1.3, 2.8), gridspec_kw={"width_ratios": [1.1, 1.1, 1.8, 1.0]})
    for a, M, ttl, ref in ((ax[0], band * 100, "share of mean |SHAP| (%)", 25), (ax[1], dsh * 100, "power (diagonal) share of band (%)", None)):
        for j in range(4): fs.strip(a, j, M[:, j], color=fs.C["blue"], width=0.18); fs.mean_ci(a, j, M[:, j], color="k")
        a.set_xticks(range(4)); a.set_xticklabels(names, fontsize=6.5); a.set_ylabel(ttl)
        if ref: a.axhline(ref, ls="--", color="k", lw=0.6)
    mch = ch.mean(0); im = ax[2].imshow(mch, aspect="auto", cmap="viridis"); ax[2].set_yticks(range(4)); ax[2].set_yticklabels(names, fontsize=6.5); ax[2].set_xlabel("EEG channel index (data order, NOT verified)"); fig.colorbar(im, ax=ax[2], fraction=0.04, pad=0.02).set_label("mean |SHAP|", fontsize=6.5)
    for j, (k, col) in enumerate((("top", fs.C["red"]), ("rand", fs.C["grey"]), ("bottom", fs.C["blue"]))): fs.strip(ax[3], j, Z[k], color=col, width=0.15); fs.mean_ci(ax[3], j, Z[k], color="k")
    ax[3].set_xticks(range(3)); ax[3].set_xticklabels(["top", "random", "bottom"], fontsize=6.5); ax[3].set_xlabel("5 channels deleted", fontsize=7); ax[3].set_ylabel("accuracy lost deleting channels"); ax[3].axhline(0, color="k", lw=0.5)
    for a, l in zip(ax, "abcd"): fs.panel(a, l)
    fig.suptitle(f"Exact SHAP of the Riemannian member (n = {n} subjects; linear model, closed form)", fontsize=8); fig.tight_layout(); return fs.save(fig, out, "fig_shap_member")


# ------------------------------------------------------------------ self-test
def _fake(rng, N=90, S=21, h=4):
    y = np.tile([0, 1, 2], N // 3); pred = y.copy(); pred[::7] = (pred[::7] + 1) % 3; prof = np.exp(-0.5 * ((np.arange(S) - 14) / 3.0) ** 2)
    cam_seg = prof[None] * (1 + 0.3 * rng.rand(N, S)); occ = 0.05 * prof[None] + 0.01 * rng.randn(N, S); x_lvl = rng.randn(N, S); shap = 0.02 * (x_lvl * prof[None]) + 0.005 * rng.randn(N, S)
    fx = 0.5 + 0.1 * rng.rand(N); return dict(y=y, pred=pred, p=fx, cam_seg=cam_seg, occ=occ, cam_map=rng.rand(N, S, h, h), shap_seg=shap, x_lvl=x_lvl, base=fx - shap.sum(1) + 0.01 * rng.randn(N), fx=fx)


def self_test():
    import tempfile; rng = np.random.RandomState(0); d = tempfile.mkdtemp(); D = _fake(rng)
    u = _up(np.arange(16.0).reshape(4, 4)); assert u.shape == (16, 16) and abs(u[0, 0]) < 1e-9 and abs(u[-1, -1] - 15) < 1e-9; print("[1] bilinear upsampling keeps corners")
    o1 = plot_gradcam(D, d, "VP000"); o2 = plot_shap_trunk(D, d, "VP000")
    n, Dm = 12, 4 * 406; np.savez(os.path.join(d, "member_shap.npz"), band=rng.dirichlet(np.ones(4), n), diag_share=rng.rand(n, 4) * .2, channel=rng.rand(n, 4, 28), top=rng.rand(n) * .05, rand=rng.rand(n) * .02, bottom=rng.rand(n) * .01)
    o3 = plot_shap_member(os.path.join(d, "member_shap.npz"), d)
    assert all(os.path.isfile(o) and os.path.isfile(o[:-4] + ".pdf") for o in (o1, o2, o3)); print("[2] three figures written (PNG + PDF):", ", ".join(os.path.basename(o) for o in (o1, o2, o3)))
    try:
        import torch; import tsfnet_model_notebook as M
        torch.manual_seed(0); m = M.TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=21, fnirs_temporal_align="egta", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0).eval()
        r = explain_spatial(m, torch.randn(6, 16, 16, 2000), torch.randn(6, 21, 16, 16, 30, 2), None, None)
        assert r["cam_map"].shape[:2] == (6, 21) and r["shap_seg"].shape == (6, 21) and np.isfinite(r["shap_seg"]).all(); print("[3] model-side computation runs on a random TSFNet:", r["cam_map"].shape)
    except ImportError as e: print(f"[3] torch/captum part SKIPPED here ({e}); it runs on your machine")
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=("collect", "plot")); ap.add_argument("--results-dir"); ap.add_argument("--data-root"); ap.add_argument("--subjects", default="1-3")
    ap.add_argument("--out-dir", default="classic"); ap.add_argument("--subject-id", default=None); ap.add_argument("--member-shap", default=None)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--max-batches", type=int, default=3); ap.add_argument("--shap-samples", type=int, default=6, help="GradientShap samples per input; completeness improves with more (try 50)"); ap.add_argument("--device", default=None); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    fs.apply(); os.makedirs(a.out_dir, exist_ok=True)
    if a.mode == "collect":
        if not (a.results_dir and a.data_root): sys.exit("need --results-dir and --data-root")
        import torch; a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu"); lo, hi = a.subjects.split("-") if "-" in a.subjects else (a.subjects, a.subjects)
        for i in range(int(lo), int(hi) + 1):
            sid = f"VP{i:03d}"
            try: D = collect_subject(sid, a); np.savez(os.path.join(a.out_dir, f"classic_{sid}.npz"), **D); print(f"[{sid}] {len(D['y'])} windows saved; GradientShap " + completeness(D), flush=True)
            except Exception as e: print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
    elif a.mode == "plot":
        sid = a.subject_id or sorted(os.path.basename(p)[8:13] for p in glob.glob(os.path.join(a.out_dir, "classic_*.npz")))[0]
        D = dict(np.load(os.path.join(a.out_dir, f"classic_{sid}.npz"))); print("figure:", plot_gradcam(D, a.out_dir, sid)); print("figure:", plot_shap_trunk(D, a.out_dir, sid))
        if a.member_shap: print("figure:", plot_shap_member(a.member_shap, a.out_dir))
    else: ap.error("give a mode: collect | plot (or --self-test)")


if __name__ == "__main__":
    main()
