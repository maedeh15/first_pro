"""
xai_attention_health.py -- health check of the attention and gating modules of the TRUNK (EGTA over the 21 fNIRS segments, CAFE gates, CAFE output projection).
Checkpoint-only part (CPU, seconds): for every checkpoint
  * sigmoid(gate_fusion), sigmoid(gate_fnirs) of CAFE (0.5 = initial value),
  * Frobenius norms of egta_q / egta_k (egta_k is ZERO-initialised: if it stays tiny the attention logits stay ~0 and the attention stays ~uniform),
  * singular values of fusion_out_proj.weight: spectrum, stable rank, participation ratio, entropy effective rank.
Data part (needs --data-root; forward passes on --max-windows random test windows per fold, AdaBN as in evaluation): normalised entropy H/ln(S) of the EGTA weights
(per head), total-variation distance to uniform and max-weight x S. References: the uniform distribution (H/lnS = 1) and Dirichlet(alpha) draws: for a symmetric Dirichlet
E[H] = psi(S*alpha+1) - psi(alpha+1) in closed form (checked by simulation); alpha = 1 is a uniformly random point of the simplex, and the observed mean is summarised as the
EFFECTIVE concentration alpha_eff that reproduces it (large alpha_eff = attention hugs the uniform distribution).
Outputs: attention_health_checkpoints.csv, attention_health_entropy.csv, attention_health_summary.json, fig_attention_health.{png,pdf}.
  python xai_attention_health.py --results-dir "...\\results_cf64a71127_seed42" [--data-root "..."] --out-dir "...\\attention_health"
"""
import argparse, glob, json, os, sys, tempfile
import numpy as np, pandas as pd
from scipy.special import digamma
from scipy.optimize import brentq


def dirichlet_mean_entropy(S, alpha):
    """closed form E[H(p)], p ~ Dirichlet(alpha * 1_S), in nats"""
    return float(digamma(S * alpha + 1) - digamma(alpha + 1))


def alpha_eff(norm_entropy, S):
    """symmetric-Dirichlet concentration whose expected normalised entropy equals the observation (inf if >= the limit 1)"""
    f = lambda a: dirichlet_mean_entropy(S, a) / np.log(S) - norm_entropy
    lo, hi = 1e-3, 1e7
    return float("inf") if f(hi) < 0 else (0.0 if f(lo) > 0 else float(brentq(f, lo, hi)))


def spectrum_stats(W):
    s = np.linalg.svd(W, compute_uv=False); p = s ** 2 / (s ** 2).sum()
    return dict(spectrum=s / s.max(), stable_rank=float((s ** 2).sum() / s[0] ** 2), participation_ratio=float((s ** 2).sum() ** 2 / (s ** 4).sum()),
                effective_rank=float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())), top_share=float(p[0]))


def checkpoint_stats(path):
    from xai_trunk import load_checkpoint, trunk_state
    sd = trunk_state(load_checkpoint(path, map_location="cpu")); g = lambda suf: next((k for k in sd if k.endswith(suf)), None)
    out = {}
    for name, suf in (("gate_fusion", "cafe.gate_fusion"), ("gate_fnirs", "cafe.gate_fnirs")):
        k = g(suf); out[f"g_{name.split('_')[1]}"] = float(1 / (1 + np.exp(-float(sd[k].reshape(-1)[0])))) if k else np.nan
    for name, suf in (("egta_q_norm", "cafe.egta_q.weight"), ("egta_k_norm", "cafe.egta_k.weight")):
        k = g(suf); out[name] = float(np.linalg.norm(sd[k].numpy())) if k else np.nan
    k = g("cafe.fusion_out_proj.weight"); sp = spectrum_stats(sd[k].numpy().astype(np.float64)) if k else None
    return out, sp


def entropy_stats(w):
    """w: (B, h, S) attention weights -> per-window arrays: normalised entropy (mean over heads), TV distance to uniform, max-weight x S"""
    B, h, S = w.shape; w = np.clip(w.astype(np.float64), 1e-12, None); w = w / w.sum(-1, keepdims=True)
    H = -(w * np.log(w)).sum(-1) / np.log(S)
    return H.mean(1), 0.5 * np.abs(w - 1.0 / S).sum(-1).mean(1), (w.max(-1) * S).mean(1)


def fold_entropy(sid, ses, a):
    import torch
    from torch.utils.data import DataLoader, Subset
    from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
    from xai_trunk import trunk_state, load_checkpoint
    full = fold_entropy.cache.setdefault(sid, NBackDataset(os.path.join(a.data_root, sid), sid)); S = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    ck = os.path.join(a.results_dir, f"{sid}_session{ses}_best.pt")
    m = TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=S, fnirs_temporal_align="egta", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0).to(a.device)
    m.load_state_dict(trunk_state(load_checkpoint(ck, map_location=a.device)), strict=True); m.eval()
    idx = np.flatnonzero(full.session_ids == ses)
    if a.adabn: adapt_batchnorm_to_target(m, DataLoader(Subset(full, idx), batch_size=16, shuffle=False), a.device, log_fn=lambda *x, **k: None)
    sel = np.random.RandomState(1000 * int(sid[2:]) + ses).permutation(idx)[:a.max_windows]; res = [[], [], []]
    with torch.no_grad():
        for batch in DataLoader(Subset(full, sel), batch_size=16, shuffle=False):
            xe, xf, y, xc, xh = _unpack_batch(batch); xe, xf = xe.to(a.device), xf.to(a.device)
            xc = xc.to(a.device) if xc is not None else None; xh = xh.to(a.device) if xh is not None else None
            m(xe, xf, xc, xh); r = entropy_stats(m.cafe._last_egta_weights.cpu().numpy())
            for i in range(3): res[i].append(r[i])
    return [float(np.concatenate(r).mean()) for r in res], S
fold_entropy.cache = {}


def _plain_log(ax):
    """readable ticks on a log axis: plain numbers at 1-2-3-5 steps instead of a single 10^0 label"""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    lo, hi = ax.get_ylim(); t = [v * 10.0 ** e for e in range(-3, 3) for v in (1, 2, 3, 5)]; t = [v for v in t if lo <= v <= hi]
    ax.yaxis.set_major_locator(FixedLocator(t)); ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def figure(ck, ent, spectra, S, out_dir):
    import matplotlib.pyplot as plt
    from fig_style import apply, save, C, panel, strip, W2
    apply(); fig, axs = plt.subplots(2, 2, figsize=(W2, 5.4)); (a, b), (c, d) = axs
    # (a) entropy
    rng = np.random.RandomState(0); dir1 = [dirichlet_mean_entropy(S, 1) / np.log(S)]
    sim = np.array([-(p * np.log(p)).sum() / np.log(S) for p in rng.dirichlet(np.ones(S), 4000)])
    if ent is not None and len(ent):
        sub = ent.groupby("subject")["norm_entropy"].mean().sort_values(); x = np.arange(len(sub))
        a.axhspan(sim.mean() - 2 * sim.std(), sim.mean() + 2 * sim.std(), color=C["sky"], alpha=0.25, lw=0, label=f"Dirichlet(1): random attention (mean ± 2 SD)")
        a.axhline(1.0, color="k", ls="--", lw=0.8, label="uniform attention")
        for i, s in enumerate(sub.index): a.scatter(np.full(sum(ent.subject == s), i), ent[ent.subject == s]["norm_entropy"], s=8, color=C["grey"], alpha=0.6, linewidths=0)
        a.plot(x, sub.values, "o", color=C["blue"], ms=3.5, label="subject mean over folds")
        a.set_ylim(min(0.8, sub.min() - 0.02), 1.012); a.set_xticks(x); a.set_xticklabels([s[2:] for s in sub.index], rotation=90, fontsize=5.5)
        a.set_xlabel("subject (sorted)"); a.set_ylabel("normalised entropy  H / ln S"); a.legend(loc="lower right", frameon=False, fontsize=6)
        a.set_title(f"EGTA attention entropy: mean {ent.norm_entropy.mean():.3f}", fontsize=8, loc="left", x=0.02)
    else:
        a.text(0.5, 0.5, "run with --data-root\nfor the entropy panel", ha="center", va="center", transform=a.transAxes); a.set_axis_off()
    panel(a, "a")
    # (b) q/k norms
    for i, (col, lab, colr) in enumerate((("egta_q_norm", "egta_q", C["blue"]), ("egta_k_norm", "egta_k (zero-init)", C["red"]))):
        v = ck[col].values; b.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color="k"), boxprops=dict(color=colr), whiskerprops=dict(color=colr), capprops=dict(color=colr))
        strip(b, i, v, colr, 0.15)
    b.set_xticks([0, 1]); b.set_xticklabels(["egta_q", "egta_k\n(0 at initialisation)"]); b.set_ylabel("Frobenius norm of the weight"); b.set_yscale("log"); b.set_title("EGTA projection norms", fontsize=8, loc="left", x=0.02); _plain_log(b); panel(b, "b")
    # (c) gates
    for i, (col, lab, colr) in enumerate((("g_fusion", "g_fusion", C["green"]), ("g_fnirs", "g_fnirs", C["orange"]))):
        v = ck[col].values; c.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color="k"), boxprops=dict(color=colr), whiskerprops=dict(color=colr), capprops=dict(color=colr)); strip(c, i, v, colr, 0.15)
    c.axhline(0.5, color="k", ls="--", lw=0.8); c.text(1.45, 0.5, "initial 0.5", ha="right", va="bottom", fontsize=6); c.set_ylim(0, 1)
    c.set_xticks([0, 1]); c.set_xticklabels(["CAFE gate\n(fusion)", "CAFE gate\n(fNIRS)"]); c.set_ylabel("sigmoid(gate)"); c.set_title("CAFE gates (dot: one checkpoint)", fontsize=8, loc="left", x=0.02); panel(c, "c")
    # (d) spectrum
    if spectra:
        M = np.stack([s["spectrum"][:min(len(s["spectrum"]) for s in spectra)] for s in spectra]); x = np.arange(1, M.shape[1] + 1)
        d.fill_between(x, np.percentile(M, 5, 0), np.percentile(M, 95, 0), color=C["blue"], alpha=0.25, lw=0); d.plot(x, np.median(M, 0), color=C["blue"])
        er = np.array([s["effective_rank"] for s in spectra]); d.set_yscale("log"); d.set_xlabel("singular value index"); d.set_ylabel("sigma_i / sigma_1")
        d.set_title(f"fusion_out_proj spectrum, eff. rank {np.median(er):.0f}/{M.shape[1]}", fontsize=8, loc="left", x=0.02); d.text(0.98, 0.97, "median, 5-95%", transform=d.transAxes, ha="right", va="top", fontsize=6.5); _plain_log(d)
    panel(d, "d"); fig.tight_layout(); return save(fig, out_dir, "fig_attention_health")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir"); ap.add_argument("--data-root", default=None); ap.add_argument("--subjects", default="1-26"); ap.add_argument("--out-dir", default="attention_health_out")
    ap.add_argument("--max-windows", type=int, default=96); ap.add_argument("--adabn", action="store_true", default=True); ap.add_argument("--no-adabn", dest="adabn", action="store_false")
    ap.add_argument("--device", default=None); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args(argv)
    if a.self_test: return self_test()
    if not a.results_dir: sys.exit("need --results-dir")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.makedirs(a.out_dir, exist_ok=True)
    import torch; a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2] if "-" in a.subjects else (a.subjects, a.subjects); rows, spectra, erows = [], [], []; S = 21
    for i in range(int(lo), int(hi) + 1):
        sid = f"VP{i:03d}"
        for ses in (1, 2, 3):
            p = os.path.join(a.results_dir, f"{sid}_session{ses}_best.pt")
            if not os.path.isfile(p): continue
            st, sp = checkpoint_stats(p); rows.append(dict(subject=sid, session=ses, **st, **({k: v for k, v in sp.items() if k != "spectrum"} if sp else {})))
            if sp: spectra.append(sp)
            if a.data_root:
                try:
                    (h, tv, mx), S = fold_entropy(sid, ses, a); erows.append(dict(subject=sid, session=ses, norm_entropy=h, tv_to_uniform=tv, maxweight_x_S=mx))
                except Exception as e: print(f"[{sid} s{ses}] entropy skipped: {type(e).__name__}: {e}")
        print(f"[{sid}] done", flush=True); fold_entropy.cache.pop(sid, None)
    if not rows: sys.exit("no checkpoints found")
    ck = pd.DataFrame(rows); ent = pd.DataFrame(erows) if erows else None
    ck.to_csv(os.path.join(a.out_dir, "attention_health_checkpoints.csv"), index=False)
    if ent is not None: ent.to_csv(os.path.join(a.out_dir, "attention_health_entropy.csv"), index=False)
    summ = {"n_checkpoints": len(ck), "gates": {c: [float(ck[c].mean()), float(ck[c].std())] for c in ("g_fusion", "g_fnirs")}, "egta_k_norm_median": float(ck.egta_k_norm.median()),
            "egta_q_norm_median": float(ck.egta_q_norm.median()), "effective_rank_median": float(ck.effective_rank.median()), "stable_rank_median": float(ck.stable_rank.median()),
            "dirichlet1_norm_entropy": dirichlet_mean_entropy(S, 1) / np.log(S)}
    if ent is not None: summ["norm_entropy"] = dict(mean=float(ent.norm_entropy.mean()), min=float(ent.norm_entropy.min()), max=float(ent.norm_entropy.max()), alpha_eff=alpha_eff(float(ent.norm_entropy.mean()), S),
                                                    tv_to_uniform=float(ent.tv_to_uniform.mean()), maxweight_x_S=float(ent.maxweight_x_S.mean()))
    json.dump(summ, open(os.path.join(a.out_dir, "attention_health_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2)); print("figure ->", figure(ck, ent, spectra, S, a.out_dir)); return summ


def self_test():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch, tsfnet_model_notebook as M
    rng = np.random.RandomState(0)
    for al in (1.0, 5.0, 50.0):                                                  # closed form vs simulation
        sim = np.mean([-(p * np.log(p)).sum() for p in rng.dirichlet(al * np.ones(21), 3000)])
        assert abs(sim - dirichlet_mean_entropy(21, al)) < 0.01, (al, sim, dirichlet_mean_entropy(21, al))
    for al in (1.0, 11.0, 80.0): assert abs(alpha_eff(dirichlet_mean_entropy(21, al) / np.log(21), 21) - al) / al < 1e-4
    print("[1] Dirichlet entropy closed form matches simulation; alpha_eff inverts it")
    w = np.full((4, 4, 21), 1 / 21.); H, tv, mx = entropy_stats(w); assert np.allclose(H, 1) and np.allclose(tv, 0) and np.allclose(mx, 1)
    w2 = np.zeros((1, 1, 21)) + 1e-9; w2[0, 0, 3] = 1; H2, tv2, _ = entropy_stats(w2); assert H2[0] < 1e-3 and tv2[0] > 0.9
    print("[2] entropy / TV / max-weight: uniform -> (1, 0, 1); one-hot -> (~0, ~1)")
    torch.manual_seed(0); m = M.TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=21, fnirs_temporal_align="egta", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0)
    d = tempfile.mkdtemp()
    for i in (1, 2):
        for ses in (1, 2, 3): torch.save({"model_state_dict": m.state_dict()}, os.path.join(d, f"VP{i:03d}_session{ses}_best.pt"))
    st, sp = checkpoint_stats(os.path.join(d, "VP001_session1_best.pt"))
    assert abs(st["g_fusion"] - 0.5) < 1e-6 and st["egta_k_norm"] == 0.0 and sp["effective_rank"] > 1 and sp["spectrum"][0] == 1.0
    print("[3] checkpoint diagnostics: gates 0.5, zero-initialised egta_k has norm 0, spectrum normalised")
    ck = pd.DataFrame([dict(subject=f"VP00{i}", session=s, **st, **{k: v for k, v in sp.items() if k != "spectrum"}) for i in (1, 2) for s in (1, 2, 3)])
    ent = pd.DataFrame([dict(subject=f"VP00{i}", session=s, norm_entropy=0.985 + 0.002 * s) for i in (1, 2) for s in (1, 2, 3)])
    ck["egta_k_norm"] = rng.uniform(0.01, 0.05, len(ck)); ck["g_fusion"] = rng.uniform(0.3, 0.7, len(ck)); ck["g_fnirs"] = rng.uniform(0.3, 0.7, len(ck))
    assert os.path.isfile(figure(ck, ent, [sp, sp], 21, d)); print("[4] figure written\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
