"""
xai_evidence_segments.py -- runner for the EXACT per-segment evidence of the fNIRS head of a model trained with --fnirs-head linear (see fnirs_evidence.py).
Needs the PATCHED tsfnet_model_notebook.py (apply_evidence_head_patch.py) and the checkpoints of the `_fhead-linear` results folder. No retraining.
For every random window of each held-out session (AdaBN as everywhere):
  c_i  = evidence of segment i for the predicted class (an exact additive term of the fNIRS-head LOGIT)       [fnirs_evidence.decompose]
  rest = the part of the logit that does not come from the segments (g*W.E + bias: the fusion-token path)
Reported (pre-stated before the run; chance/baselines in brackets):
  (1) additivity error of sum_i c_i + rest = logit                         (must be < 1e-5)
  (2) rho(evidence, occlusion impact) per window, at two levels: the fNIRS HEAD probability, and the FINAL fused prediction (Grad-CAM reference: 0.30)
  (3) deletion of the top-3 / random-3 / bottom-3 evidence segments (batch-mean replacement) on both levels
  (4) where the head logit varies from window to window: corr(logit, sum_i c_i) versus corr(logit, rest), per class, averaged
  (5) mean evidence profile by predicted class over the 21 segments (window-relative position, NOT time from block start)
LIMITS: exact only for the head LOGIT; the final prediction is a mixture of three heads (and, in the final system, of the Riemannian member); neighbouring segments share 2/3 of
their samples; attention weights move slightly when a segment is replaced, so the head-level occlusion check is approximate by construction.
  python xai_evidence_segments.py --results-dir "...\\results_..._fhead-linear" --data-root "..." --subjects 1-26 --out-dir "...\\evidence_segments"
"""
import argparse, csv, json, os, sys
import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _const(v): return np.ptp(v) <= 1e-9 * (np.abs(v).max() + 1e-300)


def rho_rows(A, B):
    r = []
    for a, b in zip(A, B):
        if not (_const(a) or _const(b)):
            v = spearmanr(a, b)[0]
            if not np.isnan(v): r.append(v)
    return r


def batch_stats(model, xe, xf, xc, xh, k, rng, decompose):
    B, S = xf.shape[:2]; ar = torch.arange(B, device=xf.device)
    out = decompose(model, xe, xf, xc, xh)
    c, rest, logit = out["contrib"], out["rest"], out["logit"]; yf, yp = out["y_fnirs"], out["y_pred"]
    cf, cp = yf.argmax(1), yp.argmax(1)
    err = float((c.sum(1) + rest - logit).abs().max())
    ev_f = c[ar, :, cf].cpu().numpy(); ev_p = c[ar, :, cp].cpu().numpy()
    rep = xf.mean(dim=0, keepdim=True).expand_as(xf); imp_f = np.zeros((B, S)); imp_p = np.zeros((B, S))
    p0f, p0p = yf[ar, cf], yp[ar, cp]
    with torch.no_grad():
        for s in range(S):
            xa = xf.clone(); xa[:, s] = rep[:, s]; o = model(xe, xa, xc, xh); imp_f[:, s] = (p0f - o[2][ar, cf]).cpu().numpy(); imp_p[:, s] = (p0p - o[3][ar, cp]).cpu().numpy()
        dele = {}
        for lvl, ev, p0, cl, idx in (("head", ev_f, p0f, cf, 2), ("full", ev_p, p0p, cp, 3)):
            order = np.argsort(-ev, axis=1)
            sets = {"top": order[:, :k], "bottom": order[:, -k:], "random": np.stack([rng.choice(S, k, replace=False) for _ in range(B)])}
            for nm, ix in sets.items():
                mask = torch.zeros(B, S, dtype=torch.bool, device=xf.device); mask.scatter_(1, torch.as_tensor(np.asarray(ix, dtype=np.int64), device=xf.device), True)
                xa = torch.where(mask.view(B, S, 1, 1, 1, 1), rep, xf); o = model(xe, xa, xc, xh)
                dele[f"{lvl}_{nm}"] = (p0 - o[idx][ar, cl]).cpu().numpy()
    return dict(err=err, rho_head=rho_rows(ev_f, imp_f), rho_full=rho_rows(ev_p, imp_p), dele=dele, logit=logit.cpu().numpy(), seg=c.sum(1).cpu().numpy(), rest=rest.cpu().numpy(),
                ev_p=ev_p, cls=cp.cpu().numpy())


def run_subject(sid, a, rng):
    from torch.utils.data import DataLoader, Subset
    from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
    from xai_trunk import trunk_state, load_checkpoint, set_xai_output
    from fnirs_evidence import decompose
    full = NBackDataset(os.path.join(a.data_root, sid), sid); S = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    acc = dict(err=0.0, rho_head=[], rho_full=[], logit=[], seg=[], rest=[], dele={}, prof={}, cnt={})
    for ts in (1, 2, 3):
        ck = os.path.join(a.results_dir, f"{sid}_session{ts}_best.pt")
        if not os.path.isfile(ck): continue
        m = TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=S, fnirs_temporal_align="egta", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0, fnirs_head="linear").to(a.device)
        m.load_state_dict(trunk_state(load_checkpoint(ck, map_location=a.device)), strict=True); set_xai_output(m); m.eval()
        idx = np.random.RandomState(1000 * int(sid[2:]) + ts).permutation(np.flatnonzero(full.session_ids == ts)); dl = DataLoader(Subset(full, idx), batch_size=a.batch_size, shuffle=False)
        adapt_batchnorm_to_target(m, dl, a.device, log_fn=lambda *x, **k: None)
        for bi, batch in enumerate(dl):
            if bi >= a.max_batches: break
            xe, xf, y, xc, xh = _unpack_batch(batch); xe, xf = xe.to(a.device), xf.to(a.device)
            xc = xc.to(a.device) if xc is not None else None; xh = xh.to(a.device) if xh is not None else None
            r = batch_stats(m, xe, xf, xc, xh, a.k, rng, decompose)
            acc["err"] = max(acc["err"], r["err"]); acc["rho_head"] += r["rho_head"]; acc["rho_full"] += r["rho_full"]
            for key in ("logit", "seg", "rest"): acc[key].append(r[key])
            for k_, v in r["dele"].items(): acc["dele"].setdefault(k_, []).extend(v)
            for cl in np.unique(r["cls"]):
                z = r["ev_p"][r["cls"] == cl]; z = (z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-12)
                acc["prof"][int(cl)] = acc["prof"].get(int(cl), 0) + z.sum(0); acc["cnt"][int(cl)] = acc["cnt"].get(int(cl), 0) + len(z)
        del m
    L, G, R = (np.concatenate(acc[k]) for k in ("logit", "seg", "rest"))
    cs = np.nanmean([np.corrcoef(L[:, c], G[:, c])[0, 1] for c in range(3)]); cr = np.nanmean([np.corrcoef(L[:, c], R[:, c])[0, 1] for c in range(3)])
    return dict(err=acc["err"], rho_head=float(np.mean(acc["rho_head"])), rho_full=float(np.mean(acc["rho_full"])), corr_sum_seg=float(cs), corr_rest=float(cr),
                **{k_: float(np.mean(v)) for k_, v in acc["dele"].items()}), {str(c): (acc["prof"][c] / acc["cnt"][c]).tolist() for c in acc["prof"]}


def report(per, prof, out_dir=None):
    ids = list(per); g = lambda k: np.array([per[s][k] for s in ids]); n = len(ids)
    print(f"\nEVIDENCE OF THE LINEAR fNIRS HEAD, {n} subjects (window-level means per subject; Wilcoxon over subjects)")
    print(f"  (1) additivity: largest |sum c_i + rest - logit| over all windows = {g('err').max():.2e}  ({'OK' if g('err').max() < 1e-5 else 'NOT additive: do not interpret'})")
    for lvl, nm in (("head", "fNIRS head"), ("full", "final fused prediction")):
        r = g(f"rho_{lvl}"); t, ra, b = g(f"{lvl}_top"), g(f"{lvl}_random"), g(f"{lvl}_bottom")
        p = float(wilcoxon(t - ra, alternative="greater").pvalue) if np.any(t - ra != 0) else 1.0; pr = float(wilcoxon(r, alternative="greater").pvalue) if np.any(r != 0) else 1.0
        print(f"  (2,3) vs {nm:24s}: rho(evidence, occlusion) {r.mean():+.3f} (positive in {int((r > 0).sum())}/{n}, p={pr:.3g}) | deletion top {t.mean():+.4f} random {ra.mean():+.4f} bottom {b.mean():+.4f} (top>random p={p:.3g})")
    print(f"       reference: Grad-CAM rho with occlusion on the fused prediction = 0.30 (validated maps, block deletion); the pre-registered bar for the evidence is rho_full > 0.30")
    print(f"  (4) window-to-window variation of the head logit: corr with the segment sum {g('corr_sum_seg').mean():+.3f} | with the rest term (fusion-token path + bias) {g('corr_rest').mean():+.3f}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True); keys = list(per[ids[0]])
        with open(os.path.join(out_dir, "evidence_segments_per_subject.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["subject"] + keys); [w.writerow([s] + [per[s][k] for k in keys]) for s in ids]
        json.dump(prof, open(os.path.join(out_dir, "evidence_profiles_by_class.json"), "w")); print("figure:", figure(per, prof, out_dir))


def figure(per, prof, out_dir):
    import fig_style as fs; fs.apply(); import matplotlib.pyplot as plt
    ids = list(per); cc = {"0": fs.C["blue"], "1": fs.C["orange"], "2": fs.C["red"]}; nm = {"0": "0-back", "1": "2-back", "2": "3-back"}
    fig, ax = plt.subplots(1, 3, figsize=(fs.W2, 2.8), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for cl in "012":
        A = np.array([prof[s][cl] for s in ids if cl in prof[s]])
        if A.size == 0: continue                                                   # a class that is never predicted (random stand-in model in the self-test)
        x = np.arange(1, A.shape[1] + 1); lo, hi = np.array([fs.boot_ci(A[:, k], n=800) for k in range(A.shape[1])]).T
        ax[0].fill_between(x, lo, hi, color=cc[cl], alpha=0.2, lw=0); ax[0].plot(x, A.mean(0), color=cc[cl], label=nm[cl])
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xlabel("Segment position in window (1 s steps)"); ax[0].set_ylabel("Evidence (within-window z)"); ax[0].legend(frameon=False, fontsize=6); fs.panel(ax[0], "a")
    for j, (k, col, lab) in enumerate((("rho_head", fs.C["sky"], "fNIRS head"), ("rho_full", fs.C["blue"], "fused"))):
        v = np.array([per[s][k] for s in ids]); fs.strip(ax[1], j, v, color=col, width=0.15); fs.mean_ci(ax[1], j, v, color="k")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].axhline(0.30, ls="--", color="k", lw=0.6); ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["fNIRS head", "fused output"], fontsize=6.5); ax[1].set_ylabel("rho(evidence, occlusion)"); fs.panel(ax[1], "b")
    for j, (k, col) in enumerate((("corr_sum_seg", fs.C["green"]), ("corr_rest", fs.C["purple"]))):
        v = np.array([per[s][k] for s in ids]); fs.strip(ax[2], j, v, color=col, width=0.15); fs.mean_ci(ax[2], j, v, color="k")
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["segment\nevidence", "rest term\n(fusion path)"], fontsize=6.5); ax[2].set_ylabel("corr with head logit"); fs.panel(ax[2], "c")
    fig.suptitle(f"Linear fNIRS head: exact per-segment evidence (n = {len(ids)} subjects)", fontsize=8); fig.tight_layout(); return fs.save(fig, out_dir, "fig_evidence_segments")


def self_test():
    import tempfile, argparse
    import tsfnet_model_notebook as M
    torch.manual_seed(0); rng = np.random.RandomState(0); d = tempfile.mkdtemp(); res = os.path.join(d, "res"); os.makedirs(res); per, prof = {}, {}
    m = M.TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=21, fnirs_temporal_align="egta", egta_k_init="noisy", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0, fnirs_head="linear")
    for i in range(1, 7):
        sid = f"VP{i:03d}"; sd = os.path.join(d, sid); os.makedirs(sd); N = 12
        np.save(f"{sd}/{sid}_eeg_windows.npy", rng.randn(N, 16, 16, 2000).astype(np.float32)); np.save(f"{sd}/{sid}_fnirs_windows.npy", rng.randn(N, 21, 16, 16, 30, 2).astype(np.float32))
        np.save(f"{sd}/{sid}_labels.npy", np.tile([0, 1, 2], 4)); np.save(f"{sd}/{sid}_session_ids.npy", np.repeat([1, 2, 3], 4)); np.save(f"{sd}/{sid}_trial_ids.npy", np.arange(N))
        for ts in (1, 2, 3): torch.save(m.state_dict(), f"{res}/{sid}_session{ts}_best.pt")
    a = argparse.Namespace(results_dir=res, data_root=d, batch_size=4, max_batches=3, k=3, device="cpu"); rg = np.random.RandomState(1)
    for i in range(1, 7):
        sid = f"VP{i:03d}"; per[sid], prof[sid] = run_subject(sid, a, rg)
    assert max(per[s]["err"] for s in per) < 1e-5, "additivity must hold on real forward passes"
    print("[1] additivity holds on real forward passes of a linear-head TSFNet (max error %.1e)" % max(per[s]["err"] for s in per))
    report(per, prof, tempfile.mkdtemp()); print("[2] report, CSV, JSON and figure written\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir"); ap.add_argument("--data-root"); ap.add_argument("--subjects", default="1-26"); ap.add_argument("--out-dir", default=None)
    ap.add_argument("--k", type=int, default=3); ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--max-batches", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    if not (a.results_dir and a.data_root): sys.exit("need --results-dir and --data-root")
    a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu"); rng = np.random.RandomState(a.seed); per, prof = {}, {}
    lo, hi = a.subjects.split("-") if "-" in a.subjects else (a.subjects, a.subjects)
    for i in range(int(lo), int(hi) + 1):
        sid = f"VP{i:03d}"
        try:
            per[sid], prof[sid] = run_subject(sid, a, rng); r = per[sid]
            print(f"[{sid}] rho head {r['rho_head']:+.3f} fused {r['rho_full']:+.3f} | corr(logit, seg-sum) {r['corr_sum_seg']:+.2f} rest {r['corr_rest']:+.2f} | additivity err {r['err']:.1e}", flush=True)
        except Exception as e: print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
    if len(per) < 6: sys.exit(f"{len(per)} usable subjects; group tests need >= 6")
    report(per, prof, a.out_dir or os.path.join(a.results_dir, "evidence_segments"))


if __name__ == "__main__":
    main()
