"""
xai_validated_maps.py -- SHAP-family, Grad-CAM, gradient and CAUSAL-OCCLUSION explanations of the fNIRS segments of the TRUNK,
each one scored by the SAME validation tests, so every method in the thesis is reported with its evidence.

Per held-out window, target class c = the model's own prediction, importance per fNIRS segment (S = 21):
  attention        EGTA weight (reference; failed faithfulness earlier)
  input_gradient   mean |dlogP(c)/dx| per segment
  grad_x_input     mean |x * dlogP(c)/dx| per segment
  gradient_shap    captum GradientShap (= SHAP's expected-gradients estimator; background = other windows of the SAME held-out session,
                   label-free). An APPROXIMATION of SHAP values, not exact.
  grad_cam         Grad-CAM (Selvaraju 2017) on fnirs_branch.conv2 (one map per segment, relu(sum_k mean-grad_k * A_k), mean over H,W,T)
  occlusion        P(c) minus P(c | segment zeroed): causal by construction (the proposed explanation)
Validation of EVERY method (pre-stated, in this order):
  V1 model randomisation (Adebayo 2018): Spearman between the explanation of the trained and of the fully re-drawn network, same windows;
     passes if below --pass-rho (0.3) in at least 2/3 of subjects.
  V2 deletion faithfulness: replace the top-k / random-k / bottom-k segments (k=3) by the window-batch MEAN segment (a different intervention
     than the zeroing behind `occlusion`, so occlusion is not graded by its own definition); mean drop of P(c) per subject;
     faithful iff top > random > bottom on the group mean AND Wilcoxon (top > random over subjects) p < 0.05.
  V3 rank agreement with the zeroing impact (all methods but occlusion), one-sided Wilcoxon rho > 0.
A method is reported as "validated" only if V1 and V2 both pass. Needs >= 6 subjects for the group tests.
Limits: segment level only (the fNIRS branch is the one place where an exact intervention is defined per segment); GradientShap is an
estimator; Grad-CAM's map is the conv2 grid and is reduced to a per-segment scalar here.

  python xai_validated_maps.py --results-dir "...\\results_cf64a71127_seed42" --data-root "..." --subjects 1-26
  python xai_validated_maps.py --self-test
"""
import argparse, csv, json, os, sys
import numpy as np, torch
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai_sanity_check import randomise_upto, GROUPS, per_segment_mean

METHODS = ("attention", "input_gradient", "grad_x_input", "gradient_shap", "grad_cam", "occlusion")


def const(v):
    return np.ptp(v) <= 1e-9 * (np.abs(v).max() + 1e-300)


def pc(yp, c):
    return yp.gather(1, c.view(-1, 1)).squeeze(1)


def explain(model, xe, xf, xc, xh, cls=None, n_bg=6, shap_samples=6, want_shap=True):
    """returns dict method -> (B, S) numpy, plus p0 (B,) and cls"""
    with torch.no_grad():
        yp0 = model(xe, xf, xc, xh)[3]
        attn = model.cafe._last_egta_weights.mean(dim=1).detach().cpu().numpy()
    c = yp0.argmax(1) if cls is None else cls
    p0 = pc(yp0, c)
    B, S = xf.shape[0], xf.shape[1]
    store = {}
    h = model.fnirs_branch.conv2.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    xg = xf.clone().requires_grad_(True)
    yp = model(xe, xg, xc, xh)[3]
    h.remove()
    tgt = torch.log(pc(yp, c).clamp_min(1e-8)).sum()
    g, ga = torch.autograd.grad(tgt, [xg, store["a"]])
    g = g.detach()
    out = {"attention": attn, "input_gradient": per_segment_mean(g).cpu().numpy(),
           "grad_x_input": per_segment_mean(g * xf).cpu().numpy()}
    a = store["a"].detach(); alpha = ga.detach().mean(dim=tuple(range(2, ga.dim())), keepdim=True)
    cam = torch.relu((alpha * a).sum(dim=1))                              # (B*S, H, W, T)
    out["grad_cam"] = cam.mean(dim=tuple(range(1, cam.dim()))).reshape(B, S).cpu().numpy()
    if want_shap:
        from captum.attr import GradientShap
        class _F(torch.nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, x, e, k, q): return s.m(e, x, k, q)[3]
        nb = min(n_bg, B)
        bg = xf[torch.randperm(B, device=xf.device)[:nb]].detach()
        at = GradientShap(_F(model)).attribute(xf.detach(), baselines=bg, n_samples=shap_samples, stdevs=0.05, target=c,
                                               additional_forward_args=(xe, xc, xh))
        out["gradient_shap"] = per_segment_mean(at.detach()).cpu().numpy()
    else:
        out["gradient_shap"] = np.full((B, S), np.nan)
    imp = np.zeros((B, S))
    with torch.no_grad():
        for s in range(S):
            xa = xf.clone(); xa[:, s] = 0.0
            imp[:, s] = (p0 - pc(model(xe, xa, xc, xh)[3], c)).cpu().numpy()
    out["occlusion"] = imp
    return out, p0.detach(), c


def deletion(model, xe, xf, xc, xh, c, p0, scores, k, rng):
    """mean drop of P(c) when top/random/bottom-k segments are replaced by the batch-mean segment; per method, per window"""
    B, S = scores["attention"].shape
    rep = xf.mean(dim=0, keepdim=True).expand_as(xf)
    res = {}
    with torch.no_grad():
        for m, sc in scores.items():
            if np.isnan(sc).all():
                continue
            order = np.argsort(-sc, axis=1)
            sets = {"top": order[:, :k], "bottom": order[:, -k:], "random": np.stack([rng.choice(S, k, replace=False) for _ in range(B)])}
            for nm, idx in sets.items():
                mask = torch.zeros(B, S, dtype=torch.bool, device=xf.device)
                mask.scatter_(1, torch.as_tensor(idx, device=xf.device), True)
                xa = torch.where(mask.view(B, S, 1, 1, 1, 1), rep, xf)
                res.setdefault(m, {})[nm] = (p0 - pc(model(xe, xa, xc, xh)[3], c)).cpu().numpy()
    return res


def rho_rows(A, B_):
    r = []
    for a, b in zip(A, B_):
        if not (const(a) or const(b)):
            v = spearmanr(a, b)[0]
            if not np.isnan(v):
                r.append(v)
    return r


def process_batch(model, rmodel, xe, xf, xc, xh, a, rng, want_shap):
    sc, p0, c = explain(model, xe, xf, xc, xh, want_shap=want_shap)
    rs, _, _ = explain(rmodel, xe, xf, xc, xh, cls=c, want_shap=want_shap)
    dele = deletion(model, xe, xf, xc, xh, c, p0, sc, a.k, rng)
    rec = {m: {"rand_rho": rho_rows(sc[m], rs[m])} for m in sc if not np.isnan(sc[m]).all()}
    for m in rec:
        if m != "occlusion":
            rec[m]["impact_rho"] = rho_rows(sc[m], sc["occlusion"])
        for nm in ("top", "random", "bottom"):
            rec[m][nm] = list(dele[m][nm])
    return rec


def run_subject(sid, a, rng):
    from torch.utils.data import DataLoader, Subset
    from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
    from xai_trunk import trunk_state, load_checkpoint
    full = NBackDataset(os.path.join(a.data_root, sid), sid)
    S = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    acc = {}
    for ts in (1, 2, 3):
        ck = os.path.join(a.results_dir, f"{sid}_session{ts}_best.pt")
        if not os.path.isfile(ck):
            print(f"[{sid} s{ts}] SKIP no checkpoint"); continue
        model = TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=S, fnirs_temporal_align="egta",
                       efgf_gate_mode="sigmoid", efgf_gate_init=-2.0).to(a.device)
        model.load_state_dict(trunk_state(load_checkpoint(ck, map_location=a.device)), strict=True); model.eval()
        dl = DataLoader(Subset(full, np.flatnonzero(full.session_ids == ts)), batch_size=a.batch_size, shuffle=False)
        if a.adabn:
            adapt_batchnorm_to_target(model, dl, a.device, log_fn=lambda *x, **k: None)
        rmodel, _ = randomise_upto(model, len(GROUPS) - 1, a.seed); rmodel = rmodel.to(a.device).eval()
        if a.adabn:
            adapt_batchnorm_to_target(rmodel, dl, a.device, log_fn=lambda *x, **k: None)
        for bi, batch in enumerate(dl):
            if bi >= a.max_batches:
                break
            xe, xf, y, xc, xh = _unpack_batch(batch)
            xe, xf = xe.to(a.device), xf.to(a.device)
            xc = xc.to(a.device) if xc is not None else None; xh = xh.to(a.device) if xh is not None else None
            rec = process_batch(model, rmodel, xe, xf, xc, xh, a, rng, a.shap)
            for m, d in rec.items():
                for k_, v in d.items():
                    acc.setdefault(m, {}).setdefault(k_, []).extend(v)
        del model, rmodel
    return {m: {k_: float(np.mean(v)) if len(v) else float("nan") for k_, v in d.items()} for m, d in acc.items()}


def summarise(per, pass_rho, verbose=True):
    ids = list(per); out = {}
    if verbose:
        print("\n" + "=" * 118 + f"\nVALIDATED EXPLANATION MAPS -- {len(ids)} subjects, fNIRS segment level, target = predicted class\n" + "=" * 118)
        print(f"{'method':15s} {'rho vs impact':>13s} | {'del top':>8s} {'random':>8s} {'bottom':>8s} {'p(top>rnd)':>11s} | {'rand-rho':>8s} {'<thr':>6s} | verdict")
    for m in METHODS:
        rows = [per[s][m] for s in ids if m in per[s]]
        if len(rows) < 6:
            continue
        g = lambda k: np.array([r.get(k, np.nan) for r in rows])
        t, r_, b_ = g("top"), g("random"), g("bottom")
        p_tr = 1.0 if not np.any(t - r_ != 0) else float(wilcoxon(t - r_, alternative="greater").pvalue)
        faith = bool(t.mean() > r_.mean() > b_.mean() and p_tr < 0.05)
        rr = g("rand_rho"); frac = float((rr < pass_rho).mean()); sane = frac >= 2 / 3
        ir = g("impact_rho"); ir = ir[~np.isnan(ir)]
        p_ir = float(wilcoxon(ir, alternative="greater").pvalue) if len(ir) >= 6 and np.any(ir != 0) else float("nan")
        verdict = "VALIDATED" if (faith and sane) else ("faithful but fails randomisation" if faith else ("passes randomisation, NOT faithful" if sane else "fails both"))
        out[m] = dict(n=len(rows), impact_rho=float(ir.mean()) if len(ir) else float("nan"), p_impact=p_ir, top=float(t.mean()), random=float(r_.mean()),
                      bottom=float(b_.mean()), p_top_gt_random=p_tr, rand_rho=float(rr.mean()), frac_below=frac, faithful=faith, sane=sane, verdict=verdict)
        if verbose:
            irs = f"{ir.mean():+.3f}" if len(ir) else "   --  "
            print(f"{m:15s} {irs:>13s} | {t.mean():+8.4f} {r_.mean():+8.4f} {b_.mean():+8.4f} {p_tr:11.4g} | {rr.mean():+8.3f} {frac:6.2f} | {verdict}")
    if verbose:
        print("\n  deletion = mean drop of P(c) after replacing 3 segments by the window-batch mean; occlusion has no 'rho vs impact' (it IS the impact).")
        print("  Report every row, including the failures; a method is 'validated' only with V1 and V2 both passed (rules in the docstring).")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir"); p.add_argument("--data-root"); p.add_argument("--subjects", default="1-26"); p.add_argument("--out-dir", default=None)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--max-batches", type=int, default=3)
    p.add_argument("--k", type=int, default=3); p.add_argument("--pass-rho", type=float, default=0.3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-shap", dest="shap", action="store_false", help="skip GradientShap (needs captum)")
    p.add_argument("--adabn", action="store_true", default=True); p.add_argument("--no-adabn", dest="adabn", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not (a.results_dir and a.data_root):
        sys.exit("need --results-dir and --data-root")
    if a.shap:
        try:
            import captum  # noqa: F401
        except ImportError:
            print("captum missing -> continuing WITHOUT GradientShap (pip install captum, then rerun to include it)"); a.shap = False
    out_dir = a.out_dir or os.path.join(a.results_dir, "xai_validated_maps"); os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(a.seed); per = {}
    for part in a.subjects.split(","):
        lo, hi = (part.split("-") + [part])[:2] if "-" in part else (part, part)
        for i in range(int(lo), int(hi) + 1):
            sid = f"VP{i:03d}"
            try:
                per[sid] = run_subject(sid, a, rng)
                print(f"[{sid}] " + " | ".join(f"{m} {per[sid][m]['top'] - per[sid][m]['random']:+.4f}" for m in per[sid]) + "   (top - random deletion drop)", flush=True)
            except Exception as e:
                print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
    if len(per) < 6:
        sys.exit(f"ERROR: {len(per)} usable subjects; group tests need >= 6 (per-subject lines above are valid).")
    res = summarise(per, a.pass_rho)
    json.dump({"per_subject": per, "group": res, "config": vars(a)}, open(os.path.join(out_dir, "validated_maps_summary.json"), "w"), indent=2, default=float)
    with open(os.path.join(out_dir, "validated_maps_group.csv"), "w", newline="") as f:
        w = csv.writer(f); keys = list(next(iter(res.values())))
        w.writerow(["method"] + keys)
        for m, d in res.items(): w.writerow([m] + [d[k] for k in keys])
    print(f"[written] {out_dir}")


def self_test():
    import tsfnet_model_notebook as M
    torch.manual_seed(0)
    model = M.TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=21, fnirs_temporal_align="egta", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0)
    with torch.no_grad():
        for n_, p_ in model.named_parameters():
            if "egta" in n_ and p_.dim() >= 1:
                p_.copy_(torch.randn_like(p_) * 0.5)
    model.eval()
    xe, xf = torch.randn(6, 16, 16, 600), torch.randn(6, 21, 16, 16, 30, 2)
    try:
        import captum; want = True
    except ImportError:
        want = False; print("[note] captum missing: GradientShap part skipped in the self-test")
    sc, p0, c = explain(model, xe, xf, None, None, want_shap=want)
    for m in METHODS:
        assert sc[m].shape == (6, 21), (m, sc[m].shape)
        if m != "gradient_shap" or want:
            assert np.isfinite(sc[m]).all(), m
    assert (sc["grad_cam"] >= 0).all()
    with torch.no_grad():                                                    # occlusion must equal the explicit re-evaluation
        xa = xf.clone(); xa[:, 4] = 0.0
        ref = (p0 - pc(model(xe, xa, None, None)[3], c)).numpy()
    assert np.allclose(sc["occlusion"][:, 4], ref, atol=1e-6), "occlusion != explicit zeroing impact"
    print("[1] all six explanations have shape (B, 21), are finite, Grad-CAM >= 0, and occlusion equals the explicit zeroing impact")
    rng = np.random.RandomState(0); rm, _ = randomise_upto(model, len(GROUPS) - 1, 3)
    args = argparse.Namespace(k=3)
    rec = process_batch(model, rm.eval(), xe, xf, None, None, args, rng, want)
    for m, d in rec.items():
        assert all(np.isfinite(v).all() for k_, v in d.items() if len(v)), m
        assert len(d["top"]) == 6
    print("[2] randomisation + deletion pipeline runs for every method: " + ", ".join(rec))
    rec2 = {m: {k_: float(np.mean(v)) if len(v) else float("nan") for k_, v in d.items()} for m, d in rec.items()}
    per = {f"VP{i:03d}": rec2 for i in range(6)}
    res = summarise(per, 0.3, verbose=False); assert set(res) >= {"occlusion", "attention"}
    print("[3] group summary runs\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
