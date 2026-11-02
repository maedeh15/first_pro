"""
xai_gradient_faithfulness.py -- is INPUT-GRADIENT saliency causally faithful? (the gap left by
attention_faithfulness_check.py, which tests only the attention weights)

Per held-out window (target = predicted class c, quantity = P(c), fixed before any ablation):
  (1) per-segment saliency s_k = mean|dP(c)/dx| over segment k  (and grad*input)
  (2) real impact_k = P(c) - P(c | segment k zeroed)           (21 extra passes)
  (3) rho_i = Spearman(s, impact) per window; per-subject mean; one-sided Wilcoxon (rho>0) over subjects
  (4) DELETION: zero the top-3 vs bottom-3 vs random-3 segments by saliency; mean drop of P(c).
      Pre-stated rule: faithful iff top-3 drop > random-3 drop > bottom-3 drop and top-vs-random Wilcoxon p<0.05.
Same checkpoints/AdaBN/loading as attention_faithfulness_check.py. Needs >= 6 subjects for the group tests.

    python xai_gradient_faithfulness.py --results-dir "...\\results_cf64a71127_seed42" --data-root "..." --subjects 1-26
"""
import argparse, csv, json, os, sys
import numpy as np, torch
from scipy.stats import spearmanr, wilcoxon
from torch.utils.data import DataLoader, Subset
from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
from xai_trunk import trunk_state, load_checkpoint, set_xai_output

p = argparse.ArgumentParser(description=__doc__.split("    python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--results-dir", required=True); p.add_argument("--data-root", required=True)
p.add_argument("--subjects", default="1-26"); p.add_argument("--out-dir", default=None)
p.add_argument("--batch-size", type=int, default=16); p.add_argument("--k", type=int, default=3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--adabn", action="store_true", default=True); p.add_argument("--no-adabn", dest="adabn", action="store_false")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
A = p.parse_args()
RNG = np.random.RandomState(A.seed)


def subjects_of(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return [f"VP{i:03d}" for i in out]


def const(v):
    return np.ptp(v) <= 1e-9 * (np.abs(v).max() + 1e-300)


def model_for(S):
    return TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=S,
                  fnirs_temporal_align="egta", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0)


def seg_mean(v):
    return v.abs().mean(dim=tuple(range(2, v.dim())))


def pc(y_pred, c):
    return y_pred.gather(1, c.view(-1, 1)).squeeze(1)


def batch_stats(model, xe, xf, xc, xh):
    with torch.no_grad():
        yp0 = model(xe, xf, xc, xh)[3]
    c = yp0.argmax(1); p0 = pc(yp0, c)
    xg = xf.clone().requires_grad_(True)
    yp = model(xe, xg, xc, xh)[3]
    g = torch.autograd.grad(torch.log(pc(yp, c).clamp_min(1e-8)).sum(), xg)[0].detach()
    sal = seg_mean(g).cpu().numpy(); gxi = seg_mean(g * xf).cpu().numpy()
    B, S = sal.shape
    imp = np.zeros((B, S))
    with torch.no_grad():
        for s in range(S):
            xa = xf.clone(); xa[:, s] = 0.0
            imp[:, s] = (p0 - pc(model(xe, xa, xc, xh)[3], c)).cpu().numpy()
        drops = {m: [] for m in ("top", "bottom", "random")}
        for b in range(B):
            order = np.argsort(-sal[b])
            sets = {"top": order[:A.k], "bottom": order[-A.k:], "random": RNG.choice(S, A.k, replace=False)}
            for m, idx in sets.items():
                xa = xf[b:b + 1].clone(); xa[:, idx] = 0.0
                drops[m].append(float(p0[b] - pc(model(xe[b:b + 1], xa, None if xc is None else xc[b:b + 1],
                                                       None if xh is None else xh[b:b + 1])[3], c[b:b + 1])[0]))
    return sal, gxi, imp, drops


def subject(sid):
    full = NBackDataset(os.path.join(A.data_root, sid), sid)
    S = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    rho = {"input_gradient": [], "grad_x_input": []}; dr = {"top": [], "bottom": [], "random": []}
    for ses in (1, 2, 3):
        ck = os.path.join(A.results_dir, f"{sid}_session{ses}_best.pt")
        if not os.path.isfile(ck):
            print(f"[{sid} s{ses}] SKIP no checkpoint"); continue
        m = model_for(S).to(A.device); m.load_state_dict(trunk_state(load_checkpoint(ck, map_location=A.device)), strict=True); set_xai_output(m); m.eval()
        dl = DataLoader(Subset(full, np.flatnonzero(full.session_ids == ses)), batch_size=A.batch_size, shuffle=False)
        if A.adabn:
            adapt_batchnorm_to_target(m, dl, A.device, log_fn=lambda *a, **k: None)
        for batch in dl:
            xe, xf, y, xc, xh = _unpack_batch(batch)
            xe, xf = xe.to(A.device), xf.to(A.device)
            xc = xc.to(A.device) if xc is not None else None; xh = xh.to(A.device) if xh is not None else None
            sal, gxi, imp, d = batch_stats(m, xe, xf, xc, xh)
            for b in range(len(imp)):
                if const(imp[b]):
                    continue
                for name, v in (("input_gradient", sal[b]), ("grad_x_input", gxi[b])):
                    if not const(v):
                        r = spearmanr(v, imp[b])[0]
                        if not np.isnan(r): rho[name].append(r)
            for k in dr: dr[k] += d[k]
        del m
    return rho, dr


def main():
    out = A.out_dir or os.path.join(A.results_dir, "gradient_faithfulness"); os.makedirs(out, exist_ok=True)
    per = {}
    for sid in subjects_of(A.subjects):
        try:
            rho, dr = subject(sid)
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}"); continue
        if not rho["input_gradient"]:
            continue
        per[sid] = {"rho_ig": float(np.mean(rho["input_gradient"])), "rho_gxi": float(np.mean(rho["grad_x_input"] or [np.nan])),
                    "n": len(rho["input_gradient"]), **{f"drop_{k}": float(np.mean(v)) for k, v in dr.items()}}
        r = per[sid]
        print(f"[{sid}] n={r['n']} rho(grad)={r['rho_ig']:+.3f} rho(grad*x)={r['rho_gxi']:+.3f} | drop top {r['drop_top']:+.4f} random {r['drop_random']:+.4f} bottom {r['drop_bottom']:+.4f}")
    ids = list(per)
    if len(ids) < 6:
        sys.exit(f"ERROR: {len(ids)} usable subjects; group tests need >= 6 (per-subject lines above are valid).")
    col = lambda k: np.array([per[s][k] for s in ids])
    print("\n" + "=" * 90 + f"\nGRADIENT FAITHFULNESS -- {len(ids)} subjects (target = predicted class, k={A.k})\n" + "=" * 90)
    summ = {"per_subject": per, "n": len(ids)}
    for name, k in (("input_gradient", "rho_ig"), ("grad*input", "rho_gxi")):
        v = col(k); v = v[~np.isnan(v)]
        pv = wilcoxon(v, alternative="greater").pvalue
        print(f"  {name:15s} mean rho {v.mean():+.3f}, positive in {(v > 0).sum()}/{len(v)}, Wilcoxon(rho>0) p={pv:.4g}")
        summ[name] = {"mean_rho": float(v.mean()), "p": float(pv)}
    t, r_, b_ = col("drop_top"), col("drop_random"), col("drop_bottom")
    p_tr = wilcoxon(t - r_, alternative="greater").pvalue; p_rb = wilcoxon(r_ - b_, alternative="greater").pvalue
    print(f"  deletion of top-{A.k} by saliency: {t.mean():+.4f}  random: {r_.mean():+.4f}  bottom: {b_.mean():+.4f}")
    print(f"    top > random in {(t > r_).sum()}/{len(t)}, p={p_tr:.4g};  random > bottom in {(r_ > b_).sum()}/{len(t)}, p={p_rb:.4g}")
    faithful = t.mean() > r_.mean() > b_.mean() and p_tr < 0.05
    print("\nVERDICT (rule stated in the docstring): " + ("input-gradient saliency is causally FAITHFUL at segment level (top > random > bottom, p<0.05)."
          if faithful else "NOT shown faithful at segment level. Report as: passes sanity checks but is not shown to mark causally important segments."))
    summ.update({"deletion": {"top": float(t.mean()), "random": float(r_.mean()), "bottom": float(b_.mean()), "p_top_gt_random": float(p_tr), "p_random_gt_bottom": float(p_rb)}, "faithful": bool(faithful)})
    json.dump(summ, open(os.path.join(out, "gradient_faithfulness_summary.json"), "w"), indent=2, default=float)
    with open(os.path.join(out, "gradient_faithfulness_per_subject.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject"] + list(per[ids[0]]))
        for s in ids: w.writerow([s] + list(per[s].values()))
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
