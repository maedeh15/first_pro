"""
xai_member_shap.py -- EXACT SHAP values for the linear member (StandardScaler + logistic regression on the multiband tangent features).

For a linear model the interventional SHAP value of feature j for the logit of class c is exact and closed-form:
    phi_cj(x) = W[c, j] * (x_j - mean_j) / scale_j        (background = the training mean; base value = b_c)
so sum_j phi_cj + b_c equals the logit exactly (completeness, asserted below). No sampling, no estimator error. Features are treated as
independent (interventional SHAP); correlated features share credit arbitrarily, which is why the CAUSAL deletion test is kept next to it.

Per subject and held-out session (head read from the checkpoint with --results-dir, else re-fitted on the training sessions):
  * |SHAP| shares per band and diagonal(power)/off-diagonal(coupling), per channel (diag + half of each coupling entry)
  * load contrast: mean SHAP of (W[3-back] - W[0-back]) over 3-back windows minus over 0-back windows
  * deletion check: accuracy lost by deleting the 5 top channels by mean|SHAP| vs random 5 vs bottom 5 (same test as xai_member_coefficients)
Group: mean over subjects, agreement between subjects (cosine), and -- only with --channel-names (28 names in DATA ORDER, one per line)
-- the two pre-stated literature directions (frontal theta up with load, parietal alpha down) as one-sided Wilcoxon over subjects, Holm over 2.
Whitened coordinates mix electrodes, so channel maps and the frontal/parietal sets are approximate.

  python xai_member_shap.py --data-root "..." --results-dir "...\\results_cf64a71127_seed42" --subjects 1-26 --out-dir "...\\xai_member_shap"
  python xai_member_shap.py --self-test
"""
import argparse, os, re, sys
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai_member_coefficients import (BANDS, layout, fit_member, head_from_checkpoint, predict, channel_feature_index, deletion_tests)


def shap_values(head, F):
    """(n, classes, D) exact interventional SHAP of the logits; also returns the logits for the completeness check"""
    Z = (F - head["mean"]) / head["scale"]
    phi = Z[:, None, :] * head["W"][None, :, :]
    return phi, Z @ head["W"].T + head["b"]


def feature_groups(v, d_chan=28):
    """v (D,) nonnegative importance -> band totals (4,), band diag share (4,), channel importance (4, d_chan)"""
    iu, nb, _ = layout(d_chan)
    diag = iu[0] == iu[1]
    band = np.array([v[b * nb:(b + 1) * nb].sum() for b in range(len(BANDS))])
    dshare = np.array([v[b * nb:(b + 1) * nb][diag].sum() for b in range(len(BANDS))])
    ch = np.zeros((len(BANDS), d_chan))
    for b in range(len(BANDS)):
        seg = v[b * nb:(b + 1) * nb]
        for k, (i, j) in enumerate(zip(*iu)):
            if i == j:
                ch[b, i] += seg[k]
            else:
                ch[b, i] += 0.5 * seg[k]; ch[b, j] += 0.5 * seg[k]
    return band, dshare, ch


def fold_stats(head, F, y, d_chan=28):
    phi, logit = shap_values(head, F)
    assert np.allclose(phi.sum(2) + head["b"], logit, atol=1e-8), "SHAP completeness violated"
    c_ = phi[:, 2, :] - phi[:, 0, :]                                    # contrast SHAP (3-back minus 0-back logit)
    mabs = np.abs(c_).mean(0)
    band, dsh, ch = feature_groups(mabs, d_chan)
    signed = c_[y == 2].mean(0) - c_[y == 0].mean(0) if (y == 2).any() and (y == 0).any() else np.zeros_like(mabs)
    return dict(mabs=mabs, band=band / band.sum(), diag_share=dsh / (band + 1e-30), channel=ch, signed=signed)


def per_subject(F, y, ses, head_for_fold, rng, d_chan=28):
    rows = []
    for t in np.unique(ses):
        te, tr = ses == t, ses != t
        head = head_for_fold(t, F[tr], y[tr])
        st = fold_stats(head, F[te], y[te], d_chan)
        base, bdel, top, bot, rnd = deletion_tests(head, F[te], y[te], st["channel"].sum(0), d_chan=d_chan)
        st.update(base=base, band_drop=bdel, top=top, bottom=bot, rand=rnd)
        rows.append(st)
    return rows


def group_report(per, names=None, d_chan=28):
    subs = sorted(per); n = len(subs)
    avg = lambda k: np.array([np.mean([r[k] for r in per[s]], axis=0) for s in subs])
    band, dsh, ch, signed = avg("band"), avg("diag_share"), avg("channel"), avg("signed")
    print(f"\nEXACT SHAP of the member, load contrast (3-back minus 0-back logit), {n} subjects")
    print("  band      share of mean|SHAP|    diagonal (power) share of the band")
    for b, nm in enumerate(BANDS):
        print(f"  {nm:6s}   {100 * band[:, b].mean():5.1f}% (SD {100 * band[:, b].std(ddof=1):4.1f})        {100 * dsh[:, b].mean():5.1f}%")
    iu, nb, _ = layout(d_chan)
    print(f"  [uniform expectation: 25% per band; diagonal = {100 * d_chan / nb:.1f}% of a band's features]")
    mv = np.array([np.mean([r["mabs"] for r in per[s]], axis=0) for s in subs])
    cos = [float(mv[i] @ mv[j] / (np.linalg.norm(mv[i]) * np.linalg.norm(mv[j]) + 1e-30)) for i in range(n) for j in range(i + 1, n)]
    print(f"  agreement between subjects (cosine of mean|SHAP| vectors, mean of all pairs): {np.mean(cos):.3f}")
    d = lambda k: np.array([[r[k] for r in per[s]] for s in subs], dtype=object)
    top = np.array([np.mean([r["top"] for r in per[s]]) for s in subs]); rnd = np.array([np.mean([r["rand"] for r in per[s]]) for s in subs])
    bot = np.array([np.mean([r["bottom"] for r in per[s]]) for s in subs])
    pv = lambda a, b: 1.0 if not np.any(a - b != 0) else float(wilcoxon(a, b, alternative="greater").pvalue)
    print(f"  CAUSAL check, accuracy lost deleting 5 channels ranked by mean|SHAP|: top {top.mean():+.4f} | random {rnd.mean():+.4f} | bottom {bot.mean():+.4f}")
    print(f"     top > random in {int((top > rnd).sum())}/{n}, p={pv(top, rnd):.4f};   top > bottom in {int((top > bot).sum())}/{n}, p={pv(top, bot):.4f}")
    bd = np.array([np.mean([r["band_drop"] for r in per[s]], axis=0) for s in subs]).mean(0)
    print("  deleting a whole band (mean accuracy loss): " + "  ".join(f"{nm} {bd[b]:+.4f}" for b, nm in enumerate(BANDS)))
    if names:
        fr = [k for k, nm in enumerate(names) if re.match(r"^(Fp|AF|F)", nm) and not re.match(r"^F[CT]", nm)]
        pa = [k for k, nm in enumerate(names) if re.match(r"^(P|PO)", nm)]
        di = [(i, k) for k, (i, j) in enumerate(zip(*iu)) if i == j]
        idx = lambda b, chans: [b * nb + k for (i, k) in di if i in chans]
        th = signed[:, idx(1, fr)].mean(1); al = signed[:, idx(2, pa)].mean(1)
        p1 = pv(th, np.zeros(n)); p2 = float(wilcoxon(-al, alternative="greater").pvalue) if np.any(al != 0) else 1.0
        lo_, hi_ = sorted([p1, p2]); a_lo = min(1.0, 2 * lo_); a_hi = max(a_lo, hi_)
        holm = {1: a_lo if p1 <= p2 else a_hi, 2: a_lo if p2 < p1 else a_hi}
        print(f"  H1 frontal theta evidence for 3-back vs 0-back (signed SHAP, diagonal): mean {th.mean():+.4f}, positive in {int((th > 0).sum())}/{n}, p={p1:.4f} (Holm {holm[1]:.4f})")
        print(f"  H2 parietal alpha evidence for 3-back vs 0-back: mean {al.mean():+.4f}, negative in {int((al < 0).sum())}/{n}, p={p2:.4f} (Holm {holm[2]:.4f})")
        print("  (sign = direction in which the classifier uses the feature, i.e. weight x displacement; not a physiological effect size)")
    else:
        print("  literature directions (frontal theta, parietal alpha) skipped: give --channel-names (28 names in data order).")
    return dict(band=band, diag_share=dsh, channel=ch, signed=signed, mabs=mv, top=top, rand=rnd, bottom=bot, band_drop=bd)


def self_test():
    rng = np.random.RandomState(0)
    n, D = 400, 4 * 406
    F = rng.randn(n, D); y = rng.randint(0, 3, n); F[y == 2, 10] += 2.0; F[y == 0, 10] -= 2.0
    head = fit_member(F, y)
    phi, logit = shap_values(head, F)
    assert np.allclose(phi.sum(2) + head["b"], logit, atol=1e-8)
    print("[1] completeness: sum of SHAP + base equals the logit to 1e-8")
    st = fold_stats(head, F, y)
    assert abs(st["band"].sum() - 1) < 1e-9 and st["channel"].shape == (4, 28)
    assert np.argmax(st["mabs"]) == 10, "the planted feature must carry the largest |SHAP|"
    print("[2] the planted feature (index 10) carries the largest mean|SHAP|; band shares sum to 1")
    ses = np.repeat([1, 2, 3], n // 3 + 1)[:n]
    rows = per_subject(F, y, ses, lambda t, Ftr, ytr: fit_member(Ftr, ytr), rng)
    assert len(rows) == 3
    per = {f"VP{i}": rows for i in range(6)}
    names = [f"F{i}" for i in range(14)] + [f"P{i}" for i in range(14)]
    group_report(per, names)
    print("[3] per-subject + group report (with literature step) run\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root"); ap.add_argument("--results-dir", default=None); ap.add_argument("--subjects", default="1-26")
    ap.add_argument("--channel-names", default=None); ap.add_argument("--out-dir", default=None); ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.data_root:
        sys.exit("need --data-root")
    names = None
    if a.channel_names:
        names = [t.strip() for t in re.split(r"[,\n]", open(a.channel_names, encoding="utf-8").read()) if t.strip()]
        if len(names) != 28:
            sys.exit(f"--channel-names has {len(names)} names, expected 28")
    rng = np.random.RandomState(0); per = {}
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2] if "-" in a.subjects else (a.subjects, a.subjects)
    for i in range(int(lo), int(hi) + 1):
        sid = f"VP{i:03d}"; d = os.path.join(a.data_root, sid)
        try:
            F = np.load(os.path.join(d, f"{sid}_eeg_riem.npy")).astype(np.float64)
            y = np.load(os.path.join(d, f"{sid}_labels.npy")); ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        except OSError as e:
            print(f"[{sid}] skipped: {e}"); continue
        hf = (lambda t, Ftr, ytr, sid=sid: head_from_checkpoint(os.path.join(a.results_dir, f"{sid}_session{int(t)}_best.pt"))) if a.results_dir \
            else (lambda t, Ftr, ytr: fit_member(Ftr, ytr))
        per[sid] = per_subject(F, y, ses, hf, rng); print(f"[{sid}] done", flush=True)
    if len(per) < 6:
        sys.exit(f"only {len(per)} subjects; group statistics need >= 6")
    o = group_report(per, names)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        np.savez(os.path.join(a.out_dir, "member_shap.npz"), subjects=np.array(sorted(per)), **{k: v for k, v in o.items()})
        print("npz ->", a.out_dir)


if __name__ == "__main__":
    main()
