"""
xai_member_coefficients.py -- explain the LINEAR MEMBER of the ~0.67 system: which EEG bands and electrodes drive its decision,
how stable that is, and whether the ranking is CAUSAL (deleting what the coefficients call important hurts more than deleting
what they call unimportant).

Why this script. The existing XAI suite explains the deep trunk. The accuracy of the final system comes from the trunk PLUS a
closed-form logistic regression on multiband tangent-space EEG features, and that member is the part that carries the EEG
information (it is 4.4 points above the trunk's own EEG branch). A linear model is the one component whose explanation is exact,
so it is explained exactly instead of with a post-hoc method.

Feature layout (verified against check_riemannian_eeg.py / build_riem_features.py): 4 bands in the order
[broad, theta 4-8, alpha 8-13, beta 13-30], each the upper triangle of a 28x28 log-covariance (log map at the identity, off-diagonal
entries multiplied by sqrt(2)); D = 4 * 406 = 1624. Every window's covariance is first re-centred and whitened with its OWN session's
mean, so a diagonal entry is the (log) variance of a whitened coordinate relative to the session average, and an off-diagonal entry
is a coupling between two whitened coordinates.

HONEST LIMITS (also printed at the end)
  * whitening mixes electrodes, so "channel c" is a whitened coordinate that is close to, not identical with, electrode c. Read the
    topography as approximate. Activation patterns (Haufe et al. 2014) are computed alongside the weights and scored in --benchmark; neither is preferred a priori.
  * coefficients live in STANDARDISED feature space (each feature rescaled by the training windows' std), the space in which the
    classifier was fitted; they rank features, they are not physical effect sizes.
  * three-class model: importance is computed for the LOAD CONTRAST (3-back minus 0-back logit weights) and for each class separately.
  * the literature checks (frontal theta, parietal alpha) are two pre-stated directional hypotheses on the diagonal of the load contrast,
    run only when electrode names are supplied; everything else is descriptive.

Inputs. Per subject: {sid}_eeg_riem.npy, {sid}_labels.npy, {sid}_session_ids.npy, {sid}_trial_ids.npy. The member of each fold is either
re-fitted exactly as the model does (StandardScaler + LogisticRegression(C=0.1) on the two training sessions; default) or read from the
checkpoints (--results-dir: the riem_head.W / b / mean / scale buffers of {sid}_session{t}_best.pt), which explains the deployed head.
CPU only.
"""
import argparse, glob, os, re, sys
import numpy as np
from scipy.stats import wilcoxon

BANDS = ("broad", "theta", "alpha", "beta")
CLASSES = ("0-back", "2-back", "3-back")


def layout(d_chan=28):
    iu = np.triu_indices(d_chan)
    per_band = len(iu[0])
    return iu, per_band, np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


def contrast(W, a=2, b=0):
    """load contrast: logit weights of class a minus class b (default 3-back minus 0-back)"""
    return W[a] - W[b]


def band_importance(v, d_chan=28):
    """v: (D,) coefficient vector -> per band: squared norm share, and its diagonal (power) / off-diagonal (coupling) split"""
    iu, nb, _ = layout(d_chan)
    diag = iu[0] == iu[1]
    tot = float(np.sum(v ** 2)) + 1e-30
    out = []
    for b in range(len(BANDS)):
        seg = v[b * nb:(b + 1) * nb]
        out.append((float(np.sum(seg ** 2)) / tot, float(np.sum(seg[diag] ** 2)) / tot, float(np.sum(seg[~diag] ** 2)) / tot))
    return np.array(out)                      # (bands, 3): total share, diagonal share, off-diagonal share


def channel_importance(v, d_chan=28):
    """(bands, d_chan): diag_c^2 + half of every coupling entry that touches c, so the rows sum to the band's squared norm"""
    iu, nb, _ = layout(d_chan)
    out = np.zeros((len(BANDS), d_chan))
    for b in range(len(BANDS)):
        seg = v[b * nb:(b + 1) * nb]
        for k, (i, j) in enumerate(zip(*iu)):
            if i == j:
                out[b, i] += seg[k] ** 2
            else:
                out[b, i] += 0.5 * seg[k] ** 2
                out[b, j] += 0.5 * seg[k] ** 2
    return out


def channel_feature_index(c, d_chan=28):
    """indices (over all bands) of every feature that involves whitened coordinate c"""
    iu, nb, _ = layout(d_chan)
    hit = np.flatnonzero((iu[0] == c) | (iu[1] == c))
    return np.concatenate([hit + b * nb for b in range(len(BANDS))])


def band_feature_index(b, d_chan=28):
    nb = layout(d_chan)[1]
    return np.arange(b * nb, (b + 1) * nb)


def fit_member(F, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(F)
    clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(F), y)
    if list(clf.classes_) != [0, 1, 2]:
        raise ValueError(f"fitting windows miss a class: {list(clf.classes_)}")
    return dict(mean=sc.mean_, scale=sc.scale_, W=clf.coef_, b=clf.intercept_)


def head_from_checkpoint(path):
    import torch
    from xai_trunk import load_checkpoint
    sd = load_checkpoint(path, map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    need = ("riem_head.W", "riem_head.b", "riem_head.mean", "riem_head.scale")
    if not all(k in sd for k in need):
        raise SystemExit(f"{path} has no closed-form head buffers {need}; it was not trained with --riem-head-fit lr / lr_all / post / post_oof")
    return dict(mean=sd["riem_head.mean"].numpy().astype(np.float64), scale=sd["riem_head.scale"].numpy().astype(np.float64),
                W=sd["riem_head.W"].numpy().astype(np.float64), b=sd["riem_head.b"].numpy().astype(np.float64))


def predict(head, F, zero_idx=None):
    Z = (F - head["mean"]) / head["scale"]
    if zero_idx is not None and len(zero_idx):
        Z = Z.copy(); Z[:, zero_idx] = 0.0          # 0 in standardised space = the training mean of that feature
    L = Z @ head["W"].T + head["b"]
    return L.argmax(1)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def haufe_pattern(Z, w):
    """Haufe et al. 2014 activation pattern of the linear score s = Z @ w: a = cov(Z, s) / var(s).
    Weights can be large on a feature that carries NO class information (a suppressor that cancels noise shared with another feature);
    the pattern cannot, because it is the covariance of each feature with the score. Z: (n, D) standardised training windows."""
    Zc = Z - Z.mean(0)
    s = Zc @ w
    return (Zc.T @ s) / ((len(s) - 1) * (s.var(ddof=1) + 1e-30))


def auc_of_ranking(importance, truth):
    """probability that a planted feature outranks a non-planted one (Mann-Whitney); 0.5 = chance"""
    from scipy.stats import rankdata
    r = rankdata(importance)
    n1, n0 = int(truth.sum()), int((~truth).sum())
    return float((r[truth].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def planted_benchmark(F, y, ses, delta, rng, n_plant=6, band=1, d_chan=28):
    """GROUND-TRUTH recovery. Add a known load effect (delta x the feature's own std, along -1/0/+1 for 0-/2-/3-back) to n_plant random
    DIAGONAL features of one band of the REAL feature matrix, refit the member on the training sessions, and ask how well the ranking by
    |weight| and by |Haufe pattern| puts the planted features first (AUC). Everything else stays real, including its correlated noise."""
    iu, nb, _ = layout(d_chan)
    diag = np.flatnonzero(iu[0] == iu[1]) + band * nb
    plant = rng.choice(diag, n_plant, replace=False)
    g = np.where(y == 0, -1.0, np.where(y == 2, 1.0, 0.0))
    Fp = F.copy()
    Fp[:, plant] += delta * F[:, plant].std(0) * g[:, None]
    truth = np.zeros(F.shape[1], dtype=bool); truth[plant] = True
    aw, ap = [], []
    for t in np.unique(ses):
        tr = ses != t
        h = fit_member(Fp[tr], y[tr]); c = contrast(h["W"])
        Z = (Fp[tr] - h["mean"]) / h["scale"]
        aw.append(auc_of_ranking(np.abs(c), truth)); ap.append(auc_of_ranking(np.abs(haufe_pattern(Z, c)), truth))
    return float(np.mean(aw)), float(np.mean(ap))


def deletion_tests(head, Fte, yte, imp_channel, k=5, n_random=10, seed=0, d_chan=28):
    """accuracy after deleting (a) each band, (b) the top-k, bottom-k and random-k channels ranked by importance"""
    rng = np.random.RandomState(seed)
    base = float((predict(head, Fte) == yte).mean())
    bands = [base - float((predict(head, Fte, band_feature_index(b, d_chan)) == yte).mean()) for b in range(len(BANDS))]
    order = np.argsort(-imp_channel)
    drop = lambda chs: base - float((predict(head, Fte, np.unique(np.concatenate([channel_feature_index(c, d_chan) for c in chs]))) == yte).mean())
    top, bot = drop(order[:k]), drop(order[-k:])
    rnd = np.mean([drop(rng.choice(d_chan, k, replace=False)) for _ in range(n_random)])
    return base, np.array(bands), top, bot, float(rnd)


def block_permuted_labels(y, ses, tid, rng):
    """shuffle the class label of whole blocks (windows of one trial move together), keeping class counts"""
    key = ses.astype(np.int64) * 100000 + tid.astype(np.int64)
    ub, first = np.unique(key, return_index=True)
    lab = y[first].copy(); rng.shuffle(lab)
    m = dict(zip(ub, lab))
    return np.array([m[k] for k in key])


def run_subject(F, y, ses, tid, head_for_fold, n_null, rng, d_chan=28):
    rows = []
    for t in np.unique(ses):
        te, tr = ses == t, ses != t
        head = head_for_fold(t, F[tr], y[tr])
        c = contrast(head["W"])
        pat = haufe_pattern((F[tr] - head["mean"]) / head["scale"], c)
        base, bdel, top, bot, rnd = deletion_tests(head, F[te], y[te], channel_importance(c, d_chan).sum(0), d_chan=d_chan)
        nulls = []
        for _ in range(n_null):
            yp = block_permuted_labels(y[tr], ses[tr], tid[tr], rng)
            nulls.append(contrast(fit_member(F[tr], yp)["W"]))
        rows.append(dict(session=int(t), W=head["W"], contrast=c, pattern=pat, base=base, band_drop=bdel, top=top, bottom=bot, rand=rnd, nulls=nulls))
    return rows


def summarise(per_subject, names=None, frontal=None, parietal=None, d_chan=28):
    subs = sorted(per_subject)
    out = {}
    # band / power-vs-coupling shares of the load contrast
    bi = np.array([np.mean([band_importance(r["contrast"], d_chan) for r in per_subject[s]], axis=0) for s in subs])
    out["band_share"] = bi.mean(0)
    # stability within subject (folds) vs a block-permutation null, and between subjects
    within, nullw = [], []
    for s in subs:
        rs = per_subject[s]
        cs = [r["contrast"] for r in rs]
        within.append(np.mean([cosine(cs[i], cs[j]) for i in range(len(cs)) for j in range(i + 1, len(cs))]))
        if rs[0]["nulls"]:
            for q in range(len(rs[0]["nulls"])):
                nc = [r["nulls"][q] for r in rs]
                nullw.append(np.mean([cosine(nc[i], nc[j]) for i in range(len(nc)) for j in range(i + 1, len(nc))]))
    out["within_cos"] = np.array(within); out["null_cos"] = np.array(nullw)
    sm = np.array([np.mean([r["contrast"] for r in per_subject[s]], axis=0) for s in subs])
    pair = [cosine(sm[i], sm[j]) for i in range(len(subs)) for j in range(i + 1, len(subs))]
    out["between_cos"] = np.array(pair)
    # causal deletion tests
    out["top"] = np.array([np.mean([r["top"] for r in per_subject[s]]) for s in subs])
    out["bottom"] = np.array([np.mean([r["bottom"] for r in per_subject[s]]) for s in subs])
    out["rand"] = np.array([np.mean([r["rand"] for r in per_subject[s]]) for s in subs])
    out["band_drop"] = np.array([np.mean([r["band_drop"] for r in per_subject[s]], axis=0) for s in subs])
    out["base"] = np.array([np.mean([r["base"] for r in per_subject[s]]) for s in subs])
    out["channel_importance"] = np.array([np.mean([channel_importance(r["contrast"], d_chan) for r in per_subject[s]], axis=0) for s in subs])
    out["subjects"] = subs
    out["pattern_band_share"] = np.array([np.mean([band_importance(r["pattern"], d_chan) for r in per_subject[s]], axis=0) for s in subs]).mean(0)
    from scipy.stats import spearmanr
    out["rank_agree"] = np.array([np.mean([spearmanr(channel_importance(r["contrast"], d_chan).sum(0), channel_importance(r["pattern"], d_chan).sum(0))[0]
                                           for r in per_subject[s]]) for s in subs])
    # pre-stated literature checks on the diagonal of the load contrast (names required)
    if names is not None and frontal and parietal:
        iu = layout(d_chan)[0]; nb = layout(d_chan)[1]; dg = np.flatnonzero(iu[0] == iu[1])
        dc = np.array([np.mean([r["contrast"] for r in per_subject[s]], axis=0) for s in subs])   # (subjects, D)
        def mean_diag(b, chans):
            return dc[:, b * nb + dg[chans]].mean(1)
        theta_f = mean_diag(1, frontal); alpha_p = mean_diag(2, parietal)
        p1 = wilcoxon(theta_f, alternative="greater").pvalue if np.any(theta_f != 0) else 1.0
        p2 = wilcoxon(alpha_p, alternative="less").pvalue if np.any(alpha_p != 0) else 1.0
        out["lit"] = dict(theta_frontal=theta_f, alpha_parietal=alpha_p, p=(p1, p2),
                          holm=(min(1.0, 2 * min(p1, p2)), min(1.0, max(2 * min(p1, p2), p1 if p1 > p2 else p2))))
    return out


def report(o):
    n = len(o["subjects"])
    print(f"\n{n} subjects; load contrast = 3-back minus 0-back logit weights, standardised feature space\n")
    print(f"  {'band':8} {'share of ||w||^2':>17} {'on diagonal (power)':>20} {'off-diagonal (coupling)':>24}   [uniform expectation: 25% / 6.9% of the band]")
    for b, nm in enumerate(BANDS):
        print(f"  {nm:8} {100 * o['band_share'][b, 0]:16.1f}% {100 * o['band_share'][b, 1]:19.1f}% {100 * o['band_share'][b, 2]:23.1f}%")
    w = o["within_cos"]
    print(f"\n  stability of the contrast vector across a subject's folds: cosine {w.mean():.3f} +/- {w.std(ddof=1):.3f}"
          + (f"   (block-permutation null {o['null_cos'].mean():.3f} +/- {o['null_cos'].std(ddof=1) if len(o['null_cos']) > 1 else 0:.3f})" if len(o["null_cos"]) else ""))
    print(f"  agreement between subjects (mean of all pairs): cosine {o['between_cos'].mean():.3f}")
    print("\n  CAUSAL CHECK -- accuracy lost by deleting features (set to the training mean), per subject:")
    def line(lbl, a, bb):
        d = a - bb
        p = 1.0 if not np.any(d != 0) else float(wilcoxon(a, bb).pvalue)
        print(f"    {lbl:46} {a.mean():+.4f} vs {bb.mean():+.4f}   diff {d.mean():+.4f}  larger in {int((d > 0).sum())}/{len(d)}  p={p:.4f}")
    line("top-5 channels by importance vs RANDOM 5", o["top"], o["rand"])
    line("top-5 channels by importance vs BOTTOM 5", o["top"], o["bottom"])
    print("    deleting a whole band (mean accuracy loss): " + "  ".join(f"{nm} {o['band_drop'][:, b].mean():+.4f}" for b, nm in enumerate(BANDS)))
    print(f"    (accuracy of the intact member on the same windows: {o['base'].mean():.4f})")
    if "lit" in o:
        L = o["lit"]
        print("\n  PRE-STATED LITERATURE CHECKS (diagonal of the load contrast, one-sided Wilcoxon over subjects, Holm over the two):")
        print(f"    H1 frontal theta increases with load : mean {L['theta_frontal'].mean():+.4f}, positive in {int((L['theta_frontal'] > 0).sum())}/{n}, p={L['p'][0]:.4f}")
        print(f"    H2 parietal alpha decreases with load: mean {L['alpha_parietal'].mean():+.4f}, negative in {int((L['alpha_parietal'] < 0).sum())}/{n}, p={L['p'][1]:.4f}")
    print(f"\n  HAUFE ACTIVATION PATTERN of the same contrast (what each feature CARRIES, not what the classifier happens to weight):")
    print("    band share of the pattern: " + "  ".join(f"{nm} {100 * o['pattern_band_share'][b, 0]:.1f}%" for b, nm in enumerate(BANDS)))
    ra = o["rank_agree"]
    print(f"    channel ranking by weights vs by pattern: Spearman {ra.mean():+.3f} +/- {ra.std(ddof=1):.3f} across subjects "
          f"(low = weights and pattern answer different questions; the planted benchmark found weights recover planted features better, so report both and prefer neither)")
    print("\n  Limits: whitening mixes electrodes (channel maps are approximate); coefficients rank standardised features, they are not physical effect")
    print("  sizes; the deletion test removes information from one fitted member and does not show the brain uses those features.")


def self_test():
    rng = np.random.RandomState(0)
    d = 6; iu = np.triu_indices(d); nb = len(iu[0]); D = 4 * nb
    # layout helpers
    assert layout(d)[1] == nb == 21 and layout(28)[1] == 406 and 4 * 406 == 1624
    v = rng.randn(D)
    assert abs(channel_importance(v, d).sum() - np.sum(v ** 2)) < 1e-9, "channel shares must sum to ||w||^2"
    assert abs(band_importance(v, d)[:, 0].sum() - 1.0) < 1e-9
    # planted: load changes the THETA diagonal of coordinates 0 and 1 (frontal) and nothing else
    n_ses, nblk, nwin = 3, 9, 15
    ses = np.repeat(np.arange(1, n_ses + 1), nblk * nwin)
    tid = np.tile(np.repeat(np.arange(nblk), nwin), n_ses)
    y = np.tile(np.repeat(np.arange(nblk) % 3, nwin), n_ses)
    F = rng.randn(len(y), D) * 1.0
    diag = np.flatnonzero(iu[0] == iu[1])
    for c in (0, 1):
        F[:, 1 * nb + diag[c]] += 1.2 * y
    heads = lambda t, Ftr, ytr: fit_member(Ftr, ytr)
    rows = run_subject(F, y, ses, tid, heads, n_null=3, rng=rng, d_chan=d)
    per = {"S1": rows}
    # make 4 "subjects" by resampling so group tests have a sample
    for q in range(2, 6):
        Fq = F + rng.randn(*F.shape) * 0.05
        per[f"S{q}"] = run_subject(Fq, y, ses, tid, heads, n_null=3, rng=rng, d_chan=d)
    o = summarise(per, names=list("abcdef"), frontal=[0, 1], parietal=[4, 5], d_chan=d)
    assert o["band_share"][1, 0] == o["band_share"][:, 0].max(), "theta must carry the planted effect"
    assert o["band_share"][1, 1] > 0.5, "...mostly on the diagonal (power), as planted"
    assert o["top"].mean() > o["rand"].mean() + 0.02, (o["top"].mean(), o["rand"].mean())
    assert o["within_cos"].mean() > o["null_cos"].mean() + 0.2, (o["within_cos"].mean(), o["null_cos"].mean())
    assert o["lit"]["theta_frontal"].mean() > 0 and o["lit"]["p"][0] < 0.1
    print(f"[1] planted frontal-theta effect recovered: theta share {o['band_share'][1, 0]:.2f}, diagonal {o['band_share'][1, 1]:.2f}")
    print(f"[2] deleting the top-5 ranked channels costs {o['top'].mean():.3f} vs {o['rand'].mean():.3f} for random 5 -> the ranking is causal here")
    print(f"[3] fold-to-fold cosine {o['within_cos'].mean():.2f} vs permutation null {o['null_cos'].mean():.2f} -> stable and non-trivial")
    # a pure-noise member must NOT show a causal ranking
    Fn = rng.randn(len(y), D); rows_n = {f"N{q}": run_subject(Fn + rng.randn(*Fn.shape) * 0.05, y, ses, tid, heads, 2, rng, d) for q in range(5)}
    on = summarise(rows_n, d_chan=d)
    assert abs(on["top"].mean() - on["rand"].mean()) < 0.03, (on["top"].mean(), on["rand"].mean())
    print(f"[4] pure noise: top-5 vs random deletion differ by {on['top'].mean() - on['rand'].mean():+.3f} (must be ~0) -> no false causal claim")
    # Haufe: a classic suppressor. Feature 0 = signal + noise n; feature 1 = n only. The best linear readout weights feature 1 NEGATIVELY
    # to cancel n, so |weight| ranks feature 1 as important although it carries no class information; the pattern does not.
    n = 4000
    yy = rng.randint(0, 2, n).astype(float); noise = rng.randn(n)
    X = np.c_[yy + noise, noise] + 0.0
    X = np.c_[X, rng.randn(n, 6)]
    Z = (X - X.mean(0)) / X.std(0)
    from sklearn.linear_model import LogisticRegression
    w = LogisticRegression(C=100.0, max_iter=2000).fit(Z, yy).coef_[0]
    a = haufe_pattern(Z, w)
    assert abs(w[1]) > 0.5 * abs(w[0]), (w[0], w[1])
    assert abs(a[1]) < 0.5 * abs(a[0]), (a[0], a[1])
    print(f"[5] suppressor: weights give the noise-only feature {abs(w[1]) / abs(w[0]):.2f}x the signal feature's weight; the pattern gives {abs(a[1]) / abs(a[0]):.2f}x")
    # planted-effect benchmark on correlated-noise features: both rankings should beat chance, and the AUCs are valid probabilities
    d6 = 6; nb6 = len(np.triu_indices(d6)[0]); D6 = 4 * nb6
    L = rng.randn(D6, D6) * 0.35 + np.eye(D6)
    Fb = rng.randn(len(y), D6) @ L
    aw, ap = planted_benchmark(Fb, y, ses, delta=1.0, rng=np.random.RandomState(1), n_plant=3, band=1, d_chan=d6)
    assert 0.5 < aw <= 1.0 and 0.5 < ap <= 1.0, (aw, ap)
    aw0, ap0 = planted_benchmark(Fb, y, ses, delta=0.0, rng=np.random.RandomState(1), n_plant=3, band=1, d_chan=d6)
    assert abs(aw0 - 0.5) < 0.2 and abs(ap0 - 0.5) < 0.2, (aw0, ap0)
    print(f"[6] planted-effect benchmark: AUC weights {aw:.2f}, pattern {ap:.2f} with the effect; {aw0:.2f}/{ap0:.2f} with none (chance)")
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", help="folder with VP001..VP026")
    ap.add_argument("--results-dir", default=None, help="read the head from <results-dir>/{sid}_session{t}_best.pt instead of re-fitting")
    ap.add_argument("--subjects", default="1-26")
    ap.add_argument("--n-null", type=int, default=3, help="block-permutation refits per fold for the stability null (0 = skip)")
    ap.add_argument("--channel-names", default=None, help="text file with the 28 electrode names in data order (one per line or comma separated)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--benchmark", action="store_true",
                    help="ground-truth recovery: plant a known load effect in real features and score weights vs Haufe patterns (no checkpoints needed)")
    ap.add_argument("--delta", type=float, default=0.3, help="planted effect in units of the feature's own std (default 0.3)")
    ap.add_argument("--self-test", action="store_true")
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
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2] if "-" in a.subjects else (a.subjects, a.subjects)
    ids = range(int(lo), int(hi) + 1)
    rng = np.random.RandomState(0)
    if a.benchmark:
        res = []
        for i in ids:
            sid = f"VP{i:03d}"; d = os.path.join(a.data_root, sid)
            try:
                F = np.load(os.path.join(d, f"{sid}_eeg_riem.npy")).astype(np.float64)
                y = np.load(os.path.join(d, f"{sid}_labels.npy")); ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
            except OSError as e:
                print(f"[{sid}] skipped: {e}"); continue
            res.append(planted_benchmark(F, y, ses, a.delta, np.random.RandomState(i)))
            print(f"[{sid}] AUC weights {res[-1][0]:.3f} | pattern {res[-1][1]:.3f}", flush=True)
        r = np.array(res); d_ = r[:, 1] - r[:, 0]
        p = 1.0 if not np.any(d_ != 0) else float(wilcoxon(r[:, 0], r[:, 1]).pvalue)
        print(f"\nPLANTED-EFFECT RECOVERY (delta={a.delta} sd, 6 diagonal theta features, real covariance structure), {len(r)} subjects")
        print(f"  AUC |weight| {r[:, 0].mean():.3f} | AUC |pattern| {r[:, 1].mean():.3f} | pattern minus weight {d_.mean():+.3f}, better in {int((d_ > 0).sum())}/{len(d_)}, p={p:.4f}   (0.5 = chance, 1 = perfect)")
        return
    per = {}
    for i in ids:
        sid = f"VP{i:03d}"; d = os.path.join(a.data_root, sid)
        try:
            F = np.load(os.path.join(d, f"{sid}_eeg_riem.npy")).astype(np.float64)
            y = np.load(os.path.join(d, f"{sid}_labels.npy")); ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
            tid = np.load(os.path.join(d, f"{sid}_trial_ids.npy"))
        except OSError as e:
            print(f"[{sid}] skipped: {e}"); continue
        if a.results_dir:
            hf = lambda t, Ftr, ytr, sid=sid: head_from_checkpoint(os.path.join(a.results_dir, f"{sid}_session{int(t)}_best.pt"))
        else:
            hf = lambda t, Ftr, ytr: fit_member(Ftr, ytr)
        per[sid] = run_subject(F, y, ses, tid, hf, a.n_null, rng)
        print(f"[{sid}] done", flush=True)
    frontal = parietal = None
    if names:
        frontal = [k for k, n in enumerate(names) if re.match(r"^(Fp|AF|F)", n) and not re.match(r"^F[CT]", n)]
        parietal = [k for k, n in enumerate(names) if re.match(r"^(P|PO)", n)]
        print(f"frontal set: {[names[k] for k in frontal]}\nparietal set: {[names[k] for k in parietal]}")
    o = summarise(per, names, frontal, parietal)
    report(o)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        np.savez(os.path.join(a.out_dir, "member_coefficients.npz"), subjects=np.array(o["subjects"]), band_share=o["band_share"],
                 channel_importance=o["channel_importance"], band_drop=o["band_drop"], top=o["top"], rand=o["rand"], bottom=o["bottom"],
                 within_cos=o["within_cos"], null_cos=o["null_cos"], between_cos=o["between_cos"])
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(BANDS, 100 * o["band_share"][:, 1], label="diagonal (power)")
            ax.bar(BANDS, 100 * o["band_share"][:, 2], bottom=100 * o["band_share"][:, 1], label="off-diagonal (coupling)")
            ax.set_ylabel("share of ||w||^2 (%)"); ax.set_title("Load contrast: where the member looks"); ax.legend()
            fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "band_share.png"), dpi=150); plt.close(fig)
            ci = o["channel_importance"].mean(0)
            fig, axs = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
            for b, ax in enumerate(axs):
                ax.bar(range(28), ci[b]); ax.set_title(BANDS[b]); ax.set_xticks(range(28)); ax.set_xticklabels(names if names else range(28), rotation=90, fontsize=6)
            fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "channel_importance.png"), dpi=150); plt.close(fig)
            print("figures + npz ->", a.out_dir)
        except Exception as e:
            print("figures skipped:", e)


if __name__ == "__main__":
    main()
