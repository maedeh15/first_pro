"""
ttime_sml_eval.py -- the ensemble half of T-TIME (Li et al., IEEE TBME): does the spectral
meta-learner (SML) weighting beat the arithmetic mean this project currently uses?

T-TIME has two halves. (1) Target model update: conditional entropy minimisation plus adaptive
marginal distribution regularisation, which needs a GPU pass per fold and is NOT in this script.
(2) Ensemble: instead of averaging the members' posteriors, weight them by the principal
eigenvector of their per-class agreement covariance, estimated on the unlabeled test data alone.
Half (2) costs nothing here because ensemble_eval.py --save-posteriors already wrote every
member's posteriors, so this reads them and re-combines. That makes it the cheap half to test
first, and it is testable on its own: if SML does not beat the mean, the ensemble half is closed
regardless of what the update half would do.

Where this differs from the paper's setting, and it matters for what a null result means:
T-TIME's members come from DIFFERENT source subjects and SML's derivation assumes the members are
conditionally independent given the label. Here the members are seeds of ONE architecture trained
on the SAME subject's two training sessions, so they are strongly correlated and SML's premise is
only weakly met. The paper's own Fig. 4 shows averaging and SML-soft within ~1 point at small M.
Expect little, and read a null as "SML does not help THESE members", not as a refutation of SML.

Two SML variants are reported because the authors' code and paper disagree:
  sml_soft  -- the paper (Sec. III-D): Q_k from the continuous per-class probabilities.
  sml_hard  -- their released ttime_ensemble.py SML_multiclass(), which one-hots the argmax first
               and runs one-vs-rest; its own comment says it "differs from paper description".
Their code takes np.linalg.eig(Q)[1][:, 0], the FIRST eigenvector, which is the principal one only
because Q is near rank one -- not guaranteed. Both variants here use eigh and take the largest
eigenvalue explicitly, so a Q that is not rank one cannot silently pick a minor direction.
"""
import argparse, glob, os, sys
import numpy as np
from scipy.stats import wilcoxon

HEADS = ("final", "eeg", "fusion", "fnirs")


def principal_eigvec(Q):
    """eigenvector of the largest eigenvalue, sign-fixed so the weights are mostly positive"""
    w, V = np.linalg.eigh(Q)                       # ascending, Q symmetric by construction
    v = V[:, -1]
    if v.sum() < 0:
        v = -v
    return v


def sml_soft(P):
    """paper Sec. III-D. P: (M, N, K) posteriors -> (N,) predictions.

    For class k, F_k is the M-vector of the members' probability for k on each sample; Q_k is its
    MxM sample covariance over the N test samples, whose principal eigenvector is proportional to
    the members' balanced accuracies. The members are then weighted per class."""
    M, N, K = P.shape
    score = np.zeros((N, K))
    for k in range(K):
        F = P[:, :, k]                              # (M, N)
        Q = np.cov(F, rowvar=True) if M > 1 else np.zeros((1, 1))
        v = principal_eigvec(np.atleast_2d(Q))
        score[:, k] = F.T @ v
    return score.argmax(1)


def sml_hard(P):
    """their released implementation: one-hot the argmax, then one-vs-rest in {-1,+1}"""
    M, N, K = P.shape
    oh = np.zeros((M, N, K))
    am = P.argmax(2)
    for m in range(M):
        oh[m, np.arange(N), am[m]] = 1
    w = np.zeros(M)
    for k in range(K):
        pred = np.where(am == k, 1.0, -1.0)         # (M, N)
        Q = np.cov(pred, rowvar=True) if M > 1 else np.zeros((1, 1))
        v = principal_eigvec(np.atleast_2d(Q))
        s = v.sum()
        w += v / s if abs(s) > 1e-12 else v
    return np.einsum("m,mnk->nk", w, oh).argmax(1)


def mean_comb(P):
    return P.mean(0).argmax(1)


COMBINERS = {"mean": mean_comb, "sml_soft": sml_soft, "sml_hard": sml_hard}


def block_score(pred_win, y_win, trial_ids):
    """majority vote of a block's window predictions, scored against the block's label"""
    ok = []
    for t in np.unique(trial_ids):
        m = trial_ids == t
        counts = np.bincount(pred_win[m], minlength=3)
        ok.append(counts.argmax() == y_win[m][0])
    return float(np.mean(ok))


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*_session*.npz"))):
        z = np.load(f)
        sid = os.path.basename(f).split("_session")[0]
        out.setdefault(sid, []).append(
            {"y": z["y"], "trial_ids": z["trial_ids"], "seeds": z["seeds"],
             **{h: z[f"P_{h}"] for h in HEADS}})
    if not out:
        raise SystemExit(f"no *_session*.npz in {d}")
    k = {len(f["seeds"]) for folds in out.values() for f in folds}
    if len(k) != 1:
        raise SystemExit(f"folds disagree on the number of members: {sorted(k)}")
    if k.pop() < 2:
        raise SystemExit("SML needs at least 2 ensemble members -- rerun ensemble_eval.py with "
                         "--seeds 42 999 2024 --save-posteriors")
    return out


def evaluate(data, head):
    """per subject: window and block accuracy for each combiner, folds pooled"""
    subs = sorted(data)
    res = {c: {"win": [], "blk": []} for c in COMBINERS}
    for s in subs:
        acc = {c: {"win": [], "blk": [], "n": []} for c in COMBINERS}
        for f in data[s]:
            P, y, tri = f[head], f["y"], f["trial_ids"]
            for c, fn in COMBINERS.items():
                pred = fn(P)
                acc[c]["win"].append(float((pred == y).mean()))
                acc[c]["blk"].append(block_score(pred, y, tri))
        for c in COMBINERS:
            res[c]["win"].append(np.mean(acc[c]["win"]))
            res[c]["blk"].append(np.mean(acc[c]["blk"]))
    return subs, {c: {k: np.array(v) for k, v in d.items()} for c, d in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posteriors", help="the posteriors/ folder written by ensemble_eval.py")
    ap.add_argument("--heads", nargs="+", default=["final"], choices=list(HEADS))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.posteriors:
        sys.exit("need --posteriors")
    data = load_dir(a.posteriors)
    K = len(next(iter(data.values()))[0]["seeds"])
    print(f"{len(data)} subjects, {K} ensemble members "
          f"(seeds {list(next(iter(data.values()))[0]['seeds'])})\n")
    for head in a.heads:
        subs, r = evaluate(data, head)
        print(f"=== head: {head} ===")
        print(f"{'combiner':9s} {'window':>8s} {'block':>8s}")
        for c in COMBINERS:
            print(f"{c:9s} {r[c]['win'].mean():8.4f} {r[c]['blk'].mean():8.4f}")
        print(f"\npaired against 'mean' over {len(subs)} subjects:")
        for c in ("sml_soft", "sml_hard"):
            for lvl in ("win", "blk"):
                d = r[c][lvl] - r["mean"][lvl]
                p = 1.0 if np.allclose(d, 0) else float(wilcoxon(r["mean"][lvl], r[c][lvl]).pvalue)
                print(f"  {c:9s} {lvl:4s} {d.mean():+.4f}  wins {int((d > 0).sum())}/{len(d)}  p={p:.4f}")
        print()
    print("Reading: this is the ENSEMBLE half of T-TIME only. The target-model-update half (CEM +")
    print("adaptive MDR) is not tested here and needs a GPU pass per fold. The members are seeds of")
    print("one architecture on one subject, so SML's conditional-independence premise is weakly met.")


def self_test():
    rng = np.random.default_rng(0)
    N, K = 600, 3
    # three members of KNOWN unequal accuracy; SML should lean on the good one, the mean cannot
    accs = [0.80, 0.50, 0.45]
    y = rng.integers(0, K, N)
    P = np.zeros((3, N, K))
    for m, acc in enumerate(accs):
        ok = rng.random(N) < acc
        pred = np.where(ok, y, (y + rng.integers(1, K, N)) % K)
        P[m] = 0.1 + rng.random((N, K)) * 0.02
        P[m, np.arange(N), pred] += 1.0
        P[m] /= P[m].sum(1, keepdims=True)
    a_mean = (mean_comb(P) == y).mean()
    a_soft = (sml_soft(P) == y).mean()
    a_hard = (sml_hard(P) == y).mean()
    print(f"members {accs} -> mean {a_mean:.4f} | sml_soft {a_soft:.4f} | sml_hard {a_hard:.4f}")
    assert a_soft > a_mean + 0.02 and a_hard > a_mean + 0.02, (a_mean, a_soft, a_hard)
    # identical members: every combiner must agree with them exactly
    Q = np.repeat(P[:1], 3, axis=0)
    assert (mean_comb(Q) == sml_soft(Q)).all() and (mean_comb(Q) == sml_hard(Q)).all()
    # principal_eigvec really takes the LARGEST eigenvalue (the released code takes column 0)
    A = np.diag([0.1, 5.0, 0.2])
    v = principal_eigvec(A)
    assert abs(abs(v[1]) - 1.0) < 1e-12, v
    # block vote: a block whose windows are mostly right is right
    tri = np.repeat(np.arange(10), 20); yb = np.repeat(np.arange(10) % 3, 20)
    pw = yb.copy(); pw[:8] = (yb[:8] + 1) % 3          # 8 of block 0's 20 windows wrong
    assert block_score(pw, yb, tri) == 1.0
    pw[:12] = (yb[:12] + 1) % 3                         # now 12 of 20 wrong -> block 0 lost
    assert abs(block_score(pw, yb, tri) - 0.9) < 1e-12
    print("self-test OK: SML beats the mean on members of unequal accuracy, all combiners agree on"
          " identical members, the largest eigenvalue is used, and the block vote counts correctly")


if __name__ == "__main__":
    main()
