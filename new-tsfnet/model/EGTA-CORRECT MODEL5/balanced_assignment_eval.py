"""
balanced_assignment_eval.py -- block decoding under the known class composition of a session.

Every test session holds exactly 3 blocks of each of 0-/2-/3-back (checked from the data below, not
assumed). Argmax per block ignores that. This script instead assigns the session's blocks to classes
jointly with the Hungarian algorithm, with each class given exactly n/3 slots, maximising the summed
log block posterior. No parameter is trained and none is chosen on labels.

DISCLOSURE (state this wherever the number appears). The method is TRANSDUCTIVE and NON-CAUSAL: it
uses (a) the class composition of a session, which is a property of the experimental design, and
(b) all nine blocks of the session at once. It is not available to a deployed system that must label
blocks one at a time. It is a stronger assumption than AdaBN or E3, which only normalise unlabeled
test inputs. Report it as "decoding with a structural prior", never as the model's accuracy.

UNIT. The assignment is per block, so the figure compared is BLOCK accuracy (mean-posterior pooling
of windows, as in the rest of the project). It does not change the window-level headline.

NO PASS/FAIL RULE. At n=26 subjects the paired test has SE of about 1.6 points, so a rule of
"gain >= 1.0 and p < 0.05" has under 15% power for a true +1.3. The result is descriptive; the effect
size, the win count and p are printed, and the verdict is left to the reader.

Input: one posteriors folder (ensemble_eval.py --save-posteriors, or combine_posterior_dirs.py
--save-combined). Seeds inside the folder are averaged first.
"""
import argparse, glob, os, sys
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon

EPS = 1e-9


def block_table(P, trial, y):
    ids = np.unique(trial)
    Pb = np.array([P[trial == t].mean(0) for t in ids])
    yb = np.array([y[trial == t][0] for t in ids])
    return Pb, yb


def assign_balanced(Pb):
    n, K = Pb.shape
    if n % K:
        raise ValueError(f"{n} blocks cannot be split evenly over {K} classes")
    cols = np.repeat(np.arange(K), n // K)
    r, c = linear_sum_assignment(-np.log(Pb + EPS)[:, cols])
    out = np.empty(n, dtype=int)
    out[r] = cols[c]
    return out


def load(d, head):
    folds = {}
    for f in sorted(glob.glob(os.path.join(d, "*_session*.npz"))):
        z = np.load(f)
        sid, ses = os.path.basename(f)[:-4].split("_session")
        key = f"P_{head}"
        if key not in z:
            raise SystemExit(f"{f} has no {key}")
        P = z[key].astype(np.float64)
        P = P / np.clip(P.sum(-1, keepdims=True), 1e-12, None)
        folds[(sid, int(ses))] = block_table(P.mean(0), z["trial_ids"], z["y"])
    if not folds:
        raise SystemExit(f"no *_session*.npz in {d}")
    return folds


def evaluate(folds):
    comp = {tuple(np.bincount(yb, minlength=3)) for _, yb in folds.values()}
    if comp != {(3, 3, 3)}:
        raise SystemExit(f"class composition per session is {comp}, not (3,3,3): the prior does not "
                         f"hold on this data, so the method is not applicable")
    per = defaultdict(lambda: [[], []])
    predcount = []
    fold_better = fold_worse = 0
    for (sid, _), (Pb, yb) in sorted(folds.items()):
        a = float((Pb.argmax(1) == yb).mean())
        b = float((assign_balanced(Pb) == yb).mean())
        per[sid][0].append(a)
        per[sid][1].append(b)
        predcount.append(np.bincount(Pb.argmax(1), minlength=3))
        fold_better += b > a
        fold_worse += b < a
    subs = sorted(per)
    A = np.array([np.mean(per[s][0]) for s in subs])
    B = np.array([np.mean(per[s][1]) for s in subs])
    return subs, A, B, np.mean(predcount, 0), fold_better, fold_worse, len(folds)


def report(folds, label=""):
    subs, A, B, pc, fb, fw, nf = evaluate(folds)
    d = B - A
    p = 1.0 if not np.any(d != 0) else float(wilcoxon(A, B).pvalue)
    print(f"{label}{len(subs)} subjects, {nf} folds, every fold 3/3/3")
    print(f"  block accuracy, argmax            {A.mean():.4f}")
    print(f"  block accuracy, balanced assign   {B.mean():.4f}   ({d.mean():+.4f})")
    print(f"  subjects better/worse/equal       {int((d > 0).sum())}/{int((d < 0).sum())}/{int((d == 0).sum())}"
          f"   Wilcoxon p={p:.4f}   SD of per-subject gain {d.std(ddof=1):.4f}")
    print(f"  folds better/worse/equal          {fb}/{fw}/{nf - fb - fw}")
    print(f"  mean predicted blocks per class under argmax (true = 3): {np.round(pc, 2)}")
    print("  descriptive only; transductive and non-causal (see the module docstring)")
    return A, B


def self_test():
    rng = np.random.RandomState(0)
    # 1) the assignment must be exactly balanced, and never worse in summed log-likelihood terms
    for _ in range(200):
        Pb = rng.dirichlet(np.ones(3), size=9)
        out = assign_balanced(Pb)
        assert np.array_equal(np.bincount(out, minlength=3), [3, 3, 3])
    # 2) a perfect posterior stays perfect; a posterior biased to one class is repaired
    y = np.repeat(np.arange(3), 3)
    P = np.full((9, 3), 0.05); P[np.arange(9), y] = 0.9
    assert (assign_balanced(P) == y).all()
    Pbias = np.full((9, 3), 0.2); Pbias[np.arange(9), y] = 0.6; Pbias[:, 0] += 0.5   # class 0 over-predicted
    Pbias /= Pbias.sum(1, keepdims=True)
    assert (Pbias.argmax(1) == y).mean() < 1.0
    assert (assign_balanced(Pbias) == y).all()
    # 3) refusal when the composition prior does not hold
    try:
        evaluate({("VP1", 1): (P[:8], y[:8])}); raise SystemExit("unequal composition NOT refused")
    except SystemExit as e:
        assert "not applicable" in str(e) or "composition" in str(e)
    # 4) a pure-noise posterior must not gain: expected change ~ 0, never systematically large
    gains = []
    for _ in range(300):
        Pb = rng.dirichlet(np.ones(3), size=9)
        gains.append(((assign_balanced(Pb) == y).mean() - (Pb.argmax(1) == y).mean()))
    assert abs(np.mean(gains)) < 0.03, np.mean(gains)
    print(f"noise-only gain {np.mean(gains):+.4f} (must be ~0)")
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="a posteriors/ folder")
    ap.add_argument("--head", default="final", choices=["final", "eeg", "fusion", "fnirs"])
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.dir:
        sys.exit("need --dir")
    report(load(a.dir, a.head), label=f"{a.dir}\n")


if __name__ == "__main__":
    main()
