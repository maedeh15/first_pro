"""
fnirs_block_position_check.py -- SHORTCUT CONTROL from a trial-level CSV (CPU, seconds). Could slow trends ("time in session": drift, fatigue) give the class away?
  1) the label sequence of every (subject, session) by trial position, and how many DISTINCT sequences exist. If the order were identical everywhere,
     position alone would identify the class (check this FIRST, from the data, not from the dataset description);
  2) a position-only classifier, leave-one-session-out: predict a trial's class as the majority class of that position in the other two sessions
     of the same subject (and, separately, across the other subjects). Its accuracy is the shortcut ceiling; chance is 1/3;
  3) the model's accuracy by trial position (Friedman across positions over subjects: do some positions get systematically more right?).
Input: ensemble_trials.csv (columns subject_id, test_session, trial_id, label, and a prediction column, --pred-col, default pred_ensemble).
  python fnirs_block_position_check.py --trials-csv ensemble_trials.csv
"""
import argparse, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare


def sequences(df):
    """label order of every (subject, session); trial_id may run 0..26 over the whole subject, so the position is the RANK inside the session"""
    seq = {}
    for (s, t), g in df.groupby(["subject_id", "test_session"]):
        seq[(s, t)] = tuple(g.sort_values("trial_id")["label"].tolist())
    return seq


def position_only(seq):
    """two shortcut ceilings from trial position alone
    (a) within subject: how often the test session's label at a position equals the label at that position in the subject's OTHER sessions (chance 1/3;
        0 means the sessions are cyclic shifts of each other, so a position->class rule learned in training sessions is anti-informative on the held-out one);
    (b) session-aware, across subjects: majority class of (session index, position) among the other subjects -- what a model could exploit only if it could
        read BOTH the session index and the position from slow trends."""
    match, hit = [], []
    for (s, t), lab in seq.items():
        for i, y in enumerate(lab):
            oth = [seq[(s, u)][i] for (s2, u) in seq if s2 == s and u != t and len(seq[(s, u)]) > i]
            if oth: match.append(np.mean([o == y for o in oth]))
            others = [seq[k][i] for k in seq if k[0] != s and k[1] == t and len(seq[k]) > i]
            if others: hit.append(Counter(others).most_common(1)[0][0] == y)
    return float(np.mean(match)), float(np.mean(hit)), len(match)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials-csv", required=True); ap.add_argument("--pred-col", default="pred_ensemble"); a = ap.parse_args(argv)
    df = pd.read_csv(a.trials_csv); seq = sequences(df); cnt = Counter(seq.values())
    print(f"{len(seq)} (subject, session) sequences, {len(cnt)} distinct label orders (labels 0/1/2 = 0-/2-/3-back)")
    for s, c in cnt.most_common(6): print(f"  {c:3d} x {''.join(map(str, s))}")
    if len(cnt) > 6: print(f"  ... and {len(cnt) - 6} more")
    by_sess = {t: Counter(v for (s, u), v in seq.items() if u == t) for t in sorted({u for _, u in seq})}
    print("  orders per session index: " + " | ".join(f"session {t}: {len(c)} distinct" for t, c in by_sess.items()))
    within, leak, n = position_only(seq)
    print(f"\n(a) within subject, other sessions: label at the same position matches {within:.3f} of the time (chance 0.333, n={n})")
    print(f"(b) session-aware, other subjects: majority class of (session, position) predicts {leak:.3f}")
    if leak > 0.9 and within < 0.2:
        print("  => the order is a fixed Latin square over sessions: (session, position) fixes the class, but a position->class rule learned on the TRAINING sessions of the same"
              "\n     subject is anti-informative on the held-out session (a). Leave-one-session-out within subject therefore does NOT reward a time-in-session shortcut;"
              "\n     a leak would need a model that knows the held-out session's index AND position (b) -- not available to a within-subject LOSO model.")
    elif max(within, leak) > 0.6:
        print("  => position carries class information that a trend could exploit: run the slow-band-only control.")
    else:
        print("  => position alone is close to chance.")
    df["pos"] = df.groupby(["subject_id", "test_session"])["trial_id"].rank(method="first").astype(int) - 1
    df["ok"] = (df[a.pred_col] == df["label"]).astype(float)
    print(f"\nmodel accuracy ({a.pred_col}) by position inside the session: " + "  ".join(f"{i}:{v:.3f}" for i, v in df.groupby("pos")["ok"].mean().items()))
    print("by session: " + "  ".join(f"{i}:{v:.3f}" for i, v in df.groupby("test_session")["ok"].mean().items()))
    M = df.groupby(["subject_id", "pos"])["ok"].mean().unstack("pos").dropna()
    if len(M) >= 6 and M.shape[1] >= 3:
        print(f"Friedman across the {M.shape[1]} positions over {len(M)} subjects: p={friedmanchisquare(*[M[c].values for c in M.columns]).pvalue:.3g} "
              "(position is confounded with the class sequence, so a difference is NOT by itself evidence of a time trend)")
    return dict(distinct=len(cnt), within=within, leak=leak)


def self_test():
    import tempfile, os
    rng = np.random.RandomState(0); rows = []
    for fixed in (True, False):
        for s in range(8):
            for t in (1, 2, 3):
                order = [0, 1, 2, 1, 2, 0, 2, 0, 1] if fixed else list(rng.permutation([0, 0, 0, 1, 1, 1, 2, 2, 2]))
                for i, y in enumerate(order):
                    rows.append(dict(subject_id=f"{'F' if fixed else 'R'}{s}", test_session=t, trial_id=i, label=y, pred_ensemble=y if rng.rand() < 0.7 else (y + 1) % 3))
    d = pd.DataFrame(rows); p = os.path.join(tempfile.mkdtemp(), "t.csv"); d[d.subject_id.str.startswith("F")].to_csv(p, index=False)
    r = main(["--trials-csv", p]); assert r["distinct"] == 1 and r["within"] > 0.99 and r["leak"] > 0.99
    d[d.subject_id.str.startswith("R")].to_csv(p, index=False); r = main(["--trials-csv", p]); assert r["within"] < 0.6 and r["leak"] < 0.7 and r["distinct"] > 5
    print("SELFTEST PASSED")


if __name__ == "__main__":
    self_test() if "--self-test" in sys.argv else main()
