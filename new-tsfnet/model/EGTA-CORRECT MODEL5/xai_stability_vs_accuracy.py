"""
xai_stability_vs_accuracy.py -- CPU only. Does explanation stability across seeds track the subject's accuracy?
(Hypothesis from the 6-subject run: VP006 has both the lowest accuracy and the least stable attention.)
Inputs: xai_stability_per_subject.csv (run xai_stability_check.py --subjects 1-26) and ensemble_folds.csv.
Spearman(ICC, subject accuracy) and Spearman(mean pairwise rho, accuracy); two-sided; exploratory, state so.

    python xai_stability_vs_accuracy.py --stability-csv xai_stability_per_subject.csv --folds-csv ensemble_folds.csv
"""
import argparse
import pandas as pd
from scipy.stats import spearmanr
p = argparse.ArgumentParser(); p.add_argument("--stability-csv", required=True); p.add_argument("--folds-csv", required=True)
a = p.parse_args()
st = pd.read_csv(a.stability_csv); fo = pd.read_csv(a.folds_csv)
scol = next(c for c in fo.columns if c.lower() in ("subject", "subject_id", "sid"))
acol = next((c for c in fo.columns if "ens" in c.lower() and "win" in c.lower() and "acc" in c.lower()),
            next(c for c in fo.columns if "acc" in c.lower()))
acc = fo.groupby(scol)[acol].mean().rename("acc")
d = st.merge(acc, left_on="subject", right_index=True)
print(f"n subjects = {len(d)} (accuracy column: {acol})")
if len(d) < 6:
    raise SystemExit("need >= 6 subjects")
for c in ("icc_2_1", "mean_pairwise_rho"):
    r, pv = spearmanr(d[c], d["acc"]); print(f"  Spearman({c}, accuracy) = {r:+.3f}  p={pv:.4g}  (exploratory, two-sided)")
