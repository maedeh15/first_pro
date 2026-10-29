# Run sheet — does the in-network Riemannian head improve without input dropout?

Written and committed BEFORE the run, so the decision rule below carries a git timestamp earlier than the result.

## Why this run
decompose_integrated.py (section 5-3-18 c) located the integrated-vs-late-fusion gap in the
in-network linear head (0.6243 window) versus the offline member (0.6449), not in the trunk
(0.6153 vs baseline 0.6167) and not in error correlation. One cause candidate is `Dropout(0.5)` on
the 1624 standardised tangent features, which the offline logistic regression does not have.
This run changes ONE variable: `--riem-dropout 0.0`. Reference: `84c18210ab`.

## Command (one line; seed 42 first)
%run run_all_subjects.py --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member --riem-weight-decay 1e-1 --riem-dropout 0.0

The folder subtag gains `do0` only when the value differs from 0.5, so existing hashes are unchanged.

## Decision rule (fixed now)
1. Primary: window accuracy vs `84c18210ab` seed 42 (0.6523), paired Wilcoxon over 26 subjects.
   Supported if difference >= +0.010 and p <= 0.05.
2. A single positive seed is NOT accepted: seed 42 is the selection seed. Run seeds 999 and 2024
   only if rule 1 holds; accept only if both are same-sign and their mean difference >= +0.010.
3. Mechanism check regardless of rule 1 (inference only): run ensemble_eval.py for the new hash with
   --save-posteriors, then decompose_integrated.py with --member from
   `late_fusion_riemann.py --save-member`. The claim "dropout explains the head gap" needs the
   head-alone accuracy to move toward 0.6449, not only the final number.

## Honest expectation
Reaching 0.67 requires the head to match the offline member (a ~2 point gap). One dropout setting
may close only part of it; a null result here leaves the other candidates (early checkpoint,
AdaBN on features, convex solve) untested and must be reported as exactly that.
