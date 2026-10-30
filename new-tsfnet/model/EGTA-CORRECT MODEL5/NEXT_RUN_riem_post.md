# Run sheet — head attached AFTER checkpoint selection (`--riem-head-fit post`)

Written and committed BEFORE the run.

## What it does
The trunk is trained and its checkpoint selected with NO Riemannian head (the loss and the validation score are those of a plain
run). Only after the best checkpoint is restored, and before any test evaluation, a closed-form StandardScaler +
LogisticRegression(C=0.1) is fitted on ALL windows of the training sessions (train + val; never the test session) and blended 0.5/0.5.
The checkpoint is re-saved with the head inside, so ensemble_eval/XAI reload one complete model.

Why: section 5-3-18 d located the gap in two additive causes -- the trunk degrades when trained against a head that already fits
the training windows (-2.4), and a head fitted on fewer windows is weaker (-2.1). This removes both.

HONEST LABEL: trunk and member are trained and fitted separately and joined by a fixed blend: TSFNet + a closed-form linear member
in one script and one checkpoint, NOT an end-to-end trained network.

## Smoke test (one subject, minutes)
%run run_all_subjects.py --subjects 1 --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member --riem-weight-decay 1e-1 --riem-head-fit post
Expect: a `[riem_head_fit] 'post'` line per subject, `adapted 6 BatchNorm layer(s)`.

## Main run (seed 42, reference `84c18210ab` = 0.6523; late-fusion reference for the same seed = 0.6708)
%run run_all_subjects.py --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member --riem-weight-decay 1e-1 --riem-head-fit post

## Decision rule (fixed now)
1. Improvement over `84c18210ab` seed 42: >= +0.010 with paired Wilcoxon p <= 0.05 over 26 subjects.
2. Expectation from the swap tests (not a result): about 0.665 to 0.671. Reaching 0.670 counts as matching late fusion.
   Exceeding 0.680 is not expected and would need an explanation before it is believed.
3. Seed 42 is the selection seed: confirm with 999 and 2024 (same sign, mean gain >= +0.010) before reporting.
4. The trunk is NOT bit-identical to the plain baseline (the unused head still consumes RNG), so differences of a few tenths
   of a point from the plain seed-42 run are noise.

## Validation attempted
test_riem_post.py (CPU, synthetic): no blend before the head exists, exact blend after, the hook runs once inside a real
train_one_fold call, the saved checkpoint holds the head, strict replay equals the reported accuracy. NOT validated here: real data.
