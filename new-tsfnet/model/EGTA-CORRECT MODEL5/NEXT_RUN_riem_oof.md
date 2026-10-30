# Run sheet — trunk trained against an HONEST head (`--riem-head-fit post_oof`)

Written and committed BEFORE the run.

## Idea
`post` removed the head during training, so the trunk is just a plain network and the blend with the member happens afterwards.
That throws away what a joint model could in principle do: learn to cover the member's mistakes. The first attempt at joint
training failed because the head had memorised the training windows (right on all of them), leaving the trunk no signal.
`post_oof` keeps the idea and removes the cause: during training the blend uses the head's posterior for each training window
computed OUT OF SESSION (head fitted on the other training session). That posterior is as imperfect as the real head will be on
the test session, so the trunk is trained to be right where the head is wrong. At the end the real head (fitted on both training
sessions) replaces it, exactly as in `post`.

HONEST EXPECTATION: this is a boosting-style change to the training target on 18 blocks, the kind of change that has failed before
in this project (capacity and trunk-training changes). I put the chance of a >= +0.010 gain over `post` at roughly 20%. The OOF head
is fitted on one session (about half the windows), so it is weaker than the final head: the trunk may over-compensate.

## Smoke test (one subject, minutes)
%run run_all_subjects.py --subjects 1 --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member --riem-weight-decay 1e-1 --riem-head-fit post_oof
Expect in the log: `[riem_head_fit] 'post_oof': out-of-session head posteriors ... (accuracy 0.6x ...)` then `'post': trunk selected without a head ...`.
The reported accuracy on that line is the honest out-of-session accuracy of the head; it should be near the head's test accuracy (~0.62-0.65), NOT near 1.0. If it is near 1.0, stop: the OOF construction leaked.

## Main run (seed 42; reference `cf64a71127` = 0.6686, the same model without the OOF training)
%run run_all_subjects.py --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member --riem-weight-decay 1e-1 --riem-head-fit post_oof

## Decision rule (fixed now)
1. Supported if window accuracy exceeds `cf64a71127` seed 42 (0.6686) by >= +0.010 with paired Wilcoxon p <= 0.05, 26 subjects.
2. Seed 42 is the selection seed: accept only if seeds 999 and 2024 are both positive vs their `cf64a71127` counterparts
   (0.6721, 0.6594) and the mean gain is >= +0.010.
3. A gain below +0.010, or a negative one, is reported as a negative result; no variant of the OOF target is tried afterwards on
   the same folds.
4. Mechanism check regardless: ensemble_eval.py --riem-head-fit post_oof --riem-branch member --save-posteriors, then
   decompose_integrated.py: the deep half should be BETTER than the plain baseline's 0.6074 for the idea to be doing what it claims.

## Validation attempted
test_riem_oof.py (CPU, synthetic): out-of-session posteriors do not leak a session-specific pattern; the training-phase blend uses the
carried columns; after fit() they are ignored; a real train_one_fold call with width D+3 features, then strict replay on plain
width-D features, reproduces the reported accuracy. NOT validated here: real data, and whether the trunk gains from it.
