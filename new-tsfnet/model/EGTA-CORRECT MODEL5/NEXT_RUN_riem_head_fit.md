# Run sheet — closed-form Riemannian head (`--riem-head-fit lr`)

Written and committed BEFORE the run.

## Why
Section 5-3-18 c: the integrated model (0.6562 window, 3 seeds) sits below late fusion (0.6765) because its
gradient-trained linear head is weaker than the offline member (0.6243 vs 0.6497). Swapping the head for the
offline member, keeping the same trunk, gives 0.6745 (+0.0183 over the integrated model, 19/26, p=0.0007).
`--riem-head-fit lr` puts that classifier inside the script: StandardScaler + LogisticRegression(C=0.1) fitted on the
fold's TRAINING windows, held in frozen buffers, blended 0.5/0.5 as before. Trunk and its RNG stream are unchanged.

HONEST LABEL: this is TSFNet plus a separately fitted linear member, not an end-to-end layer. Report it so.

## Smoke test first (one subject, minutes)
%run run_all_subjects.py --subjects 1 --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member --riem-weight-decay 1e-1 --riem-head-fit lr
Expect in the log: a `[riem_head_fit] 'lr'` line, and `adapted 6 BatchNorm layer(s)` (it was 7: the gradient head's BatchNorm is gone).

## Main run (seed 42, reference `84c18210ab` = 0.6523)
%run run_all_subjects.py --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member --riem-weight-decay 1e-1 --riem-head-fit lr

## Decision rule (fixed now)
1. Supported if window accuracy exceeds 0.6523 by >= +0.010 with paired Wilcoxon p <= 0.05 over 26 subjects.
2. Seed 42 is the selection seed: accept only if seeds 999 and 2024 are both positive and their mean gain >= +0.010.
3. Expected size if the swap evidence transfers: about 0.674 (the swap-test value). Reaching 0.670 or more on seed 42
   counts as the target being met; falling short of the swap value is reported as such.
4. Mechanism check (inference only): ensemble_eval.py --riem-head-fit lr --riem-branch member for the new hash, then
   decompose_integrated.py --member ...; the head-alone accuracy must be near the offline member's.

## Exploratory, labelled as such
`--riem-head-fit lr_all` fits on train+val windows (the offline member's exact recipe). The val windows then also inform
the head that checkpoint selection scores, so it is not used for any claim unless `lr` falls short and the gap is shown to be
the lost fitting windows.

## Validation attempted
test_riem_lr_head.py (CPU, no data): forward equals sklearn to 1e-7, no trainable parameters, state_dict round trip,
unfitted/missing-class refusals, RNG-neutral construction (identical trunk init and RNG state vs the default).
NOT validated here: the end-to-end training path on real data -- the smoke test above is the first time it runs.
