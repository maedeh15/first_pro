# Run sheet — one integrated model by attaching the closed-form head to the plain trunk

Written and committed BEFORE the run. No training: inference only (GPU or CPU).

## Why
Joint training could not reach the late-fusion accuracy (sections 5-3-18 c, d: gradient head 0.6523, no dropout 0.6432,
frozen closed-form head 0.6481; the trunk degrades when trained against a head that already fits the training windows).
The late-fusion accuracy does not need joint training: take the plain run's trunk unchanged, fit the closed-form head on the
fold's TRAINING sessions, blend 0.5/0.5. The result is ONE nn.Module and ONE checkpoint per fold, so XAI and any other
analysis runs on a single model.

HONEST LABEL: TSFNet + a separately fitted linear member, blended at a fixed weight. Not an end-to-end trained network.
Its accuracy IS the late-fusion accuracy by construction; it is a packaging, not a new result.

## Command (seed 42; writes to NEW folders, never over the baseline's)
%run ensemble_eval.py --data-root "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback" --variant-hash 40b7d8ea53 --seeds 42 --results-dirs "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback_baseline\EGTA\6\results_40b7d8ea53_seed42" --riem-branch member --riem-head-fit lr_all --attach-riem-head --out-dir "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\ensemble_eval_assembled_seed42" --save-assembled-dir "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\assembled_integrated" --save-posteriors

## Expectation (fixed now, from the swap tests in 5-3-18 d and the table in 5-3-1)
Window accuracy about 0.671 for seed 42 (late fusion of the same trunk with the EA/Riemannian member: 0.6708 / 0.6711).
Do NOT pass --expected-window-accs: the record being replayed is the plain run's, not the assembled model's.

## Checks
1. window accuracy within 0.005 of 0.6711;
2. decompose_integrated.py on the new posteriors: (1-b)*P_deep + b*P_riem == P_final (the script aborts otherwise) and P_deep
   equals the baseline posteriors (trunk unchanged).
