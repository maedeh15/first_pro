# Run sheet — XAI ground-truth and sanity runs (written and committed BEFORE the runs)

All three use the accepted baseline flags, seed 42, and a SMALL subject set (1-3): each subject costs ~6 min on the GPU.
None of the thresholds below may be changed after the results are seen.

## A. Randomisation of the TRAINED weights (no training; needs a trained checkpoint set)
%run xai_sanity_check.py --results-dir "<...\results_cf64a71127_seed42>" --data-root "<DATA_ROOT>" --subjects 1-3 --deeplift
Rule: the control `input_magnitude` must read 1.000; a method passes if its mean rho after full randomisation is below 0.3;
DeepLift is reported only if its completeness ratio is below 0.1.

## B. Randomisation of the LABELS (one extra training, 3 subjects; no p-values are used, 3 are enough)
%run run_all_subjects.py --subjects 1-3 --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --shuffle-train-labels 7
%run xai_label_randomisation.py --true-dir "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback_baseline\EGTA\6\results_40b7d8ea53_seed42" --shuffled-dir "<results_<new hash>_seed42>" --data-root "<DATA_ROOT>" --subjects 1-3
Rules: the shuffled network's accuracy on the true-labelled held-out windows must be <= 0.45 (otherwise the shuffle failed and nothing is
interpretable); control = 1.000; a method passes if rho between the two networks' explanations is below 0.3.

## C. Planted-signal recovery (one extra training per strength, SIX subjects: the pre-registered hit@3 test needs >= 6 subjects)
%run run_all_subjects.py --subjects 1-6 --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --plant-fnirs "seg=3,strength=1.0"
%run xai_plant_recovery.py --results-dir "<results_<new hash>_seed42>" --data-root "<DATA_ROOT>" --subjects 1-6 --plant "seg=3,strength=1.0"
Then the same with strength=0.3 (a weaker, more realistic cue): a SEPARATE training (~36 min for six subjects), its own results folder.
Rules: a method RECOVERS the plant when its mean rank percentile of segment 3 is >= 0.90 and hit@3 beats chance (3/21) with p < 0.05; the benchmark is
informative only if zeroing segment 3 costs clearly more accuracy than zeroing any other single segment (otherwise it is reported INCONCLUSIVE).
Note: the planted segment is not at a natural peak of attention; run `seg=17` as well if segment 3 turns out to coincide with where the network
already attends.

## Caveats stated in advance
* with 3 subjects the across-subject tests are underpowered; the per-window statistics and the effect sizes are the informative numbers;
* the trunk is trained WITHOUT any Riemannian head for these runs (the plain baseline), so the explanations are of the plain trunk;
* planting changes the data the network trains on; its accuracy on the planted data is NOT comparable with the thesis accuracies and must not be reported as such.

## D. MSTFDN (separate, not XAI)
%run MSTFDN_train_LOSO_3class.py   (from the mstfdn folder; edit PREPROCESSED_PATH and OUTPUT_PATH at the top first; SEED = 42 is now set and logged)
Report it as a SINGLE-SEED result on MSTFDN's own preprocessing (5 bands x 28 channels, channel-space fNIRS). It is not trained on the TSFNet preprocessing.
