# Run sheet -- which half of the fNIRS 0.01-0.1 Hz band-pass costs accuracy?  (final pipeline, `cf64a71127` family)

Written BEFORE the runs. One variable per run: the fNIRS filter, nothing else.

## Question
Removing the fNIRS band-pass raised accuracy by about 9 points in the earlier (older-pipeline) comparison. The band-pass is a high-pass at 0.01 Hz (drift) PLUS a
low-pass at 0.1 Hz. Which half carries the loss? Test-time band deletion already showed the trained model does not use >0.1 Hz content (cardiac, respiration/Mayer: loss ~ 0)
and depends heavily on <0.1 Hz; it cannot split drift (<0.01 Hz) from the 0.01-0.1 Hz band.

## Arms (all seeds 42 and 2024; 999 only if time allows)
| arm | preprocessing flag | what it keeps | data folder |
|---|---|---|---|
| NOFILTER (reference) | default (`off`) | everything | `TSFNET_PAPER_EXACT_nback` -> existing `cf64a71127` runs, seeds 42/999/2024 |
| HP | `--fnirs-bandpass hp001` | drift removed, everything faster kept | `..._fnirsHP001` |
| LP | `--fnirs-bandpass lp01` | >0.1 Hz removed, drift kept | `..._fnirsLP01` |
| (optional) BP | `--fnirs-bandpass on` | 0.01-0.1 only, in the FINAL pipeline | `..._fnirsBP` |
`lp001` is NOT the low-pass of this experiment (it is 0.01 Hz: the slow-band-only control). Use `lp01`.

## Predictions, fixed now (write your own next to them before running)
| outcome vs NOFILTER (paired, 26 subjects, same seeds) | reading |
|---|---|
| HP clearly lower, LP ~ NOFILTER | the 0.01 Hz high-pass (drift / DC level relative to baseline) costs the accuracy |
| LP clearly lower, HP ~ NOFILTER | removing >0.1 Hz costs it; contradicts the band-deletion result (model did not use >0.1 Hz) and would need an explanation |
| both ~ NOFILTER, BP lower | only the combination hurts (interaction / filter distortion on short segments) |
| both lower | each half costs; suspect filter distortion (`filtfilt` edge effects on 3 s segments) |
"Clearly lower" = paired Wilcoxon p <= 0.05 and mean difference <= -1.0 point on both seeds (same sign on both).

## Steps per arm (CPU preprocessing, then GPU training)
1. Preprocess (all 26 subjects; add your usual preprocessing flags): `python Preproc_tsfnet_nback_badch.py --fnirs-bandpass lp01`
2. Check the EEG did not change (must be True for a few subjects): `np.array_equal(np.load(<base>/VP001/VP001_eeg_windows.npy), np.load(<new>/VP001/VP001_eeg_windows.npy))`,
   same for `_labels.npy`, `_trial_ids.npy`, `_session_ids.npy`. If False the run is a two-variable change: stop.
3. Riemannian features for the new folder (needed by `--riem-branch member`): `python build_riem_features.py --data "<new folder>"`
4. Train with the `cf64a71127` command plus `--data-root`:
   `%run run_all_subjects.py --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member --riem-weight-decay 1e-1 --riem-head-fit post --data-root "<new folder>"`  (then `--seed 2024`)
5. Compare: `%run fnirs_filter_stats.py --a-label nofilter --a 42=<cf64a71127 seed42 csv> 2024=<cf64a71127 seed2024 csv> --b-label lp01 --b 42=<csv> 2024=<csv>`
Cost: about 3-4 h per seed (2 arms x 2 seeds = 12-16 h), plus two preprocessing passes.

## Not closed by this experiment
Within "below 0.1 Hz", real hemodynamic response and slow drift are both present; HP vs LP separates drift removal from fast-content removal, not neural from non-neural.
The defence is the 40 s block (slow by construction) and the Latin-square order (a position->class rule is anti-informative in leave-one-session-out), not a proof of neural origin.
