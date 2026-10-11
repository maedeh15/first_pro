# MGFNet: n-back, leave-one-session-out

Reimplementation of MGFNet (Zhang, Gong & Yuan, *Sensors* 2026, 26, 3402) for the
Shin et al. (2018) n-back task. Everything follows the paper **except the
evaluation protocol**: cross-session leave-one-session-out (LOSO) instead of the
paper's within-subject 80/20 trial split.

## Files

| file | role |
|---|---|
| `Preproc_mgfnet_nback.py` | Sec. 2.2 preprocessing: EEG 28×1000 (30 with `--include-eog`), HbO/HbR 36×50, 5 s / 1 s windows |
| `mgfnet_dataset.py` | Dataset, trial-level 5-fold split, train-only channel normaliser |
| `mgfnet_model.py` | Sec. 3.1–3.4: intra-modal encoders, DilatedConvNet, CGSCR, deep supervision |
| `mgfnet_train.py` | training (paper Sec. 3.5), 5-fold CV epoch selection, LOSO loop, metrics |
| `run_all_subjects_MGFNET.py` | 26-subject sweep, CSV/JSON reports |
| `mgfnet_diagnostics.py` | per-fold diagnostics, run automatically for the subjects in `DIAGNOSTIC_SUBJECTS`: t-SNE train vs test, confusion, calibration/ECE, branch agreement, CGSCR gate / coupling C / token selection / learned Morlet bands |
| `mgfnet_xai.py` | post-hoc XAI from checkpoints: SHAP (EEG topomap, fNIRS channels, temporal), SHAP of the CGSCR gate, 1-D Grad-CAM per branch, modality occlusion, EEG-noise robustness (paper Sec. 4.3 / Fig. 6), single-subject and group t-SNE |
| `crash_safe_logger.py` | same logger as the TSFNet runs |
| `make_mgfnet_fixture.py`, `verify_mgfnet_preproc.py`, `test_clocksync_unit.py`, `test_mgfnet_numerical.py` | tests |

## Run order

```bash
# 0) tests (CPU, no real data)
python test_mgfnet_numerical.py
python test_clocksync_unit.py
python make_mgfnet_fixture.py fixture_data
python Preproc_mgfnet_nback.py --subjects 1 --data-root fixture_data --out fixture_out
python verify_mgfnet_preproc.py fixture_out fixture_data

# 1) preprocessing on the real dataset
python Preproc_mgfnet_nback.py --data-root "E:\kntu\Thesis\codes\DATA" --out "E:\kntu\Thesis\codes\Results\MGFNET_PAPER_EXACT_nback_28ch"
#    -> check the log: every subject must report fnirs_clock_sync=True and
#       marker-onset check [ok]. If the onset check says MISMATCH, rerun with --task-tmin.

# 2) one subject first, then the full sweep (3 seeds)
python run_all_subjects_MGFNET.py --subjects VP001
python run_all_subjects_MGFNET.py --seed 42
python run_all_subjects_MGFNET.py --seed 999
python run_all_subjects_MGFNET.py --seed 2024

# 3) XAI after training (pip install shap scikit-learn)
python mgfnet_xai.py --results-dir <results_mgf_...> --data-root <..._28ch> --subject VP001 --session 1
python mgfnet_xai.py --results-dir <results_mgf_...> --data-root <..._28ch> --group
python mgfnet_xai.py --data-root <..._28ch> --compare-noise "MGFNet=<full dir>" "w/o CGSCR=<noCGSCR dir>"

# 4) Table 4 ablations
python run_all_subjects_MGFNET.py --use-cgscr 0
python run_all_subjects_MGFNET.py --use-cross-modal 0
python run_all_subjects_MGFNET.py --use-intra-modal 0
```

## Protocol

For each subject and each held-out session *s* ∈ {1, 2, 3}:

1. Training data are the other two sessions (18 trials). Test data are session *s* (9 trials), which is never used for training, normalisation or model selection.
2. **Model selection (`--selection cv5`, the default):** 5-fold CV at the trial level inside the training data. One model is trained per fold for 60 epochs. The epoch with the best fold-averaged validation accuracy is selected (ties go to the lower validation loss).
3. **Final model:** retrained on all training data for the selected number of epochs, then evaluated once on session *s*. `--selection fixed` instead trains exactly 60 epochs with no validation.

Training settings follow the paper: Adam, lr 1e-3, weight decay 1e-3, constant LR, batch 64, 60 epochs, no early stopping, and loss = 0.4·CE(hybrid) + 0.1·CE(each of the 6 branches).

Reported metrics: accuracy, macro precision/recall/F1, Cohen's κ, ITR (paper Eqs. 6–7, N=3, T=5 s), confusion matrices, branch accuracies and CGSCR gate statistics.

## Choices the paper leaves open

These are listed in `MGFNET_AMBIGUITIES` (model) and `MGFNET_PREPROC_AMBIGUITIES` (preprocessing). The main ones:

- **EEG channels: 28 scalp channels (default).** The paper's dims, 120 = 4×30 and 264 = 4×66, imply that HEOG/VEOG were also fed to the network (30 channels). n-back load modulates blinks and eye movements, so EOG inputs could let the model decode ocular rather than cortical activity. The default is therefore scalp-only. `--include-eog` in the preprocessing reproduces the paper's 30-channel input; it writes to a separate `_30ch` folder, and the model adapts to the channel count automatically.
- **Normalisation:** not in the paper. A per-channel z-score is fitted on training windows only, inside each LOSO fold, so it is not transductive.
- **CGSCR tokens:** `token_ij = 3·C_ij · LayerNorm(E_i + F_j)`, using a pre-norm Transformer. C is applied *after* the LayerNorm, because LN(c·x) = LN(x) would otherwise erase the coupling.
- **Gumbel-Softmax:** standard `softmax((s+g)/τ)`, hard straight-through in training and one-hot `argmax(s)` at inference.
- **Morlet init:** EEG components start at 2/6/10.5/21.5/35 Hz and fNIRS at 0.03/0.08/0.15 Hz, all inside the preprocessed band.

## Comparing with TSFNet (same preprocessing, same windows)

`Preproc_mgfnet_nback_tsfnetmatched.py` produces MGFNet inputs with preprocessing **identical** to `Preproc_tsfnet_nback_badch.py`, which must sit in the same folder. The EEG chain is not reimplemented: the script imports the TSFNet module and calls its `load_eeg_data` / `preprocess_eeg` (0.5–50 Hz order-6 Butterworth, PyPREP, CAR, ICA with EOG correlation + ICLabel + manual IC overrides, per-channel z-score). fNIRS and windowing use the TSFNet constants, in channel space:

- **fNIRS:** no band-pass (as in the TSFNet file), z-score, then baseline correction over −5…−2 s.
- **Windows:** 10 s, step 1 s, slide −2…42 s. A window is dropped if |z| > 8.
- **Per-session normalisation:** applied as the last step, as in TSFNet.

A window is kept only if TSFNet keeps it, so both models are evaluated on the same samples.

```bash
python Preproc_mgfnet_nback_tsfnetmatched.py --check-against "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"
python run_all_subjects_MGFNET.py --data-root "E:\kntu\Thesis\codes\Results\MGFNET_TSFNETMATCHED_nback"
```

Output shapes: EEG 28×2000 and HbO/HbR 36×100 (10 s).

**Verified on the fixture, against the original TSFNet script on the same data:**
- PyPREP flagged the same bad channels and ICA removed the same components.
- Both produced the same 945 windows, with identical labels, sessions and trials.
- EEG max |diff| = 0.0.
- `spatial_mapping(channel-space fNIRS)` reproduces TSFNet's fNIRS grid with max |diff| = 0.0.

**The only input difference:** MGFNet's fNIRS covers the same 10 s as its EEG window, because the model interpolates fNIRS onto the EEG time axis. TSFNet additionally sees 13 s of fNIRS after the window (21 × 3 s segments).

**Not inductive:** per-session normalisation uses the held-out session's unlabelled statistics, exactly as in the TSFNet pipeline. Disclose this alongside both models' results.
