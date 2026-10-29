# ============================================================================
# FULL BASELINE RUN -- all 26 subjects, subject-dependent (paper-exact
# protocol: Sec. 3.6), for later comparison against any changes you make.
#
# Paste this as CELL 9 in tsfnet_model_notebook.py (after CELL 8's imports/
# setup are already in the kernel), or run standalone with %run after
# %run tsfnet_model_notebook.py and %run crash_safe_logger.py.
#
# WHY THIS SHAPE:
#   - run_diagnostics_flag=False by default here: PCA/t-SNE/saliency for
#     26 subjects x 3 sessions = 78 diagnostic runs would take a very long
#     time and isn't needed for a baseline comparison run (accuracy/kappa
#     numbers are what you compare across changes). Diagnostics are for
#     drilling into ONE subject's behavior, not for a 26-subject sweep.
#     Set RUN_DIAGNOSTICS=True below if you want them anyway.
#   - Per-subject try/except: one subject crashing (e.g. a corrupted .npy,
#     an out-of-memory spike) does NOT stop the other 25 -- this matters a
#     lot for an unattended multi-hour run.
#   - CrashSafeLogger wraps the WHOLE run: every subject's full training
#     log is on disk in real time, so even a full kernel crash partway
#     through subject 15 still leaves you with subjects 1-14's complete
#     logs (and the results/checkpoints already written by
#     run_subject_cross_session for each finished subject).
#   - A combined summary (baseline_all_subjects_<timestamp>.csv/.json) is
#     written incrementally (after EACH subject, not just at the end), so
#     partial results are never lost even if the run stops partway.
# ============================================================================
import os
import csv
import json
import time
import datetime
import traceback
import argparse
import random

# Make this script self-sufficient regardless of what was run in earlier
# cells: explicitly import CrashSafeLogger here rather than relying on it
# already being in the notebook's namespace from a previous %run. Requires
# crash_safe_logger.py to be in the same folder as this script (or on
# sys.path). If CrashSafeLogger truly isn't available for any other reason,
# a plain no-op fallback context manager is used instead of crashing --
# real-time crash-safe logging is a nice-to-have, not something that should
# ever block your 26-subject run from starting.
try:
    from crash_safe_logger import CrashSafeLogger
except ImportError:
    print("WARNING: could not import CrashSafeLogger from crash_safe_logger.py "
          "(is it in the same folder as this script?). Falling back to plain "
          "print() with no crash-safe file logging for this run.")

    class CrashSafeLogger:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

import torch   # needed for torch.cuda.is_available() below; harmless re-import if already loaded

# ----------------------------------------------------------------------------
# SEED CONTROL -- for the multi-seed ablation-vs-baseline statistical comparison.
# By default this script previously had NO fixed seed at all, meaning every run
# was a random draw from the training-stochasticity distribution (weight
# initialization, batch order, dropout masks, etc.) -- this is exactly the
# noise source that made the 11-seg vs 20-seg paired comparison statistically
# non-significant despite a consistent positive mean effect. Running the SAME
# preprocessing output with 3 different fixed seeds and averaging each
# subject's result across those 3 runs reduces this noise component by
# ~1/sqrt(3) (~42% reduction in std), making the true effect (if any) easier
# to detect without collecting any new data.
#
# Usage:
#   python run_all_subjects.py --seed 42
#   python run_all_subjects.py --seed 123
#   python run_all_subjects.py --seed 2024
# (Run three times, once per seed value, for both the 11-segment and
# 20-segment pipelines -- 6 total runs -- then average_seeds_and_compare.py
# aggregates all of them.)
_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--seed", type=int, default=42,
                          help="Random seed for numpy/torch/python random, for reproducibility "
                               "and for the multi-seed noise-averaging ablation comparison.")
# RESULT OF PRIORITY-1 SUB-EXPERIMENT (gradient clipping, norm=1.0): REJECTED.
# Both seeds (42, 2024) showed a consistent drop vs. the no-clip baseline
# (overall acc -1.57pp, kappa -0.023, fusion-branch acc -2.06pp), matching
# the same "EEG branch improves alone, fusion/overall degrades" pattern seen
# in every preprocessing ablation (AutoReject, ICLabel, SSP). Default is now
# False (no clipping) so this script reproduces the accepted baseline unless
# explicitly re-enabled with --grad-clip (e.g. to revisit with a larger norm
# such as 5.0 later, if time permits).
_arg_parser.add_argument("--grad-clip", dest="grad_clip", action="store_true",
                          help="Re-enable gradient clipping (norm=1.0), which was tested "
                               "and REJECTED in the priority-1 sub-experiment. Off by default.")
_arg_parser.set_defaults(grad_clip=False)

# ISOLATED ABLATION FLAG (priority-2 sub-experiment: fusion combination formula).
# Default is now "paper_mean" (Eq. 4 exactly: independent Sigmoid per weight,
# plain mean / divide-by-3, no renormalization) -- this is the ONLY variable
# this script changes relative to the accepted priority-1 baseline (no grad
# clipping); everything else (epochs, patience, lr, weight_decay, class
# weights, CAFE dim) is untouched. Pass --fusion-combination wsum to
# reproduce the prior (renormalized-by-sum) behavior for a direct A/B run.
# ISOLATED ABLATION FLAG (priority-noiseaware). Train-time AWGN on the EEG
# stream only, per MGFNet (Zhang et al., Sensors 2026, 26, 3402) Eqs. 8-9.
# DEFAULT IS A NO-OP: --noise-aware-prob 0.0 means inject_eeg_awgn returns
# its input unchanged, so leaving these alone reproduces prior behavior
# bit-for-bit (rule 5). MGFNet's setting was prob=0.8 over
# {10,5,0,-5,-10} dB, leaving 20% of training samples clean.
_arg_parser.add_argument("--noise-aware-prob", type=float, default=0.0,
                          help="Per-sample probability of injecting AWGN into the EEG "
                               "stream during TRAINING only (0.0 = off, the default; "
                               "MGFNet used 0.8). fNIRS is never corrupted. Validation "
                               "and test passes always use clean data.")
_arg_parser.add_argument("--noise-aware-snr-db", type=float, nargs="+",
                          default=[10.0, 5.0, 0.0, -5.0, -10.0],
                          help="Discrete SNR values (dB) sampled uniformly per corrupted "
                               "training sample. Only has an effect when "
                               "--noise-aware-prob > 0. Default matches MGFNet.")

_arg_parser.add_argument("--fusion-combination", type=str, default="paper_mean",
                          choices=["paper_mean", "wsum", "gated", "stanet2", "stanet3"],
                          help="Final decision-level combination formula for y_eeg/y_fusion/"
                               "y_fnirs. 'paper_mean' (default): Eq.4 as written, sigmoid+mean/3. "
                               "'wsum': prior renormalized-by-sum behavior, for comparison only. "
                               "'gated': priority-gating experiment -- per-sample softmax weights "
                               "from a small linear layer over the concatenated branch logits "
                               "(mixture-of-experts style; see test_gating_numerical.py), instead "
                               "of the three global scalars w_eeg/w_fusion/w_fnirs. "
                               "'stanet2' / 'stanet3': priority-DECFUSION, STA-Net Eq. 5-6 -- "
                               "per-sample softmax weights from the branch classifiers' hidden "
                               "FEATURES; stanet2 = fusion + fNIRS only (EEG out of the final "
                               "decision, as in STA-Net), stanet3 = all three branches.")

# ISOLATED ABLATION FLAG (priority-3 family: class weighting). Default
# "none" so a plain `python run_all_subjects.py --seed X` still reproduces
# the accepted priority-1+2 baseline exactly.
#   --class-weighting-scheme inverse_freq       -> priority-3a: TESTED, REJECTED
#     (26 subj x 3 seeds: acc -1.17pp; 13-subject confusion-matrix check:
#     minority-class recall did NOT improve either, -0.043/-0.020; cross-
#     subject std=5.78pp, worst case -12.47pp)
#   --class-weighting-scheme effective_number   -> priority-3b: TESTED, ACCEPTED
#     (beta=0.9; 2 seeds: acc +0.20pp, kappa +0.0031 vs. baseline, stable)
#   --class-weighting-scheme focal               -> priority-3c: NEW, testing now
#     (Lin et al. 2017; per-SAMPLE (1-p_t)^gamma focusing, no per-class
#     alpha -- an independent axis from 3a/3b's fixed per-class weight)
# This is the ONLY variable this script changes relative to the accepted
# baseline when a non-"none" scheme is used; everything else (epochs,
# patience, lr, weight_decay, combination formula, CAFE dim) is untouched.
_arg_parser.add_argument("--class-weighting-scheme", type=str, default="none",
                          choices=["none", "inverse_freq", "effective_number", "focal"],
                          help="Loss class-weighting scheme (priority-3 family). 'none' "
                               "(default): unweighted, matches accepted baseline. "
                               "'inverse_freq': priority-3a, tested and REJECTED. "
                               "'effective_number': priority-3b (Cui et al. 2019), ACCEPTED "
                               "with beta=0.9. 'focal': priority-3c (Lin et al. 2017), new.")
_arg_parser.add_argument("--effective-number-beta", type=float, default=0.999,
                          help="beta hyperparameter for --class-weighting-scheme "
                               "effective_number (Cui et al. 2019 use 0.9/0.99/0.999/0.9999; "
                               "0.9 was the ACCEPTED value in this project). "
                               "Ignored for other schemes. Default 0.999.")
_arg_parser.add_argument("--focal-gamma", type=float, default=2.0,
                          help="gamma hyperparameter for --class-weighting-scheme focal "
                               "(Lin et al. 2017 use gamma=2.0 as their main setting). "
                               "Ignored for other schemes. Default 2.0.")
# ISOLATED ABLATION FLAG (priority-3d sub-experiment: SMOTE, a DATA-level
# axis independent of --class-weighting-scheme's LOSS-level axis). Default
# "none" so a plain run still reproduces the accepted baseline exactly.
_arg_parser.add_argument("--oversampling-scheme", type=str, default="none",
                          choices=["none", "smote"],
                          help="Training-data oversampling scheme (priority-3d, independent "
                               "of --class-weighting-scheme). 'none' (default): matches "
                               "accepted baseline. 'smote': Chawla et al. 2002, adapted for "
                               "paired multimodal EEG+fNIRS windows -- see "
                               "smote_oversample_fold()'s docstring for methodological caveats.")
_arg_parser.add_argument("--smote-k-neighbors", type=int, default=5,
                          help="k for SMOTE's same-class nearest-neighbor search (Chawla et "
                               "al. 2002 default: 5). Ignored unless --oversampling-scheme smote.")
# ISOLATED ABLATION FLAG (priority-5: EEG branch temporal pooling). This is
# the project's largest architecture change (documented as "RUN 3A" / item
# #12 in the architecture-changes list). Default "gap" reproduces the
# paper's exact method; pass --eeg-temporal-mode bilstm to test the
# BiLSTM-based replacement, isolated to the EEG-only classification branch
# alone (see EEGLikeBranch's docstring in the notebook for the full design).
_arg_parser.add_argument("--eeg-temporal-mode", type=str, default="gap",
                          choices=["gap", "gap_logvar", "bilstm"],
                          help="EEG branch's temporal pooling for its OWN classifier (y_eeg) "
                               "only -- never affects EFGF/Fusion/CAFE. 'gap' (default): "
                               "paper-exact Global Average Pooling over time. 'bilstm': "
                               "priority-5, replace GAP with a bidirectional LSTM's last "
                               "time-step output. 'gap_logvar': priority-LOGVAR, GAP plus "
                               "ShallowConvNet-style log-variance over time (band-power proxy), "
                               "concatenated -> 1024-dim input to the EEG classifier.")
_arg_parser.add_argument("--eeg-lstm-hidden", type=int, default=64,
                          help="hidden_size for --eeg-temporal-mode bilstm's LSTM (bidirectional, "
                               "so output dim = 2x this). Ignored for 'gap'. Default 64.")
# BUG NOTICE (see EEGLikeBranch's docstring in the notebook for full detail):
# the FIRST priority-5 run used "last_step" (lstm_out[:, -1, :]), which for a
# bidirectional LSTM only correctly summarizes the FORWARD direction over the
# whole sequence -- the backward half at that index only saw ONE time step.
# Default is now "h_n" (the bug-fixed, correct bidirectional summary using
# nn.LSTM's own final hidden states). Pass --eeg-readout-mode last_step ONLY
# to exactly reproduce that earlier (buggy) run for a direct comparison.
_arg_parser.add_argument("--eeg-readout-mode", type=str, default="h_n",
                          choices=["h_n", "last_step", "mean_pool"],
                          help="How to reduce the BiLSTM's per-time-step outputs to one feature "
                               "vector (only used with --eeg-temporal-mode bilstm). 'h_n' "
                               "(default, FIXED): correct forward+backward whole-sequence "
                               "summary. 'last_step' (BUGGY, A/B comparison only): naive "
                               "lstm_out[:,-1,:] indexing -- backward half only saw 1 time step. "
                               "'mean_pool': average the BiLSTM's outputs over the full sequence.")
# Off by default -- generating PCA/t-SNE/saliency diagnostics for all 26
# subjects x 3 sessions (78 diagnostic runs) noticeably slows down a full
# sweep. Pass --diagnostics only for a specific run you want to inspect
# closely (e.g. the final winning variant), not for every ablation sweep.
_arg_parser.add_argument("--diagnostics", dest="run_diagnostics", action="store_true",
                          help="Also generate PCA/t-SNE/saliency plots per subject/session "
                               "(slow: 78 extra diagnostic runs for 26 subjects x 3 sessions). "
                               "Off by default.")
_arg_parser.set_defaults(run_diagnostics=False)
# ISOLATED ABLATION FLAG (priority-4, deferred until after priority-5 per
# project decision). Default 64 reproduces the accepted baseline. See
# run_subject_cross_session's docstring for why 128 (not the paper's
# original 256, already independently ruled out) is the value under test.
# ISOLATED ABLATION FLAGS (priority-A1: TAFN-style asymmetric loss term).
# Motivated directly by this project's confusion-matrix audit: summed over 16
# subjects x 3 folds, 3649 of 6124 errors (59.6%) are 2-back<->3-back
# confusions, while 0-back recall is 68.1%. Every architecture/capacity change
# tried so far (AutoReject, ICLabel, SSP, BiLSTM, CAFE-128) offered only a
# GENERAL improvement and was rejected; this is the first change that targets
# the actual, localized failure mode.
# Class indices: 0 = 0-back, 1 = 2-back, 2 = 3-back. Default "1,2" therefore
# means the 2-back/3-back pair, applied symmetrically.
_arg_parser.add_argument("--asymmetric-pairs", type=str, default="",
                          help="Comma/semicolon list of class-index pairs to penalize extra, "
                               "e.g. '1,2' for the 2-back/3-back pair, or '1,2;0,1' for two "
                               "pairs. Empty (default) disables the asymmetric term entirely "
                               "and reproduces the accepted baseline.")
_arg_parser.add_argument("--asymmetric-weight", type=float, default=2.0,
                          help="Cost multiplier for the pairs given by --asymmetric-pairs. "
                               "TAFN's reference config uses asymmetric weights [1,2], i.e. 2.0. "
                               "Ignored when --asymmetric-pairs is empty.")
_arg_parser.add_argument("--ordinal-reg-lambda", type=float, default=0.0,
                          help="Weight of TAFN's smoothness regularization term (Eq.14). The "
                               "reference implementation differences ADJACENT CLASS scores, "
                               "which for this project's ordinal classes (0<2<3-back load) acts "
                               "as an ordinal prior. Reference value 1e-4; 0.0 (default) "
                               "disables it so it can be ablated separately.")
# NEW (priority-A3, isolated ablation): asymmetric branch-loss weights. Paper's
# Eq. 6 uses lambda_eeg=lambda_fnirs=0.2 (equal weight for both auxiliary
# branch losses). Motivation for testing asymmetric values: the A2 checkpoint-
# inference audit (2026-09-03) found the EEG branch's own classifier head is
# much weaker overall (48.12% vs fNIRS's 58.09%) but BOTH branches confuse the
# 2-back/3-back pair almost identically (52.66% vs 52.93% of their own
# errors) -- i.e. fNIRS's extra strength is concentrated in separating
# 0-back, not in resolving the actual bottleneck. Given that, an asymmetric
# lambda_eeg < lambda_fnirs (pushing the EEG branch's auxiliary loss to
# matter less, since Fusion/CAFE already outperform it) is the most direct
# test of whether re-balancing branch supervision helps at all, though A2's
# finding of a SHARED (not EEG-specific) 2/3-back bottleneck means this is
# not expected to be a large effect.
_arg_parser.add_argument("--lambda-eeg", type=float, default=0.2,
                          help="Weight of the EEG branch's own auxiliary NLL loss term "
                               "in tsfnet_loss (Eq. 6). Paper default 0.2. Ignored effect "
                               "on the branch's own predictions -- only affects how much "
                               "its loss contributes to backprop.")
_arg_parser.add_argument("--lambda-fnirs", type=float, default=0.2,
                          help="Weight of the fNIRS branch's own auxiliary NLL loss term "
                               "in tsfnet_loss (Eq. 6). Paper default 0.2.")
_arg_parser.add_argument("--cafe-d-model", type=int, default=64,
                          help="CAFE cross-attention dimension (priority-4). Project's "
                               "current accepted default is 64; testing 128 here. Watch "
                               "overfitting_gap_val_vs_test, not just accuracy.")
_arg_parser.add_argument("--fnirs-num-segments", type=int, default=21,
                          help="Must match FNIRS_NUM_SEGMENTS used by the currently-loaded "
                               "preprocessed .npy files (preprocessing constant). Previously "
                               "hardcoded to 21 inside TSFNet/CAFELayer -- changing the "
                               "preprocessing's segment count (e.g. to 18, per the reference-"
                               "paper-calibrated formula) without updating this would crash "
                               "with a reshape RuntimeError in CAFELayer.forward. Default 21 "
                               "reproduces prior behavior; set to 18 for the corrected "
                               "preprocessing.")
_arg_parser.add_argument("--class-count-level", type=str, default="window",
                          choices=["window", "trial"],
                          help="Unit at which class_counts is computed for "
                               "--class-weighting-scheme != 'none'. 'window' (default, legacy "
                               "behavior). 'trial': verified to be a no-op on this dataset "
                               "(trial-level counts are always exactly [6,6,6] per fold, by "
                               "fixed experimental design) -- kept only for completeness.")
_arg_parser.add_argument("--fnirs-input-gate", type=str, default="none",
                          choices=["none", "soft", "soft_prior"],
                          help="priority-FNIRSGATE isolated variable: a SOFT, LEARNABLE "
                               "weighting of the raw fNIRS input over the w (posterior.."
                               "anterior) and chromophore axes. 'none' (default) creates no "
                               "parameters and leaves forward() untouched -- bit-exact with "
                               "every prior run. 'soft' zero-initialises both logit vectors, "
                               "so the gate is exactly 1.0 at step 0 and any weighting is "
                               "LEARNED (same idiom as --egta-k-init zero). 'soft_prior' "
                               "starts biased toward anterior and toward HbO by "
                               "--fnirs-gate-prior-strength, injecting the prior measured by "
                               "map_fnirs_prior.py (anterior HbO: 2v3 0.603 vs 0.579, broad "
                               "plateau over w>=6..12, but paired p=0.136 and a -0.020 "
                               "3-class cost) while leaving the model free to undo it. "
                               "REGISTERED PREDICTION: the expected effect is higher 2-back "
                               "and 3-back RECALL in the confusion matrix, not necessarily "
                               "higher overall accuracy.")
_arg_parser.add_argument("--fnirs-gate-prior-strength", type=float, default=0.5,
                          help="priority-FNIRSGATE: logit offset used by --fnirs-input-gate "
                               "soft_prior. 0.5 gives a gate range of about 0.76..1.24 per "
                               "axis (deliberately mild). Ignored for the other modes.")
_arg_parser.add_argument("--fnirs-chroma-mode", type=str, default="both",
                          choices=["both", "hbo_only", "hbr_only", "hbt_only", "hbo_hbr", "cbsi"],
                          help="priority-A4 isolated variable: which fNIRS chromophore "
                               "channel(s) reach the network. 'both' (default) reproduces "
                               "prior behavior exactly (no-op mask). 'hbo_only' zeros the HbR "
                               "channel before FNIRSBranch; 'hbr_only' zeros HbO instead. "
                               "Motivated by Meidenbauer et al. [34]: load-dependent effects "
                               "were found only in HbR, and 3-back activity was noted as "
                               "nonlinear/unstable -- targets the 2-back/3-back bottleneck "
                               "via the chromophore axis rather than the loss function.")
_arg_parser.add_argument("--mhs-lambda", type=float, default=0.0,
                          help="priority-B1 isolated variable: weight of the CAFE multi-head "
                               "HSIC diversity regularization term (Ding et al. 2026, MSTFDN, "
                               "Eqs. 25-28), applied to both of CAFE's cross-attention modules "
                               "(4 heads each). Penalizes pairwise similarity between attention "
                               "heads' per-sample representations (batch-wise linear-kernel "
                               "HSIC/CKA), encouraging diverse subspace learning -- a general "
                               "stability improvement, NOT targeted at the 2-back/3-back pair "
                               "specifically. Reference paper's own value is 0.05 (their "
                               "lambda_mhs, tuned by grid search on their own architecture -- "
                               "treat as a starting point, not a transferred optimum). Default "
                               "0.0 disables it and reproduces prior behavior bit-for-bit "
                               "(see tsfnet_loss's mhs_lambda docstring for why).")
_arg_parser.add_argument("--data-root", type=str, default=None,
                          help="priority-GRIDALIGN: folder holding the preprocessed VP001..VP026 "
                               "data (results_* folders are written inside it). Default None = "
                               "the baseline DATA_ROOT below, prior behavior. For the shared-"
                               "grid ablation pass the '..._fnirsbboxEEG' folder produced by "
                               "Preproc_tsfnet_nback_badch.py --fnirs-grid-bbox eeg.")
_arg_parser.add_argument("--subjects", type=str, default="1-26",
                          help="Which subjects to run, e.g. '1-2' or '1,5,12-14'. Default "
                               "'1-26' (all 26, prior behavior) -- use a small range for a quick "
                               "isolated-ablation smoke test before committing to the full sweep. "
                               "Does NOT enter the variant hash/tag (it is a run-scope choice, not "
                               "a model/training change), so re-running the same flags with more "
                               "subjects later just adds results into the same results_<hash>_seed<N> "
                               "folder.")
_arg_parser.add_argument("--erp-branch", type=str, default="none", choices=["none", "eeg", "eeg_fusion"],
                          help="priority-ERP: add a digit-locked ERP pathway (needs the files from "
                               "build_erp_windows.py). 'eeg' feeds it into the EEG classifier, "
                               "'eeg_fusion' into the EEG and fusion classifiers. Zero-initialized: "
                               "the model starts exactly as without it. Default 'none' = prior "
                               "behavior bit-for-bit.")
_arg_parser.add_argument("--pretrained-dir", type=str, default=None,
                          help="priority-XSUBJ: folder written by pretrain_cross_subject.py "
                               "(contains pretrain_manifest.json). Each subject's three folds then "
                               "start from the pretrained model of the subject group that EXCLUDES "
                               "it, instead of from random weights. Default None = random init, "
                               "prior behavior. The manifest's seed must equal --seed.")
_arg_parser.add_argument("--pretrained-new-modules", type=str, nargs="*", default=(),
                          metavar="MODULE",
                          help="priority-XSUBJ-NEWMOD: modules the PRETRAINING architecture did "
                               "not have, so they start from random init instead of failing the "
                               "strict load. Use riem_head when the trunk was pretrained without "
                               "--riem-branch but this run uses it; the Riemannian features are "
                               "built per subject, so that head has nothing to transfer anyway. "
                               "Every OTHER key must still match exactly, and a named module must "
                               "be wholly absent from the checkpoint. Default () = strict load, "
                               "i.e. prior behavior.")
_arg_parser.add_argument("--fusion-fnirs-inject", type=str, default="none",
                          choices=["none", "stage1", "both"],
                          help="priority-FNIRSINJECT isolated variable: add fNIRS FEATURE maps "
                               "(segment/time-averaged, through a zero-initialized 1x1x1 conv) "
                               "into the fusion stream after EFGF1 ('stage1') or after both "
                               "EFGF1 and EFGF2 ('both'). Default 'none' = prior behavior "
                               "bit-for-bit (the fusion branch sees fNIRS only via EFGF's alpha).")
_arg_parser.add_argument("--efgf-lambda", type=float, default=1.0,
                          help="priority-PCCLAMBDA isolated variable: weight of the EFGF PCC "
                               "regularizer (Eq. 12, L_efgf = 1 - PCC(F_eeg, Phi)) in the total "
                               "loss. The paper's Eq. 5 adds it unweighted, i.e. 1.0, which is "
                               "the default and reproduces prior behavior bit-for-bit. 0.0 "
                               "removes the term entirely. Sweep candidates: 0.0, 0.01, 0.1.")
_arg_parser.add_argument("--cafe-lambda", type=float, default=1.0,
                          help="priority-PCCLAMBDA isolated variable: weight of the CAFE PCC "
                               "regularizer (Eq. 13, L_cafe = 1 - PCC(Fe_fnirs, Fe_fusion)) in "
                               "the total loss. That term pushes the two modality "
                               "representations toward being correlated (redundant). Default "
                               "1.0 = paper Eq. 5, bit-exact prior behavior. 0.0 removes it.")
_arg_parser.add_argument("--cafe-pcc-mode", type=str, default="signed", choices=["signed", "abs"],
                          help="priority-CAFEPCCABS: 'signed' (default) = the original L_cafe = 1 - mean PCC, "
                               "bit-exact. 'abs' = 1 - mean |PCC| (STA-Net's form: anti-correlation is also "
                               "'aligned', so the term stops pushing the two streams toward positive correlation "
                               "only). Loss-only change: no parameter, no RNG, forward unchanged.")
_arg_parser.add_argument("--riem-branch", type=str, default="none", choices=["none", "member"],
                          help="priority-RIEM: 'none' (default) = nothing changes, bit-exact. 'member' = add a fourth, "
                               "NON-deep decision member to the model: a linear head on per-session tangent-space EEG "
                               "features ({sid}_eeg_riem.npy from build_riem_features.py), blended into the final "
                               "decision y = (1-b)*normalise(y_deep) + b*y_riem (b = --riem-blend, fixed). The same "
                               "idea as the offline late fusion (+6 pp window), now trained jointly. Transductive "
                               "per-session centring of the features, like E3/AdaBN.")
_arg_parser.add_argument("--riem-blend", type=float, default=0.5,
                          help="fixed blend b of the Riemannian member (0.5 = the pre-registered late-fusion weight)")
_arg_parser.add_argument("--riem-lambda", type=float, default=0.2,
                          help="weight of the separate CE term on the Riemannian member's own posterior")
_arg_parser.add_argument("--riem-weight-decay", type=float, default=1e-2,
                          help="weight decay of the Riemannian head only (analogue of LogisticRegression(C=0.1))")
_arg_parser.add_argument("--riem-dropout", type=float, default=0.5,
                          help="dropout on the Riemannian head's input features (0.5 = the existing behaviour; "
                               "the folder subtag changes only when this differs from 0.5)")
_arg_parser.add_argument("--egta-pcc-lambda", type=float, default=0.0,
                          help="priority-EGTAPCC: weight of STA-Net's L_egta = 1 - mean|PCC("
                               "F_fusion, F_fnirs(ta))| on the PURE EGTA-pooled fNIRS feature "
                               "(Liu et al. 2025, Eq. 11). Requires --fnirs-temporal-align egta. "
                               "Default 0.0 = term not computed, bit-exact prior behavior. "
                               "Sweep candidates: 0.1, 0.5, 1.0.")
_arg_parser.add_argument("--eeg-filterbank", type=str, default="none",
                          choices=["none", "fixed", "learnable"],
                          help="priority-FILTERBANK: split the EEG grid into 5 bands (delta<4, "
                               "theta 4-8, alpha 8-13, beta 13-30, gamma 30-50 Hz) inside the "
                               "forward pass and feed them to conv1 as 5 input channels (MSTFDN). "
                               "'fixed' = FIR, no parameters; 'learnable' = SincNet edges (10 "
                               "params). Nothing is written to disk. Default 'none' = prior "
                               "behavior bit-for-bit.")
_arg_parser.add_argument("--eeg-filterbank-scope", type=str, default="both",
                          choices=["both", "eeg"],
                          help="priority-FILTERBANK: 'both' (default) = EEG and fusion branches "
                               "get the bands; 'eeg' = only the EEG branch. Ignored when "
                               "--eeg-filterbank none.")
_arg_parser.add_argument("--fnirs-chroma-split", type=str, default="none",
                          choices=["none", "separate", "separate_pair"],
                          help="priority-CHROMASPLIT (MGFNet): give HbO and HbR their own "
                               "FNIRSBranch with non-shared weights, added to the fNIRS DECISION "
                               "path (fn_pooled -> CAFE/EGTA -> clf_fnirs) through zero-init 1x1 "
                               "convs. EFGF / the fusion branch are untouched. 'separate_pair' "
                               "also adds an independently-weighted (HbO,HbR) pair branch. "
                               "Requires --fnirs-num-chromophores 2. Default 'none' = prior "
                               "behavior bit-for-bit.")
_arg_parser.add_argument("--label-smoothing", type=float, default=0.0,
                          help="priority-B4 isolated variable: label smoothing strength applied "
                               "to all four loss terms (l_eeg, l_fnirs, l_fusion, l_pred), "
                               "following Muller et al. 2019, 'When Does Label Smoothing Help?' "
                               "and Szegedy et al. 2016's original formulation. Replaces the "
                               "one-hot cross-entropy target with (1-eps)*onehot + eps/K, "
                               "discouraging overconfident predictions -- motivated by this "
                               "project's persistent 2-back/3-back overconfidence bottleneck. "
                               "Default 0.0 disables it and reproduces prior behavior bit-for-bit "
                               "(see label_smoothing_loss.py / tsfnet_loss's label_smoothing "
                               "docstring for why).")
_arg_parser.add_argument("--domain-adapt-lambda", type=float, default=0.0,
                          help="priority-C1 isolated variable: weight of a Deep CORAL "
                               "(Sun & Saenko 2016) feature-alignment loss between the "
                               "training sessions' and the held-out session's UNLABELED "
                               "enhanced_fnirs features, active during stage 2 training. "
                               "Deepens E3/E4's transductive session adaptation from a "
                               "post-hoc BatchNorm-statistic swap (AdaBN) to an explicit, "
                               "actively-optimized feature-covariance alignment term, "
                               "targeted at fNIRS specifically per priority-G3's finding "
                               "that its session-to-session shift is ~4x EEG's. Default "
                               "0.0 disables it and reproduces prior behavior bit-for-bit "
                               "(see coral_domain_adaptation.py for the full derivation).")
_arg_parser.add_argument("--use-fc-features", action="store_true", default=False,
                          help="priority-E1 isolated variable: concatenate an EEG functional-"
                               "connectivity (PLV, theta+alpha) embedding onto the fusion "
                               "branch's pooled feature before its classifier (see "
                               "ConnectivityBranch / TSFNet.forward). Motivated by Cao et al. "
                               "2022 (Sensors 22(19):7623), the SAME Shin et al. 2018 dataset "
                               "this project uses. REQUIRES {subject_id}_eeg_connectivity.npy "
                               "to exist for every subject (see Preproc_tsfnet_nback_badch.py's "
                               "ENABLE_EEG_CONNECTIVITY) -- if a subject is missing this file, "
                               "TSFNet raises a clear error rather than silently training on "
                               "zeros. Default False reproduces prior behavior bit-for-bit "
                               "(no new parameters constructed, clf_fusion's input dim "
                               "unchanged).")
_arg_parser.add_argument("--fc-num-channels", type=int, default=28,
                          help="priority-E1: number of EEG channels the connectivity features "
                               "were computed over (must match eeg_connectivity.py's "
                               "n_channels used during preprocessing -- 28 for this project's "
                               "montage). Only used when --use-fc-features is set.")
_arg_parser.add_argument("--fc-embed-dim", type=int, default=64,
                          help="priority-E1: output dimension of ConnectivityBranch's "
                               "connectivity embedding, concatenated onto the fusion branch's "
                               "512-dim pooled feature. Only used when --use-fc-features is set.")
_arg_parser.add_argument("--adabn", action="store_true", default=False,
                          help="priority-E4 isolated variable: AdaBN-style test-time adaptation "
                               "(Li et al. 2017). After training, recompute every BatchNorm "
                               "layer's running_mean/running_var on the held-out session's "
                               "UNLABELED inputs (one forward pass, no gradients, no labels, no "
                               "weight updates), then evaluate. Natural successor to the "
                               "per-session input normalization (ENABLE_PER_SESSION_NORM in "
                               "Preproc_tsfnet_nback_badch.py) that this project found to help: "
                               "that fixed the INPUT distribution shift across sessions, this "
                               "fixes the part of it surviving into internal feature "
                               "distributions. TRANSDUCTIVE -- must be reported as such and "
                               "never compared against a strictly inductive baseline without "
                               "disclosure. Default False reproduces prior behavior bit-for-bit.")
_arg_parser.add_argument("--tta-mode", type=str, default=None,
                          choices=[None, "adabn", "tent"],
                          help="further generalization of priority-E4. 'tent' (Wang et al. 2021, "
                               "ICLR spotlight, 'Tent: Fully Test-Time Adaptation by Entropy "
                               "Minimization') is a STRICT generalization of --adabn: it also "
                               "discards running BatchNorm statistics in favor of pure per-batch "
                               "statistics, AND additionally optimizes the BatchNorm layers' "
                               "learnable affine weight/bias by gradient descent on the entropy "
                               "of the model's own predictions -- one update per held-out-session "
                               "test batch, still using only unlabeled target inputs, never "
                               "labels. MUTUALLY EXCLUSIVE with --adabn (both modify the same "
                               "BatchNorm layers; combining them would confound which one "
                               "produced any change -- raises an error if both are set). Default "
                               "None: if --adabn is also set, behaves exactly as before; if not, "
                               "reproduces prior behavior bit-for-bit. TRANSDUCTIVE, like --adabn "
                               "-- must be reported and disclosed the same way.")
_arg_parser.add_argument("--tent-lr", type=float, default=1e-3,
                          help="Adam learning rate for tent's BatchNorm-affine adaptation. "
                               "1e-3 is the official method's own documented default "
                               "recommendation, not a project-specific tune. Only used when "
                               "--tta-mode tent.")
_arg_parser.add_argument("--tent-steps", type=int, default=1,
                          help="gradient steps per test batch for tent (official default 1: "
                               "'makes just one update per batch to not interrupt inference'). "
                               "Only used when --tta-mode tent.")
_arg_parser.add_argument("--fnirs-temporal-align", type=str, default="none",
                          choices=["none", "egta"],
                          help="priority-1 of the literature roadmap (adaptive temporal "
                               "alignment), adapted from the OFFICIAL STA-Net EGTA layer "
                               "(Liu et al. 2025, Information Fusion 119:103023). 'egta' "
                               "replaces the UNIFORM MEAN over the fNIRS segments in CAFE's "
                               "fNIRS output path (verified order-invariant; this is exactly "
                               "STA-Net's own 'No EGTA' ablation) with EEG-anchored attention "
                               "pooling. Zero-initialized keys make it identical to the mean at "
                               "step 0. Default 'none' reproduces prior behavior bit-for-bit.")
_arg_parser.add_argument("--egta-k-init", type=str, default="zero",
                          choices=["zero", "noisy"],
                          help="Only used when --fnirs-temporal-align egta. 'zero' (DEFAULT, "
                               "accepted baseline, bit-exact reproducible): egta_k's weight/bias "
                               "init to exactly 0, which makes scores=q@k^T==0 identically at "
                               "step 0 -- but ALSO makes d(scores)/d(egta_q)==k==0 identically, "
                               "a CONFIRMED dead gradient on egta_q until egta_k's own weak "
                               "gradient moves first (layer_gradient_diagnostics.py). 'noisy' "
                               "(ISOLATED ABLATION, the fix): egta_k.weight ~ N(0, 0.01^2) instead "
                               "of exact 0, breaking that bilinear deadlock so egta_q gets a live "
                               "gradient from step 0; TESTED on the accepted baseline (3 seeds, 26 "
                               "subjects): -1.34pp paired window accuracy (95% CI [-2.59,-0.10]pp, "
                               "t-test p=0.045, Wilcoxon p=0.071, 17/26 subjects worse) -- a real "
                               "but small, borderline-significant REGRESSION, not an improvement. "
                               "Fixing the dead gradient does not fix accuracy, so 'zero' stays "
                               "the default (rule 5); 'noisy' is kept available as a documented, "
                               "tested alternative.")
_arg_parser.add_argument("--lr", type=float, default=1e-3,
                          help="priority-G1: Adam learning rate. Default 1e-3 = prior behavior. "
                               "MSTFDN uses 3e-4 (with cosine annealing + warm-up).")
_arg_parser.add_argument("--weight-decay", type=float, default=1e-4,
                          help="priority-G1: Adam L2 weight decay. Default 1e-4 = prior behavior. "
                               "NOTE: MSTFDN's own L2 coefficient (0.02) is tuned for a ~34.5M-"
                               "parameter model; this project's model is ~400k parameters, so that "
                               "value is NOT directly transferable -- sweep (e.g. 1e-3, 5e-3, 1e-2) "
                               "rather than adopting 0.02 outright.")
_arg_parser.add_argument("--dropout", type=float, default=0.5,
                          help="priority-G1: dropout in the EEG/Fusion/fNIRS branches and branch "
                               "classifiers. Default 0.5 = prior behavior (MSTFDN uses 0.4).")
_arg_parser.add_argument("--conv-dropout", type=float, default=None,
                          help="priority-CONVDROP isolated variable: dropout rate for the 3D-conv "
                               "blocks of the EEG/Fusion/fNIRS branches ONLY (branch classifiers "
                               "keep --dropout). Default None = use --dropout, i.e. prior behavior "
                               "bit-for-bit. Sweep candidates: 0.1, 0.2, 0.3.")
_arg_parser.add_argument("--conv-dropout-type", type=str, default="channel",
                          choices=["channel", "element"],
                          help="priority-CONVDROP: 'channel' (default, prior behavior) = "
                               "nn.Dropout3d, zeroes whole feature channels (16 in conv1, 32 in "
                               "conv2). 'element' = nn.Dropout, zeroes individual activations.")
_arg_parser.add_argument("--cafe-dropout", type=float, default=0.3,
                          help="priority-G1: dropout inside CAFE. Default 0.3 = prior behavior.")
_arg_parser.add_argument("--stage1-max-epochs", type=int, default=100,
                          help="priority-G1 (absorbs the old B2 item): stage-1 epoch budget. "
                               "Default 100 = prior behavior. MSTFDN trains up to 1000 epochs with "
                               "early stopping, so under-training is a live hypothesis here.")
_arg_parser.add_argument("--stage1-patience", type=int, default=20,
                          help="priority-G1: stage-1 early-stopping patience (epochs without "
                               "val-accuracy improvement). Default 20 = prior behavior.")
_arg_parser.add_argument("--best-min-epoch", type=int, default=1,
                          help="priority-BESTSEL: epochs before this one cannot be chosen as the "
                               "best stage-1 checkpoint and do not count toward patience. Default 1 "
                               "= prior behavior bit-for-bit. Baseline diagnosis: validation has 3 "
                               "trials, corr(val_acc, test_acc) = -0.04, and early 'best' epochs "
                               "(<=5) went with lower test accuracy. Try 10.")
_arg_parser.add_argument("--stage1-fixed-epochs", type=int, default=0,
                          help="priority-BESTSEL: train exactly this many stage-1 epochs and keep the "
                               "FINAL weights -- no validation-based selection, no early stopping. "
                               "Default 0 = off. Baseline median 'best' epoch is 11. Mutually "
                               "exclusive with --best-min-epoch > 1.")
_arg_parser.add_argument("--stage2-max-epochs", type=int, default=80,
                          help="priority-G1: stage-2 epoch budget. Default 80 = prior behavior. "
                               "Stage 2 has no patience concept -- it stops on the stage-2 loss "
                               "target or this cap, whichever comes first.")
_arg_parser.add_argument("--warmup-epochs", type=int, default=0,
                          help="priority-G1: linear LR warm-up epochs. Default 0. Together with "
                               "--min-lr-factor 1.0 (also default) NO scheduler is created at all, "
                               "so the default path is bit-identical to prior behavior. MSTFDN "
                               "uses 10.")
_arg_parser.add_argument("--min-lr-factor", type=float, default=1.0,
                          help="priority-G1: cosine-annealing floor as a fraction of the base lr. "
                               "Default 1.0 = no annealing. MSTFDN uses 0.05.")
_arg_parser.add_argument("--fnirs-num-chromophores", type=int, default=2, choices=[2, 3],
                          help="priority-G2a: number of fNIRS chromophore channels the model "
                               "expects. 2 (default) = HbO/HbR, matching every existing "
                               "preprocessed dataset. 3 = HbO/HbR/HbT, which REQUIRES "
                               "fnirs_windows.npy regenerated with ENABLE_FNIRS_HBT=True in "
                               "Preproc_tsfnet_nback_badch.py; if the two disagree, FNIRSBranch "
                               "raises a clear shape error rather than failing silently. NOTE: "
                               "HbT is derived (HbO+HbR), not a new measurement -- roughly 90% "
                               "of its grid-level variance is already linearly reachable from "
                               "the other two, so expect a small effect at best.")
_arg_parser.add_argument("--eeg-spatial-mode", type=str, default="none",
                          choices=["none", "mhsa", "bandtok"],
                          help="'bandtok' (priority-BANDTOK): channel attention over "
                               "5-band log-POWER tokens (one pre-LN block, zero-init gates, joined to "
                               "the EEG classifier through a zero-init Linear; ~30k params). "
                               "Needs the channel-space windows. "
                               "priority-G4 isolated variable: 'mhsa' adds multi-head "
                               "self-attention over EEG CHANNELS AS TOKENS (MSTFDN's E-SFEM; "
                               "their single largest ablation effect, +22.66 points on this "
                               "exact 3-class N-back LOSO protocol) and concatenates its "
                               "embedding onto the EEG branch feature before clf_eeg. "
                               "REQUIRES {subject_id}_eeg_channel_windows.npy, i.e. "
                               "preprocessing re-run with ENABLE_EEG_CHANNEL_SPACE=True. "
                               "Scope is deliberately limited to the y_eeg path, exactly like "
                               "the existing eeg-temporal-mode ablation. Default 'none' "
                               "reproduces prior behavior bit-for-bit.")
_arg_parser.add_argument("--bandtok-norm", type=str, default="none", choices=["none", "relative"],
                          help="priority-BANDTOK: 'none' = log band power; 'relative' = log of "
                               "band/sum-over-bands power (drops overall amplitude, keeps the "
                               "spectral shape). Only used with --eeg-spatial-mode bandtok.")
_arg_parser.add_argument("--bandtok-d-model", type=int, default=32)
_arg_parser.add_argument("--bandtok-heads", type=int, default=4)
_arg_parser.add_argument("--eeg-spatial-layers", type=int, default=2,
                          help="priority-G4: Transformer encoder layers in the channel-wise "
                               "MHSA module. MSTFDN uses 2 for the 3-class N-back task.")
_arg_parser.add_argument("--eeg-spatial-heads", type=int, default=4,
                          help="priority-G4: attention heads. MSTFDN uses 2 for 3-class "
                               "N-back; 4 is this project's CAFE default. Only used when "
                               "--eeg-spatial-mode mhsa.")
# ISOLATED ABLATION FLAG (priority-C2 sub-experiment: pretrained EEG
# foundation-model backbone). Default "native" reproduces prior behavior
# bit-for-bit -- labram_eeg_branch_v2.py is never imported, self.labram_head
# is never constructed, and y_eeg is computed exactly as before via
# self.clf_eeg. "labram_frozen" replaces ONLY y_eeg's computation with a
# frozen, pretrained LaBraM backbone (Jiang et al. 2024) + a light trainable
# linear head, consuming the same raw channel-space x_chan that
# --eeg-spatial-mode mhsa already uses. Mutually exclusive with
# --eeg-spatial-mode mhsa (TSFNet raises a clear error if both are set) and
# should NOT be combined with a band-decomposed EEG input (D-1, not yet
# implemented in this script) since LaBraM was pretrained on broadband raw
# EEG -- see labram_eeg_branch_v2.py's module docstring for the full
# rationale, the checkpoint-compatibility issues found and fixed while
# building it, and why every forward() call zero-pads this project's real
# 2000-sample window to the checkpoint's own 3000-sample pretraining length.
# REQUIRES: pip install "braindecode[hub]" and network access to the
# Hugging Face Hub the first time a fold builds this model (weights are
# then cached locally -- see HF_HOME).
_arg_parser.add_argument("--eeg-backbone", type=str, default="native",
                          choices=["native", "labram_frozen", "labram_ensemble", "labram_blend"],
                          help="priority-C2 isolated variable: 'native' (default) reproduces "
                               "prior behavior bit-for-bit. 'labram_frozen': y_eeg is computed "
                               "by a frozen, pretrained LaBraM EEG foundation-model backbone "
                               "(~5.8M frozen parameters) + a light trainable linear head, "
                               "instead of self.clf_eeg -- NOTE this also severs the auxiliary "
                               "lambda_eeg gradient into the shared EEG trunk, making it a "
                               "TWO-variable change (measured: trunk aux-grad 25.94 -> 0.00). "
                               "'labram_ensemble' (C-2c, PREFERRED): keeps the native y_eeg and "
                               "its trunk supervision completely unchanged, and ADDS the LaBraM "
                               "head as a fourth Eq.4 decision-level member -- a genuinely "
                               "isolated change. Mutually exclusive with --eeg-spatial-mode mhsa.")
_arg_parser.add_argument("--eeg-band-mode", type=str, default="none",
                          choices=["none", "safe"],
                          help="priority-D-1-safe: 'none' (default) is a BIT-EXACT no-op. "
                               "'safe' adds a gated (sigmoid(-6) init) embedding of five FIXED "
                               "EEG bands (delta/theta/alpha/beta/gamma) via a non-trainable FIR "
                               "filterbank + shared-weight encoder. Reuses the same x_chan input "
                               "--eeg-spatial-mode mhsa uses -- no new preprocessing needed. "
                               "Mutually exclusive with --eeg-backbone != native. NOTE: in 'safe' "
                               "mode clf_eeg's input dim grows, which also shifts clf_fusion/"
                               "clf_fnirs init (documented limitation, rule 6).")
_arg_parser.add_argument("--eeg-band-d-model", type=int, default=32,
                          help="priority-D-1-safe: per-band temporal pooling width.")
_arg_parser.add_argument("--eeg-band-embed-dim", type=int, default=32,
                          help="priority-D-1-safe: final band-branch embedding size.")
_arg_parser.add_argument("--eeg-band-weight-decay", type=float, default=1e-2,
                          help="priority-D-1-safe (Safe-Capacity Protocol): weight_decay applied "
                               "ONLY to eeg_band_branch (100x the global 1e-4), per the project's "
                               "diagnosis that unregularized new capacity in this low-data (n=26) "
                               "regime drove 10+ prior rejections.")
_arg_parser.add_argument("--norm-mode", type=str, default="batch",
                          choices=["batch", "group"],
                          help="priority-GN: 'batch' (default) is a BIT-EXACT no-op. 'group' "
                               "replaces every BatchNorm layer with GroupNorm (per-sample "
                               "statistics, independent of batch composition). Structural fix, "
                               "not added capacity: identical parameter count and init. "
                               "Mutually exclusive with --adabn and --tta-mode tent, because "
                               "both of those operate only on BatchNorm layers.")
_arg_parser.add_argument("--norm-groups", type=int, default=8,
                          help="priority-GN: GroupNorm groups (clamped to a divisor of each "
                               "layer's channel count; 16- and 32-channel layers -> 8).")
_arg_parser.add_argument("--efgf-gate-mode", type=str, default="clamp",
                          choices=["clamp", "sigmoid"],
                          help="priority-EFGF-gatefix: 'clamp' (default) is the ORIGINAL "
                               "torch.clamp(param,0,1) parametrization -- bit-exact reproduction "
                               "of every past run, but has a confirmed dead-gradient bug (a bad "
                               "first Adam step can permanently zero out alpha/gamma; verified on "
                               "a real checkpoint: efgf1.alpha/gamma stuck at exactly 0). "
                               "'sigmoid' is the FIX: sigmoid(raw), raw init -6.0, same convention "
                               "as beta_labram/EGTA/eeg_band gates elsewhere in this file -- "
                               "gradient is never exactly zero, so the gate cannot get stuck.")
_arg_parser.add_argument("--normalize-ypred", action="store_true",
                          help="priority-ypred-normfix: normalize y_pred = "
                               "(w1*y_eeg+w2*y_fusion+w3*y_fnirs)/(w1+w2+w3) before computing "
                               "the classification loss, instead of the ORIGINAL un-normalized "
                               "(w1+w2+w3)/3 (which is not a valid probability distribution -- "
                               "only sums to 1 when w1=w2=w3=1). Default OFF reproduces the "
                               "exact prior behavior bit-for-bit (rule 5). ON removes an "
                               "undifferentiated upward pressure on all three fusion gates "
                               "(confirmed: w_eeg/w_fusion/w_fnirs sit within ~0.001 of each "
                               "other for every one of 26 subjects, seed 42 reference run).")
_arg_parser.add_argument("--val-mode", type=str, default="trial",
                          choices=["trial", "cross_session", "test_leak"],
                          help="priority-XSVAL: how the stage-1 validation set is carved out of "
                               "the two training sessions. 'trial' (default) is the ORIGINAL "
                               "behavior, bit-exact: 20%% of TRIALS pooled across both training "
                               "sessions -> only ~3 validation trials, drawn from the same "
                               "sessions as training even though the test is cross-session. "
                               "'cross_session' holds out one WHOLE training session instead -> "
                               "9 validation trials AND cross-session validation, matching the "
                               "test regime. Trade-off: training data is halved (1 session "
                               "instead of 2). 'test_leak' is a DELIBERATE-LEAKAGE DIAGNOSTIC, "
                               "never a valid result: training data is IDENTICAL to 'trial', but "
                               "best-epoch selection/early stopping use the held-out TEST session. "
                               "Quantifies how optimistic selection on test would be (cf. the "
                               "MSTFDN-paper reproduction gap). Requires --stage2-stop-mode none.")
_arg_parser.add_argument("--no-decay-scalars", action="store_true",
                          help="priority-nodecay-scalars: exclude 1-D/scalar parameters "
                               "(gates, w_eeg/w_fusion/w_fnirs, BatchNorm/GroupNorm affine "
                               "weights, pos_embed) from weight_decay entirely, instead of "
                               "applying the ORIGINAL uniform weight_decay to them. BUG FIX: "
                               "torch.optim.Adam's weight_decay is a direct L2 term added to "
                               "every parameter's gradient; for a sigmoid(raw) gate initialized "
                               "far from 0 (e.g. raw=-6.0), this is a constant, DATA-INDEPENDENT "
                               "force pulling the gate open. Verified with real torch.optim.Adam "
                               "and zero data-gradient: sigmoid(raw) drifts to 0.006 / 0.045 / "
                               "0.296 by steps 1000/3300/6500 from weight_decay ALONE. Default "
                               "OFF reproduces the exact prior behavior bit-for-bit (rule 5).")
_arg_parser.add_argument("--stage2-stop-mode", type=str, default="loss",
                          choices=["loss", "epoch_count", "none"],
                          help="priority-stage2fix: 'loss' (default) is the ORIGINAL behavior: "
                               "stop stage 2 when train_loss < stage2_target, a comparison "
                               "between losses measured in different modes (eval vs train) and, "
                               "with --val-mode cross_session, different training-set sizes -- "
                               "confirmed to nearly double the mean stopping epoch (7.6->15.3 "
                               "over 78 folds) and occasionally hit stage2_max_epochs outright. "
                               "'epoch_count': skip the loss comparison; since stage 2 already "
                               "resumes from stage 1's best checkpoint (not from scratch), "
                               "running it for another full best_epoch epochs would train "
                               "~2x best_epoch total -- so this fine-tunes for 25% of best_epoch "
                               "instead (capped at stage2_max_epochs), with lr and Adam's "
                               "moments reset. This is a documented, cheaper APPROXIMATION of "
                               "'retrain for N epochs from scratch', not that exact recipe -- "
                               "chosen to avoid ~2x wall-clock cost per fold.")
_arg_parser.add_argument("--swa-enabled", action="store_true",
                          help="priority-SWA: Stochastic Weight Averaging (Izmailov et al. 2018) "
                               "over the tail of stage-1 training. ZERO added parameters, ZERO "
                               "architecture change -- just an equal-weighted running average of "
                               "the model's own weights, which empirically sits in a wider/flatter "
                               "minimum and generalizes better. After averaging, BatchNorm stats "
                               "are recalibrated and the averaged model is compared against stage "
                               "1's own early-stopped best checkpoint on the SAME validation set; "
                               "it only replaces that checkpoint if it scores >= as well, so this "
                               "flag can never make results worse than --swa-enabled absent (see "
                               "train_one_fold's docstring for the full fallback logic). Default: "
                               "disabled (bit-identical to every existing run).")
_arg_parser.add_argument("--swa-start-epoch", type=int, default=None,
                          help="priority-SWA: epoch (within stage 1) at which weight-averaging "
                               "begins. Default None = round(0.75*stage1_max_epochs), i.e. the "
                               "last quarter of stage 1. Only used when --swa-enabled.")
_arg_parser.add_argument("--swa-lr", type=float, default=None,
                          help="priority-SWA: constant learning rate held during the averaging "
                               "tail (SWALR). Default None = same as --lr (no change). Only used "
                               "when --swa-enabled.")
_arg_parser.add_argument("--efgf-gate-init", type=float, default=-6.0,
                          help="priority-efgfinit: initial value of EFGF's raw gate parameter "
                               "when --efgf-gate-mode sigmoid is used (ignored under 'clamp'). "
                               "Default -6.0 (sigmoid=0.0025) matches the convention used "
                               "elsewhere in this file, but a real Adam simulation shows it "
                               "leaves the gate at only ~0.37 by the end of stage 1 even when "
                               "the data wants it open (vs ~0.82 from -2.0) -- exposed here so "
                               "both can be compared directly.")
_arg_parser.add_argument("--attn-conv-init-gain", type=float, default=None,
                          help="ISOLATED ABLATION ('item 3' from the gradient-flow review): "
                               "None (DEFAULT, accepted baseline, bit-exact reproducible) leaves "
                               "EFGF's eeg_attn_conv/fnirs_attn_conv at PyTorch's own default "
                               "Kaiming-uniform init. A float (e.g. 0.5) re-inits both convs' "
                               "weight with xavier_uniform_(gain=attn_conv_init_gain) instead -- "
                               "layer_gradient_diagnostics.py flagged these layers with a large "
                               "grad/weight ratio under the default init; a smaller gain keeps "
                               "their pre-sigmoid output closer to the sigmoid's linear region at "
                               "step 0. NEVER MEASURED to help or hurt accuracy -- test before "
                               "trusting.")
SEED_ARGS = _arg_parser.parse_args()
SEED = SEED_ARGS.seed
GRAD_CLIP_NORM = 1.0 if SEED_ARGS.grad_clip else None
FUSION_COMBINATION = SEED_ARGS.fusion_combination
NOISE_AWARE_PROB = SEED_ARGS.noise_aware_prob
NOISE_AWARE_SNR_DB = SEED_ARGS.noise_aware_snr_db
CLASS_WEIGHTING_SCHEME = SEED_ARGS.class_weighting_scheme
EFFECTIVE_NUMBER_BETA = SEED_ARGS.effective_number_beta
FOCAL_GAMMA = SEED_ARGS.focal_gamma
OVERSAMPLING_SCHEME = SEED_ARGS.oversampling_scheme
SMOTE_K_NEIGHBORS = SEED_ARGS.smote_k_neighbors
EEG_TEMPORAL_MODE = SEED_ARGS.eeg_temporal_mode
EEG_LSTM_HIDDEN = SEED_ARGS.eeg_lstm_hidden
EEG_READOUT_MODE = SEED_ARGS.eeg_readout_mode
CAFE_D_MODEL = SEED_ARGS.cafe_d_model
FNIRS_NUM_SEGMENTS = SEED_ARGS.fnirs_num_segments
CLASS_COUNT_LEVEL = SEED_ARGS.class_count_level
FNIRS_CHROMA_MODE = SEED_ARGS.fnirs_chroma_mode
FNIRS_INPUT_GATE = SEED_ARGS.fnirs_input_gate          # priority-FNIRSGATE
FNIRS_GATE_PRIOR_STRENGTH = SEED_ARGS.fnirs_gate_prior_strength
MHS_LAMBDA = SEED_ARGS.mhs_lambda
EFGF_LAMBDA = SEED_ARGS.efgf_lambda
CAFE_LAMBDA = SEED_ARGS.cafe_lambda
LABEL_SMOOTHING = SEED_ARGS.label_smoothing
DOMAIN_ADAPT_LAMBDA = SEED_ARGS.domain_adapt_lambda
USE_FC_FEATURES = SEED_ARGS.use_fc_features
FC_NUM_CHANNELS = SEED_ARGS.fc_num_channels
FC_EMBED_DIM = SEED_ARGS.fc_embed_dim
ADABN = SEED_ARGS.adabn
TTA_MODE = SEED_ARGS.tta_mode
FNIRS_TEMPORAL_ALIGN = SEED_ARGS.fnirs_temporal_align
EGTA_K_INIT = SEED_ARGS.egta_k_init
TENT_LR = SEED_ARGS.tent_lr
TENT_STEPS = SEED_ARGS.tent_steps
if TTA_MODE == "tent" and ADABN:
    raise ValueError(
        "--tta-mode tent and --adabn both modify the SAME BatchNorm layers and "
        "cannot be combined -- this project's isolated-variable protocol requires "
        "testing them separately. Drop --adabn to use --tta-mode tent.")
LR = SEED_ARGS.lr
WEIGHT_DECAY = SEED_ARGS.weight_decay
DROPOUT = SEED_ARGS.dropout
CAFE_DROPOUT = SEED_ARGS.cafe_dropout
CONV_DROPOUT = SEED_ARGS.conv_dropout
CONV_DROPOUT_TYPE = SEED_ARGS.conv_dropout_type
FUSION_FNIRS_INJECT = SEED_ARGS.fusion_fnirs_inject
ERP_BRANCH = SEED_ARGS.erp_branch
EGTA_PCC_LAMBDA = SEED_ARGS.egta_pcc_lambda
CAFE_PCC_MODE = SEED_ARGS.cafe_pcc_mode
RIEM_BRANCH = SEED_ARGS.riem_branch
RIEM_BLEND = SEED_ARGS.riem_blend
RIEM_LAMBDA = SEED_ARGS.riem_lambda
RIEM_WEIGHT_DECAY = SEED_ARGS.riem_weight_decay
RIEM_DROPOUT = SEED_ARGS.riem_dropout
if not (0.0 <= RIEM_DROPOUT < 1.0):
    raise SystemExit(f"--riem-dropout must be in [0, 1), got {RIEM_DROPOUT}")
EEG_FILTERBANK = SEED_ARGS.eeg_filterbank
EEG_FILTERBANK_SCOPE = SEED_ARGS.eeg_filterbank_scope
FNIRS_CHROMA_SPLIT = SEED_ARGS.fnirs_chroma_split
if EGTA_PCC_LAMBDA < 0.0:
    raise ValueError(f"--egta-pcc-lambda must be >= 0, got {EGTA_PCC_LAMBDA}")
if EGTA_PCC_LAMBDA > 0.0 and SEED_ARGS.fnirs_temporal_align != "egta":
    raise ValueError("--egta-pcc-lambda > 0 requires --fnirs-temporal-align egta")
if CONV_DROPOUT is not None and not (0.0 <= CONV_DROPOUT < 1.0):
    raise ValueError(f"--conv-dropout must be in [0, 1), got {CONV_DROPOUT}")
STAGE1_MAX_EPOCHS = SEED_ARGS.stage1_max_epochs
STAGE1_PATIENCE = SEED_ARGS.stage1_patience
BEST_MIN_EPOCH = SEED_ARGS.best_min_epoch
STAGE1_FIXED_EPOCHS = SEED_ARGS.stage1_fixed_epochs
if BEST_MIN_EPOCH < 1 or STAGE1_FIXED_EPOCHS < 0:
    raise ValueError("--best-min-epoch must be >= 1 and --stage1-fixed-epochs >= 0")
if STAGE1_FIXED_EPOCHS > 0 and BEST_MIN_EPOCH > 1:
    raise ValueError("--stage1-fixed-epochs and --best-min-epoch > 1 are mutually exclusive")
STAGE2_MAX_EPOCHS = SEED_ARGS.stage2_max_epochs
WARMUP_EPOCHS = SEED_ARGS.warmup_epochs
MIN_LR_FACTOR = SEED_ARGS.min_lr_factor
FNIRS_NUM_CHROMOPHORES = SEED_ARGS.fnirs_num_chromophores
EEG_SPATIAL_MODE = SEED_ARGS.eeg_spatial_mode
EEG_SPATIAL_LAYERS = SEED_ARGS.eeg_spatial_layers
EEG_SPATIAL_HEADS = SEED_ARGS.eeg_spatial_heads
EEG_BACKBONE = SEED_ARGS.eeg_backbone
EEG_BAND_MODE = SEED_ARGS.eeg_band_mode
EEG_BAND_D_MODEL = SEED_ARGS.eeg_band_d_model
EEG_BAND_EMBED_DIM = SEED_ARGS.eeg_band_embed_dim
EEG_BAND_WEIGHT_DECAY = SEED_ARGS.eeg_band_weight_decay
NORM_MODE = SEED_ARGS.norm_mode
NORM_GROUPS = SEED_ARGS.norm_groups
EFGF_GATE_MODE = SEED_ARGS.efgf_gate_mode
NORMALIZE_YPRED = SEED_ARGS.normalize_ypred
VAL_MODE = SEED_ARGS.val_mode
NO_DECAY_SCALARS = SEED_ARGS.no_decay_scalars
STAGE2_STOP_MODE = SEED_ARGS.stage2_stop_mode
SWA_ENABLED = SEED_ARGS.swa_enabled
SWA_START_EPOCH = SEED_ARGS.swa_start_epoch
SWA_LR = SEED_ARGS.swa_lr
EFGF_GATE_INIT = SEED_ARGS.efgf_gate_init
ATTN_CONV_INIT_GAIN = SEED_ARGS.attn_conv_init_gain
if EFGF_GATE_INIT != -6.0 and EFGF_GATE_MODE == "clamp":
    print(f"WARNING: --efgf-gate-init {EFGF_GATE_INIT} has NO EFFECT because "
          f"--efgf-gate-mode is 'clamp' (clamp mode always inits gamma/alpha to exactly "
          f"0.0, ignoring this flag). Add --efgf-gate-mode sigmoid to use a custom init.")


def _parse_pairs(spec):
    """Parse '1,2;0,1' -> [(1,2), (0,1)]. Empty string -> None (feature off)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    pairs = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [c for c in chunk.replace(",", " ").split() if c]
        if len(parts) != 2:
            raise ValueError(f"--asymmetric-pairs chunk {chunk!r} is not a pair of two "
                              f"class indices (expected e.g. '1,2').")
        i, j = int(parts[0]), int(parts[1])
        if not (0 <= i < 3 and 0 <= j < 3):
            raise ValueError(f"--asymmetric-pairs indices must be in 0..2 (got {i},{j}).")
        if i == j:
            raise ValueError(f"--asymmetric-pairs got a self-pair ({i},{j}); the diagonal is "
                              f"always 1.0 (a correct prediction is never up-weighted).")
        pairs.append((i, j))
    return pairs


ASYMMETRIC_PAIRS = _parse_pairs(SEED_ARGS.asymmetric_pairs)
ASYMMETRIC_WEIGHT = SEED_ARGS.asymmetric_weight
ORDINAL_REG_LAMBDA = SEED_ARGS.ordinal_reg_lambda
LAMBDA_EEG = SEED_ARGS.lambda_eeg
LAMBDA_FNIRS = SEED_ARGS.lambda_fnirs

random.seed(SEED)
import numpy as np
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Deterministic cuDNN (slightly slower, but removes another source of
# run-to-run noise beyond just the initial seed).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"[SEED CONTROL] Using seed={SEED} (numpy/torch/python random all seeded, cuDNN deterministic)")

# Same self-sufficiency pattern for run_subject_cross_session: if it's already in
# the notebook's namespace (from an earlier %run tsfnet_model_notebook.py), that
# version is used as-is. Otherwise, try importing it from the installed
# tsfnet_model PACKAGE (folder) if that's what you're using; if that package
# isn't present either (e.g. you're only using the single-file
# tsfnet_model_notebook.py, not the tsfnet_model/ package folder), fall back
# to running tsfnet_model_notebook.py itself, found next to THIS script --
# this covers all three ways this project can be set up without silently
# depending on cell execution order in any of them.
try:
    run_subject_cross_session
except NameError:
    try:
        from tsfnet_model.train import run_subject_cross_session
    except ModuleNotFoundError:
        try:
            _this_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # __file__ may not be defined in some exec/%run contexts -- fall back
            # to the current working directory (Jupyter's default is usually the
            # notebook's own folder, which is where these files normally live).
            _this_dir = os.getcwd()
        _notebook_path = os.path.join(_this_dir, "tsfnet_model_notebook.py")
        if not os.path.isfile(_notebook_path):
            raise ModuleNotFoundError(
                "Could not find run_subject_cross_session anywhere: no "
                "tsfnet_model package installed, and tsfnet_model_notebook.py "
                f"not found next to this script ({_notebook_path}). Put "
                "tsfnet_model_notebook.py in the same folder as "
                "run_all_subjects.py, or run it first via %run.")
        print(f"NOTE: tsfnet_model package not found; loading definitions from "
              f"{_notebook_path} instead.")
        exec(open(_notebook_path, encoding="utf-8").read(), globals())

_DEFAULT_DATA_ROOT = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"
DATA_ROOT = SEED_ARGS.data_root or _DEFAULT_DATA_ROOT
if not os.path.isdir(DATA_ROOT):
    raise FileNotFoundError(f"--data-root does not exist: {DATA_ROOT}")
def _parse_subjects_spec(spec):
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return [f"VP{i:03d}" for i in ids]


SUBJECTS = _parse_subjects_spec(SEED_ARGS.subjects)     # default "1-26" = VP001..VP026, prior behavior

# priority-XSUBJ: resolve, per subject, the pretrained checkpoint whose
# pretraining pool did NOT contain that subject. Every leakage-relevant fact
# is re-checked here rather than trusted from the manifest's file layout.
PRETRAINED_DIR = SEED_ARGS.pretrained_dir
PRETRAINED_NEW_MODULES = tuple(SEED_ARGS.pretrained_new_modules or ())
PRETRAIN_CKPT = {}
_PRETRAIN_SUBTAG = ""
def _resolve_pretrain_map(man, pretrained_dir, seed, subjects, has_data):
    """sid -> checkpoint path (pooled) or sid -> {held-out session: path} (loso).

    Every leakage-relevant fact is re-derived from the manifest and re-checked here:
      subject level : no target is in its own group's pretraining pool;
      session level : (loso) the checkpoint used when session s is held out was trained on
                      sessions that exclude s, and on exactly the other sessions."""
    if int(man["seed"]) != seed:
        raise ValueError(f"pretrained models were trained with seed {man['seed']} but this run "
                         f"uses --seed {seed}. Pretrain once per seed so the three seeds stay "
                         f"independent replicates.")
    protocol = man.get("session_protocol", "pooled")
    if protocol not in ("pooled", "loso"):
        raise ValueError(f"unknown session_protocol {protocol!r} in the manifest")
    all_sessions = {int(x) for x in man.get("sessions", [])}
    if protocol == "loso" and len(all_sessions) < 2:
        raise ValueError("loso manifest lists fewer than 2 sessions")
    out = {}
    for g, info in man["groups"].items():
        pool = set(info["pool_train"]) | set(info["pool_val"])
        if protocol == "loso":
            if {int(k) for k in info["folds"]} != all_sessions:
                raise ValueError(f"group {g}: folds {sorted(info['folds'])} do not cover sessions "
                                 f"{sorted(all_sessions)} (pretraining not finished?)")
            per_session = {}
            for k, fold in info["folds"].items():
                s_held = int(k)
                trained_on = {int(x) for x in fold["train_sessions"]}
                if s_held in trained_on:
                    raise ValueError(f"SESSION LEAKAGE: group {g} model for held-out session "
                                     f"{s_held} was trained on sessions {sorted(trained_on)}")
                if trained_on != all_sessions - {s_held}:
                    raise ValueError(f"group {g}, held-out session {s_held}: trained on "
                                     f"{sorted(trained_on)}, expected {sorted(all_sessions - {s_held})}")
                ck = os.path.join(pretrained_dir, fold["checkpoint"])
                if not os.path.isfile(ck):
                    raise FileNotFoundError(f"group {g} session {s_held} checkpoint missing: {ck}")
                per_session[s_held] = ck
        else:
            ck = os.path.join(pretrained_dir, info["checkpoint"])
            if not os.path.isfile(ck):
                raise FileNotFoundError(f"group {g} checkpoint missing: {ck} "
                                        f"(pretraining not finished?)")
        for sid in info["targets"]:
            if sid in pool:
                raise ValueError(f"LEAKAGE: {sid} is a target of group {g} but is in its "
                                 f"pretraining pool")
            if sid in out:
                raise ValueError(f"{sid} is a target of more than one group")
            out[sid] = dict(per_session) if protocol == "loso" else ck
    missing = [x for x in subjects if x not in out and has_data(x)]
    if missing:
        raise ValueError(f"no pretrained model covers {missing} -- rerun pretraining for "
                         f"their group(s)")
    return out, protocol


if PRETRAINED_DIR:
    _man_path = os.path.join(PRETRAINED_DIR, "pretrain_manifest.json")
    if not os.path.isfile(_man_path):
        raise FileNotFoundError(f"--pretrained-dir has no pretrain_manifest.json: {_man_path}")
    with open(_man_path, encoding="utf-8") as _f:
        _man = json.load(_f)
    if os.path.normpath(_man["data_root"]) != os.path.normpath(DATA_ROOT):
        print(f"[pretrained] WARNING: pretraining used data_root={_man['data_root']} but this run "
              f"uses {DATA_ROOT}")
    PRETRAIN_CKPT, _PRETRAIN_PROTOCOL = _resolve_pretrain_map(
        _man, PRETRAINED_DIR, SEED, SUBJECTS, lambda x: os.path.isdir(os.path.join(DATA_ROOT, x)))
    _PRETRAIN_SUBTAG = "_pt-" + _man["id"] + ("-loso" if _PRETRAIN_PROTOCOL == "loso" else "")
    if PRETRAINED_NEW_MODULES:
        # The subtag must record it: a run whose riem_head starts random is NOT
        # the same condition as one loaded strictly, and must not share a folder.
        _PRETRAIN_SUBTAG += "-new" + "".join(m[:4] for m in PRETRAINED_NEW_MODULES)
        print(f"[pretrained] random-init modules: {list(PRETRAINED_NEW_MODULES)}")
    print(f"[pretrained] {len(PRETRAIN_CKPT)} subjects mapped to {len(_man['groups'])} "
          f"pretrained models from {PRETRAINED_DIR} (id {_man['id']}, "
          f"session protocol: {_PRETRAIN_PROTOCOL})")
elif PRETRAINED_NEW_MODULES:
    raise ValueError("--pretrained-new-modules needs --pretrained-dir")
RUN_DIAGNOSTICS = SEED_ARGS.run_diagnostics          # True = also make PCA/t-SNE/saliency plots per subject (SLOW: 26 subjects x 3 sessions = 78 diagnostic runs). Off by default; pass --diagnostics to enable.
# Variant tag distinguishes this isolated ablation (fusion combination formula,
# plus the now-accepted no-clip setting from priority-1) from any other
# variant in every filename, so conditions can NEVER be accidentally mixed up
# or overwrite each other on disk.
_CLIP_SUBTAG = "gradclip1p0" if GRAD_CLIP_NORM is not None else "noclip"
_CW_SUBTAG = CLASS_WEIGHTING_SCHEME if CLASS_WEIGHTING_SCHEME != "none" else "noclassweight"
if CLASS_WEIGHTING_SCHEME == "effective_number":
    _CW_SUBTAG += f"beta{EFFECTIVE_NUMBER_BETA}"
elif CLASS_WEIGHTING_SCHEME == "focal":
    _CW_SUBTAG += f"gamma{FOCAL_GAMMA}"
_OS_SUBTAG = "smoteK" + str(SMOTE_K_NEIGHBORS) if OVERSAMPLING_SCHEME == "smote" else "nosmote"
_EEG_SUBTAG = (f"bilstmH{EEG_LSTM_HIDDEN}{EEG_READOUT_MODE}" if EEG_TEMPORAL_MODE == "bilstm"
               else "gaplogvar" if EEG_TEMPORAL_MODE == "gap_logvar"   # priority-LOGVAR: own folder, never "gap"'s
               else "gap")
_CAFE_SUBTAG = f"cafe{CAFE_D_MODEL}"
if ASYMMETRIC_PAIRS:
    _pair_str = "-".join(f"{i}{j}" for i, j in ASYMMETRIC_PAIRS)
    _ASYM_SUBTAG = f"asym{_pair_str}w{ASYMMETRIC_WEIGHT}"
else:
    _ASYM_SUBTAG = "noasym"
if ORDINAL_REG_LAMBDA > 0:
    _ASYM_SUBTAG += f"_ordreg{ORDINAL_REG_LAMBDA}"
# SHORTENED (2026-09-04): the original "_lambdaEEG{X}FNIRS{Y}" format pushed
# some full result-file paths past Windows' default 260-character MAX_PATH,
# causing a FileNotFoundError from CrashSafeLogger even though the folder
# had just been created successfully by os.makedirs above it (Windows simply
# refuses to open/create a file whose full path exceeds the limit, and the
# error message doesn't make that obvious). "_lam0.1-0.3" carries the same
# information in about half the characters.
_LAMBDA_SUBTAG = (f"_lam{LAMBDA_EEG}-{LAMBDA_FNIRS}"
                  if (LAMBDA_EEG != 0.2 or LAMBDA_FNIRS != 0.2) else "")
_CHROMA_SUBTAG = f"_{FNIRS_CHROMA_MODE}" if FNIRS_CHROMA_MODE != "both" else ""
_FNIRSGATE_SUBTAG = ("" if FNIRS_INPUT_GATE == "none"
                     else f"_fgate-{FNIRS_INPUT_GATE}"
                          + (f"{FNIRS_GATE_PRIOR_STRENGTH:g}"
                             if FNIRS_INPUT_GATE == "soft_prior" else ""))
_CHROMO_SUBTAG = f"_chr{FNIRS_NUM_CHROMOPHORES}" if FNIRS_NUM_CHROMOPHORES != 2 else ""
_ESP_SUBTAG = (f"_esp{EEG_SPATIAL_MODE}L{EEG_SPATIAL_LAYERS}H{EEG_SPATIAL_HEADS}"
               if EEG_SPATIAL_MODE not in ("none", "bandtok") else "")
if EEG_SPATIAL_MODE == "bandtok":
    _ESP_SUBTAG = f"_bandtok{SEED_ARGS.bandtok_norm}d{SEED_ARGS.bandtok_d_model}h{SEED_ARGS.bandtok_heads}"
_MHS_SUBTAG = f"_mhs{MHS_LAMBDA}" if MHS_LAMBDA > 0.0 else ""
_PCCLAMBDA_SUBTAG = ((f"_efgfL{EFGF_LAMBDA:g}" if EFGF_LAMBDA != 1.0 else "")
                     + (f"_cafeL{CAFE_LAMBDA:g}" if CAFE_LAMBDA != 1.0 else ""))
_LS_SUBTAG = f"_ls{LABEL_SMOOTHING}" if LABEL_SMOOTHING > 0.0 else ""
_DA_SUBTAG = f"_da{DOMAIN_ADAPT_LAMBDA}" if DOMAIN_ADAPT_LAMBDA > 0.0 else ""
_FC_SUBTAG = "_fc" if USE_FC_FEATURES else ""
_ADABN_SUBTAG = "_adabn" if ADABN else ""
_TENT_SUBTAG = "_tent" if TTA_MODE == "tent" else ""
_EGTA_SUBTAG = "_egta" if FNIRS_TEMPORAL_ALIGN == "egta" else ""
_EGTA_SUBTAG += "_kinit-noisy" if EGTA_K_INIT == "noisy" else ""   # MUST be in the tag (rule 5)
# priority-D-1-safe / priority-GN: these MUST enter the variant tag, otherwise
# a run with the flag on writes into the SAME results folder (same hash) as the
# run without it and overwrites its per-session JSONs and .pt checkpoints --
# which is exactly what happened to the E3+E4+EGTA reference folder during the
# D-1-safe runs. Non-default values only, so the default hash is unchanged.
_BAND_SUBTAG = f"_band{EEG_BAND_MODE}" if EEG_BAND_MODE != "none" else ""
_NORM_SUBTAG = (f"_gn{NORM_GROUPS}" if NORM_MODE == "group" else "")
# priority-EFGF-gatefix / priority-ypred-normfix: same class of bug as
# _BAND_SUBTAG/_NORM_SUBTAG above -- these MUST enter the variant tag or a
# run with the flag on writes into the SAME results folder (same hash) as a
# run without it. This one already happened once (harmlessly, only because
# the true reference folder had separately been moved aside by hand) before
# being caught; fixed now, before it happens again with real consequences.
_EFGF_SUBTAG = f"_efgf{EFGF_GATE_MODE}" if EFGF_GATE_MODE != "clamp" else ""
_YPREDNORM_SUBTAG = "_ypnorm" if NORMALIZE_YPRED else ""
_VALMODE_SUBTAG = {"cross_session": "_xsval", "test_leak": "_TESTLEAK"}.get(VAL_MODE, "")
_NODECAY_SUBTAG = "_nodecay" if NO_DECAY_SCALARS else ""
_STAGE2_SUBTAG = ({"epoch_count": "_s2ep", "none": "_s2none"}.get(STAGE2_STOP_MODE, "")
                  + "_s2reset")   # CRITICAL: the stage-2 optimizer lr/state reset (see
# tsfnet_model_notebook.py, ~line 2890) used to be CONDITIONAL on scheduler2 being set
# (default None -> reset never ran, matching the ORIGINAL pre-patch file). It was made
# UNCONDITIONAL when epoch_count/none modes were added (a real, correct fix -- stage 2
# should always start with fresh Adam moments regardless of whether a scheduler is
# used) -- but this silently changed the DEFAULT ('loss' mode) training dynamics
# without changing the hash. A "plain reference" run (no new flags) with the CURRENT
# file therefore trains differently from the historical reference checkpoints at hash
# b5e27ee74d, yet would get the identical hash without this permanent suffix -- risking
# silent overwrite of that reference folder. This suffix is unconditional (always
# appended, not gated by any flag) precisely because the reset itself is now always
# active and is not something a flag turns on or off.
_EFGFINIT_SUBTAG = (f"_gi{EFGF_GATE_INIT}"
                    if EFGF_GATE_INIT != -6.0 and EFGF_GATE_MODE == "sigmoid" else "")
_EFGFINIT_SUBTAG += f"_acg{ATTN_CONV_INIT_GAIN:g}" if ATTN_CONV_INIT_GAIN is not None else ""
_LABRAM_SUBTAG = {"native": "", "labram_frozen": "_labram",
                  "labram_ensemble": "_labramens",
                  "labram_blend": "_labramblend"}[EEG_BACKBONE]
# priority-G1: only non-default values enter the tag, so an unchanged run keeps
# the exact same variant hash (and therefore the same results folder) as before.
_G1_PARTS = []
for _nm, _v, _dflt in (("lr", LR, 1e-3), ("wd", WEIGHT_DECAY, 1e-4),
                        ("do", DROPOUT, 0.5), ("cdo", CAFE_DROPOUT, 0.3),
                        ("s1e", STAGE1_MAX_EPOCHS, 100), ("s1p", STAGE1_PATIENCE, 20),
                        ("s2e", STAGE2_MAX_EPOCHS, 80),
                        ("bme", BEST_MIN_EPOCH, 1), ("fix", STAGE1_FIXED_EPOCHS, 0),   # priority-BESTSEL
                        ("wu", WARMUP_EPOCHS, 0), ("mlf", MIN_LR_FACTOR, 1.0)):
    if _v != _dflt:
        _G1_PARTS.append(f"{_nm}{_v}")
_G1_SUBTAG = ("_" + "_".join(_G1_PARTS)) if _G1_PARTS else ""
_DATAROOT_SUBTAG = ("_data-" + os.path.basename(os.path.normpath(DATA_ROOT))
                    if os.path.normpath(DATA_ROOT) != os.path.normpath(_DEFAULT_DATA_ROOT) else "")
_ERP_SUBTAG = f"_erp{ERP_BRANCH}" if ERP_BRANCH != "none" else ""
_ERP_SUBTAG += f"_egtapcc{EGTA_PCC_LAMBDA:g}" if EGTA_PCC_LAMBDA > 0.0 else ""   # priority-EGTAPCC
_ERP_SUBTAG += "_cafepccabs" if CAFE_PCC_MODE == "abs" else ""   # priority-CAFEPCCABS
_ERP_SUBTAG += (f"_riem{RIEM_BLEND:g}l{RIEM_LAMBDA:g}wd{RIEM_WEIGHT_DECAY:g}" if RIEM_BRANCH == "member" else "")   # priority-RIEM
_ERP_SUBTAG += (f"do{RIEM_DROPOUT:g}" if (RIEM_BRANCH == "member" and RIEM_DROPOUT != 0.5) else "")   # priority-RIEMDROP
_ERP_SUBTAG += (f"_fb{EEG_FILTERBANK}{'' if EEG_FILTERBANK_SCOPE == 'both' else '-eegonly'}"
                if EEG_FILTERBANK != "none" else "")                                  # priority-FILTERBANK
_ERP_SUBTAG += f"_cs{FNIRS_CHROMA_SPLIT}" if FNIRS_CHROMA_SPLIT != "none" else ""      # priority-CHROMASPLIT
_FNIRSINJECT_SUBTAG = f"_fninj{FUSION_FNIRS_INJECT}" if FUSION_FNIRS_INJECT != "none" else ""
_CONVDROP_SUBTAG = ((f"_convdo{CONV_DROPOUT:g}" if CONV_DROPOUT is not None else "")
                    + ("_convdoElem" if CONV_DROPOUT_TYPE != "channel" else ""))
# priority-noiseaware: MUST be in the tag. This is the project's recurring
# bug #5 (a new flag not reflected in the results tag, risking silent
# overwrite of reference checkpoints). Both the probability AND the SNR set
# are included, since two runs differing only in the SNR list are different
# conditions. Empty string when the feature is off, so every existing tag
# (and therefore every existing results folder) is unchanged.
if NOISE_AWARE_PROB > 0.0:
    _snr_str = "-".join(f"{s:g}" for s in NOISE_AWARE_SNR_DB)
    _NOISEAWARE_SUBTAG = f"_noiseaware{NOISE_AWARE_PROB:g}snr{_snr_str}"
else:
    _NOISEAWARE_SUBTAG = ""
# priority-SWA: MUST be in the tag -- same recurring bug #5 class as
# _NOISEAWARE_SUBTAG above. swa_lr=None still matters to the tag (it changes
# the actual lr used during the averaging tail, see train_one_fold), so it is
# included whenever swa_enabled is True even if the user didn't pass it
# explicitly. Empty string when disabled, so every existing tag is unchanged.
if SWA_ENABLED:
    _swa_start_str = str(SWA_START_EPOCH) if SWA_START_EPOCH is not None else "auto"
    _swa_lr_str = f"{SWA_LR:g}" if SWA_LR is not None else "auto"
    _SWA_SUBTAG = f"_swa_s{_swa_start_str}_lr{_swa_lr_str}"
else:
    _SWA_SUBTAG = ""
import hashlib
_VARIANT_TAG_FULL = (f"{FUSION_COMBINATION}_{_CLIP_SUBTAG}_{_CW_SUBTAG}_{_OS_SUBTAG}"
                     f"_{_EEG_SUBTAG}_{_CAFE_SUBTAG}_{_ASYM_SUBTAG}{_LAMBDA_SUBTAG}{_CHROMA_SUBTAG}{_MHS_SUBTAG}{_LS_SUBTAG}{_DA_SUBTAG}{_FC_SUBTAG}{_ADABN_SUBTAG}{_TENT_SUBTAG}{_EGTA_SUBTAG}{_G1_SUBTAG}{_CHROMO_SUBTAG}{_ESP_SUBTAG}{_LABRAM_SUBTAG}{_BAND_SUBTAG}{_NORM_SUBTAG}{_EFGF_SUBTAG}{_YPREDNORM_SUBTAG}{_VALMODE_SUBTAG}{_NODECAY_SUBTAG}{_STAGE2_SUBTAG}{_EFGFINIT_SUBTAG}{_NOISEAWARE_SUBTAG}{_SWA_SUBTAG}{_PCCLAMBDA_SUBTAG}{_CONVDROP_SUBTAG}{_DATAROOT_SUBTAG}{_FNIRSINJECT_SUBTAG}{_PRETRAIN_SUBTAG}{_ERP_SUBTAG}{_FNIRSGATE_SUBTAG}")
_VARIANT_HASH = hashlib.md5(_VARIANT_TAG_FULL.encode()).hexdigest()[:10]
_VARIANT_TAG = _VARIANT_HASH
RUN_TAG = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{SEED}_{_VARIANT_TAG}"   # unique per run+seed+variant, so a later comparison run never overwrites this baseline

# Per-subject cross_session_VPxxx_summary.json + .pt checkpoints (written by
# run_subject_cross_session) previously went straight into each subject's own
# data folder (subject_dir) with a FIXED filename -- so every new run variant
# silently overwrote the previous variant's session-level results, destroying
# the confusion-matrix-level detail needed for the kind of per-subject/
# per-class analysis done for SSP and class-weighting. Every run now gets its
# own variant+seed-tagged results folder instead (NOT timestamped, so that
# re-running the SAME variant+seed on purpose still lands in the same place
# -- e.g. to resume a crashed run -- but a DIFFERENT variant or seed always
# gets a distinct, permanent folder that no later run can touch).
RESULTS_DIR_VARIANT = os.path.join(DATA_ROOT, f"results_{_VARIANT_TAG}_seed{SEED}")
os.makedirs(RESULTS_DIR_VARIANT, exist_ok=True)
print(f"[RESULTS] Per-subject session-level JSON/checkpoints for this run go to: "
      f"{RESULTS_DIR_VARIANT}")

# Moved into RESULTS_DIR_VARIANT (was directly under DATA_ROOT before) so the
# whole-run summary CSV/JSON, the run log, AND every per-subject file for
# this run all sit together in one folder instead of being split between
# DATA_ROOT's root and this subfolder.
SUMMARY_CSV = os.path.join(RESULTS_DIR_VARIANT, f"baseline_all_subjects_{RUN_TAG}.csv")
SUMMARY_JSON = os.path.join(RESULTS_DIR_VARIANT, f"baseline_all_subjects_{RUN_TAG}.json")

_csv_fields = ["subject_id", "status", "mean_acc", "std_acc", "mean_kappa", "std_kappa",
               "eeg_only_acc", "fnirs_only_acc", "fusion_only_acc",
               "w_eeg", "w_fusion", "w_fnirs",
               # NEW: per-class recall from the confusion matrix summed over this
               # subject's 3 session-folds. Added because overall accuracy alone
               # hides WHICH class the model fails on -- and for this 3-class
               # n-back task the 2-back/3-back pair is the known-hard one, so a
               # drop in recall_2back/recall_3back is the signal that actually
               # matters when judging an ablation.
               "recall_0back", "recall_2back", "recall_3back",
               "elapsed_min", "error"]


def _fmt_cm(cm):
    """Render a 3x3 confusion matrix as compact aligned text for the run log."""
    names = ["0-back", "2-back", "3-back"]
    out = ["          " + "".join(f"{('pred ' + n):>12}" for n in names)]
    for i, n in enumerate(names):
        row_total = sum(cm[i])
        rec = (cm[i][i] / row_total * 100) if row_total else float("nan")
        out.append(f"  true {n:<4}" + "".join(f"{v:>12d}" for v in cm[i]) +
                   f"   | recall {rec:5.1f}%")
    return "\n".join(out)


def _append_csv_row(row):
    file_exists = os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


all_results = {}
_all_cms = []      # one 3x3 confusion matrix per completed subject, for the grand total below

with CrashSafeLogger(os.path.join(RESULTS_DIR_VARIANT, f"baseline_run_log_{RUN_TAG}.txt"),
                      experiment_tag=f"11SEG_BASELINE_{_VARIANT_TAG_FULL.upper()}") as log:
    print(f"===== FULL BASELINE RUN started {RUN_TAG} (seed={SEED}) =====")
    print(f"[ABLATION] asymmetric_pairs={ASYMMETRIC_PAIRS!r} weight={ASYMMETRIC_WEIGHT} "
          f"ordinal_reg_lambda={ORDINAL_REG_LAMBDA} (priority-A1 isolated variable) "
          f"| lambda_eeg={LAMBDA_EEG} lambda_fnirs={LAMBDA_FNIRS} "
          f"(priority-A3 isolated variable, paper default 0.2/0.2) "
          f"| cafe_d_model={CAFE_D_MODEL!r} (priority-4 result: rejected) "
          f"| eeg_temporal_mode={EEG_TEMPORAL_MODE!r} (hidden={EEG_LSTM_HIDDEN}, "
          f"readout={EEG_READOUT_MODE!r} if bilstm) "
          f"(priority-5 result: rejected, fixed at 'gap') "
          f"| class_weighting_scheme={CLASS_WEIGHTING_SCHEME!r} "
          f"(beta={EFFECTIVE_NUMBER_BETA} if effective_number, gamma={FOCAL_GAMMA} if focal) "
          f"| oversampling_scheme={OVERSAMPLING_SCHEME!r} (k={SMOTE_K_NEIGHBORS} if smote) "
          f"| fnirs_chroma_mode={FNIRS_CHROMA_MODE!r} (priority-A4 isolated variable) "
          f"| fnirs_input_gate={FNIRS_INPUT_GATE!r} (priority-FNIRSGATE, prior_strength={FNIRS_GATE_PRIOR_STRENGTH:g}) "
          f"| fnirs_num_chromophores={FNIRS_NUM_CHROMOPHORES} (priority-G2a) "
          f"| eeg_spatial_mode={EEG_SPATIAL_MODE!r} (L={EEG_SPATIAL_LAYERS}, H={EEG_SPATIAL_HEADS}, priority-G4) "
          f"| eeg_band_mode={EEG_BAND_MODE!r} (d_model={EEG_BAND_D_MODEL}, "
          f"embed_dim={EEG_BAND_EMBED_DIM}, wd={EEG_BAND_WEIGHT_DECAY}, priority-D-1-safe) "
          f"| norm_mode={NORM_MODE!r} (groups={NORM_GROUPS}, priority-GN) "
          f"| efgf_gate_mode={EFGF_GATE_MODE!r} (priority-EFGF-gatefix) "
          f"| normalize_ypred={NORMALIZE_YPRED!r} (priority-ypred-normfix) "
          f"| val_mode={VAL_MODE!r} (priority-XSVAL) "
          f"| no_decay_scalars={NO_DECAY_SCALARS!r} (priority-nodecay-scalars) "
          f"| stage2_stop_mode={STAGE2_STOP_MODE!r} (priority-stage2fix) "
          f"| efgf_gate_init={EFGF_GATE_INIT!r} (priority-efgfinit) "
          f"| attn_conv_init_gain={ATTN_CONV_INIT_GAIN!r} (None=default, untested if set) "
          f"| mhs_lambda={MHS_LAMBDA} (priority-B1 isolated variable, MSTFDN-inspired) "
          f"| efgf_lambda={EFGF_LAMBDA} cafe_lambda={CAFE_LAMBDA} (priority-PCCLAMBDA isolated variable) "
          f"| label_smoothing={LABEL_SMOOTHING} (priority-B4 isolated variable) "
          f"| domain_adapt_lambda={DOMAIN_ADAPT_LAMBDA} (priority-C1 isolated variable) "
          f"| adabn={ADABN} (priority-E4 isolated variable, TRANSDUCTIVE) "
          f"| [G1] lr={LR} wd={WEIGHT_DECAY} dropout={DROPOUT}/{CAFE_DROPOUT} "
          f"| conv_dropout={CONV_DROPOUT} type={CONV_DROPOUT_TYPE} (priority-CONVDROP) "
          f"| fusion_fnirs_inject={FUSION_FNIRS_INJECT!r} (priority-FNIRSINJECT) "
          f"| erp_branch={ERP_BRANCH!r} (priority-ERP) "
          f"epochs={STAGE1_MAX_EPOCHS}/{STAGE1_PATIENCE}/{STAGE2_MAX_EPOCHS} "
          f"warmup={WARMUP_EPOCHS} min_lr_factor={MIN_LR_FACTOR} "
          f"| use_fc_features={USE_FC_FEATURES} (priority-E1 isolated variable, "
          f"fc_num_channels={FC_NUM_CHANNELS}, fc_embed_dim={FC_EMBED_DIM} if enabled) "
          f"(priority-3 family isolated variables) "
          f"| fusion_combination={FUSION_COMBINATION!r} (priority-2 result: accepted) "
          f"| grad_clip_norm={GRAD_CLIP_NORM!r} (priority-1 result: rejected, fixed at None) "
          f"| eeg_backbone={EEG_BACKBONE!r} (priority-C2 isolated variable) "
          f"| tta_mode={TTA_MODE!r} (priority-E4 generalization, lr={TENT_LR} "
          f"steps={TENT_STEPS} if 'tent') "
          f"| fnirs_temporal_align={FNIRS_TEMPORAL_ALIGN!r} (priority-1 roadmap, STA-Net EGTA) "
          f"| egta_k_init={EGTA_K_INIT!r} (default 'zero'=accepted baseline; 'noisy'=tested fix, -1.34pp) "
          f"| swa_enabled={SWA_ENABLED!r} (priority-SWA, start_epoch={SWA_START_EPOCH!r} "
          f"lr={SWA_LR!r})")
    print(f"Subjects: {len(SUBJECTS)} | diagnostics={RUN_DIAGNOSTICS} | device="
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Summary JSON: {SUMMARY_JSON}")
    print(f"Per-subject session-level results (JSON + checkpoints): {RESULTS_DIR_VARIANT}\n")

    run_t0 = time.time()
    for i, subject_id in enumerate(SUBJECTS, 1):
        subject_dir = os.path.join(DATA_ROOT, subject_id)
        print(f"\n{'='*70}")
        print(f"[{i}/{len(SUBJECTS)}] {subject_id}  ({time.time()-run_t0:.1f}s elapsed so far)")
        print(f"{'='*70}")

        if not os.path.isdir(subject_dir):
            print(f"  SKIP: {subject_dir} does not exist (preprocessing not run for this subject?)")
            row = {"subject_id": subject_id, "status": "skipped_no_data", "mean_acc": "",
                   "std_acc": "", "mean_kappa": "", "std_kappa": "",
                   "eeg_only_acc": "", "fnirs_only_acc": "", "fusion_only_acc": "",
                   "w_eeg": "", "w_fusion": "", "w_fnirs": "",
                   "recall_0back": "", "recall_2back": "", "recall_3back": "",
                   "elapsed_min": "", "error": ""}
            all_results[subject_id] = row
            _append_csv_row(row)
            continue

        t0 = time.time()
        try:
            mean_acc, std_acc, mean_kappa, std_kappa, summary = run_subject_cross_session(
                subject_dir, subject_id,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=16,
                run_diagnostics_flag=RUN_DIAGNOSTICS,
                results_dir=RESULTS_DIR_VARIANT,  # variant+seed-tagged folder -- never overwrites another run's session-level JSON/checkpoints
                fusion_combination=FUSION_COMBINATION,   # priority-2 result: accepted, fixed at "paper_mean"
                class_weighting_scheme=CLASS_WEIGHTING_SCHEME,  # ISOLATED ABLATION (priority-3): only this differs vs. the accepted priority-1+2 baseline
                effective_number_beta=EFFECTIVE_NUMBER_BETA,    # only used if scheme=="effective_number"
                focal_gamma=FOCAL_GAMMA,                        # only used if scheme=="focal"
                oversampling_scheme=OVERSAMPLING_SCHEME,        # ISOLATED ABLATION (priority-3d): data-level axis, independent of class_weighting_scheme
                smote_k_neighbors=SMOTE_K_NEIGHBORS,            # only used if oversampling_scheme=="smote"
                eeg_temporal_mode=EEG_TEMPORAL_MODE,            # ISOLATED ABLATION (priority-5): EEG branch's own classifier only, never touches EFGF/Fusion/CAFE
                eeg_lstm_hidden=EEG_LSTM_HIDDEN,                # only used if eeg_temporal_mode=="bilstm"
                eeg_readout_mode=EEG_READOUT_MODE,              # "h_n" (fixed, default), "last_step" (buggy, A/B only), "mean_pool"
                cafe_d_model=CAFE_D_MODEL,                       # priority-4 result: rejected -> fixed at 64
                fnirs_num_segments=FNIRS_NUM_SEGMENTS,           # must match preprocessing's FNIRS_NUM_SEGMENTS
                asymmetric_hard_pairs=ASYMMETRIC_PAIRS,          # ISOLATED ABLATION (priority-A1): targets the 2-back/3-back failure mode
                asymmetric_hard_weight=ASYMMETRIC_WEIGHT,
                ordinal_reg_lambda=ORDINAL_REG_LAMBDA,
                grad_clip_norm=GRAD_CLIP_NORM,           # priority-1 result: rejected -> default None
                class_count_level=CLASS_COUNT_LEVEL,     # "window" (legacy/verified) or "trial" (verified no-op)
                lambda_eeg=LAMBDA_EEG,                    # ISOLATED ABLATION (priority-A3): paper default 0.2
                lambda_fnirs=LAMBDA_FNIRS,                # paper default 0.2
                fnirs_chroma_mode=FNIRS_CHROMA_MODE,
                fnirs_input_gate=FNIRS_INPUT_GATE,        # ISOLATED ABLATION (priority-FNIRSGATE)
                fnirs_gate_prior_strength=FNIRS_GATE_PRIOR_STRENGTH,      # ISOLATED ABLATION (priority-A4): "both" (default)/"hbo_only"/"hbr_only"
                mhs_lambda=MHS_LAMBDA,                    # ISOLATED ABLATION (priority-B1): CAFE multi-head HSIC diversity term, 0.0 (default) disables
                efgf_lambda=EFGF_LAMBDA,                  # ISOLATED ABLATION (priority-PCCLAMBDA): 1.0 (default) = paper Eq. 5
                cafe_lambda=CAFE_LAMBDA,                  # ISOLATED ABLATION (priority-PCCLAMBDA): 1.0 (default) = paper Eq. 5
                label_smoothing=LABEL_SMOOTHING,          # ISOLATED ABLATION (priority-B4): 0.0 (default) disables
                domain_adapt_lambda=DOMAIN_ADAPT_LAMBDA,  # ISOLATED ABLATION (priority-C1): 0.0 (default) disables
                use_fc_features=USE_FC_FEATURES,          # ISOLATED ABLATION (priority-E1): EEG functional-connectivity (PLV) embedding, False (default) disables
                fc_num_channels=FC_NUM_CHANNELS,
                fc_embed_dim=FC_EMBED_DIM,
                fnirs_num_chromophores=FNIRS_NUM_CHROMOPHORES,
                bandtok_d_model=SEED_ARGS.bandtok_d_model, bandtok_heads=SEED_ARGS.bandtok_heads,
                bandtok_norm=SEED_ARGS.bandtok_norm,
                eeg_spatial_mode=EEG_SPATIAL_MODE,        # ISOLATED ABLATION (priority-G4): channel-wise MHSA on the EEG branch
                eeg_spatial_layers=EEG_SPATIAL_LAYERS,
                eeg_spatial_heads=EEG_SPATIAL_HEADS,   # ISOLATED ABLATION (priority-G2a): 2 (default) = HbO/HbR, 3 = +HbT
                eeg_backbone=EEG_BACKBONE,                # ISOLATED ABLATION (priority-C2): frozen pretrained LaBraM EEG head, "native" (default) disables
                adabn=ADABN,                              # ISOLATED ABLATION (priority-E4): AdaBN test-time BN-stat adaptation, False (default) disables
                tta_mode=TTA_MODE,                        # ISOLATED ABLATION (priority-E4 generalization): "tent" for entropy-minimization BN adaptation, None (default) disables
                tent_lr=TENT_LR, tent_steps=TENT_STEPS,   # only used when tta_mode="tent"
                fnirs_temporal_align=FNIRS_TEMPORAL_ALIGN,
                egta_k_init=EGTA_K_INIT,                      # "zero" (default, accepted) vs "noisy" (tested fix, -1.34pp)
                eeg_band_mode=EEG_BAND_MODE,                  # priority-D-1-safe
                eeg_band_d_model=EEG_BAND_D_MODEL,
                eeg_band_embed_dim=EEG_BAND_EMBED_DIM,
                eeg_band_weight_decay=EEG_BAND_WEIGHT_DECAY,
                norm_mode=NORM_MODE,                           # priority-GN
                norm_groups=NORM_GROUPS,
                efgf_gate_mode=EFGF_GATE_MODE,                 # priority-EFGF-gatefix
                normalize_ypred=NORMALIZE_YPRED,               # priority-ypred-normfix
                val_mode=VAL_MODE,                             # priority-XSVAL
                no_decay_scalars=NO_DECAY_SCALARS,             # priority-nodecay-scalars
                stage2_stop_mode=STAGE2_STOP_MODE,             # priority-stage2fix
                # priority-noiseaware: forwarded via **train_kwargs into
                # train_one_fold. prob=0.0 (default) is a strict no-op.
                noise_aware_prob=NOISE_AWARE_PROB,
                noise_aware_snr_db=NOISE_AWARE_SNR_DB,
                # priority-SWA: forwarded via **train_kwargs into train_one_fold,
                # same mechanism as noise_aware_prob/snr_db above.
                swa_enabled=SWA_ENABLED,
                swa_start_epoch=SWA_START_EPOCH,
                swa_lr=SWA_LR,
                efgf_gate_init=EFGF_GATE_INIT,                 # priority-efgfinit
                attn_conv_init_gain=ATTN_CONV_INIT_GAIN,        # "item 3": None (default) = PyTorch's own init, untested float = xavier_uniform_(gain=...)
                # ISOLATED ABLATION (roadmap priority-1): "egta" = STA-Net EGTA pooling, "none" (default) disables
                lr=LR, weight_decay=WEIGHT_DECAY,         # priority-G1 (defaults = prior behavior)
                dropout=DROPOUT, cafe_dropout=CAFE_DROPOUT,
                conv_dropout=CONV_DROPOUT,                # priority-CONVDROP: None (default) = use dropout
                conv_dropout_type=CONV_DROPOUT_TYPE,      # priority-CONVDROP: "channel" (default) = Dropout3d
                fusion_fnirs_inject=FUSION_FNIRS_INJECT,  # priority-FNIRSINJECT: "none" (default) = paper design
                init_checkpoint=PRETRAIN_CKPT.get(subject_id),  # priority-XSUBJ: None (default) = random init
                init_checkpoint_new_modules=PRETRAINED_NEW_MODULES,  # priority-XSUBJ-NEWMOD: () (default) = strict load
                erp_branch=ERP_BRANCH,                    # priority-ERP: "none" (default) = no ERP pathway
                cafe_pcc_mode=CAFE_PCC_MODE,                 # priority-CAFEPCCABS
                riem_branch=RIEM_BRANCH, riem_blend=RIEM_BLEND,   # priority-RIEM: 'none' (default) = off
                riem_lambda=RIEM_LAMBDA, riem_weight_decay=RIEM_WEIGHT_DECAY,
                riem_dropout=RIEM_DROPOUT,                # priority-RIEMDROP: 0.5 (default) = unchanged
                egta_pcc_lambda=EGTA_PCC_LAMBDA,          # priority-EGTAPCC: 0.0 (default) = term off
                eeg_filterbank=EEG_FILTERBANK,            # priority-FILTERBANK: "none" (default) = off
                fnirs_chroma_split=FNIRS_CHROMA_SPLIT,    # priority-CHROMASPLIT: "none" (default) = off
                eeg_filterbank_scope=EEG_FILTERBANK_SCOPE,
                stage1_max_epochs=STAGE1_MAX_EPOCHS, stage1_patience=STAGE1_PATIENCE,
                best_min_epoch=BEST_MIN_EPOCH, stage1_fixed_epochs=STAGE1_FIXED_EPOCHS,
                stage2_max_epochs=STAGE2_MAX_EPOCHS,
                warmup_epochs=WARMUP_EPOCHS, min_lr_factor=MIN_LR_FACTOR,
            )
            elapsed = (time.time() - t0) / 60
            branch = summary["branch_accuracy"]
            weights = summary["learned_fusion_weights"]
            _cm = summary.get("confusion_matrix_summed", [[0] * 3 for _ in range(3)])
            _pcr = summary.get("per_class_recall", [None, None, None])
            _all_cms.append(_cm)
            row = {"subject_id": subject_id, "status": "ok",
                   "mean_acc": mean_acc, "std_acc": std_acc,
                   "mean_kappa": mean_kappa, "std_kappa": std_kappa,
                   "eeg_only_acc": round(branch["eeg_only_mean"], 4),
                   "fnirs_only_acc": round(branch["fnirs_only_mean"], 4),
                   "fusion_only_acc": round(branch["fusion_only_mean"], 4),
                   "w_eeg": round(weights["w_eeg_mean"], 4),
                   "w_fusion": round(weights["w_fusion_mean"], 4),
                   "w_fnirs": round(weights["w_fnirs_mean"], 4),
                   "recall_0back": (round(_pcr[0], 4) if _pcr[0] is not None else ""),
                   "recall_2back": (round(_pcr[1], 4) if _pcr[1] is not None else ""),
                   "recall_3back": (round(_pcr[2], 4) if _pcr[2] is not None else ""),
                   "elapsed_min": round(elapsed, 2), "error": ""}
            print(f"\n[{subject_id}] DONE in {elapsed:.1f} min | "
                  f"acc={mean_acc:.4f}+-{std_acc:.4f}  kappa={mean_kappa:.4f}+-{std_kappa:.4f} | "
                  f"branch acc: eeg={branch['eeg_only_mean']:.4f} fnirs={branch['fnirs_only_mean']:.4f} "
                  f"fusion={branch['fusion_only_mean']:.4f} | "
                  f"fusion weights: w_eeg={weights['w_eeg_mean']:.4f} w_fusion={weights['w_fusion_mean']:.4f} "
                  f"w_fnirs={weights['w_fnirs_mean']:.4f}")
            print(f"  confusion matrix (summed over 3 session-folds):")
            print(_fmt_cm(_cm))

        except Exception as e:
            elapsed = (time.time() - t0) / 60
            print(f"\n[{subject_id}] FAILED after {elapsed:.1f} min: {e}")
            traceback.print_exc()
            row = {"subject_id": subject_id, "status": "error", "mean_acc": "",
                   "std_acc": "", "mean_kappa": "", "std_kappa": "",
                   "eeg_only_acc": "", "fnirs_only_acc": "", "fusion_only_acc": "",
                   "w_eeg": "", "w_fusion": "", "w_fnirs": "",
                   "recall_0back": "", "recall_2back": "", "recall_3back": "",
                   "elapsed_min": round(elapsed, 2), "error": str(e)}

        all_results[subject_id] = row
        _append_csv_row(row)               # written immediately -- survives a crash on the NEXT subject
        with open(SUMMARY_JSON, "w") as f:  # full JSON rewritten each time (cheap at 26 subjects)
            json.dump(all_results, f, indent=2)

    # ----- final aggregate across all successfully completed subjects -----
    ok_rows = [r for r in all_results.values() if r["status"] == "ok"]
    print(f"\n\n{'='*70}")
    print(f"BASELINE RUN COMPLETE: {len(ok_rows)}/{len(SUBJECTS)} subjects finished successfully")
    print(f"{'='*70}")
    if ok_rows:
        import statistics
        accs = [r["mean_acc"] for r in ok_rows]
        kappas = [r["mean_kappa"] for r in ok_rows]
        print(f"Grand mean accuracy across subjects: {statistics.mean(accs):.4f} "
              f"(+-{statistics.pstdev(accs):.4f})")
        print(f"Grand mean kappa across subjects:    {statistics.mean(kappas):.4f} "
              f"(+-{statistics.pstdev(kappas):.4f})")

        eeg_accs = [r["eeg_only_acc"] for r in ok_rows]
        fnirs_accs = [r["fnirs_only_acc"] for r in ok_rows]
        fusion_accs = [r["fusion_only_acc"] for r in ok_rows]
        w_eeg = [r["w_eeg"] for r in ok_rows]
        w_fusion = [r["w_fusion"] for r in ok_rows]
        w_fnirs = [r["w_fnirs"] for r in ok_rows]
        print(f"Grand mean branch accuracy:  eeg={statistics.mean(eeg_accs):.4f}  "
              f"fnirs={statistics.mean(fnirs_accs):.4f}  fusion={statistics.mean(fusion_accs):.4f}")
        print(f"Grand mean learned fusion weights:  w_eeg={statistics.mean(w_eeg):.4f}  "
              f"w_fusion={statistics.mean(w_fusion):.4f}  w_fnirs={statistics.mean(w_fnirs):.4f}")

    # ----- dataset-level confusion matrix, summed over every completed subject -----
    # This is the single most informative number-block for judging an ablation on
    # this task: mean accuracy can stay flat while the model quietly trades
    # 2-back recall for 3-back recall (or collapses one class entirely, as was
    # observed for some subjects in the SSP and class-weighting experiments).
    if _all_cms:
        grand_cm = [[sum(cm[i][j] for cm in _all_cms) for j in range(3)] for i in range(3)]
        print(f"\n{'='*70}")
        print(f"GRAND CONFUSION MATRIX (summed over {len(_all_cms)} subjects x 3 folds)")
        print(f"{'='*70}")
        print(_fmt_cm(grand_cm))
        _tot = sum(sum(r) for r in grand_cm)
        _diag = sum(grand_cm[i][i] for i in range(3))
        print(f"\n  overall accuracy from matrix: {_diag / _tot * 100:.2f}%  (n={_tot} windows)")
        # Off-diagonal traffic between the two hard classes, called out explicitly:
        # 2-back <-> 3-back is the pair the literature (Cao et al.) reports as the
        # bottleneck for 3-class n-back.
        _hard = grand_cm[1][2] + grand_cm[2][1]
        _errs = _tot - _diag
        if _errs > 0:
            print(f"  2-back <-> 3-back confusions: {_hard} of {_errs} total errors "
                  f"({_hard / _errs * 100:.1f}% of all errors)")

    failed = [sid for sid, r in all_results.items() if r["status"] != "ok"]
    if failed:
        print(f"\nSubjects that did NOT complete: {failed}")

    # ----- per-subject listing: mean accuracy/kappa for every subject, one line each -----
    print(f"\n{'='*70}")
    print(f"PER-SUBJECT RESULTS ({len(ok_rows)} completed / {len(SUBJECTS)} total)")
    print(f"{'='*70}")
    for subject_id in SUBJECTS:   # preserves VP001..VP026 order regardless of completion order
        r = all_results.get(subject_id)
        if r is None:
            print(f"{subject_id}: NOT RUN")
        elif r["status"] == "ok":
            print(f"{subject_id}: acc={r['mean_acc']:.4f} +- {r['std_acc']:.4f}  "
                  f"kappa={r['mean_kappa']:.4f} +- {r['std_kappa']:.4f}  |  "
                  f"branch acc: eeg={r['eeg_only_acc']:.4f} fnirs={r['fnirs_only_acc']:.4f} "
                  f"fusion={r['fusion_only_acc']:.4f}  |  "
                  f"fusion weights: w_eeg={r['w_eeg']:.4f} w_fusion={r['w_fusion']:.4f} "
                  f"w_fnirs={r['w_fnirs']:.4f}")
        elif r["status"] == "skipped_no_data":
            print(f"{subject_id}: SKIPPED (no preprocessed data)")
        else:
            print(f"{subject_id}: FAILED ({r.get('error', 'unknown error')})")

    print(f"\nFull results: {SUMMARY_CSV}")
    print(f"             {SUMMARY_JSON}")
