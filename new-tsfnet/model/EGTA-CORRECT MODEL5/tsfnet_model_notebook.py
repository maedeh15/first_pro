# ============================================================================
# TSFNet -- single-file, notebook-ready version (no relative imports).
# Combines layers.py + model.py + losses.py + dataset.py + train.py +
# model_diagnostics.py from the tsfnet_model package into one script, in
# dependency order, so it can be pasted into Jupyter cells (split at the
# "# %% CELL n" markers below, or just run the whole thing as one script
# via %run tsfnet_model_notebook.py -- RECOMMENDED over manual cell-by-cell
# copy/paste, which can corrupt indentation in Jupyter).
#
# VERSION: 2026-08-11-v5 (diagnostics_dir now defaults to a SEPARATE sibling
# folder -- <parent of subject_dir>/visualizations/<subject_id>/ -- instead
# of nesting inside subject_dir/diagnostics/, so all diagnostic PNGs stay
# out of the preprocessed-data folder, matching the same visualizations/
# tree the preprocessing script already uses for its own QC plots).
#
# See the original package's README.md for the full explanation of every
# adaptation made to the paper (3-class output, reduced model size, added
# weight decay, reduced stage1/stage2 epoch budgets), and the docstring at
# the top of model_diagnostics.py for the diagnostics-specific adaptations
# (feature-hook targets, softmax-vs-logits handling, no-RNN saliency, 3s
# EEG window time axis).
# ============================================================================

# %% CELL 1 -- imports
import argparse
import collections
import copy
import json
import math
import os
import gc
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, silhouette_score,
                              davies_bouldin_score, classification_report)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")



# %% CELL 2 -- layers.py (Conv3dSame, ConvBlock3D, EEGLikeBranch, FNIRSBranch, EFGFLayer, CAFELayer)
# ============================================================================
# TSFNet building blocks: EEG/fNIRS/Fusion 3D-CNN branches, the EFGF layer,
# and the CAFE layer. Faithfully reproduces Sec. 3.1-3.3 and Table 1 of:
#   Zhang, Y.; Yin, B.; Yuan, X. "TSFNet: Temporal-Spatial Fusion Network for
#   Hybrid Brain-Computer Interface." Sensors 2025, 25, 6111.
#
# ADAPTED FOR THIS PROJECT'S n-back DATASET (documented inline wherever a
# choice deviates from or fills a gap in the paper):
#   - 3-class output (0-/2-/3-back) instead of the paper's binary tasks.
#   - Heavier regularization (higher dropout default, weight decay in the
#     training script) because our preprocessed dataset gives far fewer
#     training windows per subject (~180 train / ~90 test per session-held-
#     out fold) than the datasets the paper was tuned on, so the paper's
#     3.3M-parameter configuration is a much higher overfitting risk here.
#
# Input tensor shapes (produced by Preprocessing_TSFNet_paper_nback.py,
# which was deliberately kept parameter-identical to the paper's Sec. 2.3
# so NO reshaping/adaptation is needed here):
#   X_eeg   : (B, 16, 16, 600)          -- H, W, T
#   X_fnirs : (B, 11, 16, 16, 30, 2)    -- Segment, H, W, T, {HbO,HbR}
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# TensorFlow-style "SAME" padding for Conv3d (verified against every entry of
# the paper's Table 1: out_dim = ceil(in_dim / stride) exactly, for all 6
# convolutions in both the main branches and both EFGF stages -- see the
# derivation check performed before writing this file).
# ----------------------------------------------------------------------------
class Conv3dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.conv = nn.Conv3d(in_channels, out_channels, self.kernel_size,
                               stride=self.stride, padding=0)

    def forward(self, x):
        # x: (B, C, D, H, W)
        pads = []
        for dim, k, s in zip(x.shape[-3:], self.kernel_size, self.stride):
            out = -(-dim // s)  # ceil division
            pad_needed = max((out - 1) * s + k - dim, 0)
            pad_before = pad_needed // 2
            pad_after = pad_needed - pad_before
            pads = [pad_before, pad_after] + pads  # F.pad wants last-dim-first order
        x = F.pad(x, pads)
        return self.conv(x)


def _triple(v):
    return (v, v, v) if isinstance(v, int) else tuple(v)


def _largest_divisor_at_most(n, cap):
    """Largest g <= cap with n % g == 0 (falls back to 1 = LayerNorm-like)."""
    for g in range(min(cap, n), 0, -1):
        if n % g == 0:
            return g
    return 1


def convert_batchnorm_to_groupnorm(module, num_groups=8):
    """
    priority-GN: replace EVERY BatchNorm{1,2,3}d in `module` (recursively, in
    place) by an nn.GroupNorm over the same channels.

    WHY (a structural fix, not added capacity -- same family as E3/E4/EGTA):
    BatchNorm normalizes each channel with statistics computed ACROSS THE
    BATCH. In this project's LOSO folds the train DataLoader has
    batch_size=16 and no drop_last, so the last batch of every epoch holds
    anywhere from 1 to 15 windows; BN then normalizes with a mean/variance
    estimated from a handful of samples, and the running statistics used at
    test time are an exponential average of those noisy estimates. GroupNorm
    computes statistics PER SAMPLE over groups of channels, so a sample's
    output is identical whatever else is in its batch, train and eval behave
    the same, and there are no running statistics to be mis-estimated.

    Parameter count: IDENTICAL (weight + bias per channel, same as BN's affine).
    Initialization:  IDENTICAL (weight=1, bias=0) and consumes NO random draws,
                     so converting after construction cannot shift any other
                     module's init (rule 6).
    num_groups:      8 by default; clamped to the largest divisor of each
                     layer's channel count (16 and 32 channels here -> 8).
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            C = child.num_features
            g = _largest_divisor_at_most(C, num_groups)
            gn = nn.GroupNorm(g, C, eps=child.eps, affine=True)
            setattr(module, name, gn)
        else:
            convert_batchnorm_to_groupnorm(child, num_groups)
    return module


class ConvBlock3D(nn.Module):
    """One 3D-conv layer as specified in Sec. 3.1: Conv3dSame -> ELU -> BN -> Dropout."""

    def __init__(self, in_ch, out_ch, kernel, stride, dropout, dropout_type="channel"):
        super().__init__()
        self.conv = Conv3dSame(in_ch, out_ch, kernel, stride)
        self.act = nn.ELU()
        self.bn = nn.BatchNorm3d(out_ch)
        # priority-CONVDROP: dropout_type="channel" (default) is the original
        # nn.Dropout3d, which zeroes WHOLE feature channels. With only 16
        # (conv1) / 32 (conv2) channels and p=0.5, half of every block's
        # channels are deleted at each step -- very aggressive for a ~415K
        # model that already hits train_acc=1.0 by epoch 4. "element" uses
        # nn.Dropout (independent per activation) instead. Neither module
        # owns parameters, so construction consumes no RNG and every weight
        # initializes identically either way (rule 6).
        if dropout_type == "channel":
            self.drop = nn.Dropout3d(dropout)
        elif dropout_type == "element":
            self.drop = nn.Dropout(dropout)
        else:
            raise ValueError(f"dropout_type must be 'channel' or 'element', got {dropout_type!r}")

    def forward(self, x):
        return self.drop(self.bn(self.act(self.conv(x))))


# ----------------------------------------------------------------------------
# EEG / Fusion branch (Table 1 "EEG" row; the Fusion branch reuses this exact
# architecture on the SAME input tensor X_eeg with its own separate weights,
# per Sec. 3.1: "since its input is entirely derived from the 3D EEG tensor,
# the same convolutional kernel parameters as in the EEG branch are adopted").
# ----------------------------------------------------------------------------
class EEGLikeBranch(nn.Module):
    def __init__(self, dropout=0.5, temporal_mode="gap", lstm_hidden=64, readout_mode="h_n",
                 dropout_type="channel"):
        """
        temporal_mode: ISOLATED ABLATION PARAMETER (priority-5, the project's
        largest and most consequential architecture change -- documented as
        "RUN 3A" in the project's architecture-changes list, item #12).
          - "gap" (default): reproduces the paper's exact Sec. 3.1 method --
            Global Average Pooling over the TIME axis only, producing a
            (B, 32, 4, 4) spatial feature map that is then flattened to a
            512-dim vector for the EEG-only classifier. Any temporal
            patterns (e.g. P300-like components) that unfold ACROSS the 50
            time steps of s2 are averaged away before classification.
          - "bilstm": priority-5 sub-experiment. Instead of averaging over
            time, s2's (B, 32, 4, 4, 50) feature map is reshaped into a
            length-50 sequence of 512-dim vectors (flattening the 32x4x4
            spatial/channel dims at each time step) and fed through a
            bidirectional LSTM (hidden_size=lstm_hidden, default 64, so
            2*64=128-dim output per time step). How the final 128-dim
            feature is READ OUT of this sequence is controlled separately
            by `readout_mode` (see below) -- this lets the classifier see
            how the spatial pattern evolves over the window rather than
            only its time-average.
          IMPORTANT: this only changes what gets returned as `pooled` (used
          for the EEG-only classifier, y_eeg). It does NOT touch s1/s2
          themselves, which are returned unchanged and are what the EFGF
          fusion module (efgf1/efgf2 in TSFNet.forward) actually consumes --
          confirmed by inspecting TSFNet.forward: efgf1(eeg_s1, ...) and
          efgf2(eeg_s2, ...) never touch `pooled`. So this change is fully
          isolated to the y_eeg branch's own classification path, exactly as
          described in the project's architecture-changes doc ("this change
          affects ONLY the EEG branch; fNIRS, Fusion, EFGF, CAFE all remain
          untouched").

        readout_mode: only used when temporal_mode="bilstm". How the (B, T,
        2*hidden) sequence of BiLSTM outputs is reduced to a single (B,
        2*hidden) feature vector. One of:
          - "h_n" (NEW DEFAULT, bug-fixed as of this edit): uses PyTorch's
            own final hidden states (h_n from nn.LSTM's return value), which
            for a bidirectional LSTM are (num_directions=2, B, hidden) --
            index 0 is the FORWARD direction's hidden state after processing
            the ENTIRE sequence left-to-right, index 1 is the BACKWARD
            direction's hidden state after processing the ENTIRE sequence
            right-to-left. Concatenating these two gives a genuine
            whole-sequence summary from both directions.
          - "last_step" (BUGGY, kept ONLY for A/B comparison against the
            already-completed/in-progress priority-5 run that used this by
            mistake): naively indexes lstm_out[:, -1, :]. For a bidirectional
            LSTM, output[:, t, :] is the CONCATENATION of the forward
            direction's state at time t and the backward direction's state
            at time t. At t=T-1 (the last time index), the forward half IS
            the correct whole-sequence forward summary (identical to h_n[0])
            -- but the backward half at t=T-1 is the backward LSTM's state
            after processing ONLY ONE time step (T-1 itself, since the
            backward pass runs T-1 -> 0), NOT the whole sequence. This means
            HALF of the 128-dim "bidirectional summary" barely used the LSTM
            at all -- verified numerically (torch.allclose against h_n[1]
            returns False) before this fix was written. This mode exists
            only so a completed/in-progress run using the old (buggy)
            behavior can be reproduced exactly for direct comparison against
            the fixed "h_n" mode -- it should NOT be used for new runs.
          - "mean_pool": averages lstm_out over the full time axis (both
            directions), i.e. mean(lstm_out, dim=1). Unlike "h_n"/
            "last_step", this doesn't privilege the sequence endpoints and
            may better capture a diagnostic pattern that is spread across
            the window rather than concentrated near one edge -- a second,
            independent sub-experiment beyond the h_n bug fix, motivated by
            the same "GAP destroys temporal information" reasoning behind
            trying BiLSTM in the first place, but pooled AFTER the LSTM has
            already enriched each time step with sequence context (unlike
            GAP, which averages the raw, context-free conv features).
        """
        super().__init__()
        self.temporal_mode = temporal_mode
        self.readout_mode = readout_mode
        self.conv1 = ConvBlock3D(1, 16, kernel=(4, 4, 12), stride=(2, 2, 6), dropout=dropout,
                                 dropout_type=dropout_type)
        self.conv2 = ConvBlock3D(16, 32, kernel=(2, 2, 6), stride=(2, 2, 2), dropout=dropout,
                                 dropout_type=dropout_type)
        if temporal_mode == "bilstm":
            # s2 is (B, 32, 4, 4, 50) -> per-timestep flattened input size is 32*4*4=512
            self.eeg_lstm = nn.LSTM(input_size=32 * 4 * 4, hidden_size=lstm_hidden,
                                     batch_first=True, bidirectional=True)
            if readout_mode not in ("h_n", "last_step", "mean_pool"):
                raise ValueError(f"Unknown readout_mode: {readout_mode!r} "
                                  f"(expected 'h_n', 'last_step', or 'mean_pool')")
        elif temporal_mode not in ("gap", "gap_logvar"):
            raise ValueError(f"Unknown temporal_mode: {temporal_mode!r} "
                             f"(expected 'gap', 'gap_logvar' or 'bilstm')")

    def forward(self, x_eeg):
        # x_eeg: (B, 16, 16, 600) -> add channel dim -> (B, 1, 16, 16, 600)
        # Conv3d expects (B, C, D, H, W); we map (H, W, T) -> (D=H, H=W, W=T)
        # consistently across the whole model, purely a labeling convention.
        x = x_eeg.unsqueeze(1)
        s1 = self.conv1(x)                 # (B, 16, 8, 8, 100)
        if self.temporal_mode == "gap_logvar":
            # priority-LOGVAR: identical ops to self.conv2(s1) (conv -> ELU ->
            # BN -> drop), unrolled only so the PRE-dropout activation is
            # available. s2 itself -- and therefore EFGF, which consumes it --
            # is computed exactly as in "gap" mode.
            s2_pre = self.conv2.bn(self.conv2.act(self.conv2.conv(s1)))
            s2 = self.conv2.drop(s2_pre)
        else:
            s2 = self.conv2(s1)            # (B, 32, 4, 4, 50)
        if self.temporal_mode == "bilstm":
            B, C, H, W, T = s2.shape
            seq = s2.permute(0, 4, 1, 2, 3).reshape(B, T, C * H * W)  # (B, T=50, 512)
            lstm_out, (h_n, c_n) = self.eeg_lstm(seq)                 # lstm_out: (B, T=50, 2*hidden); h_n: (2, B, hidden)
            if self.readout_mode == "h_n":
                # Correct whole-sequence bidirectional summary: forward's
                # final state (after t=0..T-1) concatenated with backward's
                # final state (after t=T-1..0) -- see docstring above.
                pooled = torch.cat([h_n[0], h_n[1]], dim=-1)          # (B, 2*hidden)
            elif self.readout_mode == "last_step":
                pooled = lstm_out[:, -1, :]                           # BUGGY (kept for A/B only) -> (B, 2*hidden)
            else:  # "mean_pool"
                pooled = lstm_out.mean(dim=1)                         # (B, 2*hidden)
        elif self.temporal_mode == "gap_logvar":
            # priority-LOGVAR: the n-back load markers in EEG are BAND POWER
            # (frontal-midline theta up, parietal alpha down with load), i.e.
            # signal VARIANCE over time. Plain GAP -- mean(ELU(.)) over time --
            # is a weak, poorly conditioned proxy for that, and nothing else in
            # this model squares or log-transforms anything. This adds the
            # ShallowConvNet/FBCSP-style log-variance next to the paper's GAP.
            #
            # Variance is taken on the PRE-dropout activation on purpose:
            # Dropout3d zeroes whole channels, and a zeroed channel has
            # variance 0 -> log(1e-6) = -13.8, a large train-time-only outlier
            # that never occurs at eval (kept channels are also rescaled by
            # 1/(1-p), inflating their variance by 1/(1-p)^2). The classifier
            # still has its own dropout. The mean half uses post-dropout s2,
            # exactly as "gap" does.
            gap = s2.mean(dim=-1)                                            # (B, 32, 4, 4)
            logvar = torch.log(torch.clamp(s2_pre.var(dim=-1, unbiased=False), min=1e-6))
            pooled = torch.cat([gap, logvar], dim=1)                         # (B, 64, 4, 4)
        else:
            pooled = s2.mean(dim=-1)       # GAP over TIME only (Sec. 3.1) -> (B, 32, 4, 4)
        return s1, s2, pooled              # stage1/stage2 maps (for EFGF) + final F_eeg/F_fusion


# ----------------------------------------------------------------------------
# fNIRS branch (Table 1 "fNIRS" row). The segment axis (now 21, per ablation
# #3) is processed as an extra batch dimension shared by one set of conv
# weights (a per-segment 3D CNN with tied weights across segments -- the
# only interpretation consistent with Table 1's fNIRS output size keeping
# "N x ..." unchanged through both conv layers). This module itself is
# segment-count-agnostic (S is read from the input tensor's own shape, not
# hardcoded), so it required NO code change for #3 -- only CAFELayer's
# n_segments (which fixes the pos_embed parameter's shape) needed updating.
# ----------------------------------------------------------------------------
class FNIRSBranch(nn.Module):
    def __init__(self, dropout=0.5, n_chromophores=2, dropout_type="channel"):
        super().__init__()
        # priority-G2a: n_chromophores is 2 (HbO, HbR -- the original and
        # default) or 3 (HbO, HbR, HbT). It only changes conv1's in_channels;
        # every downstream shape is identical, so nothing else in the network
        # needs to know about it.
        self.n_chromophores = n_chromophores
        self.conv1 = ConvBlock3D(n_chromophores, 16, kernel=(4, 4, 6), stride=(2, 2, 2), dropout=dropout,
                                 dropout_type=dropout_type)
        self.conv2 = ConvBlock3D(16, 32, kernel=(2, 2, 3), stride=(2, 2, 2), dropout=dropout,
                                 dropout_type=dropout_type)

    def forward(self, x_fnirs):
        # x_fnirs: (B, 11, 16, 16, 30, 2) -> (B, Seg, H, W, T, Chrom)
        B, S, H, W, T, C = x_fnirs.shape
        if C != self.n_chromophores:
            raise ValueError(
                f"FNIRSBranch was built for n_chromophores={self.n_chromophores} but received "
                f"x_fnirs with {C} chromophore channels. This almost always means the "
                f"preprocessed fnirs_windows.npy was produced with a different "
                f"ENABLE_FNIRS_HBT setting than the --fnirs-num-chromophores the model was "
                f"built with (2 = HbO/HbR, 3 = HbO/HbR/HbT).")
        x = x_fnirs.permute(0, 1, 5, 2, 3, 4).reshape(B * S, C, H, W, T)  # (B*S, C, 16, 16, 30)
        s1 = self.conv1(x)                                    # (B*S, 16, 8, 8, 15)
        s2 = self.conv2(s1)                                    # (B*S, 32, 4, 4, 8)
        s1 = s1.reshape(B, S, *s1.shape[1:])                    # (B, 11, 16, 8, 8, 15)
        s2 = s2.reshape(B, S, *s2.shape[1:])                    # (B, 11, 32, 4, 4, 8)
        pooled = s2.mean(dim=-1)                                # GAP over TIME -> (B, 11, 32, 4, 4)
        return s1, s2, pooled


# ----------------------------------------------------------------------------
# EFGF layer (Sec. 3.2, Eqs. 1-2, Table 1 "EFGF1"/"EFGF2" rows).
#
# At each of the 2 stages: a 1-channel conv is applied to the EEG-branch map
# (GAP'd over H,W -> a TEMPORAL attention vector) and to the fNIRS-branch map
# (GAP'd over Segment,T -> a SPATIAL attention map), matched to Table 1's
# exact output sizes (verified: stride=1 SAME-padding leaves H,W,T
# unchanged, only the channel dim is reduced to 1). These are combined via
# the trainable alpha into a single hybrid (H,W,T) attention map Phi, which
# gates the fusion-branch's OWN feature map at that stage; a trainable
# residual gamma additionally blends in the EEG-branch's own feature map.
# ----------------------------------------------------------------------------
class EFGFLayer(nn.Module):
    def __init__(self, channels, eeg_kernel, fnirs_kernel, gate_mode="clamp", gate_init=-6.0):
        super().__init__()
        self.eeg_attn_conv = Conv3dSame(channels, 1, eeg_kernel, stride=(1, 1, 1))
        self.fnirs_attn_conv = Conv3dSame(channels, 1, fnirs_kernel, stride=(1, 1, 1))
        # priority-EFGF-gatefix: gate_mode controls how the trainable scalars
        # gamma, alpha in [0,1] ("initialized to zero", Sec. 3.6) are
        # parametrized.
        #   "clamp" (default): the ORIGINAL torch.clamp(param, 0, 1)
        #     parametrization. BUG (confirmed on a real checkpoint --
        #     VP001_session1, seed42: efgf1.alpha=-0.0052, efgf1.gamma=-0.0046,
        #     both clamped to exactly 0): clamp's gradient is EXACTLY ZERO
        #     outside [0,1], so if Adam's first step pushes the raw parameter
        #     negative, it is permanently stuck at a clamped output of 0 --
        #     gradient can never pull it back. Kept as the default so every
        #     existing run and checkpoint remains bit-for-bit reproducible
        #     (rule 5); this is a real defect, not a good default to keep
        #     using going forward.
        #   "sigmoid": FIX. alpha = sigmoid(alpha_raw), gamma = sigmoid(gamma_raw),
        #     with alpha_raw/gamma_raw initialized to -6.0 (sigmoid(-6)=0.0025),
        #     the SAME near-zero-but-never-exactly-zero convention already
        #     used elsewhere in this codebase for beta_labram and the EGTA/
        #     eeg_band gates. sigmoid's gradient is NEVER exactly zero at any
        #     finite input, so a bad first step can shrink the gate but can
        #     never permanently kill it -- gradient descent can always recover.
        #     NOT bit-exact vs "clamp" even at init (0.0025 vs 0.0), the same
        #     honest, documented deviation already used for those other gates.
        if gate_mode not in ("clamp", "sigmoid"):
            raise ValueError(f"gate_mode must be 'clamp' or 'sigmoid', got {gate_mode!r}")
        self.gate_mode = gate_mode
        if gate_mode == "clamp":
            self.gamma = nn.Parameter(torch.zeros(1))
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            # priority-efgfinit: sigmoid(-6.0)=0.0025 was the original choice
            # (matching the convention used elsewhere in this file), but a
            # real Adam simulation shows it makes stage 1 (~3300 steps) too
            # short for the gate to open even when the DATA wants it to
            # (reaches only ~0.37 by end of stage 1, vs ~0.82 from -2.0) --
            # so early stopping can pick a checkpoint where fNIRS is still
            # barely contributing to fusion. gate_init is exposed so both can
            # be compared directly; -6.0 stays the default (rule 5).
            self.gamma_raw = nn.Parameter(torch.full((1,), float(gate_init)))
            self.alpha_raw = nn.Parameter(torch.full((1,), float(gate_init)))

    def forward(self, eeg_map, fnirs_map, fusion_map):
        """
        eeg_map:    (B, C, H, W, T)        -- EEG branch's stage-k conv output
        fnirs_map:  (B, S, C, H, W, Tf)    -- fNIRS branch's stage-k conv output
        fusion_map: (B, C, H, W, T)        -- Fusion branch's stage-k conv output
        returns:    fused (B, C, H, W, T), plus eeg_pooled/phi for PCC regularization
        """
        if self.gate_mode == "clamp":
            gamma = torch.clamp(self.gamma, 0.0, 1.0)
            alpha = torch.clamp(self.alpha, 0.0, 1.0)
        else:
            gamma = torch.sigmoid(self.gamma_raw)
            alpha = torch.sigmoid(self.alpha_raw)

        # --- EEG side: reduce to 1 channel, GAP over spatial (H, W) -> temporal vector ---
        eeg_reduced = self.eeg_attn_conv(eeg_map)                    # (B, 1, H, W, T)
        eeg_temporal = eeg_reduced.mean(dim=(2, 3))                  # (B, 1, T)
        eeg_temporal_att = torch.sigmoid(eeg_temporal)[:, :, None, None, :]  # (B,1,1,1,T)

        # --- fNIRS side: reduce to 1 channel per segment, GAP over (Segment, T) -> spatial map ---
        B, S, C, H, W, Tf = fnirs_map.shape
        fnirs_flat = fnirs_map.reshape(B * S, C, H, W, Tf)
        fnirs_reduced = self.fnirs_attn_conv(fnirs_flat)             # (B*S, 1, H, W, Tf)
        fnirs_reduced = fnirs_reduced.reshape(B, S, 1, H, W, Tf)
        fnirs_spatial = fnirs_reduced.mean(dim=(1, 5))                # (B, 1, H, W)
        fnirs_spatial_att = torch.sigmoid(fnirs_spatial)[:, :, :, :, None]  # (B,1,H,W,1)

        # --- Hybrid attention map Phi (Eq. 2), broadcasts to (B, 1, H, W, T) ---
        phi = alpha * fnirs_spatial_att + (1 - alpha) * eeg_temporal_att

        # --- Fused output (Eq. 1) ---
        fused = gamma * eeg_map + (1 - gamma) * fusion_map + phi * fusion_map

        # For PCC regularization loss (Sec. 3.5.2): average-pooled EEG map, and phi itself
        eeg_pooled_for_pcc = eeg_map.mean(dim=1)   # (B, H, W, T) -- channel-averaged
        phi_for_pcc = phi.squeeze(1)               # (B, H, W, T) (broadcast shape realized)
        phi_for_pcc = phi_for_pcc.expand(-1, H, W, eeg_pooled_for_pcc.shape[-1])
        return fused, eeg_pooled_for_pcc, phi_for_pcc


# ----------------------------------------------------------------------------
# CAFE layer (Sec. 3.3, Eq. 3). Bidirectional multi-head cross-attention
# between the fusion features (a single global token) and the 21 fNIRS
# segment features (a length-21 token sequence with learnable positional
# encoding, since "only certain segments ... contain relevant information").
# UPDATED for ablation #3 (FNIRS_NUM_SEGMENTS: 11->21, built on top of #1's
# WINDOW_LEN_EEG=10s): the token sequence length and pos_embed shape now
# track n_segments=21, so the attention window covers the wider fNIRS
# coverage (23s instead of 13s) that #3's preprocessing change produces.
# Gated fusion combines the attended and original features (paper does not
# give the exact gating formula; a learnable per-branch scalar gate,
# consistent with EFGF's gamma/alpha style, is used here and documented).
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# priority-B1 sub-experiment: HSIC-based multi-head diversity regularization
# for CAFE's cross-attention heads, adapted from Ding et al. 2026 (MSTFDN,
# Information Fusion), Eqs. (25)-(28). The paper computes this over the
# "spatial" (channel) axis inside its own Transformer-encoder SFEM modules,
# which this project's architecture does not have an exact structural
# equivalent of -- CAFE's cross-attention treats fNIRS SEGMENTS (11) or a
# single fusion token (1) as the attention sequence, not spatial channels.
# ADAPTATION (documented, not silent): rather than the paper's within-sample
# spatial-token Gram matrix, HSIC here is computed the more standard way
# (as in most other CKA/HSIC-in-training-loss implementations): each head's
# per-sample feature vector (pooled over its own attention sequence length)
# forms one row of an (n_samples=batch_size, d_head) matrix, and HSIC/CKA is
# computed over the BATCH dimension -- i.e. "how similar are head i's and
# head j's representations of the same batch of windows". This sidesteps
# cross_attn_fusion_to_fnirs's query-length-1 quirk (no per-sample spatial
# axis to use as "samples" there) while keeping the paper's core idea
# (penalize similarity between attention heads to encourage diverse
# subspace learning) exactly intact.
# ----------------------------------------------------------------------------
def _manual_mha_head_outputs(mha, query, key, value):
    """
    Recompute a torch.nn.MultiheadAttention (batch_first=True, self- or
    cross-attention, kdim==vdim==embed_dim so a single in_proj_weight covers
    Q/K/V) MANUALLY, returning the per-head context vectors BEFORE the final
    out_proj combination -- torch's own module never exposes these, only the
    already-concatenated-and-projected output.

    query: (B, Lq, E), key/value: (B, Lk, E). Returns (B, n_heads, Lq, d_head).

    IMPORTANT: this recomputation is READ-ONLY w.r.t. the module (uses its
    existing weights via F.linear, no parameters are created or mutated) and
    deliberately applies NO dropout (unlike the module's own forward, which
    does apply attention dropout during training) -- it exists solely to
    feed the auxiliary HSIC loss below, so it must not consume any of the
    model's random-number-generator state or otherwise perturb the ACTUAL
    forward path (which still goes through the real self.cross_attn_* call
    exactly as before). This is what keeps mhs_lambda=0.0 a bit-exact no-op
    on y_eeg/y_fusion/y_fnirs/y_pred regardless of whether this function
    happens to also be called.
    """
    E = mha.embed_dim
    H = mha.num_heads
    d_head = E // H
    B, Lq, _ = query.shape
    Lk = key.shape[1]

    w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)  # each (E, E)
    if mha.in_proj_bias is not None:
        b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)
    else:
        b_q = b_k = b_v = None

    q = F.linear(query, w_q, b_q)   # (B, Lq, E)
    k = F.linear(key, w_k, b_k)     # (B, Lk, E)
    v = F.linear(value, w_v, b_v)   # (B, Lk, E)

    q = q.reshape(B, Lq, H, d_head).transpose(1, 2)   # (B, H, Lq, d_head)
    k = k.reshape(B, Lk, H, d_head).transpose(1, 2)   # (B, H, Lk, d_head)
    v = v.reshape(B, Lk, H, d_head).transpose(1, 2)   # (B, H, Lk, d_head)

    attn_scores = (q @ k.transpose(-2, -1)) / (d_head ** 0.5)   # (B, H, Lq, Lk)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    head_out = attn_weights @ v                                  # (B, H, Lq, d_head)
    return head_out


def _center_gram(K):
    """Center an (n, n) Gram matrix K: Kc = H K H, H = I_n - (1/n) 11^T."""
    n = K.shape[0]
    ones = torch.ones(n, n, device=K.device, dtype=K.dtype) / n
    H = torch.eye(n, device=K.device, dtype=K.dtype) - ones
    return H @ K @ H


def hsic_linear(X, Y, eps=1e-8):
    """
    Standard linear-kernel HSIC (Gretton et al.; the same normalization used
    to define linear CKA in Kornblith et al. 2019, which Ding et al. cite):
        HSIC(K, L) = tr(Kc @ Lc) / (n-1)^2
    X: (n, d_x), Y: (n, d_y) -- n is the number of "samples" (here: the
    training batch), each row a head's feature vector for one sample.
    Returns a scalar tensor >= 0 (up to floating error); 0 iff X and Y (as
    linear-kernel Gram matrices) carry no shared variation across samples.
    """
    n = X.shape[0]
    if n < 2:
        # A batch of size 1 (e.g. a stray final mini-batch) can't form a
        # meaningful cross-sample Gram matrix -- return exactly 0 rather
        # than dividing by (1-1)^2 = 0.
        return torch.zeros((), device=X.device, dtype=X.dtype)
    K = X @ X.transpose(-2, -1)   # (n, n)
    L = Y @ Y.transpose(-2, -1)   # (n, n)
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    hsic = (Kc * Lc).sum() / ((n - 1) ** 2 + eps)
    return hsic


def cafe_multihead_hsic_loss(mha, query, key, value):
    """
    Eq. (28) adapted: average HSIC over all head PAIRS (i<j) for one
    attention module, where each head's per-sample feature vector is that
    head's attention output pooled (mean) over its own query sequence
    length -- collapsing (B, H, Lq, d_head) -> (B, H, d_head), then treating
    the batch dimension as the "samples" axis for HSIC (see module-level
    docstring above `_manual_mha_head_outputs` for why).
    """
    head_out = _manual_mha_head_outputs(mha, query, key, value)   # (B, H, Lq, d_head)
    head_feat = head_out.mean(dim=2)                               # (B, H, d_head)
    H = head_feat.shape[1]
    if H < 2:
        return torch.zeros((), device=head_feat.device, dtype=head_feat.dtype)
    pair_sum = torch.zeros((), device=head_feat.device, dtype=head_feat.dtype)
    n_pairs = 0
    for i in range(H):
        for j in range(i + 1, H):
            pair_sum = pair_sum + hsic_linear(head_feat[:, i, :], head_feat[:, j, :])
            n_pairs += 1
    return pair_sum / n_pairs


class CAFELayer(nn.Module):
    """
    fnirs_temporal_align: ISOLATED ABLATION PARAMETER (priority-1 of the
    literature roadmap, "adaptive temporal alignment"), faithfully adapted
    from the OFFICIAL STA-Net EGTA layer (Liu et al. 2025, Information Fusion
    119:103023; github.com/MutianLiu-SHU/STA-Net, sta.py, class e_f_attention).

    WHAT WAS FOUND BEFORE WRITING THIS (numerically verified on this class,
    not assumed -- see test_egta.py):
      (1) This layer's fNIRS output path pools the 21 segments by a UNIFORM
          MEAN (`fnirs_flat.mean(dim=1)`), and is exactly ORDER-INVARIANT to
          shuffling the segments (max|diff| 6e-08). That is precisely STA-Net's
          own "No EGTA" ablation condition ("substituted it with an averaging
          operation, equivalent to assigning uniform weight to each segment").
      (2) Cross-attention #1 below (fNIRS query -> ONE fusion key/value) is
          mathematically degenerate: with a single key, softmax is always 1.0,
          so its output is independent of the fNIRS input (max|diff| 0.00e+00
          for two completely different fNIRS inputs). It therefore adds NO
          segment-specific information to the fNIRS path. Left UNCHANGED here
          (one variable per experiment) -- documented as a finding.
      (3) Cross-attention #2 (fusion query -> 21 fNIRS keys/values, with the
          learnable positional embedding) IS a genuine EGTA -- but its output
          is routed to the FUSION feature, whereas official STA-Net routes the
          attention-weighted fNIRS (`fnirs_weighted`) to the fNIRS feature.
    So the precise, verified gap is: the fNIRS decision path (y_fnirs, 1/3 of
    Eq. 4) and the existing PCC regularizer both see a temporally-blind mean.

    'none' (default): bit-exact prior behavior.
    'egta': replace that uniform mean with EEG-anchored attention pooling over
        the 21 segments. Query = the EEG-guided fusion token (exactly as the
        official code: `q_eeg` comes from `eegfusion2`); keys = the
        position-embedded fNIRS tokens already built below; VALUES = the raw
        per-segment fNIRS features, so the output stays in the same 512-dim
        space and REPLACES the mean without any new value/output projection
        (a deliberate minimal-capacity choice: 10 of this project's rejected
        interventions added learnable capacity).
        The key projection is ZERO-initialized (the query projection is not):
        scores are then exactly 0 -> softmax is exactly uniform -> the pooled
        output equals the old mean at step 0 (bit-exact no-op at init, project
        rule 5). Zeroing BOTH projections would deadlock training (the
        bilinear score's gradient w.r.t. each factor is the other factor);
        zeroing only one leaves Wk a nonzero gradient -- verified in test 4.
        The EGTA parameters are created under a saved/restored torch RNG
        state so adding them does NOT shift the initialization of any other
        module (project rule 6; the C-2d finding was that 45/89 shared tensors
        silently changed initialization when new params were added).
    Not included (each is a separate, later experiment by design): BiSTF-Net's
    Bi-GRU encoder, a new PCC loss (this layer's fNIRS output already feeds
    the project's existing PCC regularizer, cafe_regularization_loss, which is
    exactly STA-Net's `1 - pearson_r(q_eeg, fnirs_weighted)` term), and
    STA-Net's HeUniform positional-embedding init.
    """

    def __init__(self, fusion_dim=4 * 4 * 32, fnirs_dim=4 * 4 * 32,
                 d_model=64, n_heads=4, n_segments=21, dropout=0.3,
                 fnirs_temporal_align="none"):
        super().__init__()
        if fnirs_temporal_align not in ("none", "egta"):
            raise ValueError(f"fnirs_temporal_align must be 'none' or 'egta', "
                             f"got {fnirs_temporal_align!r}")
        if fnirs_temporal_align == "egta" and (fnirs_dim % n_heads or d_model % n_heads):
            raise ValueError(f"fnirs_temporal_align='egta' needs fnirs_dim ({fnirs_dim}) "
                             f"and d_model ({d_model}) divisible by n_heads ({n_heads})")
        self.fnirs_temporal_align = fnirs_temporal_align
        self.fusion_dim = fusion_dim
        self.fnirs_dim = fnirs_dim
        self.d_model = d_model
        self.n_segments = n_segments
        self.n_heads = n_heads
        self._last_egta_weights = None

        self.fusion_proj = nn.Linear(fusion_dim, d_model)
        self.fnirs_proj = nn.Linear(fnirs_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_segments, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Cross-attention 1: fNIRS (query) attends to fusion (key/value)
        self.cross_attn_fnirs_to_fusion = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        # Cross-attention 2: fusion (query) attends to fNIRS (key/value)
        self.cross_attn_fusion_to_fnirs = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

        self.fusion_out_proj = nn.Linear(d_model, fusion_dim)
        self.fnirs_out_proj = nn.Linear(d_model, fnirs_dim)

        # Gated fusion (trainable scalar gates, sigmoid-constrained)
        self.gate_fusion = nn.Parameter(torch.zeros(1))
        self.gate_fnirs = nn.Parameter(torch.zeros(1))

        if fnirs_temporal_align == "egta":
            # RNG-neutral construction (project rule 6): save the global CPU
            # RNG state, build the new params from a fixed private seed, then
            # restore -- so every module built AFTER this layer gets exactly
            # the same initialization it would have had with 'none'.
            _rng_state = torch.get_rng_state()
            try:
                torch.manual_seed(20260921)
                self.egta_q = nn.Linear(d_model, d_model)
                self.egta_k = nn.Linear(d_model, d_model)
                nn.init.zeros_(self.egta_k.weight)   # => scores==0 => uniform
                nn.init.zeros_(self.egta_k.bias)     #    weights at step 0
            finally:
                torch.set_rng_state(_rng_state)

    def forward(self, fusion_feat, fnirs_feat):
        """
        fusion_feat: (B, 32, 4, 4)      -- Fusion branch's pooled F_fusion
        fnirs_feat:  (B, 11, 32, 4, 4)  -- fNIRS branch's pooled F_fnirs (per segment)
        returns: enhanced_fusion (B, fusion_dim), enhanced_fnirs (B, fnirs_dim),
                 hsic_loss (scalar tensor)  [for PCC + classifier + priority-B1]
        """
        B = fusion_feat.shape[0]
        fusion_flat = fusion_feat.reshape(B, -1)                       # (B, 512)
        fnirs_flat = fnirs_feat.reshape(B, self.n_segments, -1)        # (B, 11, 512)

        fusion_tok = self.fusion_proj(fusion_flat).unsqueeze(1)        # (B, 1, d_model)
        fnirs_tok = self.fnirs_proj(fnirs_flat) + self.pos_embed       # (B, 11, d_model)

        # Cross-attention 1: fNIRS attends to fusion
        fnirs_enhanced, _ = self.cross_attn_fnirs_to_fusion(
            query=fnirs_tok, key=fusion_tok, value=fusion_tok)         # (B, 11, d_model)
        # Cross-attention 2: fusion attends to fNIRS
        fusion_enhanced, _ = self.cross_attn_fusion_to_fnirs(
            query=fusion_tok, key=fnirs_tok, value=fnirs_tok)          # (B, 1, d_model)

        # priority-B1 (MSTFDN-inspired HSIC head-diversity term, see the
        # module-level docstring above _manual_mha_head_outputs). Computed
        # from a SEPARATE, dropout-free manual replay of both cross-
        # attentions' own weights -- read-only, does not affect
        # fnirs_enhanced/fusion_enhanced above in any way, so mhs_lambda=0.0
        # (default) is a bit-exact no-op on every other output of this layer.
        hsic_1 = cafe_multihead_hsic_loss(
            self.cross_attn_fnirs_to_fusion, fnirs_tok, fusion_tok, fusion_tok)
        hsic_2 = cafe_multihead_hsic_loss(
            self.cross_attn_fusion_to_fnirs, fusion_tok, fnirs_tok, fnirs_tok)
        hsic_loss = hsic_1 + hsic_2

        fnirs_enhanced_full = self.fnirs_out_proj(fnirs_enhanced)      # (B, 11, 512)
        fnirs_enhanced_full = fnirs_enhanced_full.mean(dim=1)          # pool segments -> (B, 512)
        fusion_enhanced_full = self.fusion_out_proj(fusion_enhanced.squeeze(1))  # (B, 512)

        g_fusion = torch.sigmoid(self.gate_fusion)
        g_fnirs = torch.sigmoid(self.gate_fnirs)
        if self.fnirs_temporal_align == "egta":
            fnirs_flat_pooled = self._egta_pool(fusion_tok, fnirs_tok, fnirs_flat)
        else:
            fnirs_flat_pooled = fnirs_flat.mean(dim=1)                 # original, pooled -> (B, 512)

        enhanced_fusion = g_fusion * fusion_enhanced_full + (1 - g_fusion) * fusion_flat
        enhanced_fnirs = g_fnirs * fnirs_enhanced_full + (1 - g_fnirs) * fnirs_flat_pooled

        return enhanced_fusion, enhanced_fnirs, hsic_loss

    def _egta_pool(self, fusion_tok, fnirs_tok, fnirs_flat):
        """
        EEG-guided temporal attention pooling over fNIRS segments (official
        STA-Net EGTA semantics: query from the EEG-fusion anchor, keys from the
        position-embedded fNIRS segment tokens, scaled dot-product softmax over
        segments, multi-head). Values are the raw per-segment fNIRS features.

        fusion_tok: (B, 1, d_model)   fnirs_tok: (B, S, d_model)
        fnirs_flat: (B, S, fnirs_dim)
        returns (B, fnirs_dim); stores per-head weights (B, h, S) for XAI.
        """
        B, S, _ = fnirs_tok.shape
        h = self.n_heads
        dk = self.d_model // h
        q = self.egta_q(fusion_tok).view(B, 1, h, dk).transpose(1, 2)    # (B,h,1,dk)
        k = self.egta_k(fnirs_tok).view(B, S, h, dk).transpose(1, 2)     # (B,h,S,dk)
        scores = (q @ k.transpose(-1, -2)).squeeze(2) / (dk ** 0.5)      # (B,h,S)
        w = torch.softmax(scores, dim=-1)                               # (B,h,S)
        self._last_egta_weights = w.detach()
        v = fnirs_flat.view(B, S, h, self.fnirs_dim // h).transpose(1, 2)  # (B,h,S,dv)
        pooled = (w.unsqueeze(-1) * v).sum(dim=2)                       # (B,h,dv)
        return pooled.reshape(B, self.fnirs_dim)                        # heads re-concatenated


# %% CELL 3 -- model.py (BranchClassifier, TSFNet)
# ============================================================================
# Full TSFNet model (Sec. 3.1-3.4), adapted for 3-class 0-/2-/3-back
# classification on this project's preprocessed n-back dataset.
# ============================================================================
import torch
import torch.nn as nn



class BranchClassifier(nn.Module):
    """Two dense layers + softmax (Sec. 3.4), shared structure for all three
    branch predictors (EEG, fusion, fNIRS)."""

    def __init__(self, in_dim, num_classes, hidden=128, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class SpatialMHSABranch(nn.Module):
    """
    priority-G4: multi-head self-attention over EEG CHANNELS AS TOKENS,
    adapted from Ding et al. 2026 (MSTFDN) E-SFEM.

    WHY THIS COMPONENT: it is the single largest effect in the reference
    paper's entire ablation, measured on this project's exact task and
    protocol (3-class N-back, leave-one-session-out): replacing the
    Transformer encoder with spatial convolution drops accuracy from 69.02%
    to 46.36%, i.e. -22.66 points. No other component of theirs comes close.

    WHY IT CANNOT BE APPLIED TO THE EXISTING 16x16 GRID: this project's
    spatial_mapping() interpolates 28 electrodes onto a smooth 16x16 image,
    after which no grid cell corresponds to a specific electrode. Attention
    over grid cells is a different operation from attention over channels,
    and would not be a test of the reference result. So this module consumes
    the RAW CHANNEL-SPACE windows (28 x 2000) that
    Preproc_tsfnet_nback_badch.py's ENABLE_EEG_CHANNEL_SPACE now saves.

    ARCHITECTURE (deliberately small -- the base model is only ~407k
    parameters, and priority-G2a already showed this dataset punishes added
    capacity that carries no new information):
        (B, CH, T)
          -> depthwise temporal conv x2  (per-channel, no cross-channel mixing,
             so channel identity is preserved for the attention that follows)
          -> adaptive average pool to exactly d_model time features
          -> (B, CH, d_model): CH tokens, d_model features each
          -> TransformerEncoder, NO positional encoding (per the reference:
             the token axis is spatial, not temporal, so position embeddings
             would impose a meaningless ordering on electrodes)
          -> 1x1 conv compressing CH -> n_compress, flatten, Linear -> embed_dim

    NOTE ON POOLING CHOICE: channels are compressed with a learned 1x1
    convolution rather than averaged. Averaging over channels would discard
    exactly the spatial pattern the attention layer just computed, which is
    the thing being tested.
    """

    def __init__(self, n_channels=28, d_model=64, n_heads=4, n_layers=2,
                 dim_feedforward=256, n_compress=4, embed_dim=64, dropout=0.3):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_heads = n_heads

        # Depthwise (groups=n_channels) so each electrode's time series is
        # filtered independently -- no cross-channel mixing before attention.
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=25, stride=10,
                      groups=n_channels, padding=12),
            nn.BatchNorm1d(n_channels),
            nn.ELU(),
            nn.Conv1d(n_channels, n_channels, kernel_size=13, stride=5,
                      groups=n_channels, padding=6),
            nn.BatchNorm1d(n_channels),
            nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.compress = nn.Conv1d(n_channels, n_compress, kernel_size=1)
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_compress * d_model, embed_dim),
            nn.ELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, x_ch):
        # x_ch: (B, CH, T)
        if x_ch.dim() != 3 or x_ch.shape[1] != self.n_channels:
            raise ValueError(
                f"SpatialMHSABranch expected (B, {self.n_channels}, T) channel-space EEG, "
                f"got {tuple(x_ch.shape)}. This usually means the preprocessed "
                f"{{subject_id}}_eeg_channel_windows.npy is missing or was produced with a "
                f"different montage -- re-run preprocessing with "
                f"ENABLE_EEG_CHANNEL_SPACE=True.")
        h = self.temporal(x_ch)          # (B, CH, T')
        h = self.pool(h)                 # (B, CH, d_model)
        h = self.encoder(h)              # (B, CH, d_model), attention over CH tokens
        h = self.compress(h)             # (B, n_compress, d_model)
        return self.out(h)               # (B, embed_dim)


# ----------------------------------------------------------------------------
# Fixed (non-trainable) FIR band-pass filterbank.
# ----------------------------------------------------------------------------
class FixedFIRBandFilter(nn.Module):
    """
    Five physiologically-standard EEG bands (delta/theta/alpha/beta/gamma),
    implemented as a FIXED, non-trainable depthwise Conv1d. Coefficients are
    precomputed with a windowed-sinc (Hamming) design -- classic linear-phase
    FIR bandpass, the "FIR zero-phase" design named in the roadmap's D-1
    description -- and registered as a buffer, so:
      - requires_grad=False: NOT a learnable filter (matches the project's
        explicit prior decision, documented in نقشه_راه_تکمیلی: "in the
        low-data regime, fixed bands beat learnable ones" -- EEGminer's
        finding that sinc-bandpass gradients are near-zero almost
        everywhere applies to LEARNED filters, not this one).
      - moves with .to(device) automatically (it's a buffer, not a bare
        tensor), and adds ZERO trainable parameters.
      - filtering happens INSIDE the forward pass (GPU, transient), never
        touching disk -- the concern about the ~14GB -> ~72GB blow-up from
        pre-storing 5 interpolated grids does not apply here, because this
        operates on the small (28, 2000) raw channel tensor, not the
        (16,16,2000) grid, and the filtered bands are never saved.
    """

    # (low_hz, high_hz) per band; standard clinical EEG convention.
    BANDS = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 45.0),   # capped below Nyquist/2 margin at 200Hz sfreq
    }

    def __init__(self, n_channels=28, sfreq=200.0, n_taps=101):
        super().__init__()
        if n_taps % 2 == 0:
            n_taps += 1  # odd length => exact linear phase, integer group delay
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.n_taps = n_taps
        self.n_bands = len(self.BANDS)
        self.pad = n_taps // 2  # 'same' padding => output aligned with input

        kernels = torch.stack(
            [self._design_bandpass(lo, hi, sfreq, n_taps) for lo, hi in self.BANDS.values()],
            dim=0,
        )  # (n_bands, n_taps)
        # Depthwise conv weight layout: (out_channels, in_channels/groups, kernel).
        # We build ONE Conv1d per band (see forward), each depthwise over the
        # 28 EEG channels with groups=n_channels, so every band's kernel is
        # replicated identically across all 28 channels (rule 4: the filter
        # itself is channel-agnostic, matching how bandpass filtering is
        # physically defined -- it is not a spatial operation).
        # Shape stored here: (n_bands, n_channels, 1, n_taps) for buffer
        # registration; expanded per-band inside forward().
        weight = kernels.view(self.n_bands, 1, 1, n_taps).repeat(1, n_channels, 1, 1)
        self.register_buffer("_weight", weight)  # (n_bands, n_channels, 1, n_taps)

    @staticmethod
    def _design_bandpass(low_hz, high_hz, sfreq, n_taps):
        """Windowed-sinc (Hamming) FIR bandpass, linear phase, unity passband gain."""
        nyq = sfreq / 2.0
        low_n, high_n = low_hz / nyq, high_hz / nyq
        n = torch.arange(n_taps, dtype=torch.float64) - (n_taps - 1) / 2.0

        def sinc_lp(cutoff_n, nn_):
            x = cutoff_n * nn_
            h = torch.where(nn_ == 0, torch.full_like(nn_, cutoff_n),
                             torch.sin(math.pi * x) / (math.pi * nn_))
            return h

        h_high = sinc_lp(high_n, n)   # low-pass at high cutoff
        h_low = sinc_lp(low_n, n)     # low-pass at low cutoff
        h_bp = h_high - h_low         # bandpass = difference of two low-passes
        window = torch.hamming_window(n_taps, periodic=False, dtype=torch.float64)
        h_bp = h_bp * window
        # Normalize passband gain to ~1 at the band's center frequency.
        center_n = (low_n + high_n) / 2.0
        freqs = torch.exp(-1j * math.pi * center_n * n)
        gain = torch.abs((h_bp.to(torch.complex128) * freqs).sum())
        h_bp = h_bp / (gain + 1e-12)
        return h_bp.to(torch.float32)

    def forward(self, x_chan):
        # x_chan: (B, C, T) raw channel-space EEG, C == self.n_channels
        if x_chan.dim() != 3 or x_chan.shape[1] != self.n_channels:
            raise ValueError(
                f"FixedFIRBandFilter expected (B, {self.n_channels}, T), "
                f"got {tuple(x_chan.shape)}.")
        B, C, T = x_chan.shape
        outs = []
        for b in range(self.n_bands):
            w = self._weight[b]  # (C, 1, n_taps)
            filtered = F.conv1d(x_chan, w, bias=None, padding=self.pad, groups=C)
            outs.append(filtered[..., :T])  # trim any off-by-one from even T
        return torch.stack(outs, dim=1)  # (B, n_bands, C, T)


# ----------------------------------------------------------------------------
# Shared-weight encoder + zero-init gate ("safe capacity" wrapper).
# ----------------------------------------------------------------------------
class SafeFiveBandAuxBranch(nn.Module):
    """
    priority-D-1-safe: five fixed EEG bands -> ONE shared-weight temporal
    encoder (applied identically to each band, so added parameter count is
    ~1x an encoder, not 5x) -> band embeddings averaged -> projected into the
    EEG branch's feature space through a sigmoid gate initialized so the
    branch contributes ~0.25% of its output at step 0 (sigmoid(-6)=0.0025,
    the SAME init used for beta_labram and cafe.egta_q's zero-init -- see
    قواعد_مهندسی_پروژه.md rule 5's discussion of this exact pattern).

    SAFE-CAPACITY MEASURES (per the project's own diagnosis of why 10+ prior
    "add capacity" interventions failed -- see conversation history):
      1. Zero-init sigmoid gate: model is NOT forced to use this path;
         must actively learn to rely on it (like LaBraM's beta, EGTA's keys).
      2. Shared encoder across bands: ~5x fewer new parameters than five
         independent per-band encoders.
      3. Small d_model/embed_dim by default (32, not 64) -- half the
         capacity of SpatialMHSABranch's default, since this project's own
         G-2a/B-3 evidence says smaller is directionally safer here.
      4. Meant to be trained with a HEAVIER weight_decay than the rest of
         the model (see run_all_subjects.py patch below) -- this class does
         not enforce that itself (optimizer-level), but exposes
         .new_params() so the training loop can build the asymmetric
         param groups described in the "Safe-Capacity Protocol" discussion.

    Input: x_chan (B, 28, 2000) -- same raw channel-space tensor
    eeg_spatial_mode='mhsa' already consumes.
    Output: (B, embed_dim) -- concatenated onto the EEG branch's pooled
    feature before clf_eeg, exactly like SpatialMHSABranch's embedding,
    but pre-scaled by the gate.
    """

    def __init__(self, n_channels=28, sfreq=200.0, n_taps=101,
                 d_model=32, embed_dim=32, dropout=0.3):
        super().__init__()
        self.filterbank = FixedFIRBandFilter(n_channels=n_channels, sfreq=sfreq, n_taps=n_taps)
        n_bands = self.filterbank.n_bands

        # Shared depthwise temporal encoder -- IDENTICAL weights applied to
        # each of the 5 bands (the "rmzgozar-e moshtarak" design agreed on
        # earlier), unlike SpatialMHSABranch which has no such sharing need
        # (it only ever sees one "band": broadband).
        self.shared_temporal = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=25, stride=10,
                      groups=n_channels, padding=12),
            nn.BatchNorm1d(n_channels),
            nn.ELU(),
            nn.Conv1d(n_channels, n_channels, kernel_size=13, stride=5,
                      groups=n_channels, padding=6),
            nn.BatchNorm1d(n_channels),
            nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(d_model)
        self.compress = nn.Conv1d(n_channels, 4, kernel_size=1)  # small: n_compress=4
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * d_model, embed_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.embed_dim = embed_dim

        # Zero-init gate: sigmoid(-6.0) ~= 0.0025, same convention as
        # beta_labram (labram_eeg_branch_v3) and cafe.egta_q's zero-init key
        # projection. NOT bit-exact zero (honestly documented, same as
        # egta_gru's test 2) -- true bit-exact no-op is guaranteed only by
        # eeg_band_mode='none' never constructing this module at all.
        self.gate = nn.Parameter(torch.full((1,), -6.0))

    def new_params(self):
        """Returns this module's parameters, for building an asymmetric
        (heavier weight_decay) optimizer param group in the training loop."""
        return list(self.parameters())

    def forward(self, x_chan):
        bands = self.filterbank(x_chan)            # (B, 5, C, T), no grad needed through here
        B, n_bands, C, T = bands.shape
        band_embeds = []
        for b in range(n_bands):
            h = self.shared_temporal(bands[:, b])   # (B, C, T') -- SAME weights every band
            h = self.pool(h)                        # (B, C, d_model)
            h = self.compress(h)                    # (B, 4, d_model)
            band_embeds.append(self.project(h))      # (B, embed_dim)
        embed = torch.stack(band_embeds, dim=1).mean(dim=1)  # (B, embed_dim), avg over bands
        gate_value = torch.sigmoid(self.gate)
        return gate_value * embed, gate_value  # scaled embedding, raw gate (for logging)


class ConnectivityBranch(nn.Module):
    """
    priority-E1: small MLP embedding the EEG functional-connectivity (PLV)
    features from eeg_connectivity.py into a fixed-size vector, concatenated
    onto the fusion branch's pooled feature before its classifier (see
    TSFNet.forward's use_fc_features branch). Kept deliberately simple (two
    linear layers) since the connectivity feature vector itself already
    encodes the theta/alpha channel-pair structure -- this branch's only job
    is dimensionality reduction + a nonlinearity, not further architectural
    sophistication (consistent with this project's "one variable per
    experiment" rule: E1 tests WHETHER connectivity features help at all,
    not how elaborate an encoder for them should be).

    Input: (B, n_bands, n_pairs) -- n_bands=2 (theta, alpha), n_pairs =
    n_channels*(n_channels-1)/2 (378 for this project's 28 EEG channels).
    Output: (B, embed_dim).
    """

    def __init__(self, n_channels=28, n_bands=2, embed_dim=64, dropout=0.3):
        super().__init__()
        n_pairs = n_channels * (n_channels - 1) // 2
        in_dim = n_bands * n_pairs
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x_conn):
        # x_conn: (B, n_bands, n_pairs) -> flatten -> (B, n_bands*n_pairs)
        B = x_conn.shape[0]
        flat = x_conn.reshape(B, -1)
        if flat.shape[1] != self.in_dim:
            raise ValueError(f"ConnectivityBranch expected flattened connectivity input "
                              f"of size {self.in_dim} (n_channels/n_bands mismatch with "
                              f"the preprocessed {{subject_id}}_eeg_connectivity.npy this "
                              f"batch came from), got {flat.shape[1]}. Check that "
                              f"fc_num_channels passed to TSFNet matches the actual "
                              f"n_channels used in eeg_connectivity.batch_connectivity_features "
                              f"during preprocessing.")
        return self.net(flat)


class TSFNet(nn.Module):
    """
    num_classes=3 (0-/2-/3-back) -- the paper's original tasks (MI, MA, WG)
    were each binary; the classifier heads and all cross-entropy losses
    below generalize directly to 3-way softmax without any other change to
    the architecture's equations.

    dropout defaults are raised relative to typical CNN practice (0.5 in the
    3D-CNN branches and dense heads, 0.3 in CAFE's attention dropout)
    because this project's cross-session hold-out gives only ~180 training
    windows per subject per fold (27 trials/subject x ~10 sliding windows
    - fewer than the paper's presumed richer per-subject sample count), so
    TSFNet's original ~3.3M-parameter configuration (Table 6) is at high
    risk of overfitting here without stronger regularization.

    fusion_combination: ISOLATED ABLATION PARAMETER (priority-2 sub-experiment).
    'paper_mean' (default, accepted as of this edit): Eq. 4 exactly as
    written -- w1/w2/w3 = independent Sigmoid(raw param) each, final
    y_pred = mean(w1*y_eeg, w2*y_fusion, w3*y_fnirs) = sum(...)/3.
    'wsum' (prior/legacy behavior, kept for A/B comparison only): the three
    sigmoid weights are instead renormalized to sum to 1 before combining --
    this was neither the paper's plain mean nor Softmax, and is now
    superseded by 'paper_mean'.
    """

    def __init__(self, num_classes=3, dropout=0.5, cafe_dropout=0.3,
                 cafe_d_model=64, cafe_n_heads=4, fusion_combination="paper_mean",
                 eeg_temporal_mode="gap", eeg_lstm_hidden=64, eeg_readout_mode="h_n",
                 fnirs_num_segments=21, fnirs_chroma_mode="both",
                 fnirs_num_chromophores=2,
                 use_fc_features=False, fc_num_channels=28, fc_embed_dim=64,
                 eeg_spatial_mode="none", eeg_spatial_channels=28,
                 eeg_spatial_d_model=64, eeg_spatial_heads=4,
                 eeg_spatial_layers=2, eeg_spatial_embed_dim=64,
                 eeg_backbone_mode="native", fnirs_temporal_align="none",
                 eeg_band_mode="none", eeg_band_channels=28,
                 eeg_band_d_model=32, eeg_band_embed_dim=32,
                 norm_mode="batch", norm_groups=8,
                 efgf_gate_mode="clamp", efgf_gate_init=-6.0,
                 conv_dropout=None, conv_dropout_type="channel"):
        super().__init__()
        if efgf_gate_mode not in ("clamp", "sigmoid"):
            raise ValueError(f"efgf_gate_mode must be 'clamp' or 'sigmoid', got {efgf_gate_mode!r}")
        if fusion_combination not in ("paper_mean", "wsum", "gated"):
            raise ValueError(
                f"fusion_combination must be 'paper_mean', 'wsum' or 'gated', "
                f"got {fusion_combination!r}")
        # priority-GN: validated here, applied at the very END of __init__.
        if norm_mode not in ("batch", "group"):
            raise ValueError(f"norm_mode must be 'batch' or 'group', got {norm_mode!r}")
        if norm_mode == "group" and eeg_backbone_mode != "native":
            raise ValueError(
                "norm_mode='group' was only verified on the native TSFNet EEG path; the "
                "LaBraM backbone has its own normalization. Use eeg_backbone_mode='native'.")
        self.norm_mode = norm_mode
        self.fusion_combination = fusion_combination
        self.eeg_temporal_mode = eeg_temporal_mode

        # eeg_backbone_mode: ISOLATED ABLATION PARAMETER (priority-C2
        # sub-experiment). "native" (default) reproduces prior behavior
        # exactly -- self.labram_head is never constructed, forward() never
        # imports labram_eeg_branch_v2, and y_eeg is computed exactly as
        # before via self.clf_eeg(eeg_flat). "labram_frozen" replaces ONLY
        # y_eeg's computation with a frozen, pretrained LaBraM EEG
        # foundation-model backbone (Jiang et al. 2024) + a light trainable
        # linear head, consuming the SAME raw channel-space x_chan
        # (B, 28, 2000) that eeg_spatial_mode='mhsa' already uses -- see
        # labram_eeg_branch_v2.py's module docstring for the full design
        # rationale (why this specific scope, why frozen, why this exact
        # checkpoint/padding choice). Mutually exclusive with
        # eeg_spatial_mode='mhsa' (both would otherwise silently compete to
        # explain the same y_eeg path, breaking the "one variable per
        # experiment" protocol this project follows everywhere else) --
        # validated below.
        if eeg_backbone_mode not in ("native", "labram_frozen", "labram_ensemble",
                                      "labram_blend"):
            raise ValueError(f"eeg_backbone_mode must be 'native', 'labram_frozen', "
                              f"'labram_ensemble' or 'labram_blend', got "
                              f"{eeg_backbone_mode!r}")
        if eeg_backbone_mode != "native" and eeg_spatial_mode == "mhsa":
            raise ValueError(
                f"eeg_backbone_mode={eeg_backbone_mode!r} and eeg_spatial_mode='mhsa' both "
                "modify the EEG decision path and cannot be combined in one run -- this "
                "project's isolated-variable protocol requires testing them separately. Set "
                "eeg_spatial_mode='none' to use a LaBraM backbone mode."
            )
        self.eeg_backbone_mode = eeg_backbone_mode
        if eeg_backbone_mode != "native":
            # PERFORMANCE FIX (see labram_eeg_branch_v3.py's "THIRD
            # PERFORMANCE ISSUE FOUND AND FIXED" note): TSFNet does NOT
            # construct a LaBraMFrozenEEGHeadV3 here (which would build and
            # load the full ~5.8M-parameter frozen backbone on every fold,
            # for a backbone that is NEVER called again after
            # precompute_embeddings has already run). x_chan reaching
            # forward() in these modes is already a PRECOMPUTED
            # (B, LABRAM_EMBED_DIM) embedding, so only the trainable head
            # itself is needed here.
            from labram_eeg_branch_v3 import LABRAM_EMBED_DIM
            self.labram_head = nn.Linear(LABRAM_EMBED_DIM, num_classes)
        if eeg_backbone_mode == "labram_ensemble":
            # C-2c: a FOURTH decision-level gate, exactly parallel to the
            # paper's own w_eeg/w_fusion/w_fnirs (Eq. 4).
            #
            # WARNING (kept available for comparison, but NOT the recommended
            # mode): adding y_labram as a fourth member silently re-weights
            # the MODALITIES. y_labram is an EEG-only classifier sitting
            # beside the EEG-only y_eeg, so the EEG share of the decision goes
            # from 1/3 to 2/4 while fNIRS drops from 1/3 to 1/4 -- and this
            # project's own numbers say that is the wrong direction (fNIRS
            # branch 58.78% vs LaBraM EEG branch 49.10%, a 9.7-point gap).
            # Use 'labram_blend' instead; see its note below.
            self.w_labram = nn.Parameter(torch.zeros(1))
        if eeg_backbone_mode == "labram_blend":
            # C-2d (RECOMMENDED): blend the two EEG-only predictions BEFORE
            # Eq. 4, leaving the three-member structure -- and therefore the
            # modality balance -- exactly as it was:
            #     y_eeg_comb = (1-b)*y_eeg_native + b*y_labram
            # `beta_labram` is a single trainable scalar through a sigmoid;
            # initialized so sigmoid(raw) = 0 exactly, making this a BIT-EXACT
            # no-op at step 0 (the model must actively learn to let LaBraM in).
            # A large negative init is used rather than 0.0 because
            # sigmoid(0)=0.5 would hand LaBraM half the EEG vote for free.
            self.beta_labram = nn.Parameter(torch.full((1,), -6.0))

        # use_fc_features: ISOLATED ABLATION PARAMETER (priority-E1
        # sub-experiment). Default False reproduces prior behavior exactly
        # (self.connectivity_branch is never constructed, clf_fusion's input
        # dim is unchanged at feat_dim=512, forward() never touches x_conn).
        # When True, an EEG functional-connectivity (PLV, theta+alpha)
        # embedding (see ConnectivityBranch) is concatenated onto the fusion
        # branch's pooled feature before clf_fusion -- see forward() below.
        # Motivated by Cao et al. 2022 (Sensors 22(19):7623), the SAME Shin
        # et al. 2018 dataset this project uses: bivariate functional-
        # connectivity features substantially improved 0-back-vs-2/3-back
        # separation over univariate per-channel power there.
        # eeg_spatial_mode: ISOLATED ABLATION PARAMETER (priority-G4).
        # "none" (default) reproduces prior behavior exactly -- the module is
        # never constructed, clf_eeg keeps its original input dim, and
        # forward() never touches x_chan.
        # "mhsa": a SpatialMHSABranch embedding of the channel-space EEG is
        # concatenated onto the EEG branch's pooled feature before clf_eeg.
        #
        # SCOPE, AND WHY IT IS THIS NARROW: exactly like the existing
        # eeg_temporal_mode ablation, this touches ONLY the y_eeg path.
        # eeg_s1/eeg_s2 are returned unchanged, so EFGF, the Fusion branch,
        # CAFE and the fNIRS branch are all bit-identical. Because y_pred is
        # the learned weighted combination of y_eeg/y_fusion/y_fnirs, an
        # improvement to the EEG branch still propagates to the final
        # prediction -- so the hypothesis is testable -- while the blast
        # radius of the change stays small enough to attribute a result to it.
        # The EEG branch is also the weakest of the three (~50% vs ~59% for
        # fNIRS), i.e. the one with the most headroom.
        self.eeg_spatial_mode = eeg_spatial_mode
        if eeg_spatial_mode not in ("none", "mhsa"):
            raise ValueError(f"eeg_spatial_mode must be 'none' or 'mhsa', "
                              f"got {eeg_spatial_mode!r}")
        if eeg_spatial_mode == "mhsa":
            self.eeg_spatial_branch = SpatialMHSABranch(
                n_channels=eeg_spatial_channels, d_model=eeg_spatial_d_model,
                n_heads=eeg_spatial_heads, n_layers=eeg_spatial_layers,
                embed_dim=eeg_spatial_embed_dim, dropout=cafe_dropout)

        # priority-D-1-safe: validation only. The branch itself is constructed
        # at the very END of __init__ (see below) so that, in the default
        # "none" mode, not a single RNG draw is consumed here and the model is
        # bit-identical to the pre-patch reference (rule 5).
        self.eeg_band_mode = eeg_band_mode
        if eeg_band_mode not in ("none", "safe"):
            raise ValueError(f"eeg_band_mode must be 'none' or 'safe', got {eeg_band_mode!r}")
        if eeg_band_mode == "safe" and eeg_backbone_mode != "native":
            raise ValueError(
                f"eeg_band_mode='safe' and eeg_backbone_mode={eeg_backbone_mode!r} both modify "
                "the EEG decision path and cannot be combined -- this project's isolated-"
                "variable protocol requires testing them separately. Set eeg_backbone_mode="
                "'native' to use eeg_band_mode='safe'.")

        self.use_fc_features = use_fc_features
        if use_fc_features:
            self.connectivity_branch = ConnectivityBranch(
                n_channels=fc_num_channels, n_bands=2, embed_dim=fc_embed_dim,
                dropout=cafe_dropout)

        # fnirs_chroma_mode: ISOLATED ABLATION PARAMETER (priority-A4
        # sub-experiment, row A of the revised priority list -- targets the
        # 2-back/3-back bottleneck via the fNIRS chromophore axis, per
        # Meidenbauer et al. [34]: the load-dependent relationship was found
        # ONLY in HbR, not HbO, and 3-back activity was noted as nonlinear/
        # unstable). One of:
        #   "both"     (default): both HbO and HbR channels reach the network
        #               unchanged -- reproduces prior behavior exactly (the
        #               chroma mask below is all-ones, a mathematical no-op).
        #   "hbo_only": the HbR channel (index 1 of the last axis of
        #               x_fnirs) is zeroed before entering FNIRSBranch.
        #   "hbr_only": the HbO channel (index 0) is zeroed instead.
        # IMPORTANT: masking is applied to the RAW x_fnirs input, upstream of
        # FNIRSBranch, so the ablation affects the ENTIRE fNIRS-derived
        # signal path -- fn_s1/fn_s2 (consumed by EFGF1/EFGF2 inside the
        # fusion branch) AND fn_pooled/clf_fnirs (the standalone fNIRS
        # branch classifier) -- not just the fNIRS branch's own auxiliary
        # loss in isolation. This matches how the ablation is meant to test
        # "what does the whole network do with only one chromophore",
        # consistent with a 3-way HbO-only/HbR-only/both split (item A4).
        # No architecture change (conv1 in_channels stays 2 in all three
        # modes) -- this is a pure data-level mask, the same pattern already
        # used elsewhere in this project (e.g. the fNIRS cardiac band-stop
        # filter ablation).
        # Channel index convention (priority-G2a): 0 = HbO, 1 = HbR, 2 = HbT.
        # "both" is kept as the default name for backward compatibility with
        # every earlier run and means "keep every available chromophore".
        _KEEP = {"both": None,            # None -> keep all (all-ones mask)
                 "hbo_only": (0,),
                 "hbr_only": (1,),
                 "hbt_only": (2,),
                 "hbo_hbr": (0, 1),       # drop HbT, keep the original pair
                 "cbsi": None}            # ISOLATED ABLATION (priority-B5): reuses `None`
                                          # (all-ones) only to pass this validation check;
                                          # forward() branches BEFORE the mask is ever
                                          # applied for this mode (see cbsi_transform.py),
                                          # so this mask is never actually used for "cbsi".
        if fnirs_chroma_mode not in _KEEP:
            raise ValueError(f"fnirs_chroma_mode must be one of {sorted(_KEEP)}, "
                              f"got {fnirs_chroma_mode!r}")
        keep = _KEEP[fnirs_chroma_mode]
        if keep is not None and max(keep) >= fnirs_num_chromophores:
            raise ValueError(
                f"fnirs_chroma_mode={fnirs_chroma_mode!r} needs at least "
                f"{max(keep) + 1} chromophore channels, but the model was built with "
                f"fnirs_num_chromophores={fnirs_num_chromophores}. Re-run preprocessing "
                f"with ENABLE_FNIRS_HBT=True (and pass --fnirs-num-chromophores 3) "
                f"if you want HbT.")
        self.fnirs_chroma_mode = fnirs_chroma_mode
        self.fnirs_num_chromophores = fnirs_num_chromophores
        # Registered as a buffer (not a Parameter) so it moves with .to(device)
        # automatically and is never touched by the optimizer. Shape
        # (1,1,1,1,1,C) broadcasts against x_fnirs: (B, Segments, H, W, T, C).
        # For fnirs_num_chromophores=2 this reproduces the original 2-channel
        # masks exactly, so every prior A4 result stays reproducible.
        _chroma_mask = torch.ones(1, 1, 1, 1, 1, fnirs_num_chromophores)
        if keep is not None:
            for c in range(fnirs_num_chromophores):
                if c not in keep:
                    _chroma_mask[..., c] = 0.0
        self.register_buffer("_fnirs_chroma_mask", _chroma_mask)
        # IMPORTANT: eeg_temporal_mode is applied ONLY to self.eeg_branch,
        # NOT to self.fusion_branch below, even though both are built from
        # the same EEGLikeBranch class. This matches the project's
        # architecture-changes documentation verbatim: "this change affects
        # ONLY the EEG branch; fNIRS, Fusion, EFGF, CAFE all remain
        # untouched" -- the fusion branch keeps plain GAP regardless of
        # eeg_temporal_mode.
        # priority-CONVDROP: conv_dropout=None (default) means the conv blocks
        # use `dropout`, exactly as before, so `dropout` keeps its old meaning
        # (conv blocks + branch classifiers) unless conv_dropout is given. When
        # it is given, it applies ONLY to the 3D-conv blocks of the EEG,
        # fusion and fNIRS branches; the branch classifiers below keep
        # `dropout`. Separating the two is the point: Dropout3d(0.5) on 16
        # channels is a very different regularizer from Dropout(0.5) on a
        # 128-unit dense layer, and they should be tunable independently.
        _conv_p = dropout if conv_dropout is None else conv_dropout
        self.eeg_branch = EEGLikeBranch(dropout=_conv_p, temporal_mode=eeg_temporal_mode,
                                         lstm_hidden=eeg_lstm_hidden, readout_mode=eeg_readout_mode,
                                         dropout_type=conv_dropout_type)
        self.fusion_branch = EEGLikeBranch(dropout=_conv_p, temporal_mode="gap",
                                           dropout_type=conv_dropout_type)   # always GAP -- separate weights, same architecture (Sec. 3.1)
        self.fnirs_branch = FNIRSBranch(dropout=_conv_p, n_chromophores=fnirs_num_chromophores,
                                        dropout_type=conv_dropout_type)

        # EFGF stage 1: operates on conv1 outputs (channels=16, EEG kernel matches Table1
        # "EFGF1 Conv1" = 2x2x6, fNIRS kernel matches "EFGF1 Conv2" = 2x2x3)
        self.efgf1 = EFGFLayer(channels=16, eeg_kernel=(2, 2, 6), fnirs_kernel=(2, 2, 3), gate_mode=efgf_gate_mode, gate_init=efgf_gate_init)
        # EFGF stage 2: operates on conv2 outputs (channels=32, both kernels = 2x2x3 per Table1)
        self.efgf2 = EFGFLayer(channels=32, eeg_kernel=(2, 2, 3), fnirs_kernel=(2, 2, 3), gate_mode=efgf_gate_mode, gate_init=efgf_gate_init)
        # NOTE: efgf1/efgf2 consume eeg_s1/eeg_s2 (the pre-pooling conv feature
        # maps from self.eeg_branch), NEVER eeg_pooled -- see TSFNet.forward.
        # This is why eeg_temporal_mode is safe to change in isolation: it
        # only affects what `pooled` (3rd return value of EEGLikeBranch) is,
        # and `pooled` is used ONLY for y_eeg (self.clf_eeg) below, not for
        # EFGF/CAFE/Fusion.

        feat_dim = 4 * 4 * 32  # = 512, shape of F_fusion / F_fnirs-per-segment after GAP
        # fnirs_num_segments: FNIRS_NUM_SEGMENTS in preprocessing. CORRECTED
        # 2026-09-01: this was hardcoded to 21 here (matching the preprocessing
        # value in effect at the time), silently breaking with a RuntimeError
        # on CAFELayer's reshape (line ~373 self.n_segments) the moment the
        # preprocessing's FNIRS_NUM_SEGMENTS changed (e.g. to 18, per the
        # corrected reference-paper-calibrated formula). It's now a real
        # constructor parameter instead, so it can be set to whatever value
        # the currently-loaded preprocessed .npy files actually use. Default
        # 21 reproduces the exact prior behavior if left unspecified.
        self.fnirs_num_segments = fnirs_num_segments
        self.cafe = CAFELayer(fusion_dim=feat_dim, fnirs_dim=feat_dim,
                               d_model=cafe_d_model, n_heads=cafe_n_heads,
                               n_segments=fnirs_num_segments, dropout=cafe_dropout,
                               fnirs_temporal_align=fnirs_temporal_align)
        self.fnirs_temporal_align = fnirs_temporal_align

        # eeg_feat_dim depends on eeg_temporal_mode: GAP flattens a (32,4,4)
        # spatial map -> 512-dim; BiLSTM's bidirectional last-timestep output
        # is 2*eeg_lstm_hidden-dim (128 for the default hidden=64), since the
        # LSTM's hidden_size is an independent hyperparameter, unrelated to
        # the conv feature map's spatial size.
        if eeg_temporal_mode == "bilstm":
            eeg_feat_dim = 2 * eeg_lstm_hidden  # bidirectional -> 2x hidden_size (128 for default hidden=64)
        elif eeg_temporal_mode == "gap_logvar":
            eeg_feat_dim = 2 * 4 * 4 * 32  # priority-LOGVAR: [GAP | log-var] -> 1024
        else:
            eeg_feat_dim = 4 * 4 * 32  # F_eeg_flatten (GAP path, unchanged)
        # priority-G4: clf_eeg's input grows by the spatial embedding only when
        # eeg_spatial_mode == "mhsa"; identical to before otherwise.
        clf_eeg_in_dim = eeg_feat_dim + (eeg_spatial_embed_dim
                                          if eeg_spatial_mode == "mhsa" else 0) \
                                       + (eeg_band_embed_dim
                                          if eeg_band_mode == "safe" else 0)
        if eeg_temporal_mode == "gap_logvar":
            # priority-LOGVAR, rule 6: a 1024-input Linear draws more random
            # numbers than the 512-input one, which would shift the init of
            # EVERY module built after this line (clf_fusion, clf_fnirs, ...)
            # and confound the comparison. Advance the global RNG exactly as
            # the baseline would (by building -- and discarding -- the
            # baseline-sized classifier), then build the real one from the
            # pre-construction state and restore the post-baseline state.
            _rng_before = torch.get_rng_state()
            BranchClassifier(clf_eeg_in_dim - 4 * 4 * 32, num_classes, dropout=dropout)
            _rng_after_baseline = torch.get_rng_state()
            torch.set_rng_state(_rng_before)
            self.clf_eeg = BranchClassifier(clf_eeg_in_dim, num_classes, dropout=dropout)
            torch.set_rng_state(_rng_after_baseline)
        else:
            self.clf_eeg = BranchClassifier(clf_eeg_in_dim, num_classes, dropout=dropout)
        # priority-E1: when use_fc_features=True, clf_fusion's input is
        # [enhanced_fusion ; connectivity_embedding] concatenated (see
        # forward()), so its input dim grows by fc_embed_dim. When False
        # (default), this is IDENTICAL to feat_dim -- the exact prior
        # architecture, same input dim, same number of clf_fusion parameters.
        clf_fusion_in_dim = feat_dim + (fc_embed_dim if use_fc_features else 0)
        self.clf_fusion = BranchClassifier(clf_fusion_in_dim, num_classes, dropout=dropout)
        self.clf_fnirs = BranchClassifier(feat_dim, num_classes, dropout=dropout)

        # Decision-level fusion weights w1 (eeg), w2 (fusion), w3 (fnirs) -- Eq. 4,
        # obtained via sigmoid of trainable raw parameters.
        self.w_eeg = nn.Parameter(torch.zeros(1))
        self.w_fusion = nn.Parameter(torch.zeros(1))
        self.w_fnirs = nn.Parameter(torch.zeros(1))

        # priority-gating: per-sample mixture-of-experts gate, built ONLY when
        # fusion_combination == "gated" so every other mode consumes zero RNG
        # draws here and stays bit-identical to before this change (rule 5/6).
        #
        # Design follows the gating network used for multimodal decision
        # fusion in mixture-of-experts literature (Jacobs et al. 1991;
        # Jordan & Jacobs 1994), concretely mirrored on the MoE gate used for
        # multimodal emotion recognition: concatenate the three branch
        # logits, pass them through one small linear layer, softmax the
        # result into per-sample mixing weights. This is DELIBERATELY NOT
        # the ECA-FusionNet Eq. 5 formula (Qin et al. 2025, J. Neural Eng.
        # 22 016030) -- that paper's w_i are described as learned network
        # weights with no stated dependence on the input, i.e. the same kind
        # of static, sample-independent scalar this project's own w_eeg /
        # w_fusion / w_fnirs already are. Copying Eq. 5 verbatim would have
        # reproduced the existing static combination under a new name, not
        # given the input-dependent weighting the diagnostics evidence
        # (VP001/VP006/VP012/VP017/VP025: best branch differs by subject)
        # calls for.
        if fusion_combination == "gated":
            self.gate_fc = nn.Linear(3 * num_classes, 3)
            # Zero-init (rule 5, "no-op at default" spirit applied to the new
            # branch's own starting point): with weight=0, bias=0, gate_fc's
            # output is exactly zero for every input, so softmax(0,0,0) is
            # uniform (1/3, 1/3, 1/3) regardless of the sample. This makes
            # the GATED model's prediction at initialization equal to the
            # *unweighted* mean of the three branch outputs -- a well-
            # defined, checkable starting point (see
            # test_gating_numerical.py) rather than an arbitrary one.
            nn.init.zeros_(self.gate_fc.weight)
            nn.init.zeros_(self.gate_fc.bias)

        # priority-D-1-safe: constructed LAST, after every pre-existing module,
        # and with NO torch.get/set_rng_state manipulation anywhere.
        #   * "none" (default): nothing below runs -> zero RNG draws consumed
        #     -> the model is BIT-IDENTICAL to the pre-patch reference for a
        #     given seed (rule 5), verified by test_d1_safe_vs_original.py.
        #   * "safe": clf_eeg's input dim grows (512 -> 544) so clf_eeg AND the
        #     modules built after it (clf_fusion, clf_fnirs) receive different
        #     initial values than the reference. This is an ACKNOWLEDGED
        #     LIMITATION, not a bug -- exactly the situation rule 6 describes
        #     for C-2d, where the prescribed remedy is to "state the limitation
        #     explicitly in the conclusion". Any accuracy difference in "safe"
        #     mode therefore reflects (new branch + a different draw of the
        #     three classifier heads), not the new branch in isolation. An
        #     earlier attempt to neutralize this with get/set_rng_state
        #     REWOUND the shared stream and silently broke the "none" default
        #     too, invalidating three full 26-subject runs; hence: no RNG
        #     manipulation here, ever.
        if eeg_band_mode == "safe":
            self.eeg_band_branch = SafeFiveBandAuxBranch(
                n_channels=eeg_band_channels, sfreq=200.0,
                d_model=eeg_band_d_model, embed_dim=eeg_band_embed_dim,
                dropout=cafe_dropout)

        # priority-GN: swap BatchNorm -> GroupNorm LAST, after every module
        # exists. GroupNorm's init (weight=1, bias=0) draws no random numbers,
        # so this cannot move any other parameter (rule 6); "batch" (default)
        # skips this entirely and is bit-identical to the pre-patch model
        # (rule 5), verified by test_groupnorm.py against the ORIGINAL file.
        if norm_mode == "group":
            convert_batchnorm_to_groupnorm(self, num_groups=norm_groups)

    def forward(self, x_eeg, x_fnirs, x_conn=None, x_chan=None):
        """
        x_eeg:   (B, 16, 16, 600)
        x_fnirs: (B, 11, 16, 16, 30, 2)
        x_conn:  (B, 2, n_pairs) or None -- priority-E1 EEG functional-
                 connectivity features (see ConnectivityBranch). Required
                 (non-None) if self.use_fc_features is True; ignored
                 entirely (may be left None) otherwise -- passing a non-None
                 x_conn when use_fc_features=False is harmless (silently
                 unused), but the reverse (use_fc_features=True with
                 x_conn=None) raises immediately rather than silently
                 training on garbage/zeros.
        """
        eeg_s1, eeg_s2, eeg_pooled = self.eeg_branch(x_eeg)
        # priority-A4 chroma ablation: mask is all-ones (mathematical no-op)
        # when fnirs_chroma_mode == "both", so this line changes nothing for
        # the default/accepted configuration.
        # priority-G2a: validate BEFORE the mask multiply below. Without this,
        # a data/model chromophore mismatch surfaces as an opaque broadcasting
        # RuntimeError from this very line, which is much harder to diagnose
        # than the actionable message FNIRSBranch would have raised a moment
        # later -- so the check is duplicated here rather than relied upon
        # downstream.
        if x_fnirs.shape[-1] != self.fnirs_num_chromophores:
            raise ValueError(
                f"TSFNet was built for fnirs_num_chromophores="
                f"{self.fnirs_num_chromophores} but received x_fnirs with "
                f"{x_fnirs.shape[-1]} chromophore channels. This almost always means the "
                f"preprocessed fnirs_windows.npy was produced with a different "
                f"ENABLE_FNIRS_HBT setting than the --fnirs-num-chromophores the model was "
                f"built with (2 = HbO/HbR, 3 = HbO/HbR/HbT).")
        if self.fnirs_chroma_mode == "cbsi":
            # ISOLATED ABLATION (priority-B5): data-DERIVED replacement of the
            # raw HbO/HbR channels (see cbsi_transform.py for the full
            # derivation/reference). Bypasses the static mask entirely --
            # channel 0 becomes the CBSI-corrected signal, channel 1 (and any
            # further chromophore channel) is zeroed, so the tensor shape and
            # FNIRSBranch's conv1 in_channels are unaffected.
            from cbsi_transform import apply_cbsi_transform
            x_fnirs = apply_cbsi_transform(x_fnirs)
        else:
            x_fnirs = x_fnirs * self._fnirs_chroma_mask
        fn_s1, fn_s2, fn_pooled = self.fnirs_branch(x_fnirs)

        # Fusion branch: manually driven through conv1 -> EFGF1 -> conv2 -> EFGF2, so each
        # EFGF stage's output becomes the INPUT to the fusion branch's next conv layer
        # (staged, deep fusion -- Sec. 3.1: "Aside from the feature-level fusion performed
        # through the EFGF layer after each convolutional layer, the rest of the processing
        # pipeline remains consistent with the EEG branch").
        x_eeg_5d = x_eeg.unsqueeze(1)
        fus_s1 = self.fusion_branch.conv1(x_eeg_5d)                     # (B, 16, 8, 8, 100)
        fused1, eeg_pcc1, phi_pcc1 = self.efgf1(eeg_s1, fn_s1, fus_s1)
        fus_s2 = self.fusion_branch.conv2(fused1)                       # (B, 32, 4, 4, 50)
        fused2, eeg_pcc2, phi_pcc2 = self.efgf2(eeg_s2, fn_s2, fus_s2)
        fusion_pooled_final = fused2.mean(dim=-1)                       # GAP over time -> (B, 32, 4, 4)

        enhanced_fusion, enhanced_fnirs, cafe_hsic_loss = self.cafe(fusion_pooled_final, fn_pooled)

        # priority-C2 / C-2c. Two distinct LaBraM modes, deliberately kept
        # separate because they are NOT the same experiment:
        #
        #   'labram_frozen'   (C-2b): REPLACES y_eeg. This severs the
        #       auxiliary lambda_eeg*CE(y_eeg) gradient path into the shared
        #       eeg_branch trunk -- measured on the real model: the trunk
        #       receives |grad|=25.94 from that term in 'native' and exactly
        #       0.0000 in 'labram_frozen' (42% less total trunk gradient).
        #       Since eeg_s1/eeg_s2 come from that SAME trunk (line ~980) and
        #       feed EFGF, replacing y_eeg silently removes deep supervision
        #       from the features fusion itself consumes. That makes C-2b a
        #       TWO-variable change, which is why its result (59.42%/0.3909)
        #       is not interpretable as a test of the foundation model alone.
        #
        #   'labram_ensemble' (C-2c): ADDS y_labram as a fourth decision-level
        #       member alongside the UNCHANGED native y_eeg. Every existing
        #       gradient path -- including the auxiliary EEG supervision on the
        #       trunk -- is preserved bit-for-bit, so this asks the intended
        #       question ("does the pretrained representation add anything to
        #       this ensemble?") as a genuinely isolated change.
        eeg_flat = eeg_pooled.reshape(eeg_pooled.shape[0], -1)     # (B, 512)
        if self.eeg_spatial_mode == "mhsa":
            if x_chan is None:
                raise ValueError(
                    "TSFNet was constructed with eeg_spatial_mode='mhsa' but forward() "
                    "received x_chan=None. The subject_dir being used has no "
                    "{subject_id}_eeg_channel_windows.npy -- re-run preprocessing with "
                    "ENABLE_EEG_CHANNEL_SPACE=True (see Preproc_tsfnet_nback_badch.py).")
            eeg_flat = torch.cat([eeg_flat, self.eeg_spatial_branch(x_chan)], dim=-1)
        band_gate_value = None
        if self.eeg_band_mode == "safe":
            if x_chan is None:
                raise ValueError(
                    "TSFNet was constructed with eeg_band_mode='safe' but forward() "
                    "received x_chan=None -- the subject_dir has no channel-space EEG. "
                    "Re-run preprocessing with ENABLE_EEG_CHANNEL_SPACE=True.")
            band_embed, band_gate_value = self.eeg_band_branch(x_chan)
            eeg_flat = torch.cat([eeg_flat, band_embed], dim=-1)
        y_eeg_native = self.clf_eeg(eeg_flat)

        y_labram = None
        if self.eeg_backbone_mode != "native":
            if x_chan is None:
                raise ValueError(
                    f"TSFNet was constructed with eeg_backbone_mode="
                    f"{self.eeg_backbone_mode!r} but forward() received x_chan=None. The "
                    "subject_dir being used has no {subject_id}_eeg_channel_windows.npy "
                    "-- re-run preprocessing with ENABLE_EEG_CHANNEL_SPACE=True "
                    "(see Preproc_tsfnet_nback_badch.py).")
            # x_chan here is the PRECOMPUTED (B, LABRAM_EMBED_DIM) frozen-backbone
            # embedding (see run_subject_cross_session's precompute step); the
            # backbone itself is never touched inside this forward pass.
            y_labram = F.softmax(self.labram_head(x_chan), dim=-1)

        if self.eeg_backbone_mode == "labram_frozen":
            y_eeg = y_labram          # C-2b: replacement (confounded -- see above)
        elif self.eeg_backbone_mode == "labram_blend":
            # C-2d: blend the two EEG-only predictions BEFORE Eq. 4, so the
            # ensemble stays three-member and the modality balance is
            # untouched. beta is sigmoid(raw) with raw initialized to -6.0, so
            # this starts as a bit-exact no-op (beta ~ 0.0025) and the model
            # must actively learn to give LaBraM weight.
            beta = torch.sigmoid(self.beta_labram)
            y_eeg = (1.0 - beta) * y_eeg_native + beta * y_labram
        else:
            y_eeg = y_eeg_native      # 'native' and 'labram_ensemble' keep this

        if self.use_fc_features:
            if x_conn is None:
                raise ValueError("TSFNet was constructed with use_fc_features=True but "
                                  "forward() received x_conn=None. This means the "
                                  "subject_dir/NBackDataset being used doesn't have a "
                                  "{subject_id}_eeg_connectivity.npy file (or _unpack_batch "
                                  "returned None because the loader's dataset had no "
                                  "connectivity loaded) -- run preprocessing with "
                                  "ENABLE_EEG_CONNECTIVITY=True first (see "
                                  "Preproc_tsfnet_nback_badch.py).")
            conn_embed = self.connectivity_branch(x_conn)                # (B, fc_embed_dim)
            fusion_clf_input = torch.cat([enhanced_fusion, conn_embed], dim=-1)
        else:
            fusion_clf_input = enhanced_fusion
        y_fusion = self.clf_fusion(fusion_clf_input)
        y_fnirs = self.clf_fnirs(enhanced_fnirs)

        w1 = torch.sigmoid(self.w_eeg)
        w2 = torch.sigmoid(self.w_fusion)
        w3 = torch.sigmoid(self.w_fnirs)
        if self.fusion_combination == "paper_mean":
            # ISOLATED ABLATION CHANGE (priority-2 step, combination-formula
            # sub-experiment). Paper Eq. 4: each of w1/w2/w3 passes through an
            # INDEPENDENT sigmoid (so they do NOT sum to 1, unlike softmax),
            # and the final prediction is the plain MEAN of the three
            # sigmoid-weighted branch outputs -- i.e. divide by the constant
            # 3, not by (w1+w2+w3). The previous "wsum" behavior instead
            # renormalized by the sum of the three sigmoid weights, which is
            # neither the paper's plain mean nor the paper's original Softmax
            # (Eq. 4 as literally written uses Sigmoid + Mean, no renorm).
            # This is the ONLY change in this edit relative to the accepted
            # priority-1 baseline (no gradient clipping): epochs, patience,
            # lr, weight_decay, loss weighting, and CAFE dim are all untouched.
            y_pred = (w1 * y_eeg + w2 * y_fusion + w3 * y_fnirs) / 3.0
            if self.eeg_backbone_mode == "labram_ensemble":
                # C-2c: Eq. 4 extended to a fourth member. The paper's own
                # formula divides by the NUMBER OF MEMBERS (a constant 3), not
                # by the sum of the sigmoid gates -- so the faithful extension
                # to four members divides by 4, keeping each member's maximum
                # contribution identical to what it is with three. w4 uses the
                # same zero init and the same independent sigmoid as w1/w2/w3,
                # so at initialization every member is weighted 0.5, exactly as
                # in the three-member case.
                w4 = torch.sigmoid(self.w_labram)
                y_pred = (w1 * y_eeg + w2 * y_fusion
                          + w3 * y_fnirs + w4 * y_labram) / 4.0
        elif self.fusion_combination == "wsum":
            # Prior (pre-ablation) behavior, kept available for direct
            # with/without comparison runs.
            w_sum = w1 + w2 + w3 + 1e-8
            y_pred = (w1 * y_eeg + w2 * y_fusion + w3 * y_fnirs) / w_sum
        elif self.fusion_combination == "gated":
            # priority-gating: per-sample weights REPLACE the three global
            # scalars w1/w2/w3 computed above -- w_eeg/w_fusion/w_fnirs play
            # no role in this mode's prediction.
            #
            # They are still nn.Parameters (so a "gated" state_dict keeps the
            # same keys as the other modes), which creates one trap: left
            # untouched they would end training with .grad == None, i.e.
            # ORPHANED, not "detached". Two concrete consequences, both
            # observed rather than assumed:
            #   (a) an optimizer still holds them in param_groups, so with
            #       weight_decay ON and no gradient ever opposing it, decay
            #       alone would walk them away from their zero init -- the
            #       exact failure mode already catalogued as this project's
            #       bug #3 (weight_decay opening sigmoid gates with no data
            #       signal). Harmless while "gated" is active, but it
            #       silently poisons the saved scalars for anyone who later
            #       loads this checkpoint in paper_mean/wsum mode.
            #   (b) grad == None (rather than a zero tensor) is what trips
            #       DDP's find_unused_parameters=False and some optimizer
            #       state-dict round-trips.
            # Adding a 0.0-coefficient term puts them in the graph, so they
            # receive an explicit ZERO gradient instead of None: the forward
            # value is bit-identical (adding exactly 0.0), the scalars stay
            # pinned at init, and both problems above disappear.
            gate_input = torch.cat([y_eeg, y_fusion, y_fnirs], dim=-1)   # (B, 3*num_classes)
            gate_logits = self.gate_fc(gate_input)                       # (B, 3)
            gate_weights = torch.softmax(gate_logits, dim=-1)            # (B, 3), sums to 1 per sample
            y_pred = (gate_weights[:, 0:1] * y_eeg
                      + gate_weights[:, 1:2] * y_fusion
                      + gate_weights[:, 2:3] * y_fnirs)
            y_pred = y_pred + 0.0 * (w1 + w2 + w3)
        else:
            raise ValueError(f"Unknown fusion_combination: {self.fusion_combination!r} "
                              f"(expected 'paper_mean', 'wsum' or 'gated')")

        aux = {
            "y_labram": y_labram,          # None unless a LaBraM mode is active
            "gate_weights": (gate_weights.detach()
                              if self.fusion_combination == "gated" else None),
            # (B,3) per-sample [w_eeg, w_fusion, w_fnirs], sums to 1 per row.
            # None in "paper_mean"/"wsum" modes -- new key, does not remove
            # or rename anything existing (rule 7).
            "egta_weights": self.cafe._last_egta_weights,  # (B,h,S) or None -- XAI only
            "beta_labram": (torch.sigmoid(self.beta_labram).detach()
                            if hasattr(self, "beta_labram") else None),
            "y_eeg_native": y_eeg_native,  # always the clf_eeg output, even in labram_frozen
            "eeg_pcc1": eeg_pcc1, "phi_pcc1": phi_pcc1,
            "eeg_pcc2": eeg_pcc2, "phi_pcc2": phi_pcc2,
            "enhanced_fusion": enhanced_fusion, "enhanced_fnirs": enhanced_fnirs,
            "cafe_hsic_loss": cafe_hsic_loss,   # priority-B1, see CAFELayer.forward
            "eeg_band_gate": (band_gate_value.detach()
                              if band_gate_value is not None else None),  # priority-D-1-safe
        }
        return y_eeg, y_fusion, y_fnirs, y_pred, aux


# %% CELL 4 -- losses.py (pearson_corrcoef, efgf/cafe regularization, tsfnet_loss)
# ============================================================================
# TSFNet loss functions (Sec. 3.5): classification loss (per-branch +
# final decision, weighted) plus PCC-based regularization for the EFGF and
# CAFE layers.
#
# ADAPTATION: the paper's cross-entropy formulas (Eqs. 7-10) are written
# generically ("yi log(y(...))") and generalize directly from the paper's
# binary tasks to our 3-class (0-/2-/3-back) softmax outputs with no change
# to the math -- we use standard multi-class cross-entropy with integer
# class labels (equivalent to the paper's one-hot formulation).
# ============================================================================
import torch
import torch.nn.functional as F


def pearson_corrcoef(u, v, eps=1e-8):
    """Batched Pearson correlation coefficient (Eq. 11), computed per-sample
    over all remaining (non-batch) dimensions, then averaged over the batch.
    u, v: (B, ...) same shape."""
    B = u.shape[0]
    u = u.reshape(B, -1)
    v = v.reshape(B, -1)
    u = u - u.mean(dim=1, keepdim=True)
    v = v - v.mean(dim=1, keepdim=True)
    num = (u * v).sum(dim=1)
    den = torch.sqrt((u ** 2).sum(dim=1) * (v ** 2).sum(dim=1) + eps)
    return num / (den + eps)   # (B,)


def efgf_regularization_loss(eeg_pcc1, phi_pcc1, eeg_pcc2, phi_pcc2):
    """Eq. 12: L_efgf = 1 - mean over samples and the 2 EFGF stages of PCC(F_eeg, Phi)."""
    pcc1 = pearson_corrcoef(eeg_pcc1, phi_pcc1)
    pcc2 = pearson_corrcoef(eeg_pcc2, phi_pcc2)
    mean_pcc = (pcc1.mean() + pcc2.mean()) / 2.0
    return 1.0 - mean_pcc


def cafe_regularization_loss(enhanced_fnirs, enhanced_fusion):
    """Eq. 13: L_cafe = 1 - mean PCC(Fe_fnirs, Fe_fusion)."""
    pcc = pearson_corrcoef(enhanced_fnirs, enhanced_fusion)
    return 1.0 - pcc.mean()


def focal_nll_loss(probs, labels, gamma, alpha=None, eps=1e-8):
    """
    Multi-class focal loss (Lin et al., 2017, ICCV, Eq. 5 generalized to
    K>2 classes), applied to already-softmaxed probabilities (mirroring how
    the rest of this file's nll_loss calls operate on `torch.log(probs+eps)`
    rather than raw logits + cross_entropy).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    probs: (B, num_classes) softmax probabilities.
    labels: (B,) integer class labels.
    gamma: focusing parameter (Lin et al. use gamma=2 as their main setting;
    gamma=0 recovers plain (optionally alpha-weighted) NLL/cross-entropy).
    alpha: optional (num_classes,) per-class weight tensor, gathered per
    sample by its true label -- same role as nll_loss's `weight` argument.
    None means alpha_t=1 for every sample (uniform), which is the
    configuration actually tested in this project's priority-3c experiment:
    isolating the effect of the (1-p_t)^gamma focusing term alone, without
    also reintroducing a per-class weight (that axis was already tested
    separately and independently as priority-3a/3b).
    Reduction is 'mean' over the batch, matching F.nll_loss's default and
    every other loss term in tsfnet_loss.
    """
    p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=eps, max=1.0 - eps)
    log_p_t = torch.log(p_t)
    focal_term = (1.0 - p_t) ** gamma
    if alpha is not None:
        alpha_t = alpha[labels]
        loss = -alpha_t * focal_term * log_p_t
    else:
        loss = -focal_term * log_p_t
    return loss.mean()


def build_asymmetric_cost_matrix(num_classes=3, hard_pairs=((1, 2),), hard_weight=2.0,
                                  device=None):
    """
    Build the (num_classes, num_classes) cost matrix C used by the asymmetric
    loss term (priority-A1). C[t, p] is the multiplier applied to a sample whose
    TRUE class is t and whose PREDICTED distribution mass sits on class p.
    C[t, t] = 1 always (correct predictions are never up-weighted).

    hard_pairs: iterable of (i, j) class-index pairs to penalize extra, applied
    SYMMETRICALLY (both i->j and j->i). Default ((1, 2),) = the 2-back / 3-back
    pair, which the confusion-matrix audit of this project identified as the
    dominant failure mode: 3649 of 6124 total errors (59.6%) are 2-back<->3-back
    confusions, while 0-back recall is comparatively healthy at 68.1%. Cao et al.
    independently report ~59% accuracy on this same pair, so this is a property
    of the task, not of this implementation.

    WHY A MATRIX AND NOT THE PAPER'S VECTOR
    ---------------------------------------
    The reference implementation (yiamcb/TAFN, CustomLoss.py) applies a per-CLASS
    vector:
        asymmetric_losses = tf.where(tf.equal(y_true, 1),
                                     asymmetric_weights * weighted_losses,
                                     weighted_losses)
    i.e. the multiplier depends only on the TRUE class, not on WHICH class the
    model confused it with. That is fine for their binary tasks (with a=[1,2],
    "penalize errors on the positive class more"), but it cannot express the
    thing this project actually needs: penalize 2->3 and 3->2 specifically,
    while leaving 2->0 at normal cost. The vector form is recoverable as the
    special case where every off-diagonal entry in row t shares one value, so
    this matrix is a strict generalization -- see asymmetric_mode="paper_vector"
    in tsfnet_loss() for the faithful reproduction.
    """
    C = torch.ones(num_classes, num_classes, dtype=torch.float32)
    for (i, j) in hard_pairs:
        C[i, j] = hard_weight
        C[j, i] = hard_weight
    C.fill_diagonal_(1.0)   # a correct prediction is never up-weighted
    if device is not None:
        C = C.to(device)
    return C


def ordinal_smoothness_loss(probs):
    """
    Regularization term from the TAFN reference implementation:
        regularization_loss = mean(|y_pred[:, 1:] - y_pred[:, :-1]|)

    NOTE ON A PAPER/CODE DISCREPANCY (this matters, so it is documented rather
    than silently resolved): Eq. 13 of the paper writes this as
        L_reg = (1/N) * sum_{i=1}^{N-1} |y_pred^(i+1) - y_pred^(i)|
    with i indexing SAMPLES, which reads like a temporal/batch smoothness term.
    The released code, however, slices axis 1 of a (batch, num_classes) tensor,
    i.e. it differences adjacent CLASS scores within each sample, not adjacent
    samples. This implementation follows the CODE, for two reasons:
      1. The code is what actually produced the paper's reported numbers.
      2. The class-axis version is the one that is meaningful here. This
         project's three classes are ORDINAL in cognitive load
         (0-back < 2-back < 3-back), so penalizing jagged score profiles across
         adjacent load levels encodes a genuine ordinal prior. A sample-axis
         term would instead couple arbitrary, shuffled batch neighbours, which
         carries no meaning under this project's shuffled DataLoader.
    """
    return (probs[:, 1:] - probs[:, :-1]).abs().mean()


def asymmetric_nll_loss(probs, labels, cost_matrix, class_weight=None, eps=1e-8):
    """
    Per-sample NLL scaled by an expected asymmetric cost.

    For each sample with true class t, the standard term is -log(p_t). We scale
    it by the cost the model's CURRENT distribution incurs:
        scale_i = sum_p C[t_i, p] * probs[i, p]
    Because C[t,t] = 1 and C[t,p] >= 1, scale_i is 1 when the model is perfectly
    confident and correct, and grows toward hard_weight as probability mass
    drifts onto a "hard-pair" class. This makes the penalty differentiable and
    proportional to HOW MUCH mass sits on the costly class, instead of a hard
    argmax-based switch (which would give zero gradient almost everywhere).

    class_weight, if given, is the per-class vector -- e.g. from
    class_weighting_scheme="inverse_freq" (effective_number was tested at
    beta=0.9 and beta=0.999 and REJECTED at both: beta=0.9's window-level
    class_counts underflow to identical weights=[1,1,1] in every single
    fold -- a mathematical no-op verified by grepping the full run log --
    and beta=0.999 was tested across 5 independent seeds, all 5 showing a
    consistent drop in both accuracy and kappa vs. the unweighted baseline)
    -- and is applied multiplicatively,
    exactly as F.nll_loss(weight=...) would -- including nll_loss's convention
    of normalizing by the sum of the weights of the samples in the batch rather
    than by the batch size, so that switching this term on/off does not silently
    rescale the loss relative to the other three branch terms.
    """
    p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=eps, max=1.0 - eps)
    nll = -torch.log(p_t)                                   # (B,)
    cost_rows = cost_matrix[labels]                          # (B, C)
    scale = (cost_rows * probs).sum(dim=1)                   # (B,)
    per_sample = scale * nll
    if class_weight is not None:
        w = class_weight[labels]
        return (w * per_sample).sum() / w.sum().clamp(min=eps)
    return per_sample.mean()


def tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, labels, aux,
                 lambda_eeg=0.2, lambda_fnirs=0.2, class_weight=None, focal_gamma=None,
                 asymmetric_cost=None, asymmetric_mode="cost_matrix",
                 ordinal_reg_lambda=0.0, mhs_lambda=0.0, label_smoothing=0.0,
                 normalize_ypred=False, efgf_lambda=1.0, cafe_lambda=1.0):
    """
    Full loss L = L_class + L_efgf + L_cafe (Eq. 5-6).
    y_*: softmax probabilities (B, num_classes) from the model's forward().
    labels: (B,) integer class labels in {0,1,2}.

    NOTE: y_* are already softmax probabilities (per Sec. 3.4's "followed by
    a softmax function"), so we use nll_loss on log(y) rather than
    cross_entropy (which expects raw logits) -- mathematically identical to
    the paper's Eqs. 7-10.

    class_weight: priority-3a/3b sub-experiments (per-CLASS fixed weight).
    Optional (num_classes,) tensor of per-class weights, passed straight
    through to F.nll_loss's `weight` argument (or used as focal_gamma's
    alpha, see below) for ALL FOUR loss terms (l_eeg, l_fnirs, l_fusion,
    l_pred) -- not just the final prediction -- since class imbalance
    (majority 0-back class) affects every branch's classifier equally, not
    only the fused output. This is NOT part of the paper's original Eq. 6
    (the paper's WG/MA/MI tasks were balanced binary tasks and had no such
    term); it is an addition specific to this project's imbalanced 3-class
    n-back setup. Default None reproduces the exact prior (unweighted)
    behavior.

    focal_gamma: ISOLATED ABLATION CHANGE (priority-3c sub-experiment, Lin
    et al. 2019 -- wait, 2017, ICCV, "Focal Loss for Dense Object
    Detection"). If not None, ALL FOUR loss terms use focal_nll_loss (with
    class_weight, if given, used as focal's per-class alpha) instead of
    plain F.nll_loss. This is a DIFFERENT axis from class_weight: it
    reweights each SAMPLE dynamically by how confidently it is already
    classified ((1-p_t)^gamma), rather than by a fixed per-class constant.
    Default None reproduces the exact prior (non-focal) behavior for every
    other class_weighting_scheme ("none", "inverse_freq", "effective_number").
    """
    eps = 1e-8
    # priority-ypred-normfix: y_pred = (w1*y_eeg + w2*y_fusion + w3*y_fnirs)/3
    # with w1,w2,w3 independent sigmoid gates (Eq. 4) is NOT guaranteed to sum
    # to 1 -- only (w1+w2+w3)/3 does. F.nll_loss(log(y_pred)) on an
    # un-normalized "probability" means d(loss)/d(w_i) is negative whenever
    # branch i's prediction for the true class is positive (almost always,
    # once training has progressed at all), regardless of that branch's
    # RELATIVE quality vs the other two -- a systematic, undifferentiated
    # pressure pushing every gate up together. Confirmed on real logged gate
    # values (26 subjects, seed 42): w_eeg/w_fusion/w_fnirs sit within ~0.001
    # of each other for every subject, consistent with this mechanism, though
    # they do NOT saturate to 1 as a naive reading of the pressure would
    # suggest (mean ~0.65) -- other regularization/early-stopping evidently
    # limits it in practice. y_eeg, y_fusion, y_fnirs are each independently
    # already-normalized per-branch softmax outputs and are NOT touched here.
    # Default False reproduces the exact prior (un-normalized) behavior (rule 5).
    if normalize_ypred:
        y_pred = y_pred / (y_pred.sum(dim=-1, keepdim=True) + eps)
    if asymmetric_cost is not None:
        # ---- priority-A1: TAFN-style asymmetric term -------------------------
        # Applied to ALL FOUR branch terms, consistent with how class_weight and
        # focal_gamma are already handled: the 2-back/3-back confusion is a
        # property of the task, so every branch classifier faces it, not just
        # the fused output.
        # Mutually exclusive with focal_gamma by design -- stacking a per-sample
        # difficulty reweighting (focal) on top of a per-confusion-pair cost
        # would make the resulting run uninterpretable as a single-variable
        # ablation, and focal was already tested and set aside as inconsistent.
        if focal_gamma is not None:
            raise ValueError("asymmetric_cost and focal_gamma are mutually exclusive "
                              "(stacking two sample-reweighting schemes would break the "
                              "one-variable-per-ablation rule this project follows).")
        l_eeg = asymmetric_nll_loss(y_eeg, labels, asymmetric_cost, class_weight, eps)
        l_fnirs = asymmetric_nll_loss(y_fnirs, labels, asymmetric_cost, class_weight, eps)
        l_fusion = asymmetric_nll_loss(y_fusion, labels, asymmetric_cost, class_weight, eps)
        l_pred = asymmetric_nll_loss(y_pred, labels, asymmetric_cost, class_weight, eps)
    elif focal_gamma is not None:
        l_eeg = focal_nll_loss(y_eeg, labels, focal_gamma, alpha=class_weight)
        l_fnirs = focal_nll_loss(y_fnirs, labels, focal_gamma, alpha=class_weight)
        l_fusion = focal_nll_loss(y_fusion, labels, focal_gamma, alpha=class_weight)
        l_pred = focal_nll_loss(y_pred, labels, focal_gamma, alpha=class_weight)
    elif label_smoothing > 0.0:
        # ISOLATED ABLATION (priority-B4): label_smoothing=0.0 (default)
        # reproduces the exact prior (unsmoothed) behavior below.
        from label_smoothing_loss import label_smoothing_loss
        num_classes = y_pred.shape[1]
        l_eeg = label_smoothing_loss(y_eeg, labels, label_smoothing, num_classes, eps, weight=class_weight)
        l_fnirs = label_smoothing_loss(y_fnirs, labels, label_smoothing, num_classes, eps, weight=class_weight)
        l_fusion = label_smoothing_loss(y_fusion, labels, label_smoothing, num_classes, eps, weight=class_weight)
        l_pred = label_smoothing_loss(y_pred, labels, label_smoothing, num_classes, eps, weight=class_weight)
    else:
        l_eeg = F.nll_loss(torch.log(y_eeg + eps), labels, weight=class_weight)
        l_fnirs = F.nll_loss(torch.log(y_fnirs + eps), labels, weight=class_weight)
        l_fusion = F.nll_loss(torch.log(y_fusion + eps), labels, weight=class_weight)
        l_pred = F.nll_loss(torch.log(y_pred + eps), labels, weight=class_weight)

    l_class = l_pred + lambda_eeg * l_eeg + lambda_fnirs * l_fnirs + l_fusion  # Eq. 6

    l_efgf = efgf_regularization_loss(aux["eeg_pcc1"], aux["phi_pcc1"],
                                       aux["eeg_pcc2"], aux["phi_pcc2"])
    l_cafe = cafe_regularization_loss(aux["enhanced_fnirs"], aux["enhanced_fusion"])

    # priority-PCCLAMBDA: Eq. 5 adds L_efgf and L_cafe to L_class with NO
    # weight. Each is 1 - PCC in [0, 2], so together they can contribute up to
    # 4.0 -- measured on a freshly initialized run60 model they are 2.02 of a
    # 4.70 total, i.e. 43% of the loss before training has even begun, on par
    # with the entire classification objective. On top of that, L_cafe drives
    # PCC(enhanced_fnirs, enhanced_fusion) toward 1, i.e. it actively pushes
    # the two modality representations toward REDUNDANCY, which works against
    # a fusion design whose value is complementarity. efgf_lambda/cafe_lambda
    # scale each term independently so that can be measured.
    # 1.0 (default) leaves both terms untouched -- bit-exact prior behavior
    # (rule 5): the multiply is skipped entirely, not merely multiplied by 1.
    # 0.0 replaces the term with an exact zero (not 0 * term), so a NaN in a
    # disabled regularizer can never leak into the total via 0 * NaN = NaN.
    # The breakdown below keeps the UNWEIGHTED values under the existing keys
    # "l_efgf"/"l_cafe" (rule 7: an existing key never changes meaning) and
    # reports the weighted contributions under new keys.
    def _weighted(term, lam):
        if lam == 1.0:
            return term
        if lam == 0.0:
            return torch.zeros((), device=term.device, dtype=term.dtype)
        return lam * term
    l_efgf_w = _weighted(l_efgf, efgf_lambda)
    l_cafe_w = _weighted(l_cafe, cafe_lambda)

    # priority-A1: TAFN Eq. 14 regularization term (see ordinal_smoothness_loss
    # for the paper-vs-code axis discrepancy). Reference uses lambda = 1e-4;
    # 0.0 here disables it so it can be ablated separately from the
    # asymmetric term.
    if ordinal_reg_lambda > 0.0:
        l_ord = ordinal_reg_lambda * ordinal_smoothness_loss(y_pred)
    else:
        l_ord = torch.zeros((), device=y_pred.device)

    # priority-B1: MSTFDN-inspired HSIC head-diversity term on CAFE's two
    # cross-attention modules (see CAFELayer.forward / cafe_multihead_hsic_loss).
    # mhs_lambda=0.0 (default) makes this an EXACT no-op: aux["cafe_hsic_loss"]
    # is always computed (cheap: at most C(4,2)=6 pairs per attention module),
    # but multiplying by 0.0 and adding contributes exactly 0.0 to `total`
    # and exactly 0.0 gradient, so every previously-run/accepted result is
    # unaffected by this addition.
    if mhs_lambda > 0.0:
        l_mhs = mhs_lambda * aux["cafe_hsic_loss"]
    else:
        l_mhs = torch.zeros((), device=y_pred.device)

    total = l_class + l_efgf_w + l_cafe_w + l_ord + l_mhs
    breakdown = {
        "total": total.item(), "l_class": l_class.item(), "l_pred": l_pred.item(),
        "l_eeg": l_eeg.item(), "l_fnirs": l_fnirs.item(), "l_fusion": l_fusion.item(),
        "l_efgf": l_efgf.item(), "l_cafe": l_cafe.item(), "l_ord": float(l_ord),
        "l_efgf_weighted": float(l_efgf_w), "l_cafe_weighted": float(l_cafe_w),
        "l_mhs": float(l_mhs), "cafe_hsic_raw": aux["cafe_hsic_loss"].item(),
    }
    return total, breakdown


# %% CELL 5 -- dataset.py (NBackDataset, session_holdout_split, trial_level_train_val_split)
# ============================================================================
# Dataset for TSFNet, reading the .npy files produced by
# Preprocessing_TSFNet_paper_nback.py.
# ============================================================================
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class _InMemoryWindowDataset(Dataset):
    """Lightweight Dataset backed by tensors already in memory (as opposed
    to NBackDataset, which loads from .npy files by index). Used to hold
    the SMOTE-augmented training set (real windows + synthetic interpolated
    windows appended), since Subset(full, indices) can only reference real,
    on-disk windows and cannot represent synthetic ones."""

    def __init__(self, eeg, fnirs, labels):
        self.eeg = eeg
        self.fnirs = fnirs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.fnirs[idx], self.labels[idx]


def smote_oversample_fold(eeg, fnirs, labels, k_neighbors=5, random_state=None):
    """
    ISOLATED ABLATION FUNCTION (priority-3d sub-experiment): SMOTE (Chawla et
    al., 2002, "SMOTE: Synthetic Minority Over-sampling Technique", JAIR),
    adapted for paired multimodal (EEG+fNIRS) spatiotemporal windows rather
    than the flat feature vectors SMOTE was originally designed for.

    IMPORTANT CAVEATS (documented here because they materially affect how
    to interpret a positive OR negative result from this experiment):
    1. Standard SMOTE assumes linear interpolation between two feature
       vectors is a physically/semantically meaningful point in the input
       space. Here, the "feature vectors" are raw (or lightly preprocessed)
       EEG/fNIRS window tensors -- there is no guarantee that the pointwise-
       linear interpolation of two EEG windows is itself a physiologically
       plausible EEG window (unlike e.g. interpolating two CNN embedding
       vectors, where the space is more likely to be locally linear/smooth).
       This is a genuine methodological risk specific to applying SMOTE to
       raw signal windows rather than to a learned feature/embedding space.
    2. EEG and fNIRS windows for a given synthetic sample are interpolated
       using the SAME donor pair and the SAME interpolation coefficient
       lambda, to preserve the (weak) cross-modal correspondence a real
       paired sample would have. Neighbor search (for picking donor pairs)
       is done on the CONCATENATION of flattened EEG+fNIRS, not on either
       modality alone, so neighbors are "close" in the joint space.

    Applied ONLY to this fold's TRAINING indices (never val/test) -- must be
    called with eeg/fnirs/labels already sliced to the training subset,
    never the full fold or held-out test session, to avoid any leakage.

    For each class with fewer samples than the fold's majority class count,
    synthetic samples are generated by: (1) picking a random real sample i
    of that class, (2) finding its k_neighbors nearest same-class neighbors
    in the joint flattened EEG+fNIRS space (Euclidean distance), (3) picking
    a random neighbor j from those, (4) synthetic = i + lambda*(j - i) for
    lambda ~ Uniform(0,1), applied identically to both eeg and fnirs.
    Classes with fewer than 2 samples in this fold cannot have neighbors
    found (k-NN needs at least 1 same-class neighbor) and are left
    un-oversampled, with a warning -- this mirrors the zero-count handling
    used in the class_weighting_scheme code paths.

    Returns (eeg_aug, fnirs_aug, labels_aug): original samples followed by
    all synthetic samples appended at the end, as new torch tensors.
    """
    rng = np.random.RandomState(random_state)
    eeg_np = eeg.numpy() if torch.is_tensor(eeg) else np.asarray(eeg)
    fnirs_np = fnirs.numpy() if torch.is_tensor(fnirs) else np.asarray(fnirs)
    labels_np_local = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)

    classes, counts = np.unique(labels_np_local, return_counts=True)
    max_count = counts.max()

    synth_eeg, synth_fnirs, synth_labels = [], [], []
    for cls, n_cls in zip(classes, counts):
        n_needed = max_count - n_cls
        if n_needed <= 0:
            continue  # this class IS the (a) majority -- nothing to add
        cls_idx = np.where(labels_np_local == cls)[0]
        if len(cls_idx) < 2:
            print(f"  WARNING [SMOTE]: class {cls} has only {len(cls_idx)} sample(s) in this "
                  f"fold's training set -- cannot find same-class neighbors, skipping "
                  f"oversampling for this class ({n_needed} samples short of majority).")
            continue
        k_eff = min(k_neighbors, len(cls_idx) - 1)  # can't have more neighbors than other same-class samples

        # Joint flattened feature space for neighbor search only (the actual
        # interpolation is still done on the full-shape eeg/fnirs tensors).
        cls_eeg_flat = eeg_np[cls_idx].reshape(len(cls_idx), -1)
        cls_fnirs_flat = fnirs_np[cls_idx].reshape(len(cls_idx), -1)
        cls_joint_flat = np.concatenate([cls_eeg_flat, cls_fnirs_flat], axis=1)
        # Pairwise Euclidean distances within this class only. BUG FIX: the
        # original version computed this via broadcasting
        # (cls_joint_flat[:,None,:] - cls_joint_flat[None,:,:]), which
        # materializes a dense (n_cls, n_cls, D) array -- D here is the
        # FULL flattened EEG+fNIRS window size (order ~1e5-1e6 elements per
        # sample for this project's window shapes), so that intermediate
        # array alone required tens of GB of RAM and crashed with
        # numpy._core._exceptions._ArrayMemoryError on every subject. Fixed
        # by using the standard ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        # identity, which needs only an (n_cls, n_cls) Gram matrix (KB-scale
        # for this project's fold sizes) rather than an (n_cls, n_cls, D)
        # array -- verified numerically to match the brute-force distances
        # to float rounding precision (~1e-6) before this fix was applied.
        sq_norms = (cls_joint_flat ** 2).sum(axis=1)
        gram = cls_joint_flat @ cls_joint_flat.T
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * gram
        np.clip(dist_sq, 0, None, out=dist_sq)  # guard tiny negative values from float rounding
        dist = np.sqrt(dist_sq)
        np.fill_diagonal(dist, np.inf)  # a sample is never its own neighbor
        nn_idx_within_cls = np.argsort(dist, axis=1)[:, :k_eff]  # (n_cls, k_eff)

        for _ in range(n_needed):
            i_local = rng.randint(len(cls_idx))
            j_local = nn_idx_within_cls[i_local, rng.randint(k_eff)]
            lam = rng.uniform(0.0, 1.0)
            i_global, j_global = cls_idx[i_local], cls_idx[j_local]
            synth_eeg.append(eeg_np[i_global] + lam * (eeg_np[j_global] - eeg_np[i_global]))
            synth_fnirs.append(fnirs_np[i_global] + lam * (fnirs_np[j_global] - fnirs_np[i_global]))
            synth_labels.append(cls)

    if synth_labels:
        eeg_aug = np.concatenate([eeg_np, np.stack(synth_eeg)], axis=0)
        fnirs_aug = np.concatenate([fnirs_np, np.stack(synth_fnirs)], axis=0)
        labels_aug = np.concatenate([labels_np_local, np.array(synth_labels)], axis=0)
    else:
        eeg_aug, fnirs_aug, labels_aug = eeg_np, fnirs_np, labels_np_local

    print(f"  [SMOTE] class counts before={dict(zip(classes.tolist(), counts.tolist()))} | "
          f"{len(synth_labels)} synthetic samples added | after oversampling total={len(labels_aug)}")

    return (torch.from_numpy(eeg_aug).float(),
            torch.from_numpy(fnirs_aug).float(),
            torch.from_numpy(labels_aug).long())


class NBackDataset(Dataset):
    """Loads one subject's preprocessed windows and exposes them as a
    standard PyTorch Dataset. session_ids/trial_ids are kept alongside so
    the training script can build the paper's session-based hold-out split
    (Sec. 3.6) and the trial-level 4:1 train/val split within a fold.

    priority-E1: if `{subject_id}_eeg_connectivity.npy` exists next to the
    other preprocessed files (see Preproc_tsfnet_nback_badch.py's
    ENABLE_EEG_CONNECTIVITY / eeg_connectivity.py), it is loaded into
    self.connectivity (shape (N, 2, n_pairs): 2 bands [theta, alpha] x
    vectorized channel-pair PLV) and __getitem__ returns a 4-tuple
    (eeg, fnirs, label, connectivity). If the file is ABSENT (every subject
    processed before E1 existed, or any run with
    ENABLE_EEG_CONNECTIVITY=False -- i.e. every existing/accepted result in
    this project), self.connectivity stays None and __getitem__ returns the
    ORIGINAL 3-tuple (eeg, fnirs, label), UNCHANGED from before this
    addition. This is a deliberate VARIABLE-ARITY design (rather than always
    padding to 4) specifically so every `for x_eeg, x_fnirs, y in loader:`
    loop elsewhere in this file keeps working byte-for-byte unmodified for
    any subject_dir without a connectivity file -- which is every
    subject_dir that existed before this change. Only train_one_fold/
    evaluate's core loops were updated (via _unpack_batch below) to
    additionally accept the 4-tuple form."""

    def __init__(self, subject_dir, subject_id, indices=None):
        eeg = np.load(os.path.join(subject_dir, f"{subject_id}_eeg_windows.npy"))
        fnirs = np.load(os.path.join(subject_dir, f"{subject_id}_fnirs_windows.npy"))
        labels = np.load(os.path.join(subject_dir, f"{subject_id}_labels.npy"))
        session_ids = np.load(os.path.join(subject_dir, f"{subject_id}_session_ids.npy"))
        trial_ids = np.load(os.path.join(subject_dir, f"{subject_id}_trial_ids.npy"))

        conn_path = os.path.join(subject_dir, f"{subject_id}_eeg_connectivity.npy")
        connectivity = np.load(conn_path) if os.path.exists(conn_path) else None
        chan_path = os.path.join(subject_dir, f"{subject_id}_eeg_channel_windows.npy")
        channels = np.load(chan_path) if os.path.exists(chan_path) else None

        if indices is not None:
            eeg, fnirs, labels = eeg[indices], fnirs[indices], labels[indices]
            session_ids, trial_ids = session_ids[indices], trial_ids[indices]
            if connectivity is not None:
                connectivity = connectivity[indices]
            if channels is not None:
                channels = channels[indices]

        self.eeg = torch.from_numpy(eeg).float()
        self.fnirs = torch.from_numpy(fnirs).float()
        self.labels = torch.from_numpy(labels).long()
        self.session_ids = session_ids
        self.trial_ids = trial_ids
        self.connectivity = (torch.from_numpy(connectivity).float()
                              if connectivity is not None else None)
        self.has_connectivity = self.connectivity is not None
        self.channels = (torch.from_numpy(channels).float()
                          if channels is not None else None)
        self.has_channels = self.channels is not None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Arity grows only when the optional files exist, so every legacy
        # subject_dir keeps returning the original 3-tuple untouched. When
        # channel-space data is present the connectivity slot is always
        # emitted too (as an empty tensor if there is no connectivity file),
        # so position 3 unambiguously means "connectivity" and position 4
        # unambiguously means "channels" -- see _unpack_batch.
        if self.channels is not None:
            conn = self.connectivity[idx] if self.connectivity is not None else torch.zeros(0)
            return self.eeg[idx], self.fnirs[idx], self.labels[idx], conn, self.channels[idx]
        if self.connectivity is not None:
            return self.eeg[idx], self.fnirs[idx], self.labels[idx], self.connectivity[idx]
        return self.eeg[idx], self.fnirs[idx], self.labels[idx]


def _unpack_batch(batch):
    """Handles both NBackDataset return arities: the original 3-tuple
    (eeg, fnirs, label) -- every existing subject_dir, every previously-run/
    accepted result -- or the priority-E1 4-tuple
    (eeg, fnirs, label, connectivity). Returns (x_eeg, x_fnirs, y,
    connectivity_or_None, channels_or_None). Centralizing this check here
    (rather than repeating `len(batch)` at every loop site) is what keeps
    use_fc_features=False / eeg_spatial_mode='none' true no-ops: for a
    3-tuple batch this returns None for both optional items and every call
    site behaves exactly as it did before this function existed."""
    def _or_none(t):
        # An empty placeholder (0-length last dim) means "not present".
        return None if (t is None or t.numel() == 0) else t

    if len(batch) == 5:
        return batch[0], batch[1], batch[2], _or_none(batch[3]), _or_none(batch[4])
    if len(batch) == 4:
        return batch[0], batch[1], batch[2], _or_none(batch[3]), None
    return batch[0], batch[1], batch[2], None, None



def session_holdout_split(session_ids, test_session):
    """Sec. 3.6: one session held out as test, the other two as train."""
    train_idx = np.where(session_ids != test_session)[0]
    test_idx = np.where(session_ids == test_session)[0]
    return train_idx, test_idx


def trial_level_train_val_split(trial_ids, labels, val_ratio=0.2, seed=42):
    """Sec. 3.6, first-stage training: 4:1 train/val split. Split at the
    TRIAL level (not window level) so sliding windows from the same trial
    never leak across train/val (they are highly correlated, being
    overlapping crops of the same ~12s epoch), stratified by class so each
    split keeps a balanced 0-/2-/3-back label distribution."""
    rng = np.random.default_rng(seed)
    unique_trials = np.unique(trial_ids)

    # majority label per trial (all windows of a trial share the same label
    # by construction, so this is just each trial's label)
    trial_to_label = {}
    for t in unique_trials:
        trial_to_label[t] = labels[trial_ids == t][0]

    val_trials = []
    for cls in np.unique(labels):
        cls_trials = [t for t in unique_trials if trial_to_label[t] == cls]
        rng.shuffle(cls_trials)
        n_val = max(1, int(round(len(cls_trials) * val_ratio)))
        val_trials.extend(cls_trials[:n_val])
    val_trials = set(val_trials)

    val_mask = np.isin(trial_ids, list(val_trials))
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx


def cross_session_train_val_split(session_ids_tv, val_session):
    """
    priority-XSVAL: hold out one WHOLE training session as validation, instead
    of carving a few trials out of the pooled training sessions.

    WHY (the defect this addresses):
    In each LOSO fold, one session is the test set and the other two are for
    training. The existing trial_level_train_val_split then takes
    val_ratio=0.2 of the TRIALS from those two POOLED sessions. With 6 trials
    per class available (3 per class per session x 2 sessions), that is
    round(6*0.2) = 1 trial per class -> a validation set of just 3 TRIALS.
    Two consequences, both bad:
      1. Early stopping and best-checkpoint selection ride on 3 trials, which
         is close to an arbitrary choice of epoch.
      2. That validation set is drawn from the SAME sessions as training, so
         it measures within-session generalization -- while the actual test
         is ACROSS sessions. The model is selected under conditions that do
         not resemble the conditions it is scored under.
    Holding out a whole session instead gives 9 validation trials (3 per
    class) AND makes validation cross-session, matching the test regime.

    COST (stated honestly): the model then trains on ONE session (9 trials)
    instead of two (18 trials) -- half the training data. Whether the better-
    matched, larger validation set outweighs the halved training set is an
    empirical question, which is exactly why this is a flag and not a silent
    change.

    session_ids_tv: (N_tv,) session id of each train+val window (test session
        already excluded by session_holdout_split).
    val_session: which of the two remaining sessions to use as validation.
    Returns (train_sub_idx, val_sub_idx), indices INTO the train_val subset.
    """
    val_mask = session_ids_tv == val_session
    train_sub_idx = np.where(~val_mask)[0]
    val_sub_idx = np.where(val_mask)[0]
    if len(train_sub_idx) == 0 or len(val_sub_idx) == 0:
        raise ValueError(
            f"cross_session_train_val_split produced an empty split for "
            f"val_session={val_session} (train={len(train_sub_idx)}, "
            f"val={len(val_sub_idx)} windows). Sessions present: "
            f"{np.unique(session_ids_tv).tolist()}")
    return train_sub_idx, val_sub_idx


# %% CELL 6 -- train.py (evaluate, cohen_kappa, detailed_test_report, train_one_fold, run_subject_cross_session)
# ============================================================================
# TSFNet training script: subject-dependent, cross-session Hold-Out
# evaluation (Sec. 3.6), two-stage training strategy, Adam lr=0.001.
#
# ADAPTATIONS for this project's n-back dataset (documented, not silent):
#   - num_classes=3 (0-/2-/3-back) instead of the paper's binary tasks.
#   - Added weight_decay (AdamW-style L2) -- NOT in the paper -- because our
#     per-subject sample count (27 trials x ~10 windows/trial = ~270
#     windows/subject, split ~180 train / ~90 test per fold) is far smaller
#     than what a 3.3M-parameter, weight-decay-free model can be trained on
#     without overfitting. Our TSFNet instance is already much smaller
#     (~400K params, see model.py's docstring) but weight decay adds a
#     second, complementary layer of protection.
#   - Reduced stage-1 patience/max-epochs from the paper's 50/300 to more
#     conservative defaults, tunable via CLI args, since with ~150
#     first-stage training windows per fold, 300 epochs is enormous
#     overkill and mainly just risks overfitting before early stopping
#     would trigger. Defaults below are a reasonable starting point, not a
#     tuned optimum -- treat them as hyperparameters to sweep.
#
# REPORTING (added): matches the pattern used in a reference exploratory
# notebook (training_v2_run_a3_with_diagnostics.ipynb) for this same
# project, adapted to our TSFNet's actual interface:
#   - Best-stage-1-val checkpoint is saved to DISK the moment it improves
#     (torch.save), not just kept in an in-memory deepcopy -- so a crash
#     during stage 2 doesn't lose the best stage-1 weights (same crash-safety
#     motivation as CrashSafeLogger, applied to model weights specifically).
#   - Per-epoch history (train_loss, train_acc, val_loss, val_acc, val_kappa)
#     is recorded for later learning-curve plotting.
#   - A detailed per-fold results dict (test accuracy/kappa, confusion
#     matrix, per-branch "zero-retraining" ablation accuracy [EEG-only /
#     fNIRS-only / fusion-only, i.e. each classifier head of the SAME
#     trained model], learned decision-fusion weights [Eq. 4], best_val_acc,
#     final train_acc, and an overfitting/generalization gap] is saved as
#     JSON per session.
#   - A cross-session summary JSON (mean/std across the 3 folds) is saved
#     once run_subject_cross_session() finishes.
# ============================================================================
import argparse
import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset



def adapt_batchnorm_to_target(model, target_loader, device, log_fn=print):
    """
    priority-E4: AdaBN-style test-time adaptation of BatchNorm statistics to
    the held-out session (Li et al. 2017, "Revisiting Batch Normalization
    For Practical Domain Adaptation").

    WHY THIS IS THE NATURAL SUCCESSOR TO E3: priority-E3 (per-session
    normalization) established empirically, on this project's own data, that
    session-to-session distribution shift is a real and recoverable source of
    loss under the leave-one-session-out protocol (+1.18 acc / +0.0182 kappa,
    consistent in all 3 seeds). E3 removed that shift at the INPUT level.
    AdaBN removes the part of it that survives into the network's INTERNAL
    feature distributions: this model's BatchNorm3d layers (2 in EEGLikeBranch
    and 2 in FNIRSBranch, instantiated once per branch, so 6 in total counting
    the Fusion branch's reuse of EEGLikeBranch) carry running_mean/running_var
    estimated ENTIRELY from the two training sessions. At test time those
    frozen statistics are used to normalize activations from a THIRD session
    whose feature distribution is measurably different. AdaBN simply
    recomputes those statistics on the target session.

    WHAT IT DOES NOT DO: no weights are trained, no gradients are computed,
    and no labels are touched. Only the non-learnable running_mean/
    running_var buffers change. This is why it costs one forward pass rather
    than a training run.

    LEAKAGE ANALYSIS (must be reported in the thesis, alongside E3's):
    This is TRANSDUCTIVE in exactly the same sense as priority-E3 -- it uses
    the held-out session's UNLABELED inputs, never its labels. It is standard
    and accepted in the BCI session-adaptation literature and corresponds to
    the realistic deployment case (a new session's raw data is available
    before decoding begins). It is nonetheless a deviation from a strictly
    inductive protocol. E3 and E4 results must be disclosed together and must
    not be compared against a strictly inductive baseline without stating
    this.

    IMPLEMENTATION NOTES (each of these is a real correctness trap):
      - momentum is set to None on each BN layer, which makes PyTorch use a
        CUMULATIVE moving average rather than an exponential one. That gives
        the EXACT mean/var over all target batches seen, independent of batch
        order -- an exponential average would weight the last batches more and
        make the result depend on shuffling.
      - reset_running_stats() is called first so the source-session statistics
        are REPLACED, not blended into. Blending would make the result depend
        on how many target batches happen to exist.
      - ONLY the BN layers are switched to train mode; the model as a whole
        stays in eval(). If model.train() were called instead, Dropout3d
        (p=0.5 in the EEG/Fusion branches) would become active during the
        adaptation pass and corrupt the very statistics being estimated.
      - The whole pass runs under torch.no_grad(): running stats are buffers
        updated in the forward pass itself, so no gradient machinery is needed
        and none should be built.
    """
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_layers:
        log_fn("  [adabn] no BatchNorm layers found -- nothing to adapt (no-op)")
        return model

    original_momenta = [m.momentum for m in bn_layers]

    model.eval()                      # dropout OFF for the whole model...
    for m in bn_layers:
        m.reset_running_stats()       # replace source stats, don't blend
        m.momentum = None             # cumulative (exact) average
        m.train()                     # ...but BN layers must be in train mode
                                      #    to update their running buffers

    n_batches = 0
    with torch.no_grad():
        for _batch in target_loader:
            x_eeg, x_fnirs, _y, x_conn, x_chan = _unpack_batch(_batch)
            x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            model(x_eeg, x_fnirs, x_conn, x_chan)
            n_batches += 1

    # Restore original momenta and put everything back in eval mode, so the
    # model that gets tested behaves exactly like a normal eval-mode model,
    # only with target-session running statistics.
    for m, mom in zip(bn_layers, original_momenta):
        m.momentum = mom
    model.eval()

    log_fn(f"  [adabn] adapted {len(bn_layers)} BatchNorm layer(s) to the held-out "
           f"session over {n_batches} batch(es) (priority-E4, transductive: "
           f"unlabeled target inputs only)")
    return model


def evaluate(model, loader, device, lambda_eeg, lambda_fnirs, class_weight=None,
             focal_gamma=None, asymmetric_cost=None, ordinal_reg_lambda=0.0,
             mhs_lambda=0.0, label_smoothing=0.0, normalize_ypred=False,
             efgf_lambda=1.0, cafe_lambda=1.0):
    """
    BUGFIX (2026-09-01): previously this function had no class_weight/
    focal_gamma/asymmetric_cost/ordinal_reg_lambda parameters at all, so
    every call always computed tsfnet_loss in the UNWEIGHTED baseline
    scale -- including the call at the end of stage 1 that recomputes
    stage1_final_train_loss (see train_one_fold), which is then compared
    via min() against stage1_best_val_train_loss (which WAS computed in
    the correctly-weighted scale, inside the training loop). For any run
    using class_weight/focal_gamma/asymmetric_cost/ordinal_reg_lambda,
    this silently mixed two different loss scales in that min() comparison,
    making stage2_target inconsistent -- e.g. asymmetric_cost with
    hard_weight=2.0 inflated the weighted scale by up to +29% relative to
    baseline, making stage 2's stopping criterion unintentionally stricter
    for weighted runs than for the unweighted baseline it was being
    compared against. Passing these through fixes it. Default values
    (None/0.0) reproduce the exact prior unweighted behavior bit-for-bit --
    this fix changes NOTHING for any run that doesn't use one of these
    four parameters (i.e. every previously-accepted or -rejected ablation
    result in this project's history remains valid; only class_weight/
    focal_gamma/asymmetric_cost/ordinal_reg_lambda runs are affected, and
    those should be rerun with this fix before drawing conclusions from
    them).

    mhs_lambda: priority-B1 sub-experiment (CAFE multi-head HSIC diversity
    term). Threaded through here for the EXACT same reason class_weight/
    focal_gamma/asymmetric_cost/ordinal_reg_lambda are: any loss-shaping
    parameter that isn't passed to evaluate() silently recreates the
    2026-09-01 stage2-scale bug documented above. Default 0.0 reproduces
    prior behavior bit-for-bit (see tsfnet_loss's mhs_lambda docstring).
    """
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for _batch in loader:
            x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(_batch)
            x_eeg, x_fnirs, y = x_eeg.to(device), x_fnirs.to(device), y.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            y_eeg, y_fusion, y_fnirs, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)
            loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux,
                                   lambda_eeg, lambda_fnirs, class_weight=class_weight,
                                   focal_gamma=focal_gamma, asymmetric_cost=asymmetric_cost,
                                   ordinal_reg_lambda=ordinal_reg_lambda, mhs_lambda=mhs_lambda, efgf_lambda=efgf_lambda, cafe_lambda=cafe_lambda,
                                   normalize_ypred=normalize_ypred,
                                   label_smoothing=label_smoothing)
            total_loss += loss.item() * y.size(0)
            preds = y_pred.argmax(dim=-1)
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)
            all_true.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    acc = n_correct / max(n_total, 1)
    avg_loss = total_loss / max(n_total, 1)
    all_true = np.concatenate(all_true) if all_true else np.array([])
    all_pred = np.concatenate(all_pred) if all_pred else np.array([])
    kappa = cohen_kappa(all_true, all_pred) if len(all_true) else float("nan")
    return avg_loss, acc, kappa


def cohen_kappa(y_true, y_pred):
    """Eq. 14, computed by hand to avoid an sklearn dependency here."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(y_true)
    if n == 0:
        return float("nan")
    po = (y_true == y_pred).mean()
    pe = 0.0
    for c in classes:
        pe += (y_true == c).mean() * (y_pred == c).mean()
    denom = 1 - pe
    return (po - pe) / denom if denom > 1e-8 else 0.0


def confusion_matrix_np(y_true, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def detailed_test_report(model, test_loader, device, num_classes=3):
    """Full test-set pass collecting the final fused prediction AND each
    branch's own prediction (zero-retraining ablation: EEG-only, fNIRS-only,
    fusion-only accuracy from the SAME trained model, no extra training
    needed since each branch already has its own classifier head), plus
    the confusion matrix and the learned Eq.4 decision-fusion weights."""
    model.eval()
    all_true, all_final, all_eeg, all_fnirs, all_fusion = [], [], [], [], []
    all_gate = []   # priority-gating: per-sample (B,3) weights, empty unless gated
    with torch.no_grad():
        for _batch in test_loader:
            x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(_batch)
            x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            y_eeg, y_fusion, y_fnirs, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)
            all_true.append(y.numpy())
            all_final.append(y_pred.argmax(dim=-1).cpu().numpy())
            all_eeg.append(y_eeg.argmax(dim=-1).cpu().numpy())
            all_fnirs.append(y_fnirs.argmax(dim=-1).cpu().numpy())
            all_fusion.append(y_fusion.argmax(dim=-1).cpu().numpy())
            if aux.get("gate_weights") is not None:
                all_gate.append(aux["gate_weights"].cpu().numpy())

    y_true = np.concatenate(all_true)
    y_final = np.concatenate(all_final)
    y_eeg = np.concatenate(all_eeg)
    y_fnirs = np.concatenate(all_fnirs)
    y_fusion = np.concatenate(all_fusion)

    test_acc = float((y_final == y_true).mean())
    kappa = cohen_kappa(y_true, y_final)
    cm = confusion_matrix_np(y_true, y_final, num_classes)

    with torch.no_grad():
        w_labram = (torch.sigmoid(model.w_labram).item()
                    if hasattr(model, "w_labram") else None)
        beta_labram = (torch.sigmoid(model.beta_labram).item()
                       if hasattr(model, "beta_labram") else None)

        if all_gate:
            # priority-gating: in "gated" mode the three nn.Parameters
            # w_eeg/w_fusion/w_fnirs are NEVER updated (they only receive an
            # explicit zero gradient), so sigmoid(w_eeg) is frozen at
            # sigmoid(0.0) = 0.5 forever. Reporting that here would write a
            # plausible-looking but completely meaningless 0.5/0.5/0.5 into
            # every row of the results CSV, hiding what the gate actually
            # learned. Report the mean of the REAL per-sample weights
            # instead -- same dict keys (rule 7), correct values.
            gate_arr = np.concatenate(all_gate, axis=0)          # (N, 3)
            gate_mean = gate_arr.mean(axis=0)
            w_eeg = float(gate_mean[0])
            w_fusion = float(gate_mean[1])
            w_fnirs = float(gate_mean[2])
            # Spread across samples: the whole point of gating is that these
            # vary per sample. If the spread is ~0 the gate has COLLAPSED to
            # a constant and is no better than the static scalars it replaced
            # -- just with extra parameters, i.e. the "added capacity" failure
            # mode this project has already hit 14+ times.
            #
            # The test must be RELATIVE, not absolute. An earlier version of
            # this check used `std > 1e-6`, which reported a fully collapsed
            # gate (std 1.9e-06 around a mean of 0.333, i.e. varying by six
            # parts per million) as "per-sample" -- a plausible-looking True
            # that would have hidden the collapse. Coefficient of variation
            # (std/mean) is scale-free and catches it: 0.01 means the weight
            # swings by ~1% of its own size across samples, which is the
            # minimum worth calling input-dependent.
            gate_std = gate_arr.std(axis=0)
            gate_cv = gate_std / np.maximum(gate_mean, 1e-8)
            gate_extra = {
                "gate_w_eeg_std": float(gate_std[0]),
                "gate_w_fusion_std": float(gate_std[1]),
                "gate_w_fnirs_std": float(gate_std[2]),
                "gate_max_cv": float(gate_cv.max()),
                "gate_is_per_sample": bool(gate_cv.max() > 0.01),
            }
        else:
            w_eeg = torch.sigmoid(model.w_eeg).item()
            w_fusion = torch.sigmoid(model.w_fusion).item()
            w_fnirs = torch.sigmoid(model.w_fnirs).item()
            gate_extra = {}

    return {
        "test_accuracy_final": test_acc,
        "kappa_final": kappa,
        "confusion_matrix_final": cm.tolist(),
        "ablation": {
            "eeg_only": float((y_eeg == y_true).mean()),
            "fnirs_only": float((y_fnirs == y_true).mean()),
            "fusion_only": float((y_fusion == y_true).mean()),
            "final_weighted": test_acc,
        },
        "learned_weights": {"w_eeg": w_eeg, "w_fusion": w_fusion, "w_fnirs": w_fnirs,
                            "w_labram": w_labram, "beta_labram": beta_labram,
                            **gate_extra},
    }


def make_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_factor):
    """
    priority-G1: linear warm-up followed by cosine annealing, as a plain
    LambdaLR multiplier on the base lr. Mirrors the schedule MSTFDN uses
    (Adam + cosine annealing with a linear warm-up phase, warm-up = 10
    epochs, minimum lr = base * 0.05).

    Returns None when warmup_epochs == 0 AND min_lr_factor == 1.0, i.e. when
    the caller asked for no schedule at all -- so the default path creates no
    scheduler object and steps nothing, leaving prior behavior bit-exact.

    The multiplier is:
        epoch < warmup           -> (epoch + 1) / warmup            (linear ramp)
        warmup <= epoch < total  -> min_f + (1 - min_f) * 0.5 * (1 + cos(pi * p))
                                    where p = (epoch - warmup) / (total - warmup)
        epoch >= total           -> min_f                            (floor, no wrap-around)

    That last clamp matters: LambdaLR keeps calling the lambda for as many
    epochs as training actually runs, and stage 1 can run FEWER epochs than
    stage1_max_epochs (early stopping) or the caller may reuse the optimizer.
    Without the clamp the cosine argument would exceed pi and the lr would
    start RISING again, which is silent and hard to notice.
    """
    if warmup_epochs == 0 and min_lr_factor == 1.0:
        return None

    warmup = max(int(warmup_epochs), 0)
    total = max(int(total_epochs), warmup + 1)
    min_f = float(min_lr_factor)

    def _lr_lambda(epoch):
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        if epoch >= total:
            return min_f
        progress = (epoch - warmup) / float(max(total - warmup, 1))
        progress = min(max(progress, 0.0), 1.0)
        return min_f + (1.0 - min_f) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def _collect_gate_values(model):
    """Read-only: current sigmoid values of every gated scalar on the model.
    Logged into history each epoch so gate adoption can be plotted afterwards
    (training_diagnostics.plot_gate_evolution). No gradient, no side effects."""
    gates = {}
    with torch.no_grad():
        if hasattr(model, "beta_labram"):
            gates["beta_labram"] = torch.sigmoid(model.beta_labram).item()
        if hasattr(model, "w_labram"):
            gates["w_labram"] = torch.sigmoid(model.w_labram).item()
        if hasattr(model, "eeg_band_branch") and hasattr(model.eeg_band_branch, "gate"):
            gates["eeg_band_gate"] = torch.sigmoid(model.eeg_band_branch.gate).item()
        # NOTE (priority-gating): when fusion_combination == "gated" these
        # three scalars are inert -- they never receive a real gradient, so
        # these entries stay pinned at sigmoid(0.0) = 0.5 for every epoch and
        # say NOTHING about what the model actually weighted. The real,
        # per-sample weights are reported by detailed_test_report() under
        # learned_weights (plus gate_*_std). Do not plot these three as
        # "gate evolution" in gated runs.
        for nm in ("w_eeg", "w_fusion", "w_fnirs"):
            if hasattr(model, nm):
                gates[f"{nm}_fusion"] = torch.sigmoid(getattr(model, nm)).item()
        # priority-gatelog: EFGF's alpha/gamma (the parameters at the center
        # of the dead-gradient bug and the weight_decay-drift finding) and
        # CAFE's gate_fnirs/gate_fusion were NOT previously logged here, even
        # though they are exactly the gates those investigations are about.
        # Both EFGF parametrizations are covered: 'clamp' exposes plain
        # alpha/gamma in [0,1] already (no sigmoid needed); 'sigmoid' exposes
        # alpha_raw/gamma_raw (apply sigmoid to get the effective value).
        for efgf_name in ("efgf1", "efgf2"):
            layer = getattr(model, efgf_name, None)
            if layer is None:
                continue
            if hasattr(layer, "alpha"):     # gate_mode='clamp'
                gates[f"{efgf_name}_alpha"] = float(torch.clamp(layer.alpha, 0.0, 1.0).item())
                gates[f"{efgf_name}_gamma"] = float(torch.clamp(layer.gamma, 0.0, 1.0).item())
            elif hasattr(layer, "alpha_raw"):   # gate_mode='sigmoid'
                gates[f"{efgf_name}_alpha"] = torch.sigmoid(layer.alpha_raw).item()
                gates[f"{efgf_name}_gamma"] = torch.sigmoid(layer.gamma_raw).item()
        if hasattr(model, "cafe"):
            if hasattr(model.cafe, "gate_fnirs"):
                gates["cafe_gate_fnirs"] = torch.sigmoid(model.cafe.gate_fnirs).item()
            if hasattr(model.cafe, "gate_fusion"):
                gates["cafe_gate_fusion"] = torch.sigmoid(model.cafe.gate_fusion).item()
    return gates


def inject_eeg_awgn(x_eeg, snr_db_choices, corrupt_prob, generator=None):
    """priority-noiseaware: additive white Gaussian noise on the EEG stream
    only, as a TRAIN-TIME augmentation. fNIRS is deliberately left untouched.

    Implements MGFNet (Zhang et al., Sensors 2026, 26, 3402) Eqs. 8-9:
        x_hat = x + N(0, sigma^2),   sigma^2 = P_signal / 10^(SNR_dB/10)
    with P_signal the per-sample signal power, and the SNR drawn per sample
    from `snr_db_choices`. Each sample is corrupted independently with
    probability `corrupt_prob` (MGFNet used 0.8, leaving 20% clean).

    WHY EEG-ONLY: the point is to stop the decision layer from over-committing
    to one modality. If BOTH streams were degraded there would be no intact
    alternative left to learn to fall back on, and the augmentation would just
    be label-preserving noise on everything.

    RELATION TO PREPROCESSING: this is orthogonal to, and does NOT replace,
    the artifact removal in the preprocessing pipeline (band-pass, EOG
    correction, MBLL, fNIRS low-pass). Those remove real, structured
    artifacts; this injects synthetic broadband noise to force redundancy.
    Disabling preprocessing "because we now add noise anyway" would conflate
    the two and make any accuracy change uninterpretable.

    Per-sample power is computed on the ALREADY-PREPROCESSED tensor, so the
    resulting SNR is defined relative to the preprocessed signal, not the raw
    recording -- worth stating explicitly in the thesis, since MGFNet defines
    it the same way but does not spell this out.

    Returns a NEW tensor; x_eeg is never modified in place (it may be a view
    into the DataLoader's pinned memory).
    """
    if not snr_db_choices or corrupt_prob <= 0.0:
        return x_eeg

    b = x_eeg.shape[0]
    flat_dims = tuple(range(1, x_eeg.dim()))

    # Per-sample signal power (variance about the mean, matching MGFNet's
    # "original signal power (variance)" wording).
    p_signal = x_eeg.var(dim=flat_dims, keepdim=True, unbiased=False)

    # Draw one SNR per sample from the discrete set.
    idx = torch.randint(len(snr_db_choices), (b,), device=x_eeg.device,
                        generator=generator)
    snr_db = torch.as_tensor(snr_db_choices, device=x_eeg.device,
                             dtype=x_eeg.dtype)[idx]
    snr_db = snr_db.view(b, *([1] * (x_eeg.dim() - 1)))

    sigma = torch.sqrt(p_signal / torch.pow(10.0, snr_db / 10.0))
    noise = torch.randn(x_eeg.shape, device=x_eeg.device, dtype=x_eeg.dtype,
                        generator=generator) * sigma

    # Bernoulli mask: which samples in this batch get corrupted at all.
    keep_clean = (torch.rand(b, device=x_eeg.device, generator=generator)
                  >= corrupt_prob)
    noise = noise.clone()
    noise[keep_clean] = 0.0
    return x_eeg + noise


@torch.no_grad()
def _update_bn_multi_input(model, loader, device):
    """Project-specific analogue of torch.optim.swa_utils.update_bn, for a
    model whose forward() takes 4 positional tensors (x_eeg, x_fnirs, x_conn,
    x_chan) rather than the single input tensor the built-in helper assumes.
    Same algorithm: reset every BatchNorm layer's running stats, then do one
    full pass over `loader` in train() mode with momentum temporarily set to
    None (cumulative moving average instead of exponential), so the final
    running_mean/var reflect a clean estimate from THIS model's current
    weights -- required before evaluating an SWA-averaged model, whose
    individual snapshots' BN stats do not describe the averaged weights.
    """
    bn_modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if not bn_modules:
        return   # e.g. norm_mode="group" -- nothing to recalibrate, GroupNorm has no running stats
    momenta = {}
    for m in bn_modules:
        momenta[m] = m.momentum
        m.reset_running_stats()
        m.momentum = None   # cumulative average over the whole pass below
    was_training = model.training
    model.train()
    for _batch in loader:
        x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(_batch)
        x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
        x_conn = x_conn.to(device) if x_conn is not None else None
        x_chan = x_chan.to(device) if x_chan is not None else None
        model(x_eeg, x_fnirs, x_conn, x_chan)
    for m in bn_modules:
        m.momentum = momenta[m]
    model.train(was_training)


def train_one_fold(model, train_loader, val_loader, test_loader, device,
                    lambda_eeg=0.2, lambda_fnirs=0.2, lr=1e-3, weight_decay=1e-4,
                    stage1_max_epochs=100, stage1_patience=20, stage2_max_epochs=80,
                    ckpt_path=None, log_fn=print, grad_clip_norm=None, class_weight=None,
                    focal_gamma=None, asymmetric_cost=None, ordinal_reg_lambda=0.0,
                    mhs_lambda=0.0, adabn=False, tta_mode=None, normalize_ypred=False,
                    warmup_epochs=0, min_lr_factor=1.0, label_smoothing=0.0,
                    domain_adapt_lambda=0.0, target_loader=None,
                    tent_lr=1e-3, tent_steps=1, eeg_band_weight_decay=1e-2,
                    no_decay_scalars=False, stage2_stop_mode="loss",
                    noise_aware_snr_db=None, noise_aware_prob=0.0,
                    swa_enabled=False, swa_start_epoch=None, swa_lr=None,
                    efgf_lambda=1.0, cafe_lambda=1.0):
    """
    swa_enabled: ISOLATED ABLATION PARAMETER (priority-SWA). Stochastic Weight
    Averaging (Izmailov et al. 2018) -- averages model weights over the TAIL of
    stage-1 training (a wider, flatter minimum empirically generalizes better,
    with ZERO added parameters and ZERO architecture change; see
    torch.optim.swa_utils docs). Default False reproduces prior behavior
    bit-for-bit: swa_model is never constructed, no extra forward pass is run,
    ckpt_path's contents are decided exactly as before.

    When True:
      1. Starting at `swa_start_epoch` (default: round(0.75*stage1_max_epochs),
         i.e. the last quarter of stage 1 -- short enough to stay cheap, long
         enough to average over several distinct SGD-noise samples), an
         AveragedModel accumulates an equal-weighted running average of the
         model's parameters once per epoch, AFTER that epoch's optimizer
         steps (so the average is of post-update weights, standard practice).
      2. During this tail phase, an SWALR scheduler overrides whatever
         `scheduler` (if any) was active, holding lr at a constant `swa_lr`
         (default: `lr`, i.e. no change from stage 1's own lr if the caller
         didn't ask for anything cyclical) -- SWA's own paper recommends a
         constant or cyclic lr during the averaging phase, not a decaying one.
      3. After stage 1's loop ends (whether by early stopping or
         stage1_max_epochs), IF the averaged model was updated at least once:
         its BatchNorm running statistics are recomputed from scratch on
         train_loader (BN running stats computed during normal training
         reflect the INDIVIDUAL weight snapshots along the way, not the
         averaged weights -- reusing them would silently evaluate the wrong
         model; torch.optim.swa_utils.update_bn documents exactly this
         requirement). This project's forward() takes 4 positional tensors
         (x_eeg, x_fnirs, x_conn, x_chan), not the single-tensor input
         torch's built-in update_bn assumes, so a small project-specific
         version is used instead (_update_bn_multi_input, defined just above
         this function) -- same algorithm (reset running stats, forward in
         train() mode with momentum=None for cumulative averaging), adapted
         to this model's actual input signature.
      4. The averaged (and BN-recalibrated) model is evaluated on val_loader
         and compared against best_val_acc (stage 1's own early-stopped best
         checkpoint, already on disk at ckpt_path). SWA's weights are saved
         to ckpt_path ONLY if they score at least as well -- this makes
         swa_enabled=True a strictly non-regressive choice by construction:
         the worst case is "SWA didn't help, the original checkpoint is used
         unchanged", never "SWA made the saved checkpoint worse". This
         fallback is logged either way via log_fn, not silent.
      Stage 2 is untouched either way -- if stage2_stop_mode != "none", stage 2
      continues from whichever checkpoint stage 1 decided on (original or SWA),
      exactly as already happens for every other stage-1 decision.

    swa_start_epoch: see above. Only meaningful when swa_enabled=True.
    swa_lr: see above (SWALR's constant lr during the averaging phase). Only
      meaningful when swa_enabled=True; None defaults to `lr`.
    """
    """ckpt_path: if given, the best stage-1 checkpoint is saved to this path
    on disk (torch.save) every time val_acc improves, in addition to being
    kept in memory -- so it survives a crash during stage 2. If None, a
    temp path derived from id(model) is used (still crash-safe, just less
    convenient to find manually).

    grad_clip_norm: priority-1 sub-experiment (gradient clipping). TESTED AND
    REJECTED (both seeds showed a consistent drop vs. no-clip baseline).
    Default None (disabled) reproduces the accepted baseline.

    class_weight: ISOLATED ABLATION CHANGE (priority-3 step, class-weighting
    sub-experiment). Optional (num_classes,) tensor of per-class inverse-
    frequency weights, computed from THIS FOLD's training-set label counts
    only (see run_subject_cross_session) -- never from validation/test
    labels, to avoid any leakage. Forwarded unchanged to tsfnet_loss in both
    stage 1 and stage 2. This is the ONLY change in this edit relative to
    the accepted priority-1+2 baseline (no grad clipping, paper_mean
    combination): epochs, patience, lr, weight_decay, combination formula,
    and CAFE dim are all untouched. Default None reproduces the exact prior
    (unweighted) behavior for direct with/without comparison.

    focal_gamma: ISOLATED ABLATION CHANGE (priority-3c sub-experiment).
    If not None, tsfnet_loss uses focal_nll_loss (Lin et al. 2017) instead
    of plain nll_loss for all four loss terms, with `class_weight` (if any)
    used as focal's per-class alpha. Default None reproduces the exact
    prior (non-focal) behavior.

    mhs_lambda: ISOLATED ABLATION PARAMETER, priority-B1 sub-experiment
    (MSTFDN-inspired HSIC head-diversity regularization on CAFE's two
    cross-attention modules -- see cafe_multihead_hsic_loss / tsfnet_loss).
    Default 0.0 reproduces prior behavior bit-for-bit (see tsfnet_loss).

    warmup_epochs / min_lr_factor: ISOLATED ABLATION PARAMETERS, priority-G1
    sub-experiment (learning-rate schedule). Defaults 0 / 1.0 create NO
    scheduler at all (see make_warmup_cosine_scheduler), so the lr stays
    constant at `lr` for every epoch exactly as before. Set e.g.
    warmup_epochs=10, min_lr_factor=0.05 to reproduce MSTFDN's schedule.
    A separate scheduler is built for each stage, because the two stages have
    different epoch budgets and stage 2 restarting from the base lr (rather
    than continuing stage 1's decayed lr) is the intended behavior -- stage 2
    is a fresh fine-tuning phase on the full training set.

    adabn: ISOLATED ABLATION PARAMETER, priority-E4 sub-experiment. If True,
    BatchNorm running statistics are recomputed on the held-out test
    session's unlabeled inputs after training and before evaluation (see
    adapt_batchnorm_to_target for the full rationale and leakage analysis).
    Default False reproduces prior behavior bit-for-bit -- the function is
    never called and no buffer is touched.

    tta_mode: ISOLATED ABLATION PARAMETER, further generalization of
    priority-E4 (see tent_adaptation.py's module docstring for the full
    rationale: Tent is a strict generalization of AdaBN -- it also discards
    running statistics in favor of pure per-batch statistics, AND
    additionally optimizes the BatchNorm layers' learnable affine
    weight/bias by gradient descent on the entropy of the model's own
    predictions on the held-out session's unlabeled inputs).
        None (default) or "adabn": behaves EXACTLY as before -- if `adabn`
            is True, adapt_batchnorm_to_target runs and detailed_test_report
            is called unmodified; if False, neither runs. tta_mode is purely
            additive here and changes NOTHING about the accepted E3+E4 path.
        "tent": MUTUALLY EXCLUSIVE with adabn=True (both modify the SAME
            BatchNorm layers; combining them would confound which one
            produced any change -- raises ValueError if both are set).
            Skips adapt_batchnorm_to_target + detailed_test_report entirely
            and instead calls tent_detailed_test_report, which adapts the
            BatchNorm affine parameters by entropy minimization DURING the
            same forward pass that produces each test batch's prediction
            (see tent_adaptation.py -- this is architecturally different
            from AdaBN's separate one-shot pre-evaluation pass, so it
            cannot share detailed_test_report's plain no_grad loop).
    tent_lr / tent_steps: passed straight to TentTSFNet when
        tta_mode="tent" (see tent_adaptation.py's own documented defaults --
        lr=1e-3, steps=1 are the official method's own recommended settings,
        not a project-specific tune). Unused otherwise.
    """
    if ckpt_path is None:
        ckpt_path = f"_tsfnet_ckpt_{id(model)}.pt"

    # priority-D-1-safe (Safe-Capacity Protocol): the new branch's parameters
    # get a separate, heavier weight_decay. When eeg_band_mode='none' the model
    # has no eeg_band_branch, so this reduces EXACTLY to the original
    # single-group optimizer -- identical behavior, same object shape.
    #
    # priority-nodecay-scalars (BUG FIX, see below): torch.optim.Adam's
    # weight_decay is a direct L2 term added to every parameter's gradient
    # (grad += wd*param), including 1-D scalar/vector parameters like the
    # EFGF/EGTA/eeg_band gates, w_eeg/w_fusion/w_fnirs, BatchNorm/GroupNorm
    # affine weights, and pos_embed. For a gate parametrized as
    # sigmoid(raw) with raw initialized far from 0 (e.g. -6.0, so
    # sigmoid(-6)=0.0025), this L2 term is NOT a regularizer in the usual
    # sense -- it is a constant, data-INDEPENDENT force pulling raw toward 0
    # every single step. Verified numerically against the real
    # torch.optim.Adam (zero data-gradient, only weight_decay acting):
    #   step 1000: sigmoid(raw) = 0.0065
    #   step 3300: sigmoid(raw) = 0.0448
    #   step 6500: sigmoid(raw) = 0.2960
    # i.e. a gate can drift to ~0.30 open by the end of stage 2 from
    # weight_decay ALONE, with zero signal that the data wants it open. This
    # means gate values (and conclusions drawn from them, e.g. "EGTA learned
    # a non-uniform attention profile") are confounded by this artifact
    # whenever weight_decay > 0 and the gate is 1-D. Standard practice
    # (e.g. AdamW usage guides) is to exclude scalar/1-D parameters and
    # embeddings from weight decay entirely; that is what no_decay_scalars=True
    # does here. Default False reproduces the exact prior behavior (rule 5) --
    # this is a real fix, but as with efgf_gate_mode/normalize_ypred it is
    # gated behind a flag rather than silently changed, so every existing run
    # stays exactly reproducible.
    _band_params = list(model.eeg_band_branch.parameters()) if hasattr(model, "eeg_band_branch") else []
    _band_ids = {id(p) for p in _band_params}

    if no_decay_scalars:
        _no_decay_params = [p for n, p in model.named_parameters()
                            if id(p) not in _band_ids and (p.dim() <= 1 or n.endswith("pos_embed"))]
        _no_decay_ids = {id(p) for p in _no_decay_params}
        _decay_params = [p for p in model.parameters()
                         if id(p) not in _band_ids and id(p) not in _no_decay_ids]
        groups = [{"params": _decay_params, "weight_decay": weight_decay}]
        if _no_decay_params:
            groups.append({"params": _no_decay_params, "weight_decay": 0.0})
        if _band_params:
            groups.append({"params": _band_params, "weight_decay": eeg_band_weight_decay})
        optimizer = torch.optim.Adam(groups, lr=lr)
    elif _band_params:
        _other = [p for p in model.parameters() if id(p) not in _band_ids]
        optimizer = torch.optim.Adam([
            {"params": _other, "weight_decay": weight_decay},
            {"params": _band_params, "weight_decay": eeg_band_weight_decay},
        ], lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # priority-G1: None unless the caller asked for a schedule (see docstring).
    scheduler = make_warmup_cosine_scheduler(
        optimizer, warmup_epochs, stage1_max_epochs, min_lr_factor)

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_kappa": [],
               "stage": [], "gates": []}   # priority-diagnostics (additive only)

    # priority-SWA: see train_one_fold's docstring for the full rationale.
    # swa_enabled=False (default) means none of this block ever runs -- no
    # AveragedModel is constructed, no extra memory/compute, stage 1 behaves
    # exactly as before (rule 5).
    swa_model = None
    swa_scheduler = None
    _swa_updated = False
    if swa_enabled:
        # BUGFIX (found on a real run, seed42, tag a7346f6959): the original
        # default round(0.75*stage1_max_epochs) = round(0.75*100) = 75 was
        # NEVER reached -- across all 78 real folds, early stopping (patience
        # 20) fired at epoch 21-63 (mean 34.2), always below 75. SWA silently
        # never activated on a single fold. Default is now a small absolute
        # epoch instead, chosen below the observed minimum stop (21), so the
        # tail phase gets a real chance to run even on the earliest-stopping
        # folds. Still fully overridable via --swa-start-epoch.
        _swa_start = swa_start_epoch if swa_start_epoch is not None else 15
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer, swa_lr=(swa_lr if swa_lr is not None else lr))

    # --- Stage 1: train on the 4/5 split, early-stop on validation accuracy ---
    best_val_acc = -1.0
    best_epoch = 0   # priority-stage2fix: epoch number at which best_val_acc was achieved
    epochs_no_improve = 0
    stage1_final_train_loss = None
    stage1_best_val_train_loss = None

    for epoch in range(1, stage1_max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        n_correct = 0
        for _batch in train_loader:
            x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(_batch)
            x_eeg, x_fnirs, y = x_eeg.to(device), x_fnirs.to(device), y.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            # priority-noiseaware: TRAIN-time only. Applied here, after the
            # batch reaches the device and before the forward pass, so it
            # touches neither the validation loop nor detailed_test_report --
            # the reported accuracy therefore stays a CLEAN-data number.
            # NOTE: x_conn/x_chan are derived from EEG during preprocessing
            # and are NOT re-derived from the corrupted signal here; with
            # both disabled in the accepted configuration this is moot, but
            # it would need revisiting before combining with E1/channel modes.
            x_eeg = inject_eeg_awgn(x_eeg, noise_aware_snr_db, noise_aware_prob)
            optimizer.zero_grad()
            y_eeg, y_fusion, y_fnirs, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)
            loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux,
                                   lambda_eeg, lambda_fnirs, class_weight=class_weight, focal_gamma=focal_gamma,
                                   asymmetric_cost=asymmetric_cost,
                                   ordinal_reg_lambda=ordinal_reg_lambda, mhs_lambda=mhs_lambda, efgf_lambda=efgf_lambda, cafe_lambda=cafe_lambda, label_smoothing=label_smoothing,
                                   normalize_ypred=normalize_ypred)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
            n_correct += (y_pred.argmax(dim=-1) == y).sum().item()
        train_loss = epoch_loss / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)
        stage1_final_train_loss = train_loss

        val_loss, val_acc, val_kappa = evaluate(model, val_loader, device, lambda_eeg, lambda_fnirs,
                                                 class_weight=class_weight, focal_gamma=focal_gamma,
                                                 asymmetric_cost=asymmetric_cost,
                                                 ordinal_reg_lambda=ordinal_reg_lambda, mhs_lambda=mhs_lambda, efgf_lambda=efgf_lambda, cafe_lambda=cafe_lambda, label_smoothing=label_smoothing,
                                                 normalize_ypred=normalize_ypred)
        log_fn(f"  [stage1] epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
               f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_kappa={val_kappa:.4f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_kappa"].append(val_kappa)
        history["stage"].append(1)   # stage 2 computes no val metrics, so history is stage-1 only
        history["gates"].append(_collect_gate_values(model))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            stage1_best_val_train_loss = train_loss
            torch.save(model.state_dict(), ckpt_path)   # disk, not just memory -- crash-safe
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        # priority-SWA: once the tail phase starts, swa_scheduler (constant
        # swa_lr) takes over from whatever `scheduler` would have done this
        # epoch, and the averaged model accumulates this epoch's post-update
        # weights. Entirely skipped when swa_enabled=False (rule 5).
        if swa_enabled and epoch >= _swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            _swa_updated = True
        elif scheduler is not None:
            scheduler.step()   # priority-G1: once per completed epoch
        if epochs_no_improve >= stage1_patience:
            log_fn(f"  [stage1] early stop at epoch {epoch} (no improvement for {stage1_patience} epochs)")
            break

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        stage1_final_train_loss = evaluate(
            model, train_loader, device, lambda_eeg, lambda_fnirs,
            class_weight=class_weight, focal_gamma=focal_gamma,
            asymmetric_cost=asymmetric_cost, ordinal_reg_lambda=ordinal_reg_lambda,
            mhs_lambda=mhs_lambda, efgf_lambda=efgf_lambda, cafe_lambda=cafe_lambda, label_smoothing=label_smoothing,
            normalize_ypred=normalize_ypred)[0]

    # priority-SWA finalization: `model` above now holds stage 1's ORIGINAL
    # early-stopped best checkpoint (best_val_acc). If swa_enabled and the
    # averaging tail actually ran at least once, recalibrate the averaged
    # model's BatchNorm stats, evaluate it on the SAME val_loader, and keep
    # whichever is at least as good -- so swa_enabled=True can never make the
    # saved checkpoint worse than the swa_enabled=False baseline would have
    # produced (see docstring point 3-4).
    if swa_enabled and _swa_updated:
        _update_bn_multi_input(swa_model, train_loader, device)
        swa_val_loss, swa_val_acc, swa_val_kappa = evaluate(
            swa_model, val_loader, device, lambda_eeg, lambda_fnirs,
            class_weight=class_weight, focal_gamma=focal_gamma,
            asymmetric_cost=asymmetric_cost, ordinal_reg_lambda=ordinal_reg_lambda,
            mhs_lambda=mhs_lambda, efgf_lambda=efgf_lambda, cafe_lambda=cafe_lambda, label_smoothing=label_smoothing,
            normalize_ypred=normalize_ypred)
        log_fn(f"  [SWA] averaged-model val_acc={swa_val_acc:.4f} val_kappa={swa_val_kappa:.4f} "
               f"vs. original best_val_acc={best_val_acc:.4f}")
        if swa_val_acc >= best_val_acc:
            log_fn(f"  [SWA] averaged model is >= original -- using SWA weights for stage 2 / final checkpoint.")
            # AveragedModel wraps the real model as .module with an extra
            # n_averaged buffer; save only the underlying model's own state_dict
            # so ckpt_path's format matches every other checkpoint in this project.
            model.load_state_dict(swa_model.module.state_dict())
            torch.save(model.state_dict(), ckpt_path)
            best_val_acc = swa_val_acc
            stage1_final_train_loss = evaluate(
                model, train_loader, device, lambda_eeg, lambda_fnirs,
                class_weight=class_weight, focal_gamma=focal_gamma,
                asymmetric_cost=asymmetric_cost, ordinal_reg_lambda=ordinal_reg_lambda,
                mhs_lambda=mhs_lambda, efgf_lambda=efgf_lambda, cafe_lambda=cafe_lambda, label_smoothing=label_smoothing,
                normalize_ypred=normalize_ypred)[0]
        else:
            log_fn(f"  [SWA] original checkpoint is better -- SWA weights discarded, "
                   f"ckpt_path unchanged (fallback, not an error).")

    # (stage2_stop_mode is validated in run_subject_cross_session, fail-fast, rule 3)
    # priority-stage2fix (BUG, default UNCHANGED for rule 5): stage1_final_train_loss
    # is measured in EVAL mode (no dropout, via evaluate()); stage1_best_val_train_loss
    # is measured DURING stage 1's own training step (train mode, dropout on).
    # Eval-mode loss is essentially always lower, so min() picks it almost every
    # time -- and stage 2's own loss is measured in TRAIN mode against that
    # eval-mode-derived target, an apples-to-oranges comparison that makes the
    # target harder to reach than intended. With --val-mode cross_session this
    # gets WORSE: the target comes from training on ONE session, but stage 2
    # trains on TWO (train+val merged), so the loss scale differs further.
    # Confirmed on a real seed-42 XSVAL run (78 folds): mean stage-2 stopping
    # epoch nearly doubled (7.6 -> 15.3) vs the same model with val_mode='trial',
    # and 2/78 folds hit stage2_max_epochs outright (0/78 did under 'trial').
    # FIX (stage2_stop_mode='epoch_count'): skip the loss-target comparison
    # entirely and run stage 2 for exactly `best_epoch` epochs (the epoch stage 1
    # judged best on validation), on the full training set. This is the
    # standard "retrain for N epochs, N chosen by validation" recipe and keeps
    # stage 1's model-selection decision intact regardless of val_mode. Default
    # 'loss' reproduces the exact prior behavior bit-for-bit (rule 5).
    # priority-stage2fix (BUG, confirmed): model.load_state_dict(ckpt_path) two
    # lines above ALREADY restores stage 1's best checkpoint before this point
    # -- so stage 2 continuing for `best_epoch` MORE epochs from there trains
    # the model for ~2*best_epoch epochs total, not best_epoch, defeating the
    # "retrain for exactly N epochs, chosen by validation" logic epoch_count
    # was meant to implement. A full reset-and-retrain-from-scratch would fix
    # this exactly but roughly doubles wall-clock cost per fold; given this
    # project's current time budget, the cheaper, explicitly-approximate fix
    # is used instead: treat epoch_count as a SHORT fine-tune on top of the
    # already-converged stage-1 checkpoint (25% of best_epoch, matching the
    # externally-reviewed suggestion), still capped at stage2_max_epochs.
    # This is a documented approximation, not the statistically "correct"
    # full-reset version -- stated explicitly here and in the CLI help text.
    stage2_target = min(stage1_final_train_loss, stage1_best_val_train_loss) \
        if stage1_best_val_train_loss is not None else stage1_final_train_loss
    if stage2_stop_mode == "epoch_count":
        stage2_epoch_budget = min(max(round(0.25 * best_epoch), 1), stage2_max_epochs)
    elif stage2_stop_mode == "none":
        stage2_epoch_budget = 0   # for epoch in range(1, 1) never executes -- stage 2 skipped
    else:
        stage2_epoch_budget = stage2_max_epochs

    # --- Stage 2: retrain on the FULL training set (train+val), stop when
    # train loss drops below stage-1's target loss (Sec. 3.6) ---
    full_train_loader = DataLoader(
        torch.utils.data.ConcatDataset([train_loader.dataset, val_loader.dataset]),
        batch_size=train_loader.batch_size, shuffle=True)

    # priority-G1 / priority-stage2fix (BUG, confirmed): the lr reset was
    # previously only applied INSIDE "if scheduler2 is not None", so with no
    # scheduler (the common default), stage 2 silently inherited stage 1's
    # decayed lr and Adam's accumulated first/second moments from stage 1.
    # Both are now reset unconditionally before stage 2 begins, regardless of
    # whether a scheduler is used.
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.state.clear()   # drop stage 1's Adam moments (keeps the same dict object), once
    scheduler2 = make_warmup_cosine_scheduler(
        optimizer, warmup_epochs, stage2_epoch_budget, min_lr_factor)

    # ISOLATED ABLATION (priority-C1): if domain_adapt_lambda > 0.0, build a
    # cycling iterator over the held-out session's UNLABELED inputs, so
    # stage 2 can pull one target minibatch per source minibatch below.
    # domain_adapt_lambda=0.0 (default) never imports coral_domain_adaptation
    # at all (matching the same conditional-import pattern already used for
    # label_smoothing_loss.py and cbsi_transform.py), so a run that never
    # enables C-1 has no dependency on this file being present, and
    # reproduces prior behavior bit-for-bit.
    _target_iter = None
    if domain_adapt_lambda > 0.0:
        from coral_domain_adaptation import coral_loss, InfiniteTargetIterator
        assert target_loader is not None, (
            "domain_adapt_lambda > 0.0 requires target_loader (the held-out "
            "session's unlabeled DataLoader) to be passed to train_one_fold."
        )
        _target_iter = InfiniteTargetIterator(target_loader)

    if stage2_stop_mode == "none":
        log_fn(f"  [stage2] skipped (stage2_stop_mode='none') -- using stage 1's best "
              f"checkpoint (epoch {best_epoch}) directly as the final model.")
    for epoch in range(1, stage2_epoch_budget + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        n_correct = 0
        for _batch in full_train_loader:
            x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(_batch)
            x_eeg, x_fnirs, y = x_eeg.to(device), x_fnirs.to(device), y.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            # priority-noiseaware: same augmentation as stage 1, so the two
            # stages see the same input distribution. With
            # --stage2-stop-mode none this loop does not execute at all.
            x_eeg = inject_eeg_awgn(x_eeg, noise_aware_snr_db, noise_aware_prob)
            optimizer.zero_grad()
            y_eeg, y_fusion, y_fnirs, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)
            loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux,
                                   lambda_eeg, lambda_fnirs, class_weight=class_weight, focal_gamma=focal_gamma,
                                   asymmetric_cost=asymmetric_cost,
                                   ordinal_reg_lambda=ordinal_reg_lambda, mhs_lambda=mhs_lambda, efgf_lambda=efgf_lambda, cafe_lambda=cafe_lambda, label_smoothing=label_smoothing,
                                   normalize_ypred=normalize_ypred)
            if domain_adapt_lambda > 0.0:
                # ISOLATED ABLATION (priority-C1): one extra forward pass on an
                # UNLABELED minibatch from the held-out session (transductive,
                # in exactly the same sense as E3/E4 -- see coral_domain_
                # adaptation.py's docstring). Deliberately NOT wrapped in
                # torch.no_grad(): CORAL needs gradients through the target
                # pass too, so the network actively learns to produce aligned
                # fNIRS features for both domains, not just features that
                # happen to already match.
                xt_eeg, xt_fnirs, _, xt_conn, xt_chan = _unpack_batch(_target_iter.next_batch())
                xt_eeg, xt_fnirs = xt_eeg.to(device), xt_fnirs.to(device)
                xt_conn = xt_conn.to(device) if xt_conn is not None else None
                xt_chan = xt_chan.to(device) if xt_chan is not None else None
                _, _, _, _, aux_t = model(xt_eeg, xt_fnirs, xt_conn, xt_chan)
                coral = coral_loss(aux["enhanced_fnirs"], aux_t["enhanced_fnirs"])
                loss = loss + domain_adapt_lambda * coral
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
            n_correct += (y_pred.argmax(dim=-1) == y).sum().item()
        train_loss = epoch_loss / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)
        log_fn(f"  [stage2] epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
               f"(stage2 target={stage2_target:.4f})")
        if scheduler2 is not None:
            scheduler2.step()   # priority-G1: once per completed epoch
        if stage2_stop_mode == "epoch_count":
            continue   # run for the full stage2_epoch_budget, no early loss-based stop
        if train_loss < stage2_target:
            log_fn(f"  [stage2] stopping: train_loss below stage2 target")
            torch.save(model.state_dict(), ckpt_path)
            break
    else:
        torch.save(model.state_dict(), ckpt_path)
        if stage2_stop_mode == "epoch_count":
            log_fn(f"  [stage2] completed all {stage2_epoch_budget} epoch(s) "
                  f"(= 25% of stage-1's best_epoch, fine-tuned from that checkpoint; "
                  f"epoch_count mode, no loss target used).")
        elif stage2_stop_mode == "none":
            pass   # already logged above; the for-else here just re-saves the unchanged checkpoint
        else:
            log_fn(f"  [stage2] reached max epochs ({stage2_max_epochs}) without meeting loss target.")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    # priority-E4 / tta_mode: applied AFTER the final checkpoint is restored
    # and BEFORE any test-set evaluation, so (a) the on-disk checkpoint keeps
    # the unadapted, source-session statistics and stays reusable/comparable
    # regardless of tta_mode, and (b) every number in `report` below (test
    # acc/kappa, confusion matrix, per-branch ablations) reflects the adapted
    # model consistently. Default tta_mode=None/adabn=False skips this
    # entirely and reproduces prior behavior bit-for-bit.
    if tta_mode == "tent":
        if adabn:
            raise ValueError(
                "tta_mode='tent' and adabn=True both modify the SAME BatchNorm "
                "layers and cannot be combined in one run -- this project's "
                "isolated-variable protocol requires testing them separately. "
                "Set adabn=False to use tta_mode='tent'.")
        from tent_adaptation import tent_detailed_test_report
        import tent_adaptation as _tent_mod
        _tent_mod._unpack_batch_for_tent = _unpack_batch  # see tent_adaptation.py's
                                                            # INTEGRATION NOTE point 1
        report = tent_detailed_test_report(model, test_loader, device,
                                           lr=tent_lr, steps=tent_steps, log_fn=log_fn,
                                           cohen_kappa_fn=cohen_kappa,
                                           confusion_matrix_fn=confusion_matrix_np)
    else:
        if adabn:
            adapt_batchnorm_to_target(model, test_loader, device, log_fn=log_fn)
        report = detailed_test_report(model, test_loader, device)
    report["best_val_acc"] = best_val_acc
    report["final_train_acc"] = train_acc
    # Primary generalization-gap metric: best (leakage-free, trial-level) val_acc vs.
    # true held-out-session test_acc. Large gap = val is not a good proxy for a fully
    # unseen session. (final_train_acc - test_acc is NOT used as the primary metric:
    # stage 2 has no validation signal and is expected to fit train_acc very high by
    # design, so that gap is misleadingly large regardless of real generalization.)
    report["overfitting_gap_val_vs_test"] = (best_val_acc - report["test_accuracy_final"])
    report["history"] = history

    log_fn(f"  -> test_acc={report['test_accuracy_final']:.4f} test_kappa={report['kappa_final']:.4f} | "
           f"ablation: eeg={report['ablation']['eeg_only']:.3f} "
           f"fnirs={report['ablation']['fnirs_only']:.3f} fusion={report['ablation']['fusion_only']:.3f} | "
           f"weights: w_eeg={report['learned_weights']['w_eeg']:.3f} "
           f"w_fusion={report['learned_weights']['w_fusion']:.3f} "
           f"w_fnirs={report['learned_weights']['w_fnirs']:.3f}")

    return report["test_accuracy_final"], report["kappa_final"], model, report


def run_subject_cross_session(subject_dir, subject_id, device="cuda" if torch.cuda.is_available() else "cpu",
                               batch_size=16, run_diagnostics_flag=False, diagnostics_dir=None,
                               results_dir=None, fusion_combination="paper_mean",
                               class_weighting_scheme="none", effective_number_beta=0.999,
                               focal_gamma=2.0, oversampling_scheme="none", smote_k_neighbors=5,
                               eeg_temporal_mode="gap", eeg_lstm_hidden=64, eeg_readout_mode="h_n",
                               cafe_d_model=64, fnirs_num_segments=21,
                               asymmetric_hard_pairs=None, asymmetric_hard_weight=2.0,
                               ordinal_reg_lambda=0.0, class_count_level="window",
                               fnirs_chroma_mode="both", mhs_lambda=0.0, label_smoothing=0.0,
                               efgf_lambda=1.0, cafe_lambda=1.0,
                               domain_adapt_lambda=0.0,
                               use_fc_features=False, fc_num_channels=28, fc_embed_dim=64,
                               adabn=False, dropout=0.5, cafe_dropout=0.3,
                               conv_dropout=None, conv_dropout_type="channel",
                               fnirs_num_chromophores=2, eeg_spatial_mode="none",
                               eeg_spatial_layers=2, eeg_spatial_heads=4,
                               eeg_spatial_d_model=64,
                               eeg_backbone="native", tta_mode=None,
                               tent_lr=1e-3, tent_steps=1,
                               fnirs_temporal_align="none",
                               eeg_band_mode="none", eeg_band_d_model=32,
                               eeg_band_embed_dim=32, eeg_band_weight_decay=1e-2,
                               norm_mode="batch", norm_groups=8,
                               efgf_gate_mode="clamp",
                               normalize_ypred=False,
                               val_mode="trial",
                               no_decay_scalars=False,
                               stage2_stop_mode="loss", efgf_gate_init=-6.0,
                               **train_kwargs):
    """Sec. 3.6: Hold-Out over all 3 sessions, average of test accuracy/kappa

    cafe_d_model: ISOLATED ABLATION PARAMETER, priority-4 sub-experiment
    (deferred until after priority-5, per project decision). Default 64
    reproduces the accepted baseline exactly. The project's architecture-
    changes doc (item #13) describes reducing CAFE's cross-attention
    dimension from 256 (which matched the paper's ~2.77M-parameter CAFE
    config but severely overfit with only 26 subjects: 99% train acc vs.
    54% val acc) down to 128 as a deliberate capacity/overfitting trade-off.
    This experiment tests 128 against the project's ACTUAL current default
    of 64 (not against the paper's 256, which was already independently
    ruled out for the same overfitting reason before this ablation project
    began) -- i.e. it asks "is 64 too small, or was it already the right
    call?" Watch overfitting_gap_val_vs_test carefully, not just accuracy,
    per this same capacity-vs-overfitting logic.

    eeg_temporal_mode / eeg_lstm_hidden / eeg_readout_mode: ISOLATED ABLATION
    PARAMETERS, priority-5 sub-experiment (the project's largest architecture
    change, documented as "RUN 3A" / item #12 in the architecture-changes
    list). "gap" (default) reproduces the paper's exact method; "bilstm"
    replaces the EEG-only classification branch's Global Average Pooling
    with a bidirectional LSTM over the time axis -- see EEGLikeBranch's
    docstring for the full design and why this is isolated to the EEG
    branch alone (fNIRS, Fusion, EFGF, CAFE all remain untouched, confirmed
    by tracing TSFNet.forward: efgf1/efgf2 consume eeg_s1/eeg_s2, never
    eeg_pooled). eeg_readout_mode controls how the BiLSTM's per-time-step
    outputs are reduced to one feature vector: "h_n" (default, bug-fixed
    correct bidirectional summary), "last_step" (BUGGY, reproduces an
    already-run/in-progress priority-5 experiment for direct comparison
    only), or "mean_pool" (a second, independent sub-experiment). See
    EEGLikeBranch's docstring for the full explanation of the "last_step"
    bug this default replaces.
    across the 3 folds reported as the subject's final performance.

    oversampling_scheme: ISOLATED ABLATION PARAMETER, priority-3d
    sub-experiment -- a DATA-level axis, independent of class_weighting_scheme
    (a LOSS-level axis). One of:
      - "none" (default): no oversampling -- reproduces the accepted
        priority-1+2(+3b) baseline exactly.
      - "smote": generates synthetic minority-class training windows via
        smote_oversample_fold() (see that function's docstring for the
        multimodal-SMOTE design and its methodological caveats). Applied
        ONLY to this fold's training subset (train_idx, i.e. after the
        trial-level train/val split) -- never to val_idx or test_idx, so
        there is no leakage. Can be combined with class_weighting_scheme,
        but this project tests it in isolation first (class_weighting_scheme
        left at its accepted "effective_number" value, or "none", per
        whatever the caller passes) to keep this a single-variable ablation
        step, consistent with every other stage in this project.
    smote_k_neighbors: k for SMOTE's same-class nearest-neighbor search
    (Chawla et al. 2002 use k=5 as their default; reduced automatically
    per-class if a fold has fewer than k_neighbors+1 samples of that class).

    class_weighting_scheme: ISOLATED ABLATION PARAMETER (priority-3 family of
    sub-experiments). One of:
      - "none" (default): no class weighting -- reproduces the accepted
        priority-1+2 baseline exactly.
      - "inverse_freq": priority-3a, TESTED AND REJECTED (26 subjects x 3
        seeds: acc -1.17pp, kappa -0.0175 vs. baseline; confusion-matrix
        check on 13 subjects showed minority-class recall did NOT improve
        either, -0.043 for class1 / -0.020 for class2, while variance across
        subjects was very high, std=5.78pp, worst case -12.47pp). Formula:
        weight_c = N / (num_classes * n_c) (He & Garcia, 2009).
      - "effective_number": priority-3b, TESTED AND ACCEPTED (beta=0.9; 2
        seeds: acc +0.20pp, kappa +0.0031 vs. baseline, stable cross-seed).
        Cui et al. (2019), "Class-Balanced Loss Based on Effective Number of
        Samples", CVPR. Formula: weight_c = (1-beta) / (1 - beta**n_c), then
        renormalized so weights average to 1 across classes (same
        normalization convention as "inverse_freq", for a fair A/B
        comparison of the two weighting SHAPES rather than their absolute
        scale). Motivation: as beta -> 1, this asymptotically approaches
        inverse_freq's 1/n_c shape; for beta < 1 it damps the weight given
        to very-low-count classes, which is the exact failure mode
        (VP008 -12.47pp, high cross-subject variance) observed with plain
        inverse_freq on this small (26-subject, 2-session-per-fold) dataset.
      - "focal": priority-3c, NEW sub-experiment. Lin et al. (2017), "Focal
        Loss for Dense Object Detection", ICCV. Instead of a fixed per-CLASS
        weight, each SAMPLE's loss is scaled by (1-p_t)**focal_gamma, where
        p_t is the model's own predicted probability for that sample's true
        class -- easy/confident samples (p_t close to 1) get down-weighted
        automatically, regardless of which class they belong to. This is
        run WITHOUT any per-class alpha (class_weight stays None) so that
        this experiment isolates the (1-p_t)^gamma focusing effect alone,
        as a variable independent of the already-separately-tested 3a/3b
        per-class weighting axis.
      In all cases (except "focal", which is per-sample), the weight tensor
      is computed SEPARATELY FOR EACH FOLD from that fold's own train+val
      labels only (tv_labels, i.e. the 2 training sessions for this fold)
      -- never from the held-out test session, so there is no leakage. The
      same fold-specific tensor is used for both stage 1 and stage 2.

    focal_gamma: only used when class_weighting_scheme="focal". Lin et al.
    use gamma=2.0 as their main setting (also tested 0.5-5 in their ablation);
    2.0 is used here as the standard literature default.

    effective_number_beta: only used when class_weighting_scheme=
    "effective_number". Cui et al. report beta in {0.9, 0.99, 0.999, 0.9999}
    as reasonable across datasets; 0.999 is used here as a moderate default
    (closer to inverse_freq's shape than 0.9999, but still visibly damped
    relative to plain 1/n_c for the small per-fold sample sizes in this
    project, roughly 200-300 windows per fold).

    run_diagnostics_flag=True: after each fold's training finishes, calls
    model_diagnostics.run_diagnostics() on that fold's trained model/loaders
    (confusion matrices, PCA/t-SNE, saliency, etc.), saved under
    diagnostics_dir/S{session}_*.png. Requires model_diagnostics.py to be
    importable (same folder as this file, or on sys.path) and scikit-learn
    installed.

    diagnostics_dir (default): now nested UNDER results_dir (the variant+seed
    -tagged results folder, see below), NOT under subject_dir's parent
    anymore. Default is:
        <results_dir>/visualizations/<subject_id>/
    e.g. if results_dir = ".../TSFNET_PAPER_EXACT_nback/results_paper_mean_
    noclip_effective_numberbeta0.9_seed42", diagnostics PNGs default to
    ".../results_paper_mean_noclip_effective_numberbeta0.9_seed42/
    visualizations/VP001/". This was CHANGED from an earlier version that
    used a fixed "<parent of subject_dir>/visualizations/<subject_id>/" path
    shared by every run variant -- that fixed path meant enabling
    run_diagnostics_flag for a second ablation variant/seed would silently
    OVERWRITE the first variant's PCA/t-SNE/saliency plots, the exact same
    class of bug that previously affected cross_session_*_summary.json
    before results_dir was made variant-tagged. Pass diagnostics_dir
    explicitly to override.

    results_dir (new): where per-session results_{subject}_session{n}.json,
    the cross_session_{subject}_summary.json, and model checkpoints (.pt)
    get saved. Defaults to subject_dir if not given, but run_all_subjects.py
    now ALWAYS passes an explicit variant+seed-tagged folder (e.g.
    "results_paper_mean_noclip_effective_numberbeta0.9_seed42/") so that
    session-level results and diagnostics from different ablation
    variants/seeds are never mixed up or overwritten. Each results JSON
    contains: test accuracy/kappa, confusion matrix, per-branch ablation
    accuracy, learned Eq.4 fusion weights, best_val_acc, final_train_acc,
    overfitting gap, and full per-epoch history (for later learning-curve
    plots)."""
    if tta_mode == "tent" and adabn:
        raise ValueError(
            "tta_mode='tent' and adabn=True both modify the SAME BatchNorm "
            "layers and cannot be combined -- this project's isolated-variable "
            "protocol requires testing them separately. Set adabn=False to use "
            "tta_mode='tent'. (Failing here, before any data loading or GPU "
            "work, rather than deep inside train_one_fold.)")
    if tta_mode not in (None, "adabn", "tent"):
        raise ValueError(f"tta_mode must be None, 'adabn' or 'tent', got {tta_mode!r}")
    if fnirs_temporal_align not in ("none", "egta"):
        raise ValueError(f"fnirs_temporal_align must be 'none' or 'egta', "
                         f"got {fnirs_temporal_align!r}")
    if val_mode not in ("trial", "cross_session"):
        raise ValueError(f"val_mode must be 'trial' or 'cross_session', got {val_mode!r}")
    if stage2_stop_mode not in ("loss", "epoch_count", "none"):
        raise ValueError(f"stage2_stop_mode must be 'loss', 'epoch_count', or 'none', "
                         f"got {stage2_stop_mode!r}")
    # priority-GN (rule 3): GroupNorm replaces EVERY BatchNorm layer, and
    # AdaBN / TENT operate ONLY on BatchNorm layers. Combined, AdaBN would
    # log "no BatchNorm layers found -- nothing to adapt" and silently do
    # nothing, so a run labelled "+adabn" would not actually contain it --
    # exactly the banner-vs-reality mismatch that already invalidated three
    # runs in this project. Fail here, before any data loading or GPU work.
    if norm_mode == "group" and adabn:
        raise ValueError(
            "norm_mode='group' and adabn=True cannot be combined: GroupNorm removes all "
            "BatchNorm layers, so AdaBN would have nothing to adapt and would silently "
            "become a no-op. GroupNorm has no running statistics at all -- it is an "
            "ALTERNATIVE to BatchNorm+AdaBN for the same small-batch/session-shift "
            "problem, not an addition to it. Drop --adabn when using --norm-mode group.")
    if norm_mode == "group" and tta_mode == "tent":
        raise ValueError(
            "norm_mode='group' and tta_mode='tent' cannot be combined: TENT adapts only "
            "BatchNorm affine parameters, and GroupNorm leaves none.")
    for _name, _lam in (("efgf_lambda", efgf_lambda), ("cafe_lambda", cafe_lambda)):
        if not (isinstance(_lam, (int, float)) and _lam >= 0.0 and _lam == _lam):
            raise ValueError(f"{_name} must be a finite number >= 0, got {_lam!r} "
                             f"(priority-PCCLAMBDA; 1.0 = paper Eq. 5, 0.0 = term disabled)")
    full = NBackDataset(subject_dir, subject_id)
    session_ids = full.session_ids
    labels_np = full.labels.numpy()
    trial_ids = full.trial_ids

    # priority-C2 PERFORMANCE FIX (see labram_eeg_branch_v2.py's
    # "PRECOMPUTING EMBEDDINGS" docstring note): the frozen LaBraM
    # backbone's output for a given window never changes during training,
    # and is IDENTICAL across all 3 session-folds of this subject (the
    # backbone is frozen and fold-independent -- there is no leakage in
    # sharing it, since it uses only pretrained weights, never fold-
    # specific labels or splits). Precompute ONCE here, before the fold
    # loop, replacing full.channels in place with (N, embed_dim)
    # embeddings instead of raw (N, 28, real_n_times) windows. Every
    # downstream consumer (Subset, DataLoader, ConcatDataset for stage 2,
    # AdaBN) keeps working unmodified, since none of them inspect what
    # "channels" contains -- they only index into NBackDataset.__getitem__.
    # A run WITHOUT --eeg-backbone labram_frozen never imports
    # labram_eeg_branch_v2 here, exactly like every other conditional
    # import in this project (coral_domain_adaptation.py, cbsi_transform.py).
    if eeg_backbone in ("labram_frozen", "labram_ensemble", "labram_blend"):
        if not full.has_channels:
            raise ValueError(
                f"eeg_backbone='labram_frozen' requires "
                f"{{subject_id}}_eeg_channel_windows.npy, which is missing for "
                f"{subject_id} -- re-run preprocessing with "
                f"ENABLE_EEG_CHANNEL_SPACE=True (see Preproc_tsfnet_nback_badch.py).")
        from labram_eeg_branch_v3 import get_shared_frozen_head, precompute_embeddings
        import time as _time
        _t_embed0 = _time.time()
        print(f"  [eeg_backbone] precomputing frozen LaBraM embeddings ONCE for all "
              f"{len(full.channels)} windows of {subject_id} (reused, unmodified, "
              f"across all 3 session-folds below -- this replaces what would otherwise "
              f"be a full transformer forward pass on every training step of every "
              f"epoch of every fold; see labram_eeg_branch_v2.py's module docstring)...")
        # The frozen backbone is IDENTICAL for every subject, so it is built
        # exactly ONCE per process and reused -- see get_shared_frozen_head's
        # docstring. This avoids 25 redundant from_pretrained calls per seed
        # (each re-fitting the MNE montage, rebuilding the spline
        # interpolation matrix, re-reading the checkpoint, and re-touching the
        # HF Hub) and removes the repeated "unauthenticated requests" warning
        # and its rate-limit risk from the middle of a multi-hour run.
        _cache_head = get_shared_frozen_head(
            device=device, pretrained=True, num_classes=3, verbose=False)
        full.channels = precompute_embeddings(
            _cache_head, full.channels, batch_size=32, device=device)
        # NOTE: deliberately NOT deleted -- it is the shared singleton, kept
        # for the next subject. Its GPU memory footprint (~23MB of frozen
        # weights) is constant for the whole run rather than churning once
        # per subject, which is also what removes the allocator-fragmentation
        # growth that showed up as a steadily rising per-epoch cost
        # (14.5s -> 17.3s -> 26.7s across the first three subjects) before
        # this was fixed.
        if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        print(f"  [eeg_backbone] done in {_time.time() - _t_embed0:.1f}s -- "
              f"full.channels is now ({tuple(full.channels.shape)}) precomputed "
              f"embeddings, not raw channel-space EEG.")

    if results_dir is None:
        results_dir = subject_dir
    os.makedirs(results_dir, exist_ok=True)

    if run_diagnostics_flag and diagnostics_dir is None:
        # Tagged under results_dir (which is now the variant+seed-specific
        # folder passed in from run_all_subjects.py) rather than a fixed
        # path derived from subject_dir's parent -- otherwise, exactly like
        # the cross_session_*_summary.json issue fixed earlier, turning on
        # diagnostics for a second variant/seed would silently overwrite the
        # first variant's PCA/t-SNE/saliency plots.
        diagnostics_dir = os.path.join(results_dir, "visualizations", subject_id)

    fold_accs, fold_kappas, all_session_results = [], [], []
    for test_session in sorted(np.unique(session_ids)):
        print(f"\n=== {subject_id} | test session {test_session} ===")
        train_val_idx, test_idx = session_holdout_split(session_ids, test_session)

        tv_trial_ids = trial_ids[train_val_idx]
        tv_labels = labels_np[train_val_idx]
        if val_mode == "trial":
            # DEFAULT -- bit-exact prior behavior (rule 5).
            train_sub_idx, val_sub_idx = trial_level_train_val_split(tv_trial_ids, tv_labels)
        else:
            # priority-XSVAL: hold out the LOWER-numbered remaining session as
            # validation, deterministically (not seed-dependent), so the same
            # fold always yields the same split and folds stay comparable
            # across seeds -- the only randomness between seeds should be the
            # model init / batch order, never the data partition itself.
            tv_sessions = np.unique(session_ids[train_val_idx])
            val_session = int(tv_sessions.min())
            train_sub_idx, val_sub_idx = cross_session_train_val_split(
                session_ids[train_val_idx], val_session)
            print(f"  [val_mode] 'cross_session' -- validating on session "
                  f"{val_session}, training on {[int(s) for s in tv_sessions if s != val_session]} "
                  f"({len(train_sub_idx)} train / {len(val_sub_idx)} val windows). Validation is "
                  f"now cross-session, matching the test regime; training set is halved.")
        train_idx = train_val_idx[train_sub_idx]
        val_idx = train_val_idx[val_sub_idx]

        train_ds = Subset(full, train_idx)
        val_ds = Subset(full, val_idx)
        test_ds = Subset(full, test_idx)

        if oversampling_scheme == "smote":
            # Applied to train_idx ONLY (never val_idx/test_idx) -- pull the
            # real training windows out of `full` by index, oversample, then
            # replace train_ds with an in-memory dataset containing the
            # original + synthetic windows. val_ds/test_ds are untouched.
            eeg_aug, fnirs_aug, labels_aug = smote_oversample_fold(
                full.eeg[train_idx], full.fnirs[train_idx], full.labels[train_idx],
                k_neighbors=smote_k_neighbors,
                random_state=hash((subject_id, int(test_session))) % (2**31))
            train_ds = _InMemoryWindowDataset(eeg_aug, fnirs_aug, labels_aug)
        elif oversampling_scheme != "none":
            raise ValueError(f"Unknown oversampling_scheme: {oversampling_scheme!r} "
                              f"(expected 'none' or 'smote')")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        class_weight = None
        num_classes = 3
        # "focal" is intentionally EXCLUDED from this per-class-weight block:
        # priority-3c isolates the (1-p_t)^gamma per-SAMPLE focusing effect
        # alone (class_weight stays None, used as focal's alpha=uniform),
        # independent of the per-class weight axis already tested in 3a/3b.
        if class_weighting_scheme not in ("none", "focal"):
            if class_count_level == "window":
                class_counts = np.bincount(tv_labels, minlength=num_classes)
            elif class_count_level == "trial":
                # Verified (2026-08-31 audit) to be a no-op on this dataset:
                # trial-level counts are exactly [6,6,6] in every single one
                # of the 78 folds (26 subjects x 3 sessions), by fixed
                # experimental design (TRIALS_PER_SESSION=9, 3 trials/class/
                # session in preprocessing) -- not naturally-occurring
                # imbalance. Kept only so this axis can be reproduced/audited;
                # do not expect it to change any effective_number/inverse_freq
                # result relative to class_weighting_scheme="none".
                class_counts = np.array([
                    len(np.unique(tv_trial_ids[tv_labels == c]))
                    for c in range(num_classes)
                ])
            else:
                raise ValueError(f"Unknown class_count_level: {class_count_level!r} "
                                  f"(expected 'window' or 'trial')")
            if np.any(class_counts == 0):
                print(f"  WARNING [{subject_id} session {test_session}]: class(es) with zero "
                      f"training samples in this fold (counts={class_counts.tolist()}); "
                      f"using weight=0 for those classes' nll_loss term (no gradient signal "
                      f"from a class that never appears, rather than dividing by zero).")

            if class_weighting_scheme == "inverse_freq":
                # Priority-3a (He & Garcia, 2009). REJECTED -- kept only for
                # A/B reproducibility. weight_c = N / (num_classes * n_c);
                # classes seen less often get proportionally larger weight,
                # unbounded as n_c -> small (this unboundedness is exactly
                # what caused the instability observed in the 3a experiment).
                with np.errstate(divide="ignore"):
                    raw_weight = np.where(class_counts > 0,
                                           len(tv_labels) / (num_classes * np.maximum(class_counts, 1)),
                                           0.0)

            elif class_weighting_scheme == "effective_number":
                # Priority-3b (Cui et al., 2019, CVPR). "Effective number" of
                # samples for a class with n_c raw samples is
                # E_c = (1 - beta**n_c) / (1 - beta), which saturates (rather
                # than growing linearly with n_c) as n_c increases -- and,
                # critically, damps the weight assigned to LOW-count classes
                # compared to plain 1/n_c, because (1-beta**n_c) itself
                # shrinks for small n_c. Raw weight is the inverse of the
                # effective number: w_c = (1-beta) / (1-beta**n_c).
                # beta=0 recovers unweighted (E_c=n_c for all beta->0 limit,
                # i.e. w_c constant); beta->1 recovers plain inverse_freq's
                # 1/n_c shape. We use beta=effective_number_beta (default
                # 0.999) as a middle ground, motivated directly by 3a's
                # failure mode (VP008 -12.47pp, cross-subject std=5.78pp)
                # which plain inverse_freq's unbounded 1/n_c weighting is
                # believed to have caused.
                beta = effective_number_beta
                with np.errstate(divide="ignore", invalid="ignore"):
                    eff_num = np.where(class_counts > 0,
                                        (1.0 - beta ** np.maximum(class_counts, 1)) / (1.0 - beta),
                                        np.inf)  # zero-count class -> inf effective number -> weight 0 below
                    raw_weight = np.where(class_counts > 0, 1.0 / eff_num, 0.0)

            else:
                raise ValueError(f"Unknown class_weighting_scheme: {class_weighting_scheme!r} "
                                  f"(expected 'none', 'inverse_freq', 'effective_number', or 'focal')")

            # Renormalize so the SAMPLE-weighted average weight equals 1
            # (sum_c n_c * weight_c == n_present_samples) -- this is the
            # exact convention the original priority-3a inverse_freq formula
            # already satisfied (verified: counts=[150,60,50] ->
            # weights=[0.578,1.444,1.733], sum(count*weight)=260=N), so
            # applying the SAME convention here means inverse_freq's numbers
            # in this generalized function are bit-for-bit identical to the
            # already-analyzed priority-3a run, and effective_number is
            # normalized on a fair, comparable basis (same total loss mass),
            # isolating the WEIGHT SHAPE as the only difference between the
            # two schemes rather than an incidental scale difference.
            n_samples_present = class_counts.sum()
            weighted_mass = (class_counts * raw_weight).sum()
            if weighted_mass > 0:
                raw_weight = raw_weight * (n_samples_present / weighted_mass)
            class_weight = torch.tensor(raw_weight, dtype=torch.float32, device=device)
            print(f"  [class_weight:{class_weighting_scheme}|count_level={class_count_level}] "
                  f"fold test_session={test_session} | "
                  f"train+val class counts={class_counts.tolist()} | weights={raw_weight.round(3).tolist()}")

        if class_weighting_scheme == "focal":
            print(f"  [focal_loss] fold test_session={test_session} | gamma={focal_gamma} "
                  f"| alpha=uniform (class_weight=None, isolating the focusing term alone)")

        model = TSFNet(num_classes=3, fusion_combination=fusion_combination,
                        fnirs_temporal_align=fnirs_temporal_align,
                        eeg_temporal_mode=eeg_temporal_mode, eeg_lstm_hidden=eeg_lstm_hidden,
                        eeg_readout_mode=eeg_readout_mode, cafe_d_model=cafe_d_model,
                        fnirs_num_segments=fnirs_num_segments,
                        fnirs_chroma_mode=fnirs_chroma_mode,
                        use_fc_features=use_fc_features, fc_num_channels=fc_num_channels,
                        fc_embed_dim=fc_embed_dim,
                        dropout=dropout, cafe_dropout=cafe_dropout,
                        conv_dropout=conv_dropout, conv_dropout_type=conv_dropout_type,
                        fnirs_num_chromophores=fnirs_num_chromophores,
                        eeg_spatial_mode=eeg_spatial_mode,
                        eeg_spatial_layers=eeg_spatial_layers,
                        eeg_spatial_heads=eeg_spatial_heads,
                        eeg_spatial_d_model=eeg_spatial_d_model,
                        eeg_backbone_mode=eeg_backbone,
                        eeg_band_mode=eeg_band_mode,
                        eeg_band_d_model=eeg_band_d_model,
                        eeg_band_embed_dim=eeg_band_embed_dim,
                        norm_mode=norm_mode, norm_groups=norm_groups,
                        efgf_gate_mode=efgf_gate_mode,
                        efgf_gate_init=efgf_gate_init).to(device)
        ckpt_path = os.path.join(results_dir, f"{subject_id}_session{test_session}_best.pt")
        active_focal_gamma = focal_gamma if class_weighting_scheme == "focal" else None
        if fnirs_chroma_mode != "both" and test_session == sorted(np.unique(session_ids))[0]:
            print(f"  [fnirs_chroma] mode={fnirs_chroma_mode!r} -- masking applied to raw "
                  f"x_fnirs before FNIRSBranch (priority-A4 isolated variable)")
        if mhs_lambda > 0.0 and test_session == sorted(np.unique(session_ids))[0]:
            print(f"  [mhs_lambda] {mhs_lambda} -- CAFE multi-head HSIC diversity term active "
                  f"(priority-B1 isolated variable, MSTFDN-inspired)")
        if label_smoothing > 0.0 and test_session == sorted(np.unique(session_ids))[0]:
            print(f"  [label_smoothing] {label_smoothing} -- smoothed cross-entropy target "
                  f"active on all four loss terms (priority-B4 isolated variable)")
        if domain_adapt_lambda > 0.0 and test_session == sorted(np.unique(session_ids))[0]:
            print(f"  [domain_adapt_lambda] {domain_adapt_lambda} -- Deep CORAL alignment "
                  f"of enhanced_fnirs features against the held-out session's unlabeled "
                  f"inputs, active during stage 2 (priority-C1 isolated variable, "
                  f"transductive -- see thesis note alongside E3/E4)")
        if eeg_spatial_mode != "none" and test_session == sorted(np.unique(session_ids))[0]:
            print(f"  [eeg_spatial] mode={eeg_spatial_mode!r} layers={eeg_spatial_layers} "
                  f"heads={eeg_spatial_heads} -- channel-wise MHSA on the EEG branch "
                  f"(priority-G4 isolated variable)")
        if eeg_backbone != "native" and test_session == sorted(np.unique(session_ids))[0]:
            print(f"  [eeg_backbone] mode='labram_frozen' -- y_eeg computed by a frozen, "
                  f"pretrained LaBraM EEG foundation-model backbone + a light trainable "
                  f"linear head, consuming raw channel-space x_chan (priority-C2 isolated "
                  f"variable; NOT transductive -- no test-session data used during "
                  f"training, only the frozen backbone's pretrained weights and this "
                  f"fold's own labeled training windows for the head)")
        if adabn and test_session == sorted(np.unique(session_ids))[0]:
            print(f"  [adabn] True -- BatchNorm running stats will be recomputed on the "
                  f"held-out session's unlabeled inputs before testing "
                  f"(priority-E4 isolated variable, TRANSDUCTIVE -- see thesis note)")
        if fnirs_temporal_align == "egta" and test_session == sorted(np.unique(session_ids))[0]:
            print(f"  [fnirs_temporal_align] 'egta' -- the fNIRS decision path pools the "
                  f"{fnirs_num_segments} segments by EEG-anchored attention (STA-Net EGTA) "
                  f"instead of a uniform mean; zero-initialized keys => identical to the "
                  f"uniform mean at step 0, so any change is LEARNED")
        if tta_mode == "tent" and test_session == sorted(np.unique(session_ids))[0]:
            print(f"  [tta_mode] 'tent' -- BatchNorm affine params (weight/bias) will be "
                  f"adapted by entropy minimization on the held-out session's unlabeled "
                  f"inputs, one gradient step per test batch, lr={tent_lr}, "
                  f"steps={tent_steps} (further generalization of priority-E4, "
                  f"TRANSDUCTIVE -- see thesis note; mutually exclusive with adabn)")
        if use_fc_features and test_session == sorted(np.unique(session_ids))[0]:
            print(f"  [use_fc_features] True -- EEG functional-connectivity (PLV) embedding "
                  f"concatenated into clf_fusion input (priority-E1 isolated variable, "
                  f"fc_num_channels={fc_num_channels}, fc_embed_dim={fc_embed_dim})")
        if asymmetric_hard_pairs:
            _asym_cost = build_asymmetric_cost_matrix(
                num_classes=3, hard_pairs=asymmetric_hard_pairs,
                hard_weight=asymmetric_hard_weight, device=device)
            if test_session == sorted(np.unique(session_ids))[0]:
                print(f"  [asymmetric_cost] pairs={asymmetric_hard_pairs} "
                      f"weight={asymmetric_hard_weight} | cost matrix rows="
                      f"{_asym_cost.cpu().numpy().tolist()}")
        else:
            _asym_cost = None

        acc, kappa, trained_model, report = train_one_fold(
            model, train_loader, val_loader, test_loader, device,
            ckpt_path=ckpt_path, class_weight=class_weight,
            focal_gamma=active_focal_gamma, asymmetric_cost=_asym_cost,
            ordinal_reg_lambda=ordinal_reg_lambda, mhs_lambda=mhs_lambda, efgf_lambda=efgf_lambda, cafe_lambda=cafe_lambda,
            label_smoothing=label_smoothing,
            domain_adapt_lambda=domain_adapt_lambda, target_loader=test_loader,
            adabn=adabn, tta_mode=tta_mode, tent_lr=tent_lr, tent_steps=tent_steps,
            eeg_band_weight_decay=eeg_band_weight_decay,
            normalize_ypred=normalize_ypred,
            no_decay_scalars=no_decay_scalars,
            stage2_stop_mode=stage2_stop_mode,
            **train_kwargs)
        fold_accs.append(acc)
        fold_kappas.append(kappa)

        report["test_session"] = int(test_session)
        report["train_sessions"] = [int(s) for s in np.unique(session_ids) if s != test_session]

        # ---- behavioural / representational diagnostics (whitelist-gated) ----
        # Saves t-SNE of the fusion embedding, the train-vs-test overlay that
        # visualizes cross-session shift, confusion heatmap, confidence and
        # calibration, branch-agreement / oracle analysis, and the EGTA
        # temporal attention profile -- but ONLY for the handful of subjects
        # listed in model_diagnostics.DIAGNOSTIC_SUBJECTS, since doing it for
        # all 26 x 3 folds is slow and produces files nobody reads. Failures
        # are caught inside and never abort the run.
        try:
            from model_diagnostics import run_fold_diagnostics
            _diag = run_fold_diagnostics(
                trained_model, train_loader, test_loader, device,
                subject_id=subject_id, fold_idx=int(test_session),
                out_dir=results_dir, unpack_batch=_unpack_batch)
            if _diag is not None:
                report["diagnostics"] = _diag
        except ImportError:
            pass   # model_diagnostics.py not present -> silently skip

        results_path = os.path.join(results_dir, f"results_{subject_id}_session{test_session}.json")
        with open(results_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  results saved to: {results_path}")
        print(f"  checkpoint saved to: {ckpt_path}")
        all_session_results.append(report)

        if run_diagnostics_flag:
            # run_diagnostics is already defined earlier in this same file (CELL 7)
            # NOTE: train_loader here is the 4/5 stage-1 split, not the full
            # train+val set used in stage 2 -- fine for diagnostics purposes
            # (it's only used as a reference "train" distribution for
            # PCA/t-SNE/distribution-shift plots, not for further training).
            run_diagnostics(trained_model, train_loader, test_loader, device,
                             session_id=test_session, output_dir=diagnostics_dir)

    mean_acc, std_acc = float(np.mean(fold_accs)), float(np.std(fold_accs))
    mean_kappa, std_kappa = float(np.mean(fold_kappas)), float(np.std(fold_kappas))
    print(f"\n{subject_id} FINAL: acc={mean_acc:.4f}+-{std_acc:.4f}  "
          f"kappa={mean_kappa:.4f}+-{std_kappa:.4f}")

    # ------------------------------------------------------------------
    # Aggregate per-branch ("ablation") accuracy and the learned Eq.4 fusion
    # weights across the 3 session-folds, so they're available at the TOP
    # LEVEL of the summary (not just nested inside each session_results[i]),
    # matching how mean_acc/mean_kappa are already surfaced. This is what
    # feeds run_all_subjects.py's per-subject CSV/console listing below.
    eeg_accs = [r["ablation"]["eeg_only"] for r in all_session_results]
    fnirs_accs = [r["ablation"]["fnirs_only"] for r in all_session_results]
    fusion_accs = [r["ablation"]["fusion_only"] for r in all_session_results]
    w_eeg_vals = [r["learned_weights"]["w_eeg"] for r in all_session_results]
    w_fusion_vals = [r["learned_weights"]["w_fusion"] for r in all_session_results]
    w_fnirs_vals = [r["learned_weights"]["w_fnirs"] for r in all_session_results]

    print(f"{subject_id} BRANCH ACCURACY (mean over {len(all_session_results)} folds): "
          f"eeg={np.mean(eeg_accs):.4f}+-{np.std(eeg_accs):.4f}  "
          f"fnirs={np.mean(fnirs_accs):.4f}+-{np.std(fnirs_accs):.4f}  "
          f"fusion={np.mean(fusion_accs):.4f}+-{np.std(fusion_accs):.4f}")
    # The label must match what was actually computed: in "gated" mode these
    # are means of the PER-SAMPLE softmax weights, not Eq.4's sigmoid gates
    # (which are inert there). Calling them Eq.4 gates in the log would be
    # wrong in a way that is easy to copy into a results table unnoticed.
    _cvs = [r["learned_weights"].get("gate_max_cv") for r in all_session_results]
    _cvs = [c for c in _cvs if c is not None]
    if _cvs:
        _label = "mean per-sample softmax gate weights"
        _suffix = (f" | max CV across samples={max(_cvs):.4f}"
                   + ("" if max(_cvs) > 0.01 else
                      "  <-- WARNING: gate is effectively CONSTANT "
                      "(collapsed to static fusion with extra parameters)"))
    else:
        _label = "sigmoid-activated Eq.4 gates"
        _suffix = ""
    print(f"{subject_id} LEARNED FUSION WEIGHTS (mean over {len(all_session_results)} folds, "
          f"{_label}): "
          f"w_eeg={np.mean(w_eeg_vals):.4f}  w_fusion={np.mean(w_fusion_vals):.4f}  "
          f"w_fnirs={np.mean(w_fnirs_vals):.4f}{_suffix}")

    # Sum the per-fold confusion matrices into one subject-level matrix.
    # Rows = true class, columns = predicted class, consistent with
    # confusion_matrix_np(). Guard against a fold whose matrix is missing or
    # malformed rather than crashing the whole 26-subject sweep.
    cm_summed = np.zeros((3, 3), dtype=np.int64)
    for _r in all_session_results:
        _cm = np.asarray(_r.get("confusion_matrix_final", []), dtype=np.int64)
        if _cm.shape == (3, 3):
            cm_summed += _cm
    _row_totals = cm_summed.sum(axis=1)
    per_class_recall = [
        float(cm_summed[i, i] / _row_totals[i]) if _row_totals[i] > 0 else None
        for i in range(3)
    ]

    summary = {
        "subject_id": subject_id,
        "session_results": all_session_results,
        "final_average_accuracy": mean_acc,
        "final_std_accuracy": std_acc,
        "final_average_kappa": mean_kappa,
        "final_std_kappa": std_kappa,
        # NEW: aggregated per-branch ablation accuracy (mean+-std over the 3
        # session-folds) -- previously only available nested inside each
        # individual session_results[i]["ablation"], not summarized here.
        "branch_accuracy": {
            "eeg_only_mean": float(np.mean(eeg_accs)), "eeg_only_std": float(np.std(eeg_accs)),
            "fnirs_only_mean": float(np.mean(fnirs_accs)), "fnirs_only_std": float(np.std(fnirs_accs)),
            "fusion_only_mean": float(np.mean(fusion_accs)), "fusion_only_std": float(np.std(fusion_accs)),
        },
        # NEW: aggregated learned Eq.4 fusion-gate weights (mean over the 3
        # session-folds) -- the final classifier's learned per-branch weights
        # (sigmoid-activated w_eeg/w_fusion/w_fnirs combining the three
        # branch predictions into the final y_pred).
        "learned_fusion_weights": {
            "w_eeg_mean": float(np.mean(w_eeg_vals)),
            "w_fusion_mean": float(np.mean(w_fusion_vals)),
            "w_fnirs_mean": float(np.mean(w_fnirs_vals)),
        },
        # NEW: confusion matrix SUMMED over this subject's 3 session-folds,
        # plus the per-class recall derived from it. Previously the confusion
        # matrices existed only inside each individual session_results[i]
        # ["confusion_matrix_final"], so answering "which classes does this
        # model actually confuse?" required opening the JSON and adding them up
        # by hand -- which is exactly the question that matters most for this
        # 3-class n-back task, where 2-back vs 3-back is the known-hard pair
        # (Cao et al. report ~59% on that pair alone).
        "confusion_matrix_summed": cm_summed.tolist(),
        "per_class_recall": per_class_recall,
    }
    summary_path = os.path.join(results_dir, f"cross_session_{subject_id}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Full cross-session summary saved to: {summary_path}")

    return mean_acc, std_acc, mean_kappa, std_kappa, summary

# %% CELL 7 -- model_diagnostics.py (run_diagnostics and all its helper plots)
"""
model_diagnostics.py
================================
Diagnostic visualization for THIS PROJECT'S TSFNet implementation
(tsfnet_model/model.py -- 3D-CNN + EFGF + CAFE, 3-class 0-/2-/3-back).

Adapted from a diagnostics script written for a different (BiLSTM-based)
architecture -- retargeted here to match our model's actual interface:

  - EEG branch:    Conv3d x2 -> GAP(time) -> eeg_flat (512-dim: 4x4x32)
  - fNIRS branch:  Conv3d x2 (per segment) -> GAP(time) -> CAFE -> enhanced_fnirs (512-dim)
  - Fusion branch: Conv3d x2 + EFGF (2 stages) -> CAFE -> enhanced_fusion (512-dim)
  - Classifiers:   clf_eeg / clf_fnirs / clf_fusion = BranchClassifier
                       (Linear(512,128) -> ELU -> Dropout -> Linear(128,3)),
                   forward() of BranchClassifier applies softmax INTERNALLY
                   -- so model.clf_*.net(...) gives raw logits (net[-1] is
                   the last Linear, pre-softmax), while model.clf_*(...)
                   gives probabilities. This matters throughout below.
  - forward() returns a 5-tuple:
        (y_eeg, y_fusion, y_fnirs, y_pred, aux)
    where y_eeg/y_fusion/y_fnirs/y_pred are already SOFTMAX PROBABILITIES
    (not logits) -- unlike the original script this was adapted from, we
    must NOT re-apply softmax to y_pred, only to the per-branch raw logits
    grabbed via forward hooks.
  - aux dict contains: eeg_pcc1, phi_pcc1, eeg_pcc2, phi_pcc2,
    enhanced_fusion, enhanced_fnirs (used as the 512-dim feature vectors
    for PCA/t-SNE below, in place of the original script's 64-dim
    BiLSTM/CAFE read-outs).

Analyses performed (same 10 as the original script, LSTM-specific saliency
trick removed since our EEG branch is Conv3d, not an RNN):
  1.  Confusion matrix (normalized + raw) per branch
  2.  Per-class accuracy per branch
  3.  PCA of 512-dim features (train vs test)
  4.  t-SNE of 512-dim features (train vs test) + misclassified overlay
  5.  Train vs Test feature distribution shift (mean +/- std per dimension)
  6.  Per-class centroid distance matrix (class separability)
  7.  Logit confidence & entropy histograms (final branch)
  8.  Silhouette score & Davies-Bouldin index per feature space
  9.  Input gradient saliency (EEG) -- plain eval-mode backward (no RNN,
      so no cudnn training-mode workaround needed)
  10. Text summary report

HOW TO USE
----------
Call this right after evaluating a fold's test set in train_one_fold()
(tsfnet_model/train.py), or standalone:

    from model_diagnostics import run_diagnostics

    run_diagnostics(
        model        = model,             # trained TSFNet instance
        train_loader = train_loader,
        test_loader  = test_loader,
        device       = device,
        session_id   = test_session,
        output_dir   = os.path.join(data_dir, subject_id, "diagnostics"),
    )
"""

import os
import gc
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, silhouette_score,
                              davies_bouldin_score, classification_report)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CLASS_NAMES = ["0-back", "2-back", "3-back"]
COLORS = ["#2a78d6", "#e24b4a", "#1baf7a"]

# EEG window length used by Preprocessing_TSFNet_paper_nback_v2_EEG10s.py.
# UPDATED to match ablation item #1 (WINDOW_LEN_EEG: 3.0 -> 10.0 s, 600 -> 2000
# samples @ 200Hz). Only used to label the saliency plot's x-axis in seconds
# (the Conv3d/GAP architecture itself is shape-agnostic and needs no change --
# see note near the saliency-map computation below); update again if
# WINDOW_LEN_EEG changes in preprocessing.
EEG_WINDOW_SECONDS = 10.0


# ──────────────────────────────────────────────────────────────────────────
# 1. Feature extraction
#    Hooks on clf_eeg.net[0] / clf_fnirs.net[0] / clf_fusion.net[0]: since
#    BranchClassifier.net = Sequential(Linear(512,128), ELU, Dropout,
#    Linear(128,3)), inp[0] of the hook on net[0] is the 512-dim vector fed
#    into the first Linear -- i.e. eeg_flat / enhanced_fnirs / enhanced_fusion
#    exactly as computed in TSFNet.forward(). This is the correct feature
#    point for PCA/t-SNE (analogous to the original script's 64-dim
#    BiLSTM/CAFE read-out point).
# ──────────────────────────────────────────────────────────────────────────

def _extract_features(model, loader, device):
    """
    One forward pass over `loader`, collecting:
      feat_eeg, feat_fnirs, feat_fusion : each (N, 512)
      feat_combined                     : (N, 1536)
      probs                             : (N, 3) -- final y_pred (already softmax)
      labels                            : (N,)
    """
    model.eval()
    feat_store = {"eeg": [], "fnirs": [], "fusion": []}

    def _hook_factory(key):
        def _hook(module, inp, out):
            feat_store[key].append(inp[0].detach().cpu().numpy())
        return _hook

    h1 = model.clf_eeg.net[0].register_forward_hook(_hook_factory("eeg"))
    h2 = model.clf_fnirs.net[0].register_forward_hook(_hook_factory("fnirs"))
    h3 = model.clf_fusion.net[0].register_forward_hook(_hook_factory("fusion"))

    all_probs, all_labels = [], []
    with torch.no_grad():
        for eeg_b, nirs_b, lbl_b in loader:
            y_eeg, y_fusion, y_fnirs, y_pred, aux = model(eeg_b.to(device), nirs_b.to(device))
            all_probs.append(y_pred.cpu().numpy())   # already softmax probabilities
            all_labels.append(lbl_b.numpy())

    h1.remove(); h2.remove(); h3.remove()

    feat_eeg = np.concatenate(feat_store["eeg"], axis=0)
    feat_fnirs = np.concatenate(feat_store["fnirs"], axis=0)
    feat_fusion = np.concatenate(feat_store["fusion"], axis=0)
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    feat_combined = np.concatenate([feat_eeg, feat_fnirs, feat_fusion], axis=1)

    del feat_store, all_probs, all_labels
    gc.collect()

    return {
        "feat_eeg": feat_eeg, "feat_fnirs": feat_fnirs, "feat_fusion": feat_fusion,
        "feat_combined": feat_combined, "probs": probs, "labels": labels,
    }


# ──────────────────────────────────────────────────────────────────────────
# 2. Per-branch logits (separate pass -- hooks on the LAST Linear of each
#    head, giving raw pre-softmax logits; the "final" branch has no logit
#    layer of its own since y_pred is a weighted MEAN of already-softmaxed
#    branch probabilities (Eq. 4) -- so "final" below stores y_pred
#    directly, not a hooked logit tensor).
# ──────────────────────────────────────────────────────────────────────────

def _collect_branch_logits(model, loader, device):
    branch_logits = {"eeg": [], "fnirs": [], "fusion": [], "final": []}

    def _hook_logit(key):
        def _h(m, inp, out):
            branch_logits[key].append(out.detach().cpu())
        return _h

    # net[-1] = Linear(128,3), the final logit layer of each classifier head
    h1 = model.clf_eeg.net[-1].register_forward_hook(_hook_logit("eeg"))
    h2 = model.clf_fnirs.net[-1].register_forward_hook(_hook_logit("fnirs"))
    h3 = model.clf_fusion.net[-1].register_forward_hook(_hook_logit("fusion"))

    model.eval()
    with torch.no_grad():
        for eeg_b, nirs_b, _ in loader:
            y_eeg, y_fusion, y_fnirs, y_pred, aux = model(eeg_b.to(device), nirs_b.to(device))
            branch_logits["final"].append(y_pred.cpu())   # already probabilities, NOT logits

    h1.remove(); h2.remove(); h3.remove()
    return {k: torch.cat(v, dim=0).numpy() for k, v in branch_logits.items()}


# ──────────────────────────────────────────────────────────────────────────
# 3. Confusion matrix
# ──────────────────────────────────────────────────────────────────────────

def _plot_confusion(branch_logits, labels, session_id, output_dir):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    key_map = {"Final": "final", "EEG": "eeg", "fNIRS": "fnirs", "Fusion": "fusion"}

    for ax, name in zip(axes, ["Final", "EEG", "fNIRS", "Fusion"]):
        preds = np.argmax(branch_logits[key_map[name]], axis=1)  # argmax is valid for both
        cm = confusion_matrix(labels, preds)                     # logits and probabilities
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cmn, vmin=0, vmax=100, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name}  (acc={cm.diagonal().sum()/cm.sum()*100:.1f}%)", fontsize=10)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i,j]:.1f}%\n({cm[i,j]})",
                        ha="center", va="center",
                        color="white" if cmn[i,j] > 55 else "black", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Confusion Matrices -- Session {session_id}", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_01_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────────
# 4. Per-class accuracy per branch
# ──────────────────────────────────────────────────────────────────────────

def _plot_per_class_acc(branch_logits, labels, session_id, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    x, width = np.arange(3), 0.2
    branch_colors = ["#2a78d6", "#e24b4a", "#1baf7a", "#f5a623"]
    key_map = {"Final": "final", "EEG": "eeg", "fNIRS": "fnirs", "Fusion": "fusion"}

    for bi, (bname, bcolor) in enumerate(zip(["Final", "EEG", "fNIRS", "Fusion"], branch_colors)):
        preds = np.argmax(branch_logits[key_map[bname]], axis=1)
        accs = [(preds[labels == c] == c).mean() * 100 for c in range(3)]
        bars = ax.bar(x + bi * width, accs, width, label=bname, color=bcolor,
                       alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", fontsize=7.5)

    ax.set_xticks(x + 1.5 * width); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
    ax.axhline(33.3, ls="--", color="gray", lw=0.8, label="Chance")
    ax.legend(fontsize=9)
    ax.set_title(f"Per-class Accuracy per Branch -- Session {session_id}")
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_02_per_class_acc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────────
# 5. PCA
# ──────────────────────────────────────────────────────────────────────────

def _plot_pca(train_data, test_data, session_id, output_dir):
    for feat_key, title in [("feat_eeg", "EEG branch (512-dim, Conv3D GAP)"),
                             ("feat_fnirs", "fNIRS branch (512-dim, CAFE-enhanced)"),
                             ("feat_combined", "All branches (1536-dim)")]:
        X_tr, y_tr = train_data[feat_key], train_data["labels"]
        X_te, y_te = test_data[feat_key], test_data["labels"]

        scaler = StandardScaler().fit(X_tr)
        pca = PCA(n_components=2, random_state=42).fit(scaler.transform(X_tr))
        Z_tr = pca.transform(scaler.transform(X_tr))
        Z_te = pca.transform(scaler.transform(X_te))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (Z, y, split) in zip(axes, [(Z_tr, y_tr, "Train"), (Z_te, y_te, "Test")]):
            for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
                mask = y == c
                ax.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.5, color=col, label=cname)
            ax.set_title(f"PCA -- {split} | {title}")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax.legend(markerscale=3, fontsize=8)

        plt.suptitle(f"Session {session_id} -- PCA Feature Space", fontsize=12)
        plt.tight_layout()
        tag = feat_key.replace("feat_", "")
        path = os.path.join(output_dir, f"S{session_id}_03_pca_{tag}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────────
# 6. t-SNE + misclassified overlay
# ──────────────────────────────────────────────────────────────────────────

def _plot_tsne(test_data, session_id, output_dir, perplexity=30):
    for feat_key, title in [("feat_eeg", "EEG branch (Conv3D GAP)"),
                             ("feat_fnirs", "fNIRS branch (CAFE-enhanced)"),
                             ("feat_combined", "All branches")]:
        X = test_data[feat_key]
        y = test_data["labels"]
        pred = np.argmax(test_data["probs"], axis=1)

        X_s = StandardScaler().fit_transform(X)
        n_max = min(3000, len(X_s))
        idx = np.random.choice(len(X_s), n_max, replace=False)
        X_s, y, pred = X_s[idx], y[idx], pred[idx]

        # NOTE: with only ~270 windows/subject total (far fewer than the
        # 3000-sample cap above), perplexity=30 may be too large relative to
        # n_max for small folds -- t-SNE requires perplexity < n_samples;
        # this is auto-corrected below rather than left to error out.
        eff_perplexity = min(perplexity, max(5, n_max // 4))

        print(f"    t-SNE {feat_key} ({n_max} samples, perplexity={eff_perplexity})...",
              end="", flush=True)
        Z = TSNE(n_components=2, perplexity=eff_perplexity, random_state=42,
                 max_iter=1000).fit_transform(X_s)
        print(" done")

        wrong = pred != y
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            mask = y == c
            ax.scatter(Z[mask, 0], Z[mask, 1], s=7, alpha=0.45, color=col, label=cname)
        ax.set_title("Colour = true class"); ax.legend(markerscale=3, fontsize=9)

        ax = axes[1]
        ax.scatter(Z[~wrong, 0], Z[~wrong, 1], s=6, alpha=0.35, color="#1baf7a", label="Correct")
        ax.scatter(Z[wrong, 0], Z[wrong, 1], s=10, alpha=0.7, color="#e24b4a",
                   label="Misclassified", marker="x")
        ax.set_title(f"Correct vs Misclassified (error {wrong.mean()*100:.1f}%)")
        ax.legend(markerscale=3, fontsize=9)

        fig.suptitle(f"t-SNE -- Session {session_id} -- {title}", fontsize=12)
        plt.tight_layout()
        tag = feat_key.replace("feat_", "")
        path = os.path.join(output_dir, f"S{session_id}_04_tsne_{tag}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────────
# 7. Train vs Test distribution shift
# ──────────────────────────────────────────────────────────────────────────

def _plot_distribution_shift(train_data, test_data, session_id, output_dir):
    for feat_key, title in [("feat_eeg", "EEG branch (512-dim, Conv3D GAP)"),
                             ("feat_fnirs", "fNIRS branch (512-dim, CAFE-enhanced)")]:
        X_tr = train_data[feat_key]; X_te = test_data[feat_key]
        mu_tr, mu_te = X_tr.mean(0), X_te.mean(0)
        std_tr, std_te = X_tr.std(0), X_te.std(0)
        dims = np.arange(X_tr.shape[1])

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        ax = axes[0]
        ax.plot(dims, mu_tr, color="#2a78d6", lw=1.2, label="Train mean")
        ax.plot(dims, mu_te, color="#e24b4a", lw=1.2, label="Test mean")
        ax.fill_between(dims, mu_tr - std_tr, mu_tr + std_tr, color="#2a78d6", alpha=0.15)
        ax.fill_between(dims, mu_te - std_te, mu_te + std_te, color="#e24b4a", alpha=0.15)
        ax.set_xlabel("Feature dimension"); ax.set_ylabel("Activation")
        ax.set_title(f"Mean +/- Std -- {title}"); ax.legend()

        ax = axes[1]
        shift = np.abs(mu_tr - mu_te)
        ax.bar(dims, shift, color="#ef9f27", width=0.8)
        ax.set_xlabel("Feature dimension")
        ax.set_ylabel("|Train mean - Test mean|")
        ax.set_title("Distribution shift per dimension (high = domain shift)")
        top5 = np.argsort(shift)[-5:]
        for d in top5:
            ax.axvline(d, color="red", lw=0.8, ls="--")
        ax.text(0.98, 0.95, f"Top-5 shifted dims: {sorted(top5.tolist())}",
                transform=ax.transAxes, ha="right", fontsize=8, color="red")

        plt.suptitle(f"Session {session_id} -- Distribution Shift", fontsize=12)
        plt.tight_layout()
        tag = feat_key.replace("feat_", "")
        path = os.path.join(output_dir, f"S{session_id}_05_dist_shift_{tag}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────────
# 8. Class centroid distance matrix
# ──────────────────────────────────────────────────────────────────────────

def _plot_centroid_distance(train_data, test_data, session_id, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (data, split) in zip(axes, [(train_data, "Train"), (test_data, "Test")]):
        X = data["feat_combined"]; y = data["labels"]
        centroids = np.array([X[y == c].mean(0) for c in range(3)])
        dist = np.array([[np.linalg.norm(centroids[i] - centroids[j]) for j in range(3)]
                         for i in range(3)])
        im = ax.imshow(dist, cmap="YlOrRd")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{dist[i,j]:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{split} -- Centroid Distance (larger = more separable)")

    plt.suptitle(f"Session {session_id} -- Class Centroid Distance (1536-dim)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_06_centroid_dist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────────
# 9. Confidence & Entropy
#    NOTE (adaptation): test_data["probs"] is already softmax output
#    (y_pred from forward()) -- unlike the original script, we must NOT
#    re-apply torch.softmax here, or the distribution gets distorted.
# ──────────────────────────────────────────────────────────────────────────

def _plot_confidence(test_data, session_id, output_dir):
    probs = test_data["probs"]; labels = test_data["labels"]
    preds = np.argmax(probs, axis=1)
    correct = preds == labels

    max_prob = probs.max(axis=1)
    entropy = -(probs * np.log(probs + 1e-9)).sum(axis=1)
    max_ent = np.log(3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(max_prob[correct], bins=40, alpha=0.7, color="#1baf7a",
            label=f"Correct ({correct.sum()})", density=True)
    ax.hist(max_prob[~correct], bins=40, alpha=0.7, color="#e24b4a",
            label=f"Wrong ({(~correct).sum()})", density=True)
    ax.set_xlabel("Max softmax probability"); ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution (Final branch)"); ax.legend()
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)

    ax = axes[1]
    ax.hist(entropy[correct] / max_ent, bins=40, alpha=0.7, color="#1baf7a",
            label="Correct", density=True)
    ax.hist(entropy[~correct] / max_ent, bins=40, alpha=0.7, color="#e24b4a",
            label="Wrong", density=True)
    ax.set_xlabel("Normalised entropy (0=certain, 1=uniform)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Uncertainty (Entropy)"); ax.legend()

    plt.suptitle(f"Session {session_id} -- Confidence & Uncertainty", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_07_confidence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────────
# 10. Cluster metrics
# ──────────────────────────────────────────────────────────────────────────

def _compute_cluster_metrics(data, session_id):
    results = {}
    for feat_key in ["feat_eeg", "feat_fnirs", "feat_combined"]:
        X = data[feat_key]; y = data["labels"]
        n = min(2000, len(X))
        idx = np.random.choice(len(X), n, replace=False)
        Xs, ys = StandardScaler().fit_transform(X[idx]), y[idx]
        results[feat_key] = {
            "silhouette": silhouette_score(Xs, ys, metric="euclidean"),
            "davies_bouldin": davies_bouldin_score(Xs, ys),
        }
    return results


# ──────────────────────────────────────────────────────────────────────────
# 11. Input Gradient Saliency (EEG)
#
#     ADAPTATION: the original script's cudnn-RNN-backward-in-eval-mode
#     workaround (switching to model.train() with BN forced to eval) is NOT
#     NEEDED here -- our EEG branch is a plain Conv3d stack (EEGLikeBranch
#     in layers.py), which has no such cudnn restriction. Standard
#     model.eval() + requires_grad on the input works directly.
#
#     Time axis: our EEG window is EEG_WINDOW_SECONDS=10.0s (2000 samples
#     @200Hz, per Preprocessing_TSFNet_paper_nback_v2_EEG10s.py's
#     WINDOW_LEN_EEG) -- see EEG_WINDOW_SECONDS definition above.
#
#     Spatial axis: our EEG tensor is (B,16,16,2000) -- H,W,T (a spatially-
#     mapped grid, not a flat channel axis), so saliency is averaged over
#     BOTH spatial dims (H,W) to get a pure temporal profile, matching what
#     the original script's 1D "channel x time" saliency did conceptually.
# ──────────────────────────────────────────────────────────────────────────

def _plot_saliency(model, test_loader, device, session_id, output_dir, n_samples=200):
    model.eval()

    _grads = {c: [] for c in range(3)}
    collected = 0

    for eeg_b, nirs_b, lbl_b in test_loader:
        if collected >= n_samples:
            break
        eeg_b = eeg_b.to(device).float()
        nirs_b = nirs_b.to(device).float()
        lbl_np = lbl_b.numpy()
        eeg_b.requires_grad_(True)

        y_eeg, y_fusion, y_fnirs, y_pred, aux = model(eeg_b, nirs_b)

        for c in range(3):
            mask = (lbl_np == c)
            if mask.sum() == 0:
                continue
            if eeg_b.grad is not None:
                eeg_b.grad.zero_()
            model.zero_grad()
            # Use the FINAL prediction's class score (y_pred), matching the
            # original script's use of final_out for saliency.
            score = y_pred[mask, c].sum()
            score.backward(retain_graph=(c < 2))
            if eeg_b.grad is not None:
                grad = eeg_b.grad.detach().cpu().abs()   # (B, 16, 16, 600)
                _grads[c].append(grad[mask].mean(0))       # (16, 16, 600)

        eeg_b = eeg_b.detach()
        collected += len(lbl_np)

    if not any(_grads[c] for c in range(3)):
        print("  [saliency] no gradients collected -- skipped.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for c, ax in enumerate(axes):
        if not _grads[c]:
            ax.set_ylabel(f"|Gradient|\n{CLASS_NAMES[c]}", fontsize=9)
            continue
        g = torch.stack(_grads[c]).mean(0)          # (16, 16, 600)
        g_time = g.mean(dim=(0, 1)).numpy()          # average over H,W -> (600,)
        t = np.linspace(0, EEG_WINDOW_SECONDS, len(g_time))
        ax.plot(t, g_time, color=COLORS[c], lw=1.0)
        ax.fill_between(t, 0, g_time, color=COLORS[c], alpha=0.3)
        ax.set_ylabel(f"|Gradient|\n{CLASS_NAMES[c]}", fontsize=9)
        ax.set_xlim(0, EEG_WINDOW_SECONDS)

    axes[-1].set_xlabel("Time within window (seconds)")
    plt.suptitle(f"Session {session_id} -- Input Gradient Saliency (EEG)\n"
                 "(which time points drive the final prediction)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_08_saliency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────────
# 12. Text report
# ──────────────────────────────────────────────────────────────────────────

def _write_report(test_data, cluster_metrics, session_id, output_dir):
    probs = test_data["probs"]; labels = test_data["labels"]
    preds = np.argmax(probs, axis=1)

    lines = [
        f"DIAGNOSTIC REPORT -- Session {session_id}", "=" * 60,
        "\n[Classification Report -- Final Branch]",
        classification_report(labels, preds, target_names=CLASS_NAMES),
        "\n[Cluster Separability Metrics]",
        f"{'Feature':20s}  {'Silhouette':>12}  {'Davies-Bouldin':>15}",
        "-" * 52,
    ]
    for k, v in cluster_metrics.items():
        tag = k.replace("feat_", "")
        lines.append(f"{tag:20s}  {v['silhouette']:12.4f}  {v['davies_bouldin']:15.4f}")
    lines += [
        "\nNote: Silhouette in [-1,1] -- higher is better separable.",
        "      Davies-Bouldin >= 0    -- lower is better separable.",
        "\n[Model Architecture Notes]",
        "  EEG branch:   Conv3d x2 (Sec. 3.1 Table 1) -> GAP(time) -> 512-dim (4x4x32)",
        "  fNIRS branch: Conv3d x2 (per segment) -> GAP(time) -> CAFE-enhanced -> 512-dim",
        "  Fusion:       Conv3d x2 + EFGF (2 stages) -> CAFE-enhanced -> 512-dim",
        "  Classifier:   Linear(512->128) -> ELU -> Dropout -> Linear(128->3) -> softmax",
        "  Loss:         weighted CrossEntropy (Eq. 6) + EFGF PCC reg. (Eq. 12) "
        "+ CAFE PCC reg. (Eq. 13)",
    ]

    path = os.path.join(output_dir, f"S{session_id}_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [saved] {path}")
    print("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────

def run_diagnostics(model, train_loader, test_loader, device, session_id, output_dir):
    """
    Parameters
    ----------
    model        : trained TSFNet instance (tsfnet_model.model.TSFNet)
    train_loader : DataLoader for the training sessions (yields eeg,fnirs,label)
    test_loader  : DataLoader for the test session
    device       : torch.device or "cuda"/"cpu" string
    session_id   : int (1, 2, or 3) -- which session was held out as test
    output_dir   : str -- folder where all PNGs and the report are saved
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTICS -- Session {session_id}")
    print(f"{'='*60}")

    print("\n[1/9] Extracting features -- train set...")
    train_data = _extract_features(model, train_loader, device)

    print("[2/9] Extracting features -- test set...")
    test_data = _extract_features(model, test_loader, device)

    print("[3/9] Collecting per-branch logits...")
    branch_logits = _collect_branch_logits(model, test_loader, device)

    labels = test_data["labels"]

    print("[4/9] Confusion matrices...")
    _plot_confusion(branch_logits, labels, session_id, output_dir)

    print("[5/9] Per-class accuracy...")
    _plot_per_class_acc(branch_logits, labels, session_id, output_dir)

    print("[6/9] PCA...")
    _plot_pca(train_data, test_data, session_id, output_dir)

    print("[7/9] t-SNE (1-2 min)...")
    _plot_tsne(test_data, session_id, output_dir)

    print("[8/9] Distribution shift...")
    _plot_distribution_shift(train_data, test_data, session_id, output_dir)

    _plot_centroid_distance(train_data, test_data, session_id, output_dir)
    _plot_confidence(test_data, session_id, output_dir)

    print("[+] Cluster metrics...")
    cluster_metrics = _compute_cluster_metrics(test_data, session_id)

    print("[+] Gradient saliency (EEG)...")
    _plot_saliency(model, test_loader, device, session_id, output_dir)

    print("[+] Writing report...")
    _write_report(test_data, cluster_metrics, session_id, output_dir)

    del train_data, test_data, branch_logits, labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  All diagnostics saved -> {output_dir}\n")


# %% CELL 8 -- run it
# Edit these two paths/values and run:
#
# subject_dir = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\VP001"
# subject_id = "VP001"
#
# run_subject_cross_session(
#     subject_dir, subject_id,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     batch_size=16,
#     run_diagnostics_flag=True,          # set False to skip PCA/t-SNE/saliency plots
#     diagnostics_dir=None,               # defaults to <results_dir>/visualizations/<subject_id>/
#     results_dir=None,                   # defaults to subject_dir if not given; run_all_subjects.py
#                                          # always passes a variant+seed-tagged folder explicitly
# )
