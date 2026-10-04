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
import copy
import json
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


class ConvBlock3D(nn.Module):
    """One 3D-conv layer as specified in Sec. 3.1: Conv3dSame -> ELU -> BN -> Dropout."""

    def __init__(self, in_ch, out_ch, kernel, stride, dropout):
        super().__init__()
        self.conv = Conv3dSame(in_ch, out_ch, kernel, stride)
        self.act = nn.ELU()
        self.bn = nn.BatchNorm3d(out_ch)
        self.drop = nn.Dropout3d(dropout)

    def forward(self, x):
        return self.drop(self.bn(self.act(self.conv(x))))


# ----------------------------------------------------------------------------
# EEG / Fusion branch (Table 1 "EEG" row; the Fusion branch reuses this exact
# architecture on the SAME input tensor X_eeg with its own separate weights,
# per Sec. 3.1: "since its input is entirely derived from the 3D EEG tensor,
# the same convolutional kernel parameters as in the EEG branch are adopted").
# ----------------------------------------------------------------------------
class EEGLikeBranch(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = ConvBlock3D(1, 16, kernel=(4, 4, 12), stride=(2, 2, 6), dropout=dropout)
        self.conv2 = ConvBlock3D(16, 32, kernel=(2, 2, 6), stride=(2, 2, 2), dropout=dropout)

    def forward(self, x_eeg):
        # x_eeg: (B, 16, 16, 600) -> add channel dim -> (B, 1, 16, 16, 600)
        # Conv3d expects (B, C, D, H, W); we map (H, W, T) -> (D=H, H=W, W=T)
        # consistently across the whole model, purely a labeling convention.
        x = x_eeg.unsqueeze(1)
        s1 = self.conv1(x)                 # (B, 16, 8, 8, 100)
        s2 = self.conv2(s1)                # (B, 32, 4, 4, 50)
        pooled = s2.mean(dim=-1)           # GAP over TIME only (Sec. 3.1) -> (B, 32, 4, 4)
        return s1, s2, pooled              # stage1/stage2 maps (for EFGF) + final F_eeg/F_fusion


# ----------------------------------------------------------------------------
# fNIRS branch (Table 1 "fNIRS" row). The 11-segment axis is processed as an
# extra batch dimension shared by one set of conv weights (a per-segment 3D
# CNN with tied weights across segments -- the only interpretation consistent
# with Table 1's fNIRS output size keeping "11 x ..." unchanged through both
# conv layers).
# ----------------------------------------------------------------------------
class FNIRSBranch(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = ConvBlock3D(2, 16, kernel=(4, 4, 6), stride=(2, 2, 2), dropout=dropout)
        self.conv2 = ConvBlock3D(16, 32, kernel=(2, 2, 3), stride=(2, 2, 2), dropout=dropout)

    def forward(self, x_fnirs):
        # x_fnirs: (B, 11, 16, 16, 30, 2) -> (B, Seg, H, W, T, Chrom)
        B, S, H, W, T, C = x_fnirs.shape
        x = x_fnirs.permute(0, 1, 5, 2, 3, 4).reshape(B * S, C, H, W, T)  # (B*S, 2, 16, 16, 30)
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
    def __init__(self, channels, eeg_kernel, fnirs_kernel):
        super().__init__()
        self.eeg_attn_conv = Conv3dSame(channels, 1, eeg_kernel, stride=(1, 1, 1))
        self.fnirs_attn_conv = Conv3dSame(channels, 1, fnirs_kernel, stride=(1, 1, 1))
        # Trainable scalars gamma, alpha in [0,1], "initialized to zero" (Sec. 3.6).
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, eeg_map, fnirs_map, fusion_map):
        """
        eeg_map:    (B, C, H, W, T)        -- EEG branch's stage-k conv output
        fnirs_map:  (B, S, C, H, W, Tf)    -- fNIRS branch's stage-k conv output
        fusion_map: (B, C, H, W, T)        -- Fusion branch's stage-k conv output
        returns:    fused (B, C, H, W, T), plus eeg_pooled/phi for PCC regularization
        """
        gamma = torch.clamp(self.gamma, 0.0, 1.0)
        alpha = torch.clamp(self.alpha, 0.0, 1.0)

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
# between the fusion features (a single global token) and the 11 fNIRS
# segment features (a length-11 token sequence with learnable positional
# encoding, since "only certain segments ... contain relevant information").
# Gated fusion combines the attended and original features (paper does not
# give the exact gating formula; a learnable per-branch scalar gate,
# consistent with EFGF's gamma/alpha style, is used here and documented).
# ----------------------------------------------------------------------------
class CAFELayer(nn.Module):
    def __init__(self, fusion_dim=4 * 4 * 32, fnirs_dim=4 * 4 * 32,
                 d_model=64, n_heads=4, n_segments=11, dropout=0.3):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.fnirs_dim = fnirs_dim
        self.d_model = d_model
        self.n_segments = n_segments

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

    def forward(self, fusion_feat, fnirs_feat):
        """
        fusion_feat: (B, 32, 4, 4)      -- Fusion branch's pooled F_fusion
        fnirs_feat:  (B, 11, 32, 4, 4)  -- fNIRS branch's pooled F_fnirs (per segment)
        returns: enhanced_fusion (B, fusion_dim), enhanced_fnirs (B, fnirs_dim)  [for PCC + classifier]
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

        fnirs_enhanced_full = self.fnirs_out_proj(fnirs_enhanced)      # (B, 11, 512)
        fnirs_enhanced_full = fnirs_enhanced_full.mean(dim=1)          # pool segments -> (B, 512)
        fusion_enhanced_full = self.fusion_out_proj(fusion_enhanced.squeeze(1))  # (B, 512)

        g_fusion = torch.sigmoid(self.gate_fusion)
        g_fnirs = torch.sigmoid(self.gate_fnirs)
        fnirs_flat_pooled = fnirs_flat.mean(dim=1)                     # original, pooled -> (B, 512)

        enhanced_fusion = g_fusion * fusion_enhanced_full + (1 - g_fusion) * fusion_flat
        enhanced_fnirs = g_fnirs * fnirs_enhanced_full + (1 - g_fnirs) * fnirs_flat_pooled

        return enhanced_fusion, enhanced_fnirs


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
    """

    def __init__(self, num_classes=3, dropout=0.5, cafe_dropout=0.3,
                 cafe_d_model=64, cafe_n_heads=4):
        super().__init__()
        self.eeg_branch = EEGLikeBranch(dropout=dropout)
        self.fusion_branch = EEGLikeBranch(dropout=dropout)   # separate weights, same architecture (Sec. 3.1)
        self.fnirs_branch = FNIRSBranch(dropout=dropout)

        # EFGF stage 1: operates on conv1 outputs (channels=16, EEG kernel matches Table1
        # "EFGF1 Conv1" = 2x2x6, fNIRS kernel matches "EFGF1 Conv2" = 2x2x3)
        self.efgf1 = EFGFLayer(channels=16, eeg_kernel=(2, 2, 6), fnirs_kernel=(2, 2, 3))
        # EFGF stage 2: operates on conv2 outputs (channels=32, both kernels = 2x2x3 per Table1)
        self.efgf2 = EFGFLayer(channels=32, eeg_kernel=(2, 2, 3), fnirs_kernel=(2, 2, 3))

        feat_dim = 4 * 4 * 32  # = 512, shape of F_fusion / F_fnirs-per-segment after GAP
        self.cafe = CAFELayer(fusion_dim=feat_dim, fnirs_dim=feat_dim,
                               d_model=cafe_d_model, n_heads=cafe_n_heads,
                               n_segments=11, dropout=cafe_dropout)

        eeg_feat_dim = 4 * 4 * 32  # F_eeg_flatten
        self.clf_eeg = BranchClassifier(eeg_feat_dim, num_classes, dropout=dropout)
        self.clf_fusion = BranchClassifier(feat_dim, num_classes, dropout=dropout)
        self.clf_fnirs = BranchClassifier(feat_dim, num_classes, dropout=dropout)

        # Decision-level fusion weights w1 (eeg), w2 (fusion), w3 (fnirs) -- Eq. 4,
        # obtained via sigmoid of trainable raw parameters.
        self.w_eeg = nn.Parameter(torch.zeros(1))
        self.w_fusion = nn.Parameter(torch.zeros(1))
        self.w_fnirs = nn.Parameter(torch.zeros(1))

    def forward(self, x_eeg, x_fnirs):
        """
        x_eeg:   (B, 16, 16, 600)
        x_fnirs: (B, 11, 16, 16, 30, 2)
        """
        eeg_s1, eeg_s2, eeg_pooled = self.eeg_branch(x_eeg)
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

        enhanced_fusion, enhanced_fnirs = self.cafe(fusion_pooled_final, fn_pooled)

        eeg_flat = eeg_pooled.reshape(eeg_pooled.shape[0], -1)     # (B, 512)
        y_eeg = self.clf_eeg(eeg_flat)
        y_fusion = self.clf_fusion(enhanced_fusion)
        y_fnirs = self.clf_fnirs(enhanced_fnirs)

        w1 = torch.sigmoid(self.w_eeg)
        w2 = torch.sigmoid(self.w_fusion)
        w3 = torch.sigmoid(self.w_fnirs)
        w_sum = w1 + w2 + w3 + 1e-8
        y_pred = (w1 * y_eeg + w2 * y_fusion + w3 * y_fnirs) / w_sum

        aux = {
            "eeg_pcc1": eeg_pcc1, "phi_pcc1": phi_pcc1,
            "eeg_pcc2": eeg_pcc2, "phi_pcc2": phi_pcc2,
            "enhanced_fusion": enhanced_fusion, "enhanced_fnirs": enhanced_fnirs,
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


def tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, labels, aux,
                 lambda_eeg=0.2, lambda_fnirs=0.2):
    """
    Full loss L = L_class + L_efgf + L_cafe (Eq. 5-6).
    y_*: softmax probabilities (B, num_classes) from the model's forward().
    labels: (B,) integer class labels in {0,1,2}.

    NOTE: y_* are already softmax probabilities (per Sec. 3.4's "followed by
    a softmax function"), so we use nll_loss on log(y) rather than
    cross_entropy (which expects raw logits) -- mathematically identical to
    the paper's Eqs. 7-10.
    """
    eps = 1e-8
    l_eeg = F.nll_loss(torch.log(y_eeg + eps), labels)
    l_fnirs = F.nll_loss(torch.log(y_fnirs + eps), labels)
    l_fusion = F.nll_loss(torch.log(y_fusion + eps), labels)
    l_pred = F.nll_loss(torch.log(y_pred + eps), labels)

    l_class = l_pred + lambda_eeg * l_eeg + lambda_fnirs * l_fnirs + l_fusion  # Eq. 6

    l_efgf = efgf_regularization_loss(aux["eeg_pcc1"], aux["phi_pcc1"],
                                       aux["eeg_pcc2"], aux["phi_pcc2"])
    l_cafe = cafe_regularization_loss(aux["enhanced_fnirs"], aux["enhanced_fusion"])

    total = l_class + l_efgf + l_cafe
    breakdown = {
        "total": total.item(), "l_class": l_class.item(), "l_pred": l_pred.item(),
        "l_eeg": l_eeg.item(), "l_fnirs": l_fnirs.item(), "l_fusion": l_fusion.item(),
        "l_efgf": l_efgf.item(), "l_cafe": l_cafe.item(),
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


class NBackDataset(Dataset):
    """Loads one subject's preprocessed windows and exposes them as a
    standard PyTorch Dataset. session_ids/trial_ids are kept alongside so
    the training script can build the paper's session-based hold-out split
    (Sec. 3.6) and the trial-level 4:1 train/val split within a fold."""

    def __init__(self, subject_dir, subject_id, indices=None):
        eeg = np.load(os.path.join(subject_dir, f"{subject_id}_eeg_windows.npy"))
        fnirs = np.load(os.path.join(subject_dir, f"{subject_id}_fnirs_windows.npy"))
        labels = np.load(os.path.join(subject_dir, f"{subject_id}_labels.npy"))
        session_ids = np.load(os.path.join(subject_dir, f"{subject_id}_session_ids.npy"))
        trial_ids = np.load(os.path.join(subject_dir, f"{subject_id}_trial_ids.npy"))

        if indices is not None:
            eeg, fnirs, labels = eeg[indices], fnirs[indices], labels[indices]
            session_ids, trial_ids = session_ids[indices], trial_ids[indices]

        self.eeg = torch.from_numpy(eeg).float()
        self.fnirs = torch.from_numpy(fnirs).float()
        self.labels = torch.from_numpy(labels).long()
        self.session_ids = session_ids
        self.trial_ids = trial_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.fnirs[idx], self.labels[idx]


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



def evaluate(model, loader, device, lambda_eeg, lambda_fnirs):
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for x_eeg, x_fnirs, y in loader:
            x_eeg, x_fnirs, y = x_eeg.to(device), x_fnirs.to(device), y.to(device)
            y_eeg, y_fusion, y_fnirs, y_pred, aux = model(x_eeg, x_fnirs)
            loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux,
                                   lambda_eeg, lambda_fnirs)
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
    with torch.no_grad():
        for x_eeg, x_fnirs, y in test_loader:
            x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
            y_eeg, y_fusion, y_fnirs, y_pred, aux = model(x_eeg, x_fnirs)
            all_true.append(y.numpy())
            all_final.append(y_pred.argmax(dim=-1).cpu().numpy())
            all_eeg.append(y_eeg.argmax(dim=-1).cpu().numpy())
            all_fnirs.append(y_fnirs.argmax(dim=-1).cpu().numpy())
            all_fusion.append(y_fusion.argmax(dim=-1).cpu().numpy())

    y_true = np.concatenate(all_true)
    y_final = np.concatenate(all_final)
    y_eeg = np.concatenate(all_eeg)
    y_fnirs = np.concatenate(all_fnirs)
    y_fusion = np.concatenate(all_fusion)

    test_acc = float((y_final == y_true).mean())
    kappa = cohen_kappa(y_true, y_final)
    cm = confusion_matrix_np(y_true, y_final, num_classes)

    with torch.no_grad():
        w_eeg = torch.sigmoid(model.w_eeg).item()
        w_fusion = torch.sigmoid(model.w_fusion).item()
        w_fnirs = torch.sigmoid(model.w_fnirs).item()

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
        "learned_weights": {"w_eeg": w_eeg, "w_fusion": w_fusion, "w_fnirs": w_fnirs},
    }


def train_one_fold(model, train_loader, val_loader, test_loader, device,
                    lambda_eeg=0.2, lambda_fnirs=0.2, lr=1e-3, weight_decay=1e-4,
                    stage1_max_epochs=100, stage1_patience=20, stage2_max_epochs=80,
                    ckpt_path=None, log_fn=print):
    """ckpt_path: if given, the best stage-1 checkpoint is saved to this path
    on disk (torch.save) every time val_acc improves, in addition to being
    kept in memory -- so it survives a crash during stage 2. If None, a
    temp path derived from id(model) is used (still crash-safe, just less
    convenient to find manually)."""
    if ckpt_path is None:
        ckpt_path = f"_tsfnet_ckpt_{id(model)}.pt"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_kappa": []}

    # --- Stage 1: train on the 4/5 split, early-stop on validation accuracy ---
    best_val_acc = -1.0
    epochs_no_improve = 0
    stage1_final_train_loss = None
    stage1_best_val_train_loss = None

    for epoch in range(1, stage1_max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        n_correct = 0
        for x_eeg, x_fnirs, y in train_loader:
            x_eeg, x_fnirs, y = x_eeg.to(device), x_fnirs.to(device), y.to(device)
            optimizer.zero_grad()
            y_eeg, y_fusion, y_fnirs, y_pred, aux = model(x_eeg, x_fnirs)
            loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux,
                                   lambda_eeg, lambda_fnirs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
            n_correct += (y_pred.argmax(dim=-1) == y).sum().item()
        train_loss = epoch_loss / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)
        stage1_final_train_loss = train_loss

        val_loss, val_acc, val_kappa = evaluate(model, val_loader, device, lambda_eeg, lambda_fnirs)
        log_fn(f"  [stage1] epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
               f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_kappa={val_kappa:.4f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_kappa"].append(val_kappa)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stage1_best_val_train_loss = train_loss
            torch.save(model.state_dict(), ckpt_path)   # disk, not just memory -- crash-safe
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= stage1_patience:
            log_fn(f"  [stage1] early stop at epoch {epoch} (no improvement for {stage1_patience} epochs)")
            break

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        stage1_final_train_loss = evaluate(model, train_loader, device, lambda_eeg, lambda_fnirs)[0]

    # Stage-2 target: the smaller of (final stage-1 train loss, train loss at
    # the best-val epoch) -- guards against stage 2 having nothing to do if
    # early stopping triggered on a noisy uptick in train_loss right before
    # stopping (same rationale as the reference notebook this is adapted from).
    stage2_target = min(stage1_final_train_loss, stage1_best_val_train_loss) \
        if stage1_best_val_train_loss is not None else stage1_final_train_loss

    # --- Stage 2: retrain on the FULL training set (train+val), stop when
    # train loss drops below stage-1's target loss (Sec. 3.6) ---
    full_train_loader = DataLoader(
        torch.utils.data.ConcatDataset([train_loader.dataset, val_loader.dataset]),
        batch_size=train_loader.batch_size, shuffle=True)

    for epoch in range(1, stage2_max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        n_correct = 0
        for x_eeg, x_fnirs, y in full_train_loader:
            x_eeg, x_fnirs, y = x_eeg.to(device), x_fnirs.to(device), y.to(device)
            optimizer.zero_grad()
            y_eeg, y_fusion, y_fnirs, y_pred, aux = model(x_eeg, x_fnirs)
            loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux,
                                   lambda_eeg, lambda_fnirs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
            n_correct += (y_pred.argmax(dim=-1) == y).sum().item()
        train_loss = epoch_loss / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)
        log_fn(f"  [stage2] epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
               f"(stage2 target={stage2_target:.4f})")
        if train_loss < stage2_target:
            log_fn(f"  [stage2] stopping: train_loss below stage2 target")
            torch.save(model.state_dict(), ckpt_path)
            break
    else:
        torch.save(model.state_dict(), ckpt_path)
        log_fn(f"  [stage2] reached max epochs ({stage2_max_epochs}) without meeting loss target.")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
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
                               results_dir=None, **train_kwargs):
    """Sec. 3.6: Hold-Out over all 3 sessions, average of test accuracy/kappa
    across the 3 folds reported as the subject's final performance.

    run_diagnostics_flag=True: after each fold's training finishes, calls
    model_diagnostics.run_diagnostics() on that fold's trained model/loaders
    (confusion matrices, PCA/t-SNE, saliency, etc.), saved under
    diagnostics_dir/S{session}_*.png. Requires model_diagnostics.py to be
    importable (same folder as this file, or on sys.path) and scikit-learn
    installed.

    diagnostics_dir (default): kept SEPARATE from subject_dir (the
    preprocessed-DATA folder for this subject -- .npy/.fif/.json), mirroring
    the same "visualizations OUT of the data folder" structure used by the
    preprocessing script. Default is a sibling folder:
        <parent of subject_dir>/visualizations/<subject_id>/
    e.g. if subject_dir = ".../TSFNET_PAPER_EXACT_nback/VP001", diagnostics
    PNGs default to ".../TSFNET_PAPER_EXACT_nback/visualizations/VP001/"
    (the SAME visualizations/ tree the preprocessing script already uses for
    its own QC plots) -- NOT ".../VP001/diagnostics/" as in an earlier
    version of this function, which nested plot images inside the data
    folder. Pass diagnostics_dir explicitly to override.

    results_dir (new): where per-session results_{subject}_session{n}.json,
    the cross_session_{subject}_summary.json, and model checkpoints (.pt)
    get saved. Defaults to subject_dir -- these are DATA (structured
    metrics / model weights), not images, so they stay alongside the .npy
    files by default, unlike diagnostics_dir above. Each results JSON
    contains: test accuracy/kappa, confusion matrix, per-branch ablation
    accuracy, learned Eq.4 fusion weights, best_val_acc, final_train_acc,
    overfitting gap, and full per-epoch history (for later learning-curve
    plots)."""
    full = NBackDataset(subject_dir, subject_id)
    session_ids = full.session_ids
    labels_np = full.labels.numpy()
    trial_ids = full.trial_ids

    if run_diagnostics_flag and diagnostics_dir is None:
        parent_dir = os.path.dirname(os.path.normpath(subject_dir))
        diagnostics_dir = os.path.join(parent_dir, "visualizations", subject_id)
    if results_dir is None:
        results_dir = subject_dir
    os.makedirs(results_dir, exist_ok=True)

    fold_accs, fold_kappas, all_session_results = [], [], []
    for test_session in sorted(np.unique(session_ids)):
        print(f"\n=== {subject_id} | test session {test_session} ===")
        train_val_idx, test_idx = session_holdout_split(session_ids, test_session)

        tv_trial_ids = trial_ids[train_val_idx]
        tv_labels = labels_np[train_val_idx]
        train_sub_idx, val_sub_idx = trial_level_train_val_split(tv_trial_ids, tv_labels)
        train_idx = train_val_idx[train_sub_idx]
        val_idx = train_val_idx[val_sub_idx]

        train_ds = Subset(full, train_idx)
        val_ds = Subset(full, val_idx)
        test_ds = Subset(full, test_idx)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = TSFNet(num_classes=3).to(device)
        ckpt_path = os.path.join(results_dir, f"{subject_id}_session{test_session}_best.pt")
        acc, kappa, trained_model, report = train_one_fold(
            model, train_loader, val_loader, test_loader, device,
            ckpt_path=ckpt_path, **train_kwargs)
        fold_accs.append(acc)
        fold_kappas.append(kappa)

        report["test_session"] = int(test_session)
        report["train_sessions"] = [int(s) for s in np.unique(session_ids) if s != test_session]
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
    print(f"{subject_id} LEARNED FUSION WEIGHTS (mean over {len(all_session_results)} folds, "
          f"sigmoid-activated Eq.4 gates): "
          f"w_eeg={np.mean(w_eeg_vals):.4f}  w_fusion={np.mean(w_fusion_vals):.4f}  "
          f"w_fnirs={np.mean(w_fnirs_vals):.4f}")

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
#     diagnostics_dir=None,               # defaults to <parent>/visualizations/<subject_id>/
#     results_dir=None,                   # defaults to subject_dir; JSON results + .pt checkpoints saved here
# )
