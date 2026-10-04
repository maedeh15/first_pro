# ============================================================================
# TSFNet -- single-file, notebook-ready version (no relative imports).
# Combines layers.py + model.py + losses.py + dataset.py + train.py from the
# tsfnet_model package into one script, in dependency order, so it can be
# pasted into Jupyter cells (split at the "# %% CELL n" markers below, or
# just run the whole thing as one script/cell).
#
# See the original package's README.md for the full explanation of every
# adaptation made to the paper (3-class output, reduced model size, added
# weight decay, reduced stage1/stage2 epoch budgets).
# ============================================================================

# %% CELL 1 -- imports
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset



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


# %% CELL 6 -- train.py (evaluate, cohen_kappa, train_one_fold, run_subject_cross_session)
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
# ============================================================================
import argparse
import copy
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


def train_one_fold(model, train_loader, val_loader, test_loader, device,
                    lambda_eeg=0.2, lambda_fnirs=0.2, lr=1e-3, weight_decay=1e-4,
                    stage1_max_epochs=100, stage1_patience=20, stage2_max_epochs=80,
                    log_fn=print):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Stage 1: train on the 4/5 split, early-stop on validation accuracy ---
    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0
    stage1_final_train_loss = None

    for epoch in range(1, stage1_max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
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
        train_loss = epoch_loss / max(n_seen, 1)
        stage1_final_train_loss = train_loss

        val_loss, val_acc, val_kappa = evaluate(model, val_loader, device, lambda_eeg, lambda_fnirs)
        log_fn(f"  [stage1] epoch {epoch:3d} | train_loss={train_loss:.4f} | "
               f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_kappa={val_kappa:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= stage1_patience:
            log_fn(f"  [stage1] early stop at epoch {epoch} (no improvement for {stage1_patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        stage1_final_train_loss = evaluate(model, train_loader, device, lambda_eeg, lambda_fnirs)[0]

    # --- Stage 2: retrain on the FULL training set (train+val), stop when
    # train loss drops below stage-1's final train loss (Sec. 3.6) ---
    full_train_loader = DataLoader(
        torch.utils.data.ConcatDataset([train_loader.dataset, val_loader.dataset]),
        batch_size=train_loader.batch_size, shuffle=True)

    for epoch in range(1, stage2_max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
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
        train_loss = epoch_loss / max(n_seen, 1)
        log_fn(f"  [stage2] epoch {epoch:3d} | train_loss={train_loss:.4f} "
               f"(stage1 target={stage1_final_train_loss:.4f})")
        if train_loss < stage1_final_train_loss:
            log_fn(f"  [stage2] stopping: train_loss below stage1 target")
            break

    test_loss, test_acc, test_kappa = evaluate(model, test_loader, device, lambda_eeg, lambda_fnirs)
    return test_acc, test_kappa


def run_subject_cross_session(subject_dir, subject_id, device="cuda" if torch.cuda.is_available() else "cpu",
                               batch_size=16, **train_kwargs):
    """Sec. 3.6: Hold-Out over all 3 sessions, average of test accuracy/kappa
    across the 3 folds reported as the subject's final performance."""
    full = NBackDataset(subject_dir, subject_id)
    session_ids = full.session_ids
    labels_np = full.labels.numpy()
    trial_ids = full.trial_ids

    fold_accs, fold_kappas = [], []
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
        acc, kappa = train_one_fold(model, train_loader, val_loader, test_loader, device,
                                     **train_kwargs)
        print(f"  -> test_acc={acc:.4f} test_kappa={kappa:.4f}")
        fold_accs.append(acc)
        fold_kappas.append(kappa)

    mean_acc, std_acc = float(np.mean(fold_accs)), float(np.std(fold_accs))
    mean_kappa, std_kappa = float(np.mean(fold_kappas)), float(np.std(fold_kappas))
    print(f"\n{subject_id} FINAL: acc={mean_acc:.4f}+-{std_acc:.4f}  "
          f"kappa={mean_kappa:.4f}+-{std_kappa:.4f}")
    return mean_acc, std_acc, mean_kappa, std_kappa

# %% CELL 7 -- run it
# Edit these two paths/values and run:
#
# subject_dir = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\VP001"
# subject_id = "VP001"
#
# run_subject_cross_session(
#     subject_dir, subject_id,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     batch_size=16,
# )
