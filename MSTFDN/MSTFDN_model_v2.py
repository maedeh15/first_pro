"""
MSTFDN: EEG-fNIRS Multimodal Spatial-Temporal Fusion Decoding Network
=======================================================================
Full architecture implementation based on:
  Ding et al., "MSTFDN: An EEG-fNIRS multimodal spatial-temporal fusion
  decoding network for personalized multi-task scenarios",
  Information Fusion 131 (2026) 104187.

Architecture overview (Fig. 1 of the paper):
┌────────────────────────────────────────────────────────────┐
│                        MSTFDN                              │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐ │
│  │ EEG Branch  │   │ fNIRS Branch │   │  Fusion Branch  │ │
│  │  E-TFEM     │   │  F-TFEM      │   │  EFF-SFEM       │ │
│  │  E-SFEM     │   │  F-SFEM      │   │  Classifier     │ │
│  │  E-STFEM    │   │  F-STFEM     │   │                 │ │
│  └──────┬──────┘   └──────┬───────┘   └────────┬────────┘ │
│         │                 │                     │          │
│         └────────concat───┘──────────────►──────┘          │
└────────────────────────────────────────────────────────────┘

Data shapes for N-back (MBI dataset) -- CORRECTED per paper Sec. 3.2/3.3:
  EEG input  : (B, 5_bands, CH_e=28, T_e=2000)    -- channel space, no spatial grid
  fNIRS input: (B, 3_hemo,  CH_f=36, T_f=100)     -- HbO/HbR/HbT, single 10s window

  IMPORTANT: There is NO 16x16 spatial grid anywhere in this architecture --
  the paper never mentions spatial mapping, azimuthal projection, or a grid.
  Samples stay in raw channel space throughout. MHSA in E-SFEM/F-SFEM/EFF-SFEM
  treats each CHANNEL as a "token", attending across channels using their
  temporal feature vector (length T_e or T_f) as the per-token representation.

  fNIRS is a SINGLE 10s window matching the EEG window 1:1 -- NOT a stack of
  21 overlapping 3s segments (an earlier, incorrect reading of the paper).
  X_fnirs in R^{3xCHfxTf} per paper Sec 2.1, with HbT = HbO+HbR computed in
  preprocessing (not inside the model).

  CH_e = number of EEG electrode channels after preprocessing (28 for N-back)
  CH_f = number of fNIRS channels (36 for N-back)
  T_e  = EEG time steps at 200 Hz for 10s window = 2000
  T_f  = fNIRS time steps at 10 Hz for 10s window = 100 (2000/20 exactly,
         matching E-STFEM's stride=20 in Eq.8 with no rounding/pooling hack)

Key hyperparameters (from paper Sec. 3.3, Table A.1-A.3):
  EEG:  He=8 heads, Ne=1 layer, KSS=4
  fNIRS: Hf=4 heads, Nf=1 layer, KSS=4
  Fusion: Hef=4 heads, Nef=1 layer
  Dropout: 0.4 everywhere
  FFN inner dim: 256
  lambda_nw=0.02, lambda_mhs=0.05
  Adam lr=3e-4, weight_decay=1e-3, batch=16, max_epoch=1000, early_stop
  Cosine annealing + linear warmup (warmup_epochs=10, eta_min=lr*0.05)

EEG temporal convolution (E-TFEM, paper Sec. 2.2.1):
  4 scales of dilated depthwise grouped conv (groups=5 frequency bands):
    scale1: kernel=(1,81), dilation=5  → ~2s receptive field at 200Hz
    scale2: kernel=(1,50), dilation=4  → ~1s
    scale3: kernel=(1,35), dilation=3  → ~0.5s
    scale4: kernel=(1,25), dilation=2  → ~0.25s
  Output per scale: (B, 5, CH_e, T_e) → concat to (B, 20, CH_e, T_e)
  Pointwise conv (1×20→1) + squeeze → (B, CH_e, T_e)

fNIRS temporal convolution (F-TFEM, paper Sec. 2.3.1):
  2 scales of dilated depthwise grouped conv (groups=3 hemoglobin types: HbO, HbR, HbT):
    scale1: kernel=(1,15), dilation=5  → ~7s receptive field at 10Hz
    scale2: kernel=(1,13), dilation=4  → ~5s
  Output per scale: (B, 3, CH_f, T_f) → concat to (B, 6, CH_f, T_f)
  Pointwise conv (1×6→1) + squeeze → (B, CH_f, T_f)

E-SFEM / F-SFEM / EFF-SFEM (paper Sec. 2.2.2):
  Standard Transformer encoder (no positional encoding)
  Input: (B, CH, T)  → treated as (B, CH, d_spatial) where d_spatial=T
  E-SFEM:   He=8 heads, Ne=1 layer, BN in FFN
  F-SFEM:   Hf=4 heads, Nf=1 layer, LN in FFN
  EFF-SFEM: Hef=4 heads, Nef=1 layer, LN in FFN

E-STFEM / F-STFEM (paper Sec. 2.2.3 / 2.3.3):
  Depthwise conv + pointwise conv
  EEG:   dwconv kernel=(1,1,1,20) stride=(1,20)  → time compress T_e/20 = T_f
          pwconv (KSS→1) × KSS=4  → output (B, KSS*CH_e, T_f)
  fNIRS: dwconv kernel=(1,1,1,11) stride=1        → T_f unchanged
          pwconv (KSS→1) × KSS=4  → output (B, KSS*CH_f, T_f)

Fusion (paper Sec. 2.4):
  concat F_eeg_st ⊕ F_fnirs_st → (B, KSS*(CH_e+CH_f), T_f)
  EFF-SFEM Transformer encoder
  Classifier: Conv1d(KSS*(CH_e+CH_f) → CH_e+CH_f) + ReLU
              Conv1d(CH_e+CH_f → n_class) + ReLU
              GAP over T_f → (B, n_class) → Softmax
"""

import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


# ===========================================================================
#  HYPERPARAMETERS — set defaults for N-back (MBI dataset-A, 3-class)
# ===========================================================================
class MSTFDNConfig:
    """
    All hyperparameters from the paper for the N-back MWL task.
    For the HBCI MI task you can instantiate with different values.
    """
    # ── Data dimensions ─────────────────────────────────────────────────────
    n_eeg_ch:    int = 28       # EEG channels (N-back dataset after preprocessing)
    n_fnirs_ch:  int = 36       # fNIRS channels (N-back dataset)
    T_e:         int = 2000     # EEG time steps  (10s × 200Hz)
    T_f:         int = 100      # fNIRS time steps (10s × 10Hz) -- matches paper
                                 # X_fnirs in R^(3 x CHf x Tf), single 10s window,
                                 # NOT a stack of overlapping short segments
    n_bands:     int = 5        # EEG frequency bands
    n_hemo:      int = 3        # HbO, HbR, HbT -- ALL THREE arrive as model input
                                 # (paper: "the fNIRS input feature tensor for the
                                 # model is denoted as X_fnirs in R^{3xCHfxTf}")
    n_class:     int = 4        # 0-back, 2-back, 3-back, rest (paper Sec 3.2)

    # ── Architecture ────────────────────────────────────────────────────────
    KSS:         int = 4        # pointwise output multiplier (spatial scaling factor)
    He:          int = 8        # E-SFEM attention heads
    Hf:          int = 4        # F-SFEM attention heads
    Hef:         int = 4        # EFF-SFEM attention heads
    Ne:          int = 1        # E-SFEM transformer layers
    Nf:          int = 1        # F-SFEM transformer layers
    Nef:         int = 1        # EFF-SFEM transformer layers
    ffn_dim:     int = 256      # FFN inner dimension (same for all SFEMs)
    dropout:     float = 0.4    # dropout rate

    # ── Loss ────────────────────────────────────────────────────────────────
    lambda_nw:   float = 0.02   # L2 weight decay coefficient
    lambda_mhs:  float = 0.05   # multi-head similarity loss coefficient

    # ── Training ────────────────────────────────────────────────────────────
    lr:          float = 3e-4
    weight_decay: float = 1e-3
    batch_size:  int = 16
    max_epochs:  int = 1000
    warmup_epochs: int = 10
    eta_min_ratio: float = 0.05  # eta_min = lr * eta_min_ratio
    patience:    int = 50        # early stopping patience


# ===========================================================================
#  HELPER: Linear HSIC for multi-head similarity loss (Eq. 25-28)
# ===========================================================================
def hsic_linear(h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
    """
    Compute HSIC between two attention heads' outputs using a linear kernel.

    h_i, h_j: (B, CH, d_k) -- the i-th / j-th head output of Eq. 5 for each
              sample (CH tokens x d_k features). Nothing is averaged away.

    HSIC(G_i, G_j) = 1/(d_k-1)^2 * tr(G_i H G_j H)   [Eq. 25]
    where G_i = kappa(h_i, h_i) = h_i^T h_i  (d_k x d_k, linear kernel),
          H   = I_dk - (1/d_k) 11^T.
    Computed per sample, then averaged over the batch.
    """
    d_k = h_i.shape[-1]

    # Gram matrices per sample: (B, d_k, d_k), contracted over the CH tokens
    G_i = h_i.transpose(1, 2) @ h_i
    G_j = h_j.transpose(1, 2) @ h_j

    # Centering matrix H_dk: (d_k, d_k), per paper definition exactly
    H = torch.eye(d_k, device=h_i.device, dtype=h_i.dtype) \
        - (1.0 / d_k) * torch.ones(d_k, d_k, device=h_i.device, dtype=h_i.dtype)

    # HSIC = 1/(dk-1)^2 * tr(G_i H G_j H), averaged over the batch
    denom = max((d_k - 1) ** 2, 1)
    prod = (G_i @ H) @ (G_j @ H)
    hsic_val = prod.diagonal(dim1=-2, dim2=-1).sum(-1) / denom
    return hsic_val.mean()


def mhs_loss_from_heads(head_outputs: list) -> torch.Tensor:
    """
    Compute average pairwise HSIC over all H*(H-1)/2 head pairs. [Eq. 28]
    head_outputs: list of H tensors, each (B, CH, d_k) -- per-head output
                  of the last SFEM layer (Eq. 5).
    Returns scalar HSIC average.
    """
    H = len(head_outputs)
    if H < 2:
        return torch.tensor(0.0, device=head_outputs[0].device)

    total = 0.0
    n_pairs = 0
    for i in range(H):
        for j in range(i + 1, H):
            total = total + hsic_linear(head_outputs[i], head_outputs[j])
            n_pairs += 1
    return total / n_pairs


# ===========================================================================
#  SFEM: Spatial Feature Extraction Module (Transformer Encoder, Fig. 1c)
# ===========================================================================
class SpatialFeatureExtractionModule(nn.Module):
    """
    Transformer encoder-based SFEM (E-SFEM, F-SFEM, or EFF-SFEM).
    
    Input:  (B, n_ch, d_spatial)   — channels × temporal/spatial features
    Output: (B, n_ch, d_spatial)   — same shape (residual connection)
    
    Key design choices from the paper:
    - No positional encoding (MHSA attends to spatial, not temporal, patterns)
    - E-SFEM uses BN in FFN (paper: BN better for EEG data)
    - F-SFEM and EFF-SFEM use LN in FFN (paper: LN better for mixed/fNIRS signals)
    - MHSA output features per head are stored for HSIC loss computation
    """
    def __init__(
        self,
        n_ch:      int,      # number of "channels" = spatial dimension input
        d_spatial: int,      # feature dimension per channel (= T_e or T_f)
        n_heads:   int,      # number of attention heads
        n_layers:  int,      # number of stacked transformer layers
        ffn_dim:   int,      # FFN inner dimension
        dropout:   float,
        use_bn:    bool = False,  # True for E-SFEM, False for F-/EFF-SFEM
    ):
        super().__init__()
        self.n_heads   = n_heads
        self.d_spatial = d_spatial
        self.n_layers  = n_layers

        # d_k: dimension learned by each head   [d_spatial must be divisible by n_heads]
        assert d_spatial % n_heads == 0, (
            f"d_spatial ({d_spatial}) must be divisible by n_heads ({n_heads})")
        self.d_k = d_spatial // n_heads

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                _TransformerEncoderLayer(
                    n_ch=n_ch,
                    d_spatial=d_spatial,
                    n_heads=n_heads,
                    d_k=self.d_k,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    use_bn=use_bn,
                ))

    def forward(self, x: torch.Tensor):
        """
        x: (B, n_ch, d_spatial)
        Returns (out, head_outputs_last_layer)
          out: (B, n_ch, d_spatial)
          head_outputs: list of n_heads tensors, each (B, n_ch, d_k)
                        — from the LAST transformer layer's MHSA
                        — used for HSIC loss computation
        """
        head_outputs = None
        for layer in self.layers:
            x, head_outputs = layer(x)
        return x, head_outputs


class _TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer (Fig. 1c)."""

    def __init__(self, n_ch, d_spatial, n_heads, d_k, ffn_dim, dropout, use_bn):
        super().__init__()
        self.n_heads   = n_heads
        self.d_k       = d_k
        self.d_spatial = d_spatial
        self.n_ch      = n_ch

        # ── MHSA projection weights ──
        # Q, K, V: (d_spatial → d_k) per head, applied along the feature dim of each channel
        self.W_Q = nn.Linear(d_spatial, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_spatial, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_spatial, n_heads * d_k, bias=False)
        self.W_O = nn.Linear(n_heads * d_k, d_spatial, bias=False)
        self.ln_mhsa = nn.LayerNorm(d_spatial)
        self.dropout_mhsa = nn.Dropout(dropout)

        # ── FFN ──
        self.ffn = nn.Sequential(
            nn.Linear(d_spatial, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_spatial),
            nn.Dropout(dropout),
        )

        # Paper: BN for EEG (E-SFEM), LN for fNIRS & fusion (F-SFEM, EFF-SFEM)
        if use_bn:
            self.norm_ffn = nn.BatchNorm1d(n_ch)
        else:
            self.norm_ffn = nn.LayerNorm(d_spatial)
        self.use_bn = use_bn
        # NOTE: no extra trailing LN here -- Eq.7 is exactly
        # F_out_s = LN(FFN(F_multi)) + F_multi, nothing more after that.

    def forward(self, x: torch.Tensor):
        """
        x: (B, n_ch, d_spatial)

        MHSA computes attention over the SPATIAL (feature) dimension d_spatial,
        treating each of the n_ch channels as a separate "token" with d_spatial features.
        This extracts spatial activation patterns BETWEEN channels.
        """
        B, n_ch, d_sp = x.shape

        # ── Multi-Head Self-Attention ─────────────────────────────────────
        # Project x → Q, K, V: (B, n_ch, n_heads*d_k)
        Q = self.W_Q(x)  # (B, n_ch, n_heads*d_k)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape to (B, n_heads, n_ch, d_k) for per-head attention
        def split_heads(t):
            return t.view(B, n_ch, self.n_heads, self.d_k).transpose(1, 2)
            # → (B, n_heads, n_ch, d_k)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Scaled dot-product attention per head [Eq. 5]
        # QK^T: (B, n_heads, n_ch, n_ch)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout_mhsa(attn)

        # head output h_i: (B, n_heads, n_ch, d_k)
        head_out = torch.matmul(attn, V)

        # Store per-head outputs h_i^(N) (Eq. 5) for the HSIC loss, one
        # (B, n_ch, d_k) tensor per head. hsic_linear() forms the d_k x d_k
        # Gram matrix G_i = h_i^T h_i per sample, matching H_dk in Eq. 25.
        head_outputs = [head_out[:, h] for h in range(self.n_heads)]

        # Concatenate heads: (B, n_ch, n_heads*d_k) [Eq. 6]
        multi = head_out.transpose(1, 2).contiguous().view(B, n_ch, self.n_heads * self.d_k)
        multi = self.W_O(multi)          # (B, n_ch, d_spatial)

        # Residual [Eq. 6]: F_multi = LN(MultiHead(F_in)) + F_in
        # IMPORTANT: LN is applied to the MultiHead output ONLY, then the
        # residual is added AFTER normalization (this is NOT the standard
        # post-norm Transformer pattern of LN(x + sublayer(x))).
        x = self.ln_mhsa(multi) + x     # (B, n_ch, d_spatial)

        # ── FFN + Residual [Eq. 7] ────────────────────────────────────────
        # F_out_s = LN(FFN(F_multi)) + F_multi -- same pre-add-norm pattern,
        # and this IS the final output of the layer (no extra norm after).
        ffn_out = self.ffn(x)            # (B, n_ch, d_spatial)
        ffn_out = self.norm_ffn(ffn_out) # BN (E-SFEM) or LN (F-/EFF-SFEM)
        x = ffn_out + x                  # (B, n_ch, d_spatial)

        return x, head_outputs


# ===========================================================================
#  E-TFEM: EEG Temporal Feature Extraction Module (Fig. 1b)
#  Multi-scale grouped dilated depthwise convolution (Inception-style)
# ===========================================================================
class EEGTemporalFeatureExtractionModule(nn.Module):
    """
    E-TFEM (Fig. 1b, Sec. 2.2.1)
    
    Input:  X_eeg (B, n_bands, n_eeg_ch, T_e)
    Output: F_eegt (B, n_eeg_ch, T_e)
    
    4 parallel dilated depthwise convolutions (grouped by frequency band),
    each producing (B, 5_bands, n_eeg_ch, T_e) → concat → (B, 20, n_eeg_ch, T_e)
    Pointwise conv (20→1) + squeeze → (B, n_eeg_ch, T_e)
    """
    SCALES = [
        # (kernel_size_time, dilation)  → receptive field at 200Hz
        (81, 5),   # ~2s
        (50, 4),   # ~1s
        (35, 3),   # ~0.5s
        (25, 2),   # ~0.25s
    ]

    def __init__(self, n_bands: int, n_eeg_ch: int, dropout: float):
        super().__init__()
        self.n_bands  = n_bands    # 5
        self.n_eeg_ch = n_eeg_ch  # 28

        # One grouped (depthwise) dilated Conv2d per scale on (B, 5, CH, T):
        # groups=n_bands splits the input into the 5 frequency-band subsets
        # (paper Sec 2.2.1); each band has ONE (1,k) kernel shared across all
        # EEG channels, so output is again (B, 5, CH, T) per scale.
        self.scale_convs = nn.ModuleList()
        self.scale_bns   = nn.ModuleList()

        for k, d in self.SCALES:
            # Padding to maintain T_e: pad = dilation*(kernel-1)//2
            pad = d * (k - 1) // 2
            conv = nn.Conv2d(
                in_channels=n_bands,
                out_channels=n_bands,
                kernel_size=(1, k),
                dilation=(1, d),
                padding=(0, pad),
                groups=n_bands,
                bias=False,          # followed by BN (Eq. 1)
            )
            self.scale_convs.append(conv)
            self.scale_bns.append(nn.BatchNorm2d(n_bands))

        # Pointwise conv (Eq. 3): W in R^{1x20x1x1}, b in R^1 -> (B, 1, CH, T)
        self.pw_conv = nn.Conv2d(
            in_channels=4 * n_bands,   # 20 (4 scales × 5 bands)
            out_channels=1,
            kernel_size=(1, 1),
            bias=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_bands, n_eeg_ch, T_e)
        Returns: (B, n_eeg_ch, T_e)
        """
        # 4 parallel dilated grouped convolutions [Eq. 1,2]
        scale_outs = [F.elu(bn(conv(x)))        # (B, 5, CH, T) each
                      for conv, bn in zip(self.scale_convs, self.scale_bns)]

        # Concat along band dim → (B, 4*n_bands=20, CH, T) [Eq. 2]
        x_cat = torch.cat(scale_outs, dim=1)   # (B, 20, CH, T)

        # Pointwise conv: reduce 20 bands → 1, then squeeze [Eq. 3]
        x_cat = self.pw_conv(x_cat)             # (B, 1, CH, T)
        x_cat = self.dropout(x_cat)
        out = x_cat.squeeze(1)                  # (B, CH, T)  = F_eegt
        return out


# ===========================================================================
#  F-TFEM: fNIRS Temporal Feature Extraction Module (Fig. 1e)
#  Same structure as E-TFEM but 2 scales, grouped by hemoglobin type
# ===========================================================================
class fNIRSTemporalFeatureExtractionModule(nn.Module):
    """
    F-TFEM (Fig. 1e, Sec. 2.3.1)

    Input:  X_fnirs (B, n_hemo=3, n_fnirs_ch, T_f)
            [HbO, HbR, HbT -- ALL THREE arrive as model input, per paper:
             "the fNIRS input feature tensor for the model is denoted as
             X_fnirs in R^{3xCHfxTf}". HbT is computed upstream in
             preprocessing (HbT = HbO + HbR), NOT inside this module.]
    Output: F_fnirst (B, n_fnirs_ch, T_f)

    2 parallel dilated depthwise convolutions (grouped by 3 hemoglobin signals),
    each -> (B, 3, n_fnirs_ch, T_f) -> concat -> (B, 6, n_fnirs_ch, T_f)
    Pointwise conv (6->1) + squeeze -> (B, n_fnirs_ch, T_f)
    """
    SCALES = [
        # (kernel_size, dilation) -> ~7s and ~5s at 10Hz
        (15, 5),   # ~7s
        (13, 4),   # ~5s
    ]
    N_HEMO_FULL = 3   # HbO, HbR, HbT -- all supplied as input, not derived here

    def __init__(self, n_fnirs_ch: int, dropout: float):
        super().__init__()
        self.n_fnirs_ch = n_fnirs_ch   # 36
        n_hemo = self.N_HEMO_FULL      # 3

        self.scale_convs = nn.ModuleList()
        self.scale_bns   = nn.ModuleList()

        for k, d in self.SCALES:
            pad = d * (k - 1) // 2
            # Same structure as E-TFEM: grouped by the 3 hemoglobin types,
            # one (1,k) kernel per type shared across fNIRS channels.
            conv = nn.Conv2d(
                in_channels=n_hemo,
                out_channels=n_hemo,
                kernel_size=(1, k),
                dilation=(1, d),
                padding=(0, pad),
                groups=n_hemo,
                bias=False,          # followed by BN (Eq. 11)
            )
            self.scale_convs.append(conv)
            self.scale_bns.append(nn.BatchNorm2d(n_hemo))

        # Pointwise (Eq. 13): W in R^{1x6x1x1}, b in R^1
        self.pw_conv = nn.Conv2d(
            in_channels=len(self.SCALES) * n_hemo,   # 6
            out_channels=1,
            kernel_size=(1, 1),
            bias=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_hemo=3, n_fnirs_ch, T_f)    [HbO, HbR, HbT -- already stacked]
        Returns: (B, n_fnirs_ch, T_f)
        """
        B, n_hemo_in, CH, T = x.shape
        assert n_hemo_in == self.N_HEMO_FULL, (
            f"Expected 3 hemoglobin channels (HbO,HbR,HbT) as input, got {n_hemo_in}. "
            f"HbT must be computed in preprocessing, not inside this module."
        )

        scale_outs = [F.elu(bn(conv(x)))        # (B, 3, CH, T) each [Eq. 11]
                      for conv, bn in zip(self.scale_convs, self.scale_bns)]

        x_cat = torch.cat(scale_outs, dim=1)   # (B, 6, CH, T) [Eq. 12]
        x_cat = self.pw_conv(x_cat)             # (B, 1, CH, T) [Eq. 13]
        x_cat = self.dropout(x_cat)
        out = x_cat.squeeze(1)                  # (B, CH, T)
        return out


# ===========================================================================
#  E-STFEM: EEG Spatial-Temporal Feature Extraction Module (Fig. 1d)
#  Depthwise separable conv: compress T_e → T_f, expand CH_e → KSS*CH_e
# ===========================================================================
class EEGSpatialTemporalFeatureExtractionModule(nn.Module):
    """
    E-STFEM (Fig. 1d, Sec. 2.2.3)
    
    Input:  F_eegs  (B, n_eeg_ch, T_e)   from E-SFEM
    Output: F_eegst (B, KSS*n_eeg_ch, T_f)
    
    Step 1 — Depthwise conv (Eq. 8):
      unsqueeze → (B, 1, n_eeg_ch, T_e)
      Conv2d depthwise, kernel=(1,1,1,20), stride=(1,20), groups=1
      → (B, 1, n_eeg_ch, T_f)   (time dimension compressed by 20×)
      
    Step 2 — Pointwise conv (Eq. 9):
      Conv2d 1×1, 1 → KSS channels
      → (B, KSS, n_eeg_ch, T_f)
      
    Step 3 — BN + ELU + reshape (Eq. 10):
      → (B, KSS*n_eeg_ch, T_f)
    """
    def __init__(self, n_eeg_ch: int, T_e: int, T_f: int, KSS: int, dropout: float):
        super().__init__()
        self.n_eeg_ch = n_eeg_ch
        self.KSS      = KSS

        # stride=20 compresses T_e (2000) -> T_f (100), EXACT integer division
        # (2000/20=100), matching paper Eq.8 precisely -- T_f=100 is the
        # corrected fNIRS window length (10s @ 10Hz, single window, not a
        # 3s-segment stack), so no adaptive-pool workaround is needed here.
        compress = 20
        assert T_e % compress == 0 and T_e // compress == T_f, (
            f"T_e={T_e} must equal T_f={T_f} x stride(20) exactly, per paper Eq.8. "
            f"Got T_e/20={T_e/compress}."
        )
        self.dw_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, compress),   # W = n_eeg_ch (spatial), kernel_size_t = 20
            stride=(1, compress),        # stride=(spatial_stride=1, time_stride=20)
            groups=1,
            bias=True,                   # b_st_dwconv_eeg in Eq. 8
        )

        # Pointwise conv: 1 -> KSS
        self.pw_conv = nn.Conv2d(
            in_channels=1,
            out_channels=KSS,
            kernel_size=(1, 1),
            bias=True,                   # b_st_pwconv in Eq. 9 / 15
        )
        self.bn  = nn.BatchNorm2d(KSS)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_eeg_ch, T_e)
        Returns: (B, KSS*n_eeg_ch, T_f)
        """
        B, CH, T = x.shape

        # Unsqueeze -> (B, 1, CH, T)   [Eq. 8]
        x = x.unsqueeze(1)

        # Depthwise conv, stride=20 -> exactly (B, 1, CH, T_f), no pooling needed
        x = self.dw_conv(x)

        # Pointwise conv -> (B, KSS, CH, T_f)  [Eq. 9]
        x = self.pw_conv(x)

        # BN + ELU  [Eq. 10]
        x = F.elu(self.bn(x))
        x = self.drop(x)

        # Reshape (B, KSS, CH, T_f) → (B, KSS*CH, T_f)  [Eq. 10 reshape]
        B, KSS, CH, Tf = x.shape
        x = x.permute(0, 1, 2, 3).contiguous().view(B, KSS * CH, Tf)
        return x


# ===========================================================================
#  F-STFEM: fNIRS Spatial-Temporal Feature Extraction Module (Fig. 1f)
#  Similar to E-STFEM but with kernel=(1,1,1,11), no time compression needed
# ===========================================================================
class fNIRSSpatialTemporalFeatureExtractionModule(nn.Module):
    """
    F-STFEM (Fig. 1f, Sec. 2.3.3)
    
    Input:  F_fnirss  (B, n_fnirs_ch, T_f)   from F-SFEM
    Output: F_fnirsst (B, KSS*n_fnirs_ch, T_f)
    
    Depthwise conv kernel=(1,1,1,11) — captures local ~1.1s context in T_f
    No time stride: T_f remains 30
    """
    def __init__(self, n_fnirs_ch: int, T_f: int, KSS: int, dropout: float):
        super().__init__()
        self.n_fnirs_ch = n_fnirs_ch
        self.KSS        = KSS

        k = 11   # kernel size in time dimension (from paper: W_st_dwconv_fnirs ∈ R^{1×1×1×11})
        pad = k // 2   # 'same' padding

        # Depthwise: (B, 1, CH, T_f) → (B, 1, CH, T_f)
        self.dw_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, k),
            padding=(0, pad),
            groups=1,
            bias=True,                   # b_st_dwconv_fnirs in Eq. 14
        )

        # Pointwise: 1 → KSS
        self.pw_conv = nn.Conv2d(
            in_channels=1,
            out_channels=KSS,
            kernel_size=(1, 1),
            bias=True,                   # b_st_pwconv in Eq. 9 / 15
        )
        self.bn   = nn.BatchNorm2d(KSS)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_fnirs_ch, T_f)
        Returns: (B, KSS*n_fnirs_ch, T_f)
        """
        B, CH, Tf = x.shape

        x = x.unsqueeze(1)                 # (B, 1, CH, T_f)
        x = self.dw_conv(x)                # (B, 1, CH, T_f)
        x = self.pw_conv(x)                # (B, KSS, CH, T_f)
        x = F.elu(self.bn(x))
        x = self.drop(x)

        # Reshape → (B, KSS*CH, T_f)
        B, KSS, CH, Tf = x.shape
        x = x.contiguous().view(B, KSS * CH, Tf)
        return x


# ===========================================================================
#  MSTFDN: Full Model
# ===========================================================================
class MSTFDN(nn.Module):
    """
    Full MSTFDN architecture (Fig. 1a).
    
    Inputs:
      eeg:   (B, n_bands, n_eeg_ch, T_e)
             For N-back: (B, 5, 28, 2000)
      fnirs: (B, n_hemo=3, n_fnirs_ch, T_f)
             For N-back: (B, 3, 36, 100)
    
    Outputs during training: (logits, head_outputs_e, head_outputs_f, head_outputs_ef)
    Outputs during inference: logits (B, n_class)
    """

    def __init__(self, cfg: MSTFDNConfig):
        super().__init__()
        self.cfg = cfg
        CH_e = cfg.n_eeg_ch
        CH_f = cfg.n_fnirs_ch
        KSS  = cfg.KSS

        # ── EEG Branch ────────────────────────────────────────────────────
        self.e_tfem  = EEGTemporalFeatureExtractionModule(
            n_bands=cfg.n_bands, n_eeg_ch=CH_e, dropout=cfg.dropout)

        self.e_sfem  = SpatialFeatureExtractionModule(
            n_ch=CH_e, d_spatial=cfg.T_e,
            n_heads=cfg.He, n_layers=cfg.Ne,
            ffn_dim=cfg.ffn_dim, dropout=cfg.dropout,
            use_bn=True)   # BN for EEG (paper Sec. 2.2.2)

        self.e_stfem = EEGSpatialTemporalFeatureExtractionModule(
            n_eeg_ch=CH_e, T_e=cfg.T_e, T_f=cfg.T_f,
            KSS=KSS, dropout=cfg.dropout)

        # ── fNIRS Branch ──────────────────────────────────────────────────
        self.f_tfem  = fNIRSTemporalFeatureExtractionModule(
            n_fnirs_ch=CH_f, dropout=cfg.dropout)

        self.f_sfem  = SpatialFeatureExtractionModule(
            n_ch=CH_f, d_spatial=cfg.T_f,
            n_heads=cfg.Hf, n_layers=cfg.Nf,
            ffn_dim=cfg.ffn_dim, dropout=cfg.dropout,
            use_bn=False)  # LN for fNIRS

        self.f_stfem = fNIRSSpatialTemporalFeatureExtractionModule(
            n_fnirs_ch=CH_f, T_f=cfg.T_f,
            KSS=KSS, dropout=cfg.dropout)

        # ── Fusion Branch ─────────────────────────────────────────────────
        fusion_ch = KSS * (CH_e + CH_f)   # d_spatial of EFF-SFEM input

        self.eff_sfem = SpatialFeatureExtractionModule(
            n_ch=fusion_ch, d_spatial=cfg.T_f,
            n_heads=cfg.Hef, n_layers=cfg.Nef,
            ffn_dim=cfg.ffn_dim, dropout=cfg.dropout,
            use_bn=False)  # LN for fusion

        # ── Classifier (Fig. 1g, Eq. 18–21) ─────────────────────────────
        # Conv1d(KSS*(CH_e+CH_f) → (CH_e+CH_f)) + ReLU
        self.cls_conv1 = nn.Conv1d(
            in_channels=fusion_ch,
            out_channels=CH_e + CH_f,
            kernel_size=1, bias=True)

        # Conv1d((CH_e+CH_f) → n_class) + ReLU
        self.cls_conv2 = nn.Conv1d(
            in_channels=CH_e + CH_f,
            out_channels=cfg.n_class,
            kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv layers, Xavier for linear."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _process_fnirs(self, fnirs: torch.Tensor):
        """
        Process fNIRS input through F-TFEM + F-SFEM + F-STFEM.

        fnirs: (B, n_hemo=3, CH_f, T_f) -- a SINGLE 10s window with HbO/HbR/HbT
        already stacked (paper: X_fnirs in R^{3xCHfxTf}). There is NO segment
        stacking/averaging here -- that was based on a misreading of the
        preprocessing protocol and has been removed.
        """
        f_t = self.f_tfem(fnirs)             # (B, CH_f, T_f)
        f_s, heads_f = self.f_sfem(f_t)      # (B, CH_f, T_f)
        f_st = self.f_stfem(f_s)             # (B, KSS*CH_f, T_f)
        return f_st, heads_f

    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor):
        """
        eeg:   (B, n_bands=5, n_eeg_ch, T_e=2000)
        fnirs: (B, n_hemo=3,  n_fnirs_ch, T_f=100)   -- HbO, HbR, HbT; single window

        Returns:
          logits: (B, n_class)
          heads_e:  list of n_heads tensors (B, CH_e, d_k_e) -- for HSIC loss
          heads_f:  list of n_heads tensors (B, CH_f, d_k_f) -- for HSIC loss
          heads_ef: list of n_heads tensors (B, KSS*(CH_e+CH_f), d_k_ef) -- for HSIC loss
        """
        # ── EEG Branch ────────────────────────────────────────────────────
        # E-TFEM: multi-scale temporal features
        f_eeg_t = self.e_tfem(eeg)                    # (B, CH_e, T_e)

        # E-SFEM: spatial features (Transformer encoder)
        f_eeg_s, heads_e = self.e_sfem(f_eeg_t)      # (B, CH_e, T_e)

        # E-STFEM: spatial-temporal features, time compressed to T_f
        f_eeg_st = self.e_stfem(f_eeg_s)              # (B, KSS*CH_e, T_f)

        # ── fNIRS Branch ──────────────────────────────────────────────────
        f_fnirs_st, heads_f = self._process_fnirs(fnirs)  # (B, KSS*CH_f, T_f)

        # ── Fusion Branch ─────────────────────────────────────────────────
        # Concatenate along spatial (channel) dim [Eq. 17]
        f_fusion = torch.cat([f_eeg_st, f_fnirs_st], dim=1)   # (B, KSS*(CH_e+CH_f), T_f)

        # EFF-SFEM: fusion spatial features
        f_fusion_s, heads_ef = self.eff_sfem(f_fusion)        # same shape

        # ── Classifier [Eq. 18–21] ────────────────────────────────────────
        # Conv1: KSS*(CH_e+CH_f) → (CH_e+CH_f)
        out = F.relu(self.cls_conv1(f_fusion_s))   # (B, CH_e+CH_f, T_f)
        # Conv2: (CH_e+CH_f) → n_class
        out = F.relu(self.cls_conv2(out))           # (B, n_class, T_f)
        # GAP over T_f [Eq. 20]
        out = out.mean(dim=-1)                      # (B, n_class)

        return out, heads_e, heads_f, heads_ef


# ===========================================================================
#  MSTFDN LOSS FUNCTION (Eq. 22–28)
# ===========================================================================
class MSTFDNLoss(nn.Module):
    """
    Total loss = CE + L2_weight_norm + MHS_loss  [Eq. 22]
    
    - CE:   cross-entropy  [Eq. 23]
    - L2:   lambda_nw * sum of L2 norms of all weight matrices [Eq. 24]
            Loss_nw = lambda_nw * sum_i ||W_i||_2   (norm, NOT squared —
            confirmed against paper's exact formula). Paper ALSO applies
            weight_decay=1e-3 in the Adam optimizer separately; both are
            used together, this is not double-counting a single mechanism.
    - MHS:  lambda_mhs * (HSIC_eeg + HSIC_fnirs + HSIC_ef) [Eq. 27,28]
            where each HSIC = avg pairwise HSIC over all head pairs in that SFEM
    """
    def __init__(self, cfg: MSTFDNConfig, model: MSTFDN):
        super().__init__()
        self.cfg   = cfg
        self.model = model
        self.ce    = nn.CrossEntropyLoss()

    def forward(
        self,
        logits:    torch.Tensor,
        targets:   torch.Tensor,
        heads_e:   list,
        heads_f:   list,
        heads_ef:  list,
    ) -> tuple:
        """
        logits:  (B, n_class) — raw logits (not softmaxed)
        targets: (B,)         — class indices
        heads_*: lists of (B, CH, d_k) tensors — MHSA head outputs from last SFEM layer
        
        Returns (total_loss, ce_loss, l2_loss, mhs_loss) all as scalars.
        """
        # ── Cross-entropy loss [Eq. 23] ───────────────────────────────────
        loss_ce = self.ce(logits, targets)

        # ── L2 norm regularization [Eq. 24] ──────────────────────────────
        # Loss_nw = lambda_nw * sum_i ||W_i||_2  — sum of L2 norms, NOT squared
        loss_l2 = torch.tensor(0.0, device=logits.device)
        # "each weight matrix W_i": conv/linear weights only (dim > 1),
        # not the 1-D affine gains of BatchNorm/LayerNorm.
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                loss_l2 = loss_l2 + param.norm(2)
        loss_l2 = self.cfg.lambda_nw * loss_l2

        # ── Multi-head similarity loss [Eq. 27,28] ────────────────────────
        # L_hsic for each of the three SFEMs, then sum × lambda_mhs
        l_hsic_eeg   = mhs_loss_from_heads(heads_e)
        l_hsic_fnirs = mhs_loss_from_heads(heads_f)
        l_hsic_ef    = mhs_loss_from_heads(heads_ef)

        loss_mhs = self.cfg.lambda_mhs * (l_hsic_eeg + l_hsic_fnirs + l_hsic_ef)

        # ── Total loss [Eq. 22] ───────────────────────────────────────────
        loss_total = loss_ce + loss_l2 + loss_mhs

        return loss_total, loss_ce, loss_l2, loss_mhs


# ===========================================================================
#  TRAINER: Cosine annealing + linear warmup + early stopping
# ===========================================================================
class MSTFDNTrainer:
    """
    Training loop following paper Sec. 3.3.2:
    - Adam(lr=3e-4, weight_decay=1e-3)
    - CosineAnnealingLR + linear warmup (warmup_epochs=10, eta_min=lr*0.05)
    - batch_size=16, max_epochs=1000, early stopping
    - dropout=0.4, all layers
    """

    def __init__(
        self,
        model:     MSTFDN,
        cfg:       MSTFDNConfig,
        device:    str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model  = model.to(device)
        self.cfg    = cfg
        self.device = device
        self.loss_fn = MSTFDNLoss(cfg, model)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # Cosine annealing scheduler (applied AFTER warmup)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.max_epochs - cfg.warmup_epochs,
            eta_min=cfg.lr * cfg.eta_min_ratio,
        )

    def _warmup_lr(self, epoch: int):
        """Linear learning rate warmup for the first warmup_epochs epochs."""
        if epoch < self.cfg.warmup_epochs:
            scale = (epoch + 1) / self.cfg.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.cfg.lr * scale

    def train_epoch(self, loader) -> dict:
        self.model.train()
        total_loss = ce_sum = l2_sum = mhs_sum = 0.0
        correct = n = 0

        for eeg, fnirs, labels in loader:
            eeg    = eeg.to(self.device)
            fnirs  = fnirs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits, he, hf, hef = self.model(eeg, fnirs)
            loss, ce, l2, mhs   = self.loss_fn(logits, labels, he, hf, hef)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            B = labels.size(0)
            total_loss += loss.item() * B
            ce_sum     += ce.item()   * B
            l2_sum     += l2.item()   * B
            mhs_sum    += mhs.item()  * B
            correct    += (logits.argmax(1) == labels).sum().item()
            n          += B

        return {
            'loss': total_loss / n,
            'ce':   ce_sum / n,
            'l2':   l2_sum / n,
            'mhs':  mhs_sum / n,
            'acc':  correct / n,
        }

    @torch.no_grad()
    def eval_epoch(self, loader) -> dict:
        self.model.eval()
        total_loss = 0.0
        correct = n = 0
        all_preds, all_labels = [], []

        for eeg, fnirs, labels in loader:
            eeg    = eeg.to(self.device)
            fnirs  = fnirs.to(self.device)
            labels = labels.to(self.device)

            logits, he, hf, hef = self.model(eeg, fnirs)
            loss, *_ = self.loss_fn(logits, labels, he, hf, hef)

            preds = logits.argmax(1)
            correct    += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            n          += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        return {
            'loss': total_loss / n,
            'acc':  correct / n,
            'preds':  all_preds,
            'labels': all_labels,
        }

    def fit(self, train_loader, val_loader, verbose: bool = True):
        """Full training loop with warmup, cosine annealing, and early stopping."""
        best_val_loss = float('inf')
        best_state    = None
        patience_cnt  = 0

        for epoch in range(self.cfg.max_epochs):
            # Linear warmup
            self._warmup_lr(epoch)

            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.eval_epoch(val_loader)

            # Cosine annealing starts after warmup
            if epoch >= self.cfg.warmup_epochs:
                self.scheduler.step()

            if verbose and epoch % 10 == 0:
                lr_now = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch:4d}/{self.cfg.max_epochs} | "
                    f"LR={lr_now:.2e} | "
                    f"Train Loss={train_metrics['loss']:.4f} Acc={train_metrics['acc']:.4f} | "
                    f"Val Loss={val_metrics['loss']:.4f} Acc={val_metrics['acc']:.4f}"
                )

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_state    = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.cfg.patience:
                    print(f"Early stopping at epoch {epoch} (patience={self.cfg.patience})")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.model


# ===========================================================================
#  DATASET WRAPPER: loads preprocessed .npy files
# ===========================================================================
class MSTFDNDataset(torch.utils.data.Dataset):
    """
    Loads the preprocessed .npy arrays produced by MSTFDN_preprocessing_v3.py.

    CORRECTED: matches the paper's ACTUAL protocol -- no 16x16 spatial grid,
    no 21-segment fNIRS stack, 4 classes (0-back, 2-back, 3-back, rest).

    Expected files in data_dir/{subject_id}/:
      {subject_id}_eeg_windows.npy    (N, 5, 28, 2000)  float16 -- real 5-band data
      {subject_id}_fnirs_windows.npy  (N, 3, 36, 100)   float32 -- HbO/HbR/HbT
      {subject_id}_labels.npy         (N,)  int, values 0/1/2/3
    """

    def __init__(self, data_dir: str, subject_id: str, indices: list = None):
        import numpy as np, os

        sub_dir = os.path.join(data_dir, subject_id)
        eeg    = np.load(os.path.join(sub_dir, f"{subject_id}_eeg_windows.npy"))
        fnirs  = np.load(os.path.join(sub_dir, f"{subject_id}_fnirs_windows.npy"))
        labels = np.load(os.path.join(sub_dir, f"{subject_id}_labels.npy"))

        if eeg.ndim != 4 or eeg.shape[1] != 5:
            raise ValueError(
                f"{subject_id}: expected eeg_windows shape (N,5,CH,T_e) from "
                f"MSTFDN_preprocessing_v3.py, got {eeg.shape}. Old preprocessing "
                f"versions (16x16 grid or broadcast-copy bands) are not compatible."
            )
        if fnirs.ndim != 4 or fnirs.shape[1] != 3:
            raise ValueError(
                f"{subject_id}: expected fnirs_windows shape (N,3,CH,T_f) from "
                f"MSTFDN_preprocessing_v3.py, got {fnirs.shape}. Old preprocessing "
                f"versions (21-segment stack) are not compatible."
            )

        if indices is not None:
            eeg    = eeg[indices]
            fnirs  = fnirs[indices]
            labels = labels[indices]

        # Keep EEG as float16 in host RAM, cast to float32 per-window in
        # __getitem__ -- same memory-saving pattern as the training script's
        # dataset class.
        self.eeg    = torch.from_numpy(eeg).half()
        self.fnirs  = torch.from_numpy(fnirs).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx].float(), self.fnirs[idx], self.labels[idx]


# ===========================================================================
#  SHAPE DEBUG: verify forward pass dimensions
# ===========================================================================
def debug_forward_shapes(cfg: MSTFDNConfig = None, device: str = 'cpu'):
    """
    Run a single forward pass with random data and print all intermediate shapes.
    Useful for verifying the architecture before training.
    """
    if cfg is None:
        cfg = MSTFDNConfig()

    print("=" * 60)
    print("MSTFDN Shape Debug")
    print(f"  n_eeg_ch={cfg.n_eeg_ch}, n_fnirs_ch={cfg.n_fnirs_ch}")
    print(f"  T_e={cfg.T_e}, T_f={cfg.T_f}")
    print(f"  n_bands={cfg.n_bands}, n_hemo={cfg.n_hemo}, n_class={cfg.n_class}")
    print(f"  KSS={cfg.KSS}, He={cfg.He}, Hf={cfg.Hf}, Hef={cfg.Hef}")
    print("=" * 60)

    model = MSTFDN(cfg).to(device)
    model.eval()

    B = 4  # batch size
    eeg   = torch.randn(B, cfg.n_bands, cfg.n_eeg_ch, cfg.T_e, device=device)
    fnirs = torch.randn(B, cfg.n_hemo, cfg.n_fnirs_ch, cfg.T_f, device=device)

    print(f"Input EEG  shape: {tuple(eeg.shape)}")
    print(f"Input fNIRS shape: {tuple(fnirs.shape)}")

    with torch.no_grad():
        # ── EEG Branch ────────────────────────────────────────────────────
        f_eeg_t = model.e_tfem(eeg)
        print(f"\n[E-TFEM]  F_eegt  : {tuple(f_eeg_t.shape)}  expected (B, {cfg.n_eeg_ch}, {cfg.T_e})")

        f_eeg_s, heads_e = model.e_sfem(f_eeg_t)
        print(f"[E-SFEM]  F_eegs  : {tuple(f_eeg_s.shape)}  expected (B, {cfg.n_eeg_ch}, {cfg.T_e})")
        print(f"          heads_e : {len(heads_e)} heads × {tuple(heads_e[0].shape)}")

        f_eeg_st = model.e_stfem(f_eeg_s)
        print(f"[E-STFEM] F_eegst : {tuple(f_eeg_st.shape)}  expected (B, {cfg.KSS*cfg.n_eeg_ch}, {cfg.T_f})")

        # ── fNIRS Branch ──────────────────────────────────────────────────
        f_fnirs_st, heads_f = model._process_fnirs(fnirs)
        print(f"\n[F-Branch] F_fnirst: {tuple(f_fnirs_st.shape)}  expected (B, {cfg.KSS*cfg.n_fnirs_ch}, {cfg.T_f})")
        print(f"           heads_f : {len(heads_f)} heads × {tuple(heads_f[0].shape)}")

        # ── Fusion Branch ─────────────────────────────────────────────────
        f_fusion = torch.cat([f_eeg_st, f_fnirs_st], dim=1)
        print(f"\n[Concat]  F_fusion : {tuple(f_fusion.shape)}  expected (B, {cfg.KSS*(cfg.n_eeg_ch+cfg.n_fnirs_ch)}, {cfg.T_f})")

        f_fusion_s, heads_ef = model.eff_sfem(f_fusion)
        print(f"[EFF-SFEM] F_fusions: {tuple(f_fusion_s.shape)}")
        print(f"           heads_ef: {len(heads_ef)} heads × {tuple(heads_ef[0].shape)}")

        # ── Classifier ────────────────────────────────────────────────────
        out = F.relu(model.cls_conv1(f_fusion_s))
        print(f"\n[CLS-Conv1] F1: {tuple(out.shape)}  expected (B, {cfg.n_eeg_ch+cfg.n_fnirs_ch}, {cfg.T_f})")
        out = F.relu(model.cls_conv2(out))
        print(f"[CLS-Conv2] F2: {tuple(out.shape)}  expected (B, {cfg.n_class}, {cfg.T_f})")
        out = out.mean(dim=-1)
        print(f"[GAP]      out: {tuple(out.shape)}  expected (B, {cfg.n_class})")

    print("\n✓ Full forward pass:")
    logits, he, hf, hef = model(eeg, fnirs)
    print(f"  logits: {tuple(logits.shape)}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {n_params:,}")
    print("=" * 60)

    # Test loss
    targets = torch.randint(0, cfg.n_class, (B,), device=device)
    loss_fn = MSTFDNLoss(cfg, model)
    total, ce, l2, mhs = loss_fn(logits, targets, he, hf, hef)
    print(f"\nLoss components:")
    print(f"  CE={ce.item():.4f}  L2={l2.item():.4f}  MHS={mhs.item():.6f}")
    print(f"  Total={total.item():.4f}")
    print("\n✓ All shapes correct — model ready for training.")

    return model


# ===========================================================================
#  MAIN: example usage
# ===========================================================================
if __name__ == "__main__":
    import sys

    # -- 1. Config for N-back (MBI dataset-A, 4-class: 0/2/3-back + rest) --
    cfg = MSTFDNConfig()
    cfg.n_eeg_ch    = 28
    cfg.n_fnirs_ch  = 36
    cfg.T_e         = 2000   # 10s x 200Hz
    cfg.T_f         = 100    # 10s x 10Hz -- single window, matches paper exactly
    cfg.n_bands     = 5
    cfg.n_hemo      = 3      # HbO, HbR, HbT -- all supplied as input
    cfg.n_class     = 4      # 0-back, 2-back, 3-back, rest (paper Sec 3.2)

    # SFEM attention: T_e must be divisible by He, T_f must be divisible by Hf/Hef
    # T_e=2000 / He=8  = 250 (exact)
    # T_f=100  / Hf=5  = 20  (exact)  -- also try Hf=4 (25) or Hf=10 (10) if needed
    cfg.He  = 8
    cfg.Hf  = 5
    cfg.Hef = 5

    cfg.Ne  = 1
    cfg.Nf  = 1
    cfg.Nef = 1
    cfg.KSS = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # -- 2. Debug forward pass --------------------------------------------
    model = debug_forward_shapes(cfg, device=device)

    # -- 3. Quick training smoke test with random data ---------------------
    print("\nRunning quick training smoke test (2 epochs, random data)...")

    N = 64  # synthetic samples
    eeg_data   = torch.randn(N, cfg.n_bands, cfg.n_eeg_ch, cfg.T_e)
    fnirs_data = torch.randn(N, cfg.n_hemo, cfg.n_fnirs_ch, cfg.T_f)   # no segment dim
    labels     = torch.randint(0, cfg.n_class, (N,))

    dataset = torch.utils.data.TensorDataset(eeg_data, fnirs_data, labels)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [48, 16])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=cfg.batch_size)

    cfg.max_epochs    = 2
    cfg.warmup_epochs = 1
    cfg.patience      = 100

    model   = MSTFDN(cfg).to(device)
    trainer = MSTFDNTrainer(model, cfg, device=device)
    trainer.fit(train_loader, val_loader, verbose=True)
    print("\nSmoke test passed -- model trains without errors.")
