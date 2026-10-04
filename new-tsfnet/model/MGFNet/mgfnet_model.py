"""
mgfnet_model.py -- from-scratch reimplementation of MGFNet
(Zhang, Gong & Yuan, Sensors 2026, 26, 3402), Sections 3.1-3.4.

NO OFFICIAL CODE EXISTS. Everything here is reconstructed from the paper
text. Every place where the text is underdetermined is marked # AMBIGUITY-n
and listed in MGFNET_AMBIGUITIES at the bottom of this file.

INPUT CONTRACT (channel x time):
    x_eeg  : (B, 30, T_eeg)     EEG, 200 Hz   (paper: T_eeg = 1000 for a 5 s window)
    x_hbo  : (B, 36, T_nirs)    HbO,  10 Hz   (paper: T_nirs = 50)
    x_hbr  : (B, 36, T_nirs)    HbR,  10 Hz
Inputs are expected z-scored per channel (mgfnet_dataset.ChannelNormalizer).

OUTPUT:
    y_hyb   : (B, C)  logits of the main hybrid classifier
    y_aux   : dict with logits of the deeply-supervised branches, keyed
              "1","2","3","12","13","23"  (paper Eq. 1's index set)
    aux     : diagnostics -- coupling matrix, gate weights, routing feature,
              component selection, temperature.

Every classifier head returns raw LOGITS (use F.cross_entropy).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Channel-width schedule
# ---------------------------------------------------------------------------
# The paper gives only the FINAL feature dims: x1 in R^120 (EEG, 30 ch),
# x2,x3 in R^144 (fNIRS, 36 ch), x12,x13 in R^264 (66 ch), x23 in R^288
# (72 ch) -- all exactly 4x the input channel count. Reconstructed as
#     C_in -> 2*C_in (x3) -> 4*C_in (x3)
# widening at the two strided layers.  # AMBIGUITY-1
def _width_schedule(c_in, n_blocks, widen_at):
    widths, cur = [], c_in
    for i in range(n_blocks):
        if i in widen_at:
            cur = cur * 2
        widths.append(cur)
    return widths


class CBR(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU (paper Sec. 3.1, "CBR module")."""

    def __init__(self, c_in, c_out, kernel, stride=1, dilation=1):
        super().__init__()
        # 'same'-style padding so length shrinks only by the stride. # AMBIGUITY-2
        pad = ((kernel - 1) * dilation) // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel, stride=stride,
                              padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class IntraModalEncoder(nn.Module):
    """Sec. 3.1. Six stacked CBR blocks + GAP, non-shared across modalities."""

    def __init__(self, c_in, kernels, strides):
        super().__init__()
        assert len(kernels) == len(strides) == 6
        widen_at = {i for i, s in enumerate(strides) if s > 1}   # {0, 3}
        widths = _width_schedule(c_in, 6, widen_at)
        blocks, prev = [], c_in
        for k, s, w in zip(kernels, strides, widths):
            blocks.append(CBR(prev, w, k, stride=s))
            prev = w
        self.blocks = nn.Sequential(*blocks)
        self.out_dim = prev

    def forward(self, x):                       # (B, C_in, T)
        return self.blocks(x).mean(dim=-1)      # GAP -> (B, out_dim)


class DilatedConvNet(nn.Module):
    """Sec. 3.2 / Fig. 2. k=9,s=4 -> two k=3,s=1 -> three dilated k=3 with
    dilation [2, 2, 4] -> GAP."""

    def __init__(self, c_in):
        super().__init__()
        kernels = [9, 3, 3, 3, 3, 3]
        strides = [4, 1, 1, 1, 1, 1]
        dilations = [1, 1, 1, 2, 2, 4]
        # widen at the strided layer and where the dilated stage begins:
        # 66 -> 264 and 72 -> 288, matching the paper.  # AMBIGUITY-1
        widths = _width_schedule(c_in, 6, {0, 3})
        blocks, prev = [], c_in
        for k, s, d, w in zip(kernels, strides, dilations, widths):
            blocks.append(CBR(prev, w, k, stride=s, dilation=d))
            prev = w
        self.blocks = nn.Sequential(*blocks)
        self.out_dim = prev

    def forward(self, x):
        return self.blocks(x).mean(dim=-1)


# ---------------------------------------------------------------------------
# CGSCR
# ---------------------------------------------------------------------------
def _logit(p):
    return math.log(p / (1.0 - p))


class MorletFilterBank(nn.Module):
    """Sec. 3.3 step 1: "learnable Morlet-like convolutional filters".

    Complex Morlet kernels (Gaussian envelope x cos / sin carrier) with a
    learnable centre frequency f0 and bandwidth bw (frequency-domain std).
    The output is the quadrature magnitude, so it is phase-invariant.

    f0 is kept inside [f_lo, f_hi] through a sigmoid (smooth, no clamp, so
    the gradient never becomes exactly zero), which stops a centre frequency
    from drifting outside the band the preprocessing left in the data.
    bw > 0 through softplus. Reflect padding avoids the zero-padding edge
    artefact a long kernel would otherwise create.  # AMBIGUITY-3
    """

    def __init__(self, f0_init, bw_init, fs, kernel_size, f_lo, f_hi):
        super().__init__()
        assert len(f0_init) == len(bw_init)
        for f in f0_init:
            if not (f_lo < f < f_hi):
                raise ValueError(f"f0 init {f} outside ({f_lo}, {f_hi})")
        self.n_components = len(f0_init)
        self.fs = float(fs)
        self.kernel_size = int(kernel_size) | 1          # odd
        self.f_lo, self.f_hi = float(f_lo), float(f_hi)

        p = [(f - f_lo) / (f_hi - f_lo) for f in f0_init]
        self.f0_raw = nn.Parameter(torch.tensor([_logit(v) for v in p], dtype=torch.float32))
        self.bw_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(bw_init, dtype=torch.float32))))

        t = (torch.arange(self.kernel_size, dtype=torch.float32)
             - (self.kernel_size - 1) / 2.0) / self.fs
        self.register_buffer("t", t)

    @property
    def f0(self):
        return self.f_lo + (self.f_hi - self.f_lo) * torch.sigmoid(self.f0_raw)

    @property
    def bw(self):
        return F.softplus(self.bw_raw) + 1e-4

    def _kernels(self):
        f0 = self.f0.unsqueeze(1)                             # (K,1) Hz
        sigma_t = 1.0 / (2.0 * math.pi * self.bw.unsqueeze(1))
        t = self.t.unsqueeze(0)                               # (1,L)
        env = torch.exp(-0.5 * (t / sigma_t) ** 2)
        env = env / (env.sum(dim=1, keepdim=True) + 1e-8)
        phase = 2.0 * math.pi * f0 * t
        return (env * torch.cos(phase)).unsqueeze(1), (env * torch.sin(phase)).unsqueeze(1)

    def forward(self, x):
        """(B, C, T) -> (B, K, C, T): each channel filtered by each component."""
        B, C, T = x.shape
        kr, ki = self._kernels()                              # (K,1,L)
        pad = self.kernel_size // 2
        xf = x.reshape(B * C, 1, T)
        xf = F.pad(xf, (pad, pad), mode="reflect" if pad < T else "constant")
        re = F.conv1d(xf, kr)
        im = F.conv1d(xf, ki)
        mag = torch.sqrt(re * re + im * im + 1e-8)            # phase-invariant
        return mag.reshape(B, C, self.n_components, T).transpose(1, 2)


# Initial component centres/bandwidths, chosen INSIDE what preprocessing keeps.
# EEG (1-40 Hz band-pass): delta / theta / alpha / beta / gamma(30-40).
EEG_F0_INIT = (2.0, 6.0, 10.5, 21.5, 35.0)
EEG_BW_INIT = (1.5, 2.0, 2.5, 8.5, 5.0)
EEG_F_RANGE = (1.0, 40.0)
EEG_KERNEL_S = 1.0
# fNIRS (0.2 Hz low-pass, 5 s window): three scales below the cut-off.
# With a 5 s window the slowest scale acts as a smooth trend (envelope wider
# than the kernel); the faster two resolve oscillatory content.  # AMBIGUITY-5
NIRS_F0_INIT = (0.03, 0.08, 0.15)
NIRS_BW_INIT = (0.02, 0.04, 0.06)
NIRS_F_RANGE = (0.005, 0.25)
NIRS_KERNEL_S = 4.9


class BandEncoder(nn.Module):
    """Sec. 3.3 step 2: "lightweight band encoders" mapping each component to
    a 64-d embedding; shared across components within a modality.
    # AMBIGUITY-4"""

    def __init__(self, c_in, d_out=64):
        super().__init__()
        self.net = nn.Sequential(
            CBR(c_in, 64, 7, stride=4),
            CBR(64, d_out, 3, stride=1),
        )

    def forward(self, x):                        # (B, K, C, T)
        B, K, C, T = x.shape
        h = self.net(x.reshape(B * K, C, T)).mean(dim=-1)
        return h.reshape(B, K, -1)               # (B, K, 64)


class CGSCR(nn.Module):
    """Sec. 3.3.

    Stage A  frequency decomposition       -> Morlet banks (EEG: 5, fNIRS: 3)
    Stage B  cross-component coupling      -> multi-head affinity, C in (B,5,3)
    Stage C  sparse routing + branch gate  -> Transformer -> Mean(Abs) ->
                                              Gumbel-Softmax -> x_scr -> 6-way gate
    """

    N_EEG_COMP = 5          # paper: E in R^{B x 5 x 64}
    N_NIRS_COMP = 3         # paper: F in R^{B x 3 x 64}
    D_LATENT = 128
    N_BRANCHES = 6

    def __init__(self, eeg_channels, nirs_channels, eeg_fs=200.0,
                 nirs_fs=10.0, n_heads=8, dropout=0.1):
        super().__init__()
        self.eeg_bank = MorletFilterBank(EEG_F0_INIT, EEG_BW_INIT, eeg_fs,
                                         int(round(EEG_KERNEL_S * eeg_fs)), *EEG_F_RANGE)
        self.hbo_bank = MorletFilterBank(NIRS_F0_INIT, NIRS_BW_INIT, nirs_fs,
                                         int(round(NIRS_KERNEL_S * nirs_fs)), *NIRS_F_RANGE)
        self.hbr_bank = MorletFilterBank(NIRS_F0_INIT, NIRS_BW_INIT, nirs_fs,
                                         int(round(NIRS_KERNEL_S * nirs_fs)), *NIRS_F_RANGE)

        self.eeg_enc = BandEncoder(eeg_channels, 64)
        # HbO and HbR component embeddings are averaged into the single
        # F in R^{Bx3x64} the paper specifies.  # AMBIGUITY-6
        self.nirs_enc = BandEncoder(nirs_channels, 64)

        self.proj_e = nn.Linear(64, self.D_LATENT)
        self.proj_f = nn.Linear(64, self.D_LATENT)
        # Multi-head AFFINITY: only Q and K are parameterised, because the
        # product the paper uses is the attention-weight matrix C itself; a
        # value/output projection would never receive a gradient.
        # # AMBIGUITY-11
        assert self.D_LATENT % n_heads == 0
        self.n_heads = n_heads
        self.d_head = self.D_LATENT // n_heads
        self.q_proj = nn.Linear(self.D_LATENT, self.D_LATENT)
        self.k_proj = nn.Linear(self.D_LATENT, self.D_LATENT)
        self.attn_drop = nn.Dropout(dropout)

        # Cross-modal tokens: one per (EEG comp i, fNIRS comp j) pair = 15.
        #   token_ij = (N_NIRS_COMP * C_ij) * LayerNorm(E_i + F_j)
        # The coupling scale is applied AFTER the LayerNorm: LN(c*x) == LN(x)
        # for any c > 0, so scaling before it would erase C entirely (and
        # starve q_proj/k_proj of gradient). The N_NIRS_COMP factor makes a
        # uniform coupling (C_ij = 1/3) an identity scale.  # AMBIGUITY-7
        self.token_norm = nn.LayerNorm(self.D_LATENT)
        # Pre-norm Transformer (no final norm): the residual stream keeps the
        # coupling-dependent token magnitudes, so the Mean(Abs) importance
        # score below reflects them. A post-norm encoder would re-normalise
        # every token and flatten the scores.  # AMBIGUITY-12
        layer = nn.TransformerEncoderLayer(
            d_model=self.D_LATENT, nhead=n_heads, dim_feedforward=512,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2,
                                                 enable_nested_tensor=False)

        # Learnable Gumbel temperature, init 1.0 (paper). Log-parameterised
        # so it stays strictly positive.  # AMBIGUITY-8
        self.log_tau = nn.Parameter(torch.zeros(1))

        self.routing_gate = nn.Sequential(
            nn.Linear(self.D_LATENT, self.D_LATENT), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(self.D_LATENT, self.N_BRANCHES))

    @property
    def tau(self):
        return self.log_tau.exp().clamp(min=1e-2, max=1e2)

    def coupling(self, Ep, Fp):
        """Multi-head affinity of EEG components (queries) over fNIRS
        components (keys), averaged over heads -> (B,5,3)."""
        B = Ep.shape[0]
        q = self.q_proj(Ep).view(B, self.N_EEG_COMP, self.n_heads, self.d_head)
        k = self.k_proj(Fp).view(B, self.N_NIRS_COMP, self.n_heads, self.d_head)
        logits = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head)
        return self.attn_drop(F.softmax(logits, dim=-1)).mean(dim=1)

    def route(self, Ep, Fp, C):
        """Tokens -> Transformer -> Mean(Abs) -> Gumbel-Softmax selection."""
        B = Ep.shape[0]
        base = self.token_norm(Ep.unsqueeze(2) + Fp.unsqueeze(1))           # (B,5,3,128)
        tokens = (self.N_NIRS_COMP * C).unsqueeze(-1) * base
        z = self.transformer(tokens.reshape(B, -1, self.D_LATENT))          # (B,15,128)
        scores = z.abs().mean(dim=-1)                                       # Mean(Abs) -> (B,15)

        if self.training:
            # Standard Gumbel-Softmax: softmax((s + g) / tau), straight-through
            # hard one-hot forward ("discrete yet differentiable").
            sel = F.gumbel_softmax(scores, tau=self.tau, hard=True, dim=-1)
        else:
            # Deterministic inference with the SAME discrete form as training:
            # one-hot of the most important component (the mode of the
            # Gumbel-max distribution).  # AMBIGUITY-9
            sel = F.one_hot(scores.argmax(dim=-1), scores.shape[-1]).to(z.dtype)
        x_scr = torch.einsum("bn,bnd->bd", sel, z)                          # (B,128)
        return x_scr, sel, scores

    def forward(self, x_eeg, x_hbo, x_hbr):
        E = self.eeg_enc(self.eeg_bank(x_eeg))                    # (B,5,64)
        F_hbo = self.nirs_enc(self.hbo_bank(x_hbo))               # (B,3,64)
        F_hbr = self.nirs_enc(self.hbr_bank(x_hbr))               # (B,3,64)
        Fc = 0.5 * (F_hbo + F_hbr)                                # (B,3,64)

        Ep = self.proj_e(E)                                       # (B,5,128)
        Fp = self.proj_f(Fc)                                      # (B,3,128)
        C = self.coupling(Ep, Fp)                                 # (B,5,3)
        x_scr, sel, scores = self.route(Ep, Fp, C)
        gate = F.softmax(self.routing_gate(x_scr), dim=-1)        # (B,6)
        return x_scr, gate, C, sel, scores


# ---------------------------------------------------------------------------
# MGFNet
# ---------------------------------------------------------------------------
class MGFNet(nn.Module):
    BRANCH_KEYS = ("1", "2", "3", "12", "13", "23")

    def __init__(self, num_classes=3, eeg_channels=30, nirs_channels=36,
                 eeg_fs=200.0, nirs_fs=10.0, dropout=0.1,
                 use_cgscr=True, use_cross_modal=True, use_intra_modal=True):
        super().__init__()
        if not (use_intra_modal or use_cross_modal):
            raise ValueError(
                "use_intra_modal and use_cross_modal cannot both be False: "
                "MGFNet would have no branch features to gate or classify.")
        self.use_cgscr = use_cgscr
        self.use_cross_modal = use_cross_modal
        self.use_intra_modal = use_intra_modal
        self.num_classes = num_classes
        self.eeg_channels = eeg_channels
        self.nirs_channels = nirs_channels

        if use_intra_modal:
            self.enc_eeg = IntraModalEncoder(eeg_channels, [9, 3, 3, 9, 3, 3],
                                             [4, 1, 1, 4, 1, 1])
            self.enc_hbo = IntraModalEncoder(nirs_channels, [5, 3, 3, 5, 3, 3],
                                             [2, 1, 1, 2, 1, 1])
            self.enc_hbr = IntraModalEncoder(nirs_channels, [5, 3, 3, 5, 3, 3],
                                             [2, 1, 1, 2, 1, 1])
        if use_cross_modal:
            self.enc_12 = DilatedConvNet(eeg_channels + nirs_channels)
            self.enc_13 = DilatedConvNet(eeg_channels + nirs_channels)
            self.enc_23 = DilatedConvNet(nirs_channels + nirs_channels)

        dims = {}
        if use_intra_modal:
            dims.update({"1": self.enc_eeg.out_dim, "2": self.enc_hbo.out_dim,
                         "3": self.enc_hbr.out_dim})
        if use_cross_modal:
            dims.update({"12": self.enc_12.out_dim, "13": self.enc_13.out_dim,
                         "23": self.enc_23.out_dim})
        self.active_keys = tuple(k for k in self.BRANCH_KEYS if k in dims)
        self.branch_dims = dims

        # Deep supervision (Sec. 3.4): one auxiliary head per active branch.
        self.aux_heads = nn.ModuleDict(
            {k: nn.Linear(d, num_classes) for k, d in dims.items()})

        if use_cgscr:
            self.cgscr = CGSCR(eeg_channels, nirs_channels, eeg_fs, nirs_fs,
                               dropout=dropout)
            # The gate is always 6-wide (paper). Under a branch ablation only
            # the active slots are used (after the 6-way softmax).
            self._gate_idx = [self.BRANCH_KEYS.index(k) for k in self.active_keys]
            fused_dim = sum(dims.values()) + CGSCR.D_LATENT
        else:
            fused_dim = sum(dims.values())

        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(fused_dim, num_classes))

    def forward(self, x_eeg, x_hbo, x_hbr):
        if x_eeg.shape[1] != self.eeg_channels or x_hbo.shape[1] != self.nirs_channels \
                or x_hbr.shape[1] != self.nirs_channels:
            raise ValueError(
                f"channel mismatch: model expects EEG={self.eeg_channels}, "
                f"fNIRS={self.nirs_channels}; got EEG={x_eeg.shape[1]}, "
                f"HbO={x_hbo.shape[1]}, HbR={x_hbr.shape[1]}")
        T_eeg = x_eeg.shape[-1]
        feats = {}
        if self.use_intra_modal:
            feats["1"] = self.enc_eeg(x_eeg)
            feats["2"] = self.enc_hbo(x_hbo)
            feats["3"] = self.enc_hbr(x_hbr)
        if self.use_cross_modal:
            # Sec. 3.2: fNIRS is aligned to the EEG time axis by linear
            # interpolation BEFORE pairwise concatenation.
            hbo_up = F.interpolate(x_hbo, size=T_eeg, mode="linear", align_corners=False)
            hbr_up = F.interpolate(x_hbr, size=T_eeg, mode="linear", align_corners=False)
            feats["12"] = self.enc_12(torch.cat([x_eeg, hbo_up], dim=1))
            feats["13"] = self.enc_13(torch.cat([x_eeg, hbr_up], dim=1))
            feats["23"] = self.enc_23(torch.cat([hbo_up, hbr_up], dim=1))

        y_aux = {k: self.aux_heads[k](feats[k]) for k in self.active_keys}

        aux = {"coupling": None, "gate_weights": None, "routing_feature": None,
               "component_selection": None, "importance_scores": None,
               "gumbel_tau": None, "branch_keys": self.active_keys}

        if self.use_cgscr:
            x_scr, gate, C, sel, scores = self.cgscr(x_eeg, x_hbo, x_hbr)
            parts = [feats[k] * gate[:, i:i + 1]
                     for k, i in zip(self.active_keys, self._gate_idx)]
            fused = torch.cat(parts + [x_scr], dim=1)
            aux.update({"coupling": C, "gate_weights": gate,
                        "routing_feature": x_scr, "component_selection": sel,
                        "importance_scores": scores,
                        "gumbel_tau": float(self.cgscr.tau.item())})
        else:
            # Table 4 "w/o CGSCR": plain static concatenation, no gating.
            fused = torch.cat([feats[k] for k in self.active_keys], dim=1)

        return self.classifier(fused), y_aux, aux


def mgfnet_deep_supervision_loss(y_hyb, y_aux, target,
                                 lambda_hyb=0.4, lambda_aux=0.1):
    """Paper Eq. 1: L = 0.4 * CE(y_hyb) + sum_i 0.1 * CE(y_i), on LOGITS.
    Fixed absolute weights; not renormalised under a branch ablation.
    # AMBIGUITY-10"""
    loss = lambda_hyb * F.cross_entropy(y_hyb, target)
    for logits in y_aux.values():
        loss = loss + lambda_aux * F.cross_entropy(logits, target)
    return loss


MGFNET_AMBIGUITIES = {
    1: "Intermediate channel widths are never stated; only the final dims "
       "(120/144/264/288 = 4x input channels). C -> 2C -> 4C, widening at "
       "the strided layers (and at the dilated stage in DilatedConvNet).",
    2: "Padding is not specified. 'same'-style padding.",
    3: "'Morlet-like learnable filters': parameterisation and kernel length "
       "not given. Complex Morlet, learnable f0 (sigmoid-bounded to the "
       "preprocessed band) and bandwidth (softplus), quadrature magnitude, "
       "reflect padding; EEG kernel 1 s, fNIRS kernel 4.9 s.",
    4: "'Lightweight band encoders' are unspecified; one 2-layer CBR encoder "
       "shared across components within a modality.",
    5: "Component initialisation is not given. EEG: delta/theta/alpha/beta/"
       "gamma centres inside 1-40 Hz; fNIRS: 0.03/0.08/0.15 Hz, below the "
       "0.2 Hz low-pass.",
    6: "The paper reports F in R^{Bx3x64} for fNIRS as a whole. HbO and HbR "
       "component embeddings (shared encoder) are averaged.",
    7: "How 'cross-modal tokens are constructed' from the coupling matrix is "
       "not described. 15 tokens (5x3): token_ij = 3*C_ij * LN(E_i + F_j). "
       "C is applied after the LayerNorm so it is not normalised away.",
    8: "Whether the learnable Gumbel temperature is constrained is not "
       "stated; log-parameterised to stay positive.",
    9: "Gumbel-Softmax variant and inference behaviour are not stated. "
       "Standard softmax((s+g)/tau) with straight-through hard one-hot in "
       "training; deterministic one-hot argmax(s) at inference, so train "
       "and test use the same discrete routing form.",
    10: "Eq. 1's weights are absolute; under a branch ablation the total "
        "weight drops below 1.0.",
    11: "'Multi-head attention ... to compute the affinity matrix': only the "
        "Q/K affinity is used (heads averaged); no value/output projection.",
    12: "Transformer norm placement is not stated. Pre-norm (norm_first) "
        "without a final norm, so the Mean(Abs) scores keep the "
        "coupling-dependent token magnitudes.",
}
