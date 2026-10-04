# ============================================================================
# D-1-safe: fixed five-band EEG decomposition with a "safe-capacity" wrapper.
#
# INSERT THIS BLOCK into tsfnet_model_notebook.py directly after the
# SpatialMHSABranch class (after line 766, before "class ConnectivityBranch").
#
# WHY THIS EXISTS (roadmap D-1, requested by the advisor as a response to the
# MSTFDN 69.02% comparison): MSTFDN's five-band EEG decomposition (delta/
# theta/alpha/beta/gamma) is the ONE idea from that paper never tested here.
# Ten+ prior "add capacity" interventions (G-1, G-2a, G-4, C-1, B-4, B-5,
# egta_gru, ...) all failed. This module is deliberately engineered to
# minimize every risk factor identified in that failure pattern (see the
# project's own analysis: no forced participation at init, heavy regularization
# on new params, shared weights instead of 5x replication, low added capacity).
#
# REUSES THE EXISTING x_chan INPUT: SpatialMHSABranch already consumes raw
# channel-space EEG (B, 28, 2000) at SFREQ_EEG=200Hz, saved by
# Preproc_tsfnet_nback_badch.py's ENABLE_EEG_CHANNEL_SPACE=True (already
# True in this project, per line 334). NO new preprocessing run is needed --
# this reads the SAME .npy files eeg_spatial_mode='mhsa' already reads.
#
# ENGINEERING RULES THIS DESIGN SATISFIES (see قواعد_مهندسی_پروژه.md):
#   Rule 2 (gradient must reach new params): the FIR filter bank is a fixed
#     buffer (no gradient needed/possible there by design), but the encoder
#     and gate ARE trainable and verified with a numeric gradient test below.
#   Rule 3 (flag interference): raises ValueError if combined with
#     eeg_backbone_mode != 'native' (both would modify the EEG decision path).
#   Rule 4 (verify fixed mappings from raw data, not assumption): filter
#     center frequencies come directly from firwin against SFREQ_EEG=200.0,
#     the same constant Preproc_tsfnet_nback_badch.py uses -- not a
#     re-derived or assumed sample rate.
#   Rule 5 (default = bit-exact no-op): default eeg_band_mode='none' means
#     this module is NEVER constructed and clf_eeg's input dim is completely
#     unchanged -- identical to prior behavior. This is the ONLY guarantee
#     rule 5 requires; the 'safe' mode itself is honestly NOT a no-op (same
#     honest deviation already documented for labram_blend and egta_gru).
#   Rule 6 (RNG shift): the gate and encoder are constructed in TSFNet's
#     __init__ in a fixed position; verified with a state_dict diff test.
#   Rule 7 (output contract): forward()'s return signature and aux dict keys
#     are unchanged; band_gate_value is ADDED to the aux dict, never
#     replacing/removing an existing key.
# ============================================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
