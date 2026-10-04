"""
labram_eeg_branch.py -- Priority C-2: a pretrained-EEG-foundation-model
classification head (LaBraM, Jiang et al. 2024, ICLR spotlight) as an
ISOLATED, low-risk alternative to TSFNet's native EEG-only head.

WHY THIS IS THE NATURAL NEXT STEP AFTER C-1 (Deep CORAL) WAS REJECTED
(see coral_domain_adaptation.py's own docstring, which this project has
already established as the "how domain-adapted can we make TSFNet's own
capacity go" ceiling):
- C-1 tried to make TSFNet's OWN, task-trained features generalize better
  across sessions. It failed (all 3 seeds worse than E3+E4 on both
  accuracy and kappa -- see run logs seed42/2024/999, domain_adapt_lambda
  =0.1). The project's own capacity, however aligned, could not do better.
- The project's central finding (documented seven times now: B3, cafe_d_model
  =128, larger MLP, HbT, G4 at two capacities, C-1) is that EVERY attempt to
  give the model MORE to learn FROM THIS PROJECT'S OWN LIMITED DATA makes
  things worse. LaBraM is the only remaining candidate that breaks this
  failure mode structurally: it brings capacity that was learned from
  ~2,500 hours of EEG data collected elsewhere, not from these 26 subjects.
  The model gets more capacity WITHOUT having to learn it from the scarce
  resource that has burned every other attempt.

WHY THIS IS THE "LOWEST-RISK VERSION" DESIGN FROM THE ROADMAP:
- ONLY the EEG-only auxiliary head (the `y_eeg` branch that feeds the
  paper_mean ensemble alongside `y_fusion` and `y_fnirs`) is touched.
- `eeg_s1`/`eeg_s2` (the internal temporal features that feed the EFGF
  gated-fusion layer and the CAFE cross-attention layer) are NEVER
  replaced. Swapping those would restructure the model's core fusion
  pathway -- the same "isolated domain" argument already used to justify
  G4's channel-selection experiment applies here verbatim: touch the
  EEG-only decision path, leave the fusion backbone alone.
- The backbone is FROZEN (zero gradient into its ~5.8M pretrained
  parameters). Only a light linear head (embed_dim -> 3 classes) is
  trained. This is deliberately the cheapest possible way to test whether
  foundation-model EEG capacity helps here at all, before ever considering
  a costlier fine-tuning variant.

DEFAULT IS AN EXACT NO-OP: `eeg_backbone='native'` (the default) never
imports LaBraMFrozenEEGHead, never touches `y_eeg`, and reproduces prior
behavior bit-for-bit. Only `eeg_backbone='labram_frozen'` activates this
path (`--eeg-backbone labram_frozen` CLI flag, see integration note below).

WHERE THE PRETRAINED WEIGHTS COME FROM (this module does NOT ship them):
This project uses the `braindecode` library's own `InterpolatedLaBraM`
implementation (Guetschel et al., an adaptation of the original LaBraM
code faithful to Jiang et al. 2024's released checkpoint format), NOT a
hand-rolled reimplementation. Two reasons this is preferable to writing a
custom loader against the original https://github.com/935963004/LaBraM
checkpoint directly:
  1. `InterpolatedLaBraM` already solves exactly the channel-mismatch
     problem this project's roadmap flagged ("نگاشت ۲۸ کانال با یک Conv1D
     (روش مقاله‌ی LEAD)") via `ChannelInterpolationLayer`: for every one of
     this project's 28 channels whose 10-5-system NAME matches one of
     LaBraM's 128 canonical channels (26 of 28 -- verified below), the
     projection is an exact one-hot permutation, not a learned/approximate
     mapping. Only the 2 channels without a canonical name match (AFF5h,
     AFF6h) fall back to spline interpolation from this project's own 3D
     electrode coordinates (already available -- see thesis methods
     chapter section 1-3). This is a STRICTLY more faithful channel
     mapping than a from-scratch learned Conv1D would be for 26/28
     channels, while still handling the 2 exceptions principled-ly.
  2. It is tested, maintained code, reducing the risk of a subtle
     patch-embedding or positional-encoding bug that a hand port from the
     original TensorFlow-esque checkpoint format could introduce --
     directly serving this project's standing methodology rule of
     "test the real integration, not just an isolated function."

Install once (this exact pinned combination was validated for this file):
    pip install "braindecode[hub]" --break-system-packages

CORRECTED CHECKPOINT SOURCE (this matters -- an earlier draft of this file
got this wrong and it is worth recording why, per this project's standing
rule of being explicit about engineering mistakes and how they were
caught): as of braindecode >=1.6, `braindecode.models.Labram` REQUIRES
`chs_info` to match `LABRAM_CHANNEL_ORDER` exactly (128 canonical
channels) -- this is what `InterpolatedLaBraM` builds internally. The
community-contributed checkpoint at
https://huggingface.co/braindecode/Labram-Braindecode
(`braindecode_labram_base.pt`) is a "shallow conversion" predating this
change: it was converted for a plain 64-channel / 1600-sample
configuration, and loading it into the current 128-channel
`InterpolatedLaBraM` raises a hard shape mismatch on both
`position_embedding` (checkpoint: 64+1 channels; model: 128+1) and
`temporal_embedding` (checkpoint: 8+1 patches i.e. 1600 samples at
patch_size=200; model: whatever `n_times` this file requests) -- this
was caught empirically (see the integration log this file's git history
was validated against) rather than assumed, exactly per this project's
"test the real integration" rule.

The CORRECT, currently-maintained checkpoint is
`"braindecode/labram-pretrained"` on the Hugging Face Hub, loaded via
`InterpolatedLaBraM.from_pretrained(...)` (NOT a manual
`torch.load` + `load_state_dict`). This one call downloads the checkpoint
AND resolves both the channel-count and time-length adaptation correctly
for whatever `chs_info`/`n_times` this project passes in -- see
`LaBraMFrozenEEGHead.__init__` below. Reference:
https://braindecode.org/stable/auto_examples/model_building/plot_channel_interpolation.html
("Loading Pretrained Foundation Models on Arbitrary Channel Sets").

Actual pretrained weights: `InterpolatedLaBraM` builds LaBraM's published
architecture (embed_dim=200, depth=12, heads=10, patch_size=200 -- i.e.
1-second patches at this project's 200 Hz -- confirmed against the
official repo's `labram_base_patch200_1600_8k_vocab` config).

REFERENCE:
Jiang, W.-B., Zhao, L.-M., & Lu, B.-L. (2024). Large Brain Model for
Learning Generic Representations with Tremendous EEG Data in BCI. ICLR
2024 (spotlight). https://openreview.net/forum?id=QzTpTRVtrP
Code: https://github.com/935963004/LaBraM

CONFLICT NOTE (already flagged in the roadmap, repeated here for
visibility at the point of use): LaBraM was pretrained on broadband raw
EEG. Therefore C-2 (this file) and D-1 (five-band frequency decomposition)
are mutually exclusive -- do not enable both `--eeg-backbone labram_frozen`
and a band-decomposed EEG input in the same run.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# This project's fixed 28-channel montage (10-5 system positions), in the
# exact channel order `_eeg_channel_windows.npy` was written in (see G4 /
# Preproc_tsfnet_nback_badch.py). This order MUST match the array's
# channel axis exactly, or every channel gets silently mapped to the
# wrong LaBraM canonical position.
# ----------------------------------------------------------------------
PROJECT_EEG_CHANNELS = [
    "Fp1", "Fp2", "AFF5h", "AFF6h", "AFz", "F1", "F2", "FC1", "FC2", "FC5",
    "FC6", "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6", "Pz",
    "P3", "P4", "P7", "P8", "POz", "O1", "O2",
]
assert len(PROJECT_EEG_CHANNELS) == 28, (
    f"expected 28 channels (this project's fixed EEG montage), got "
    f"{len(PROJECT_EEG_CHANNELS)}"
)

PROJECT_SFREQ = 200.0     # Hz, this project's fixed EEG sampling rate
PROJECT_N_TIMES = 2000    # samples per 10s window at 200 Hz
LABRAM_PATCH_SIZE = 200   # 1 second per patch, matches PROJECT_SFREQ exactly


def _build_chs_info(ch_names, sfreq):
    """
    Build MNE-style chs_info (list[dict] with 'ch_name' and 'loc' keys) for
    this project's montage, via the standard 10-5 system positions.
    Returns the list, plus a diagnostic dict reporting how many channels
    will be exact-name-matched to LaBraM's canonical list vs. how many
    will fall back to spline interpolation from 3D position.
    """
    import mne
    from braindecode.models.labram import LABRAM_CHANNEL_ORDER

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1005")
        info.set_montage(montage, match_case=False, on_missing="warn")

    canon_lower = {c.lower() for c in LABRAM_CHANNEL_ORDER}
    matched = [c for c in ch_names if c.lower() in canon_lower]
    interpolated = [c for c in ch_names if c.lower() not in canon_lower]

    diagnostic = {
        "n_total": len(ch_names),
        "n_exact_name_match": len(matched),
        "n_spline_interpolated": len(interpolated),
        "interpolated_channels": interpolated,
    }
    return info["chs"], diagnostic


class LaBraMFrozenEEGHead(nn.Module):
    """
    Frozen LaBraM backbone (pretrained EEG foundation model) + a single
    trainable linear classification head, producing a probability vector
    over this project's 3 classes (0-/2-/3-back) from raw channel-space
    EEG -- i.e. from the SAME `(N, 28, 2000)` array as
    `_eeg_channel_windows.npy` (already on disk from priority-G4), not
    from TSFNet's internal `eeg_s1`/`eeg_s2` grid features.

    This is designed as a DROP-IN REPLACEMENT for the EEG-only head's
    output `y_eeg` (see the integration note at the bottom of this file)
    -- it is NOT a replacement for the EEG branch feeding EFGF/CAFE.

    pretrained : bool, default True. If True (the only valid setting for
        an actual accept/reject run), the backbone is built AND its
        pretrained weights loaded in one call via
        `InterpolatedLaBraM.from_pretrained(hub_repo_id, ...)` -- this
        requires network access to the Hugging Face Hub and
        `pip install "braindecode[hub]"`. If False (used ONLY by this
        file's own structural self-test below, never for a real run), the
        backbone is randomly initialized with no network access required.
    hub_repo_id : str, default "braindecode/labram-pretrained" -- the
        CURRENT, correctly-mapped checkpoint (see the module docstring's
        "CORRECTED CHECKPOINT SOURCE" note for why this is NOT the same as
        the community `braindecode/Labram-Braindecode` repo).
    num_classes : int, default 3 (this project's 0-/2-/3-back task).
    """

    def __init__(self, pretrained=True, hub_repo_id="braindecode/labram-pretrained",
                 num_classes=3, ch_names=None, sfreq=PROJECT_SFREQ,
                 n_times=PROJECT_N_TIMES, verbose=True):
        super().__init__()
        from braindecode.models.labram import InterpolatedLaBraM

        ch_names = ch_names if ch_names is not None else PROJECT_EEG_CHANNELS
        chs_info, diag = _build_chs_info(ch_names, sfreq)
        if verbose:
            print(
                f"[LaBraMFrozenEEGHead] channel mapping: "
                f"{diag['n_exact_name_match']}/{diag['n_total']} channels "
                f"matched to LaBraM's canonical 128-channel layout by exact "
                f"name (one-hot, no interpolation); "
                f"{diag['n_spline_interpolated']} channel(s) "
                f"{diag['interpolated_channels']} spline-interpolated from "
                f"this project's 3D electrode positions."
            )

        if pretrained:
            # ONE call both builds the 128-channel canonical backbone AND
            # loads its pretrained weights, correctly handling this
            # project's chs_info/n_times -- see the module docstring's
            # "CORRECTED CHECKPOINT SOURCE" note for why this replaced an
            # earlier (broken) manual torch.load + load_state_dict path.
            self.backbone = InterpolatedLaBraM.from_pretrained(
                hub_repo_id,
                chs_info=chs_info,
                n_times=n_times,
                n_outputs=num_classes,   # placeholder; final_layer replaced below
                patch_size=LABRAM_PATCH_SIZE,
                strict=False,  # the interpolation matrix is not in the checkpoint
            )
            if verbose:
                print(
                    f"[LaBraMFrozenEEGHead] loaded pretrained backbone from "
                    f"Hugging Face Hub repo '{hub_repo_id}' "
                    f"(chs_info + n_times={n_times} resolved via "
                    f"InterpolatedLaBraM.from_pretrained, strict=False since "
                    f"the interpolation matrix is deliberately not part of "
                    f"the checkpoint)."
                )
        else:
            self.backbone = InterpolatedLaBraM(
                chs_info=chs_info,
                n_outputs=num_classes,   # placeholder; final_layer replaced below
                n_times=n_times,
                sfreq=sfreq,
                n_chans=len(ch_names),
                trainable=False,  # the channel-interpolation matrix itself is fixed
            )
            if verbose:
                warnings.warn(
                    "[LaBraMFrozenEEGHead] pretrained=False -- backbone is "
                    "RANDOMLY INITIALIZED. This is only valid for this "
                    "file's own structural self-test, never for an actual "
                    "C-2 accept/reject run.",
                    stacklevel=2,
                )

        # The wrapped model's own final_layer (Linear(embed_dim, n_outputs))
        # is TASK-TRAINED-FROM-SCRATCH by construction and was never part of
        # the pretrained checkpoint; replace it with Identity so we get the
        # raw pooled embed_dim=200 feature, and attach our OWN light linear
        # head on top. This makes explicit (rather than accidentally true)
        # that only this head is trainable.
        embed_dim = self.backbone.final_layer.in_features
        self.backbone.final_layer = nn.Identity()

        # ---- Freeze the entire backbone (including the channel-interpolation
        # buffer, which is non-trainable by construction already). ----
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ---- The ONLY trainable parameters in this module. ----
        self.head = nn.Linear(embed_dim, num_classes)

    def train(self, mode: bool = True):
        """
        Override train() so the frozen backbone is ALWAYS kept in eval()
        mode (freezing its dropout/drop-path stochasticity, matching a
        genuinely frozen feature extractor), regardless of whether the
        outer training loop calls model.train() on this module. Only
        `self.head` toggles normally with the requested mode.

        This guards against the standard frozen-backbone bug: calling
        `.train()` on a parent module recursively sets `.training = True`
        on every submodule by default, silently re-enabling dropout inside
        a backbone that is supposed to be fully frozen.
        """
        self.training = mode
        self.backbone.eval()       # always -- backbone never trains
        self.head.train(mode)      # only the head follows the outer mode
        return self

    def forward(self, x_eeg_channels, return_probs=True):
        """
        x_eeg_channels : (N, 28, 2000) raw channel-space EEG, same format
            as `_eeg_channel_windows.npy`.
        return_probs : if True (default, matching this project's
            convention that all sub-head outputs are POST-softmax
            probabilities -- see label_smoothing_loss's own docstring
            note on this), applies softmax before returning. If False,
            returns raw logits (e.g. for use with
            nn.CrossEntropyLoss(label_smoothing=...) directly, which
            expects logits).
        """
        assert x_eeg_channels.dim() == 3 and x_eeg_channels.shape[1:] == (
            len(PROJECT_EEG_CHANNELS), PROJECT_N_TIMES,
        ), (
            f"expected (N, {len(PROJECT_EEG_CHANNELS)}, {PROJECT_N_TIMES}), "
            f"got {tuple(x_eeg_channels.shape)}"
        )
        with torch.no_grad():
            # Backbone is frozen: no need to build a graph through it at
            # all, saving both memory and compute (this is NOT the same as
            # requires_grad=False alone -- that would still build and then
            # discard the graph; no_grad skips building it in the first
            # place, which matters a lot here given the transformer's
            # O((n_channels * n_patches)^2) attention memory cost -- see
            # the batch-size note in this file's self-test output).
            features = self.backbone(x_eeg_channels)  # (N, embed_dim), pooled
        logits = self.head(features)
        return F.softmax(logits, dim=-1) if return_probs else logits


# ============================================================================
# Numeric debug / verification script -- run this file directly
# (`python labram_eeg_branch.py`) before wiring it into training, per this
# project's standing methodology rule. Uses a RANDOMLY INITIALIZED backbone
# (no pretrained checkpoint required) since these checks are purely
# structural: shape correctness, gradient isolation, and frozen-mode
# correctness are independent of which weights the backbone holds.
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 70)
    print("Building LaBraMFrozenEEGHead (random-init backbone, structural "
          "self-test only -- pretrained=False, no network access needed)...")
    print("=" * 70)
    model = LaBraMFrozenEEGHead(pretrained=False, num_classes=3)

    # ---- Check 1: output shape and valid probability simplex. ----
    # NOTE: kept at batch_size=2 deliberately -- see the memory note printed
    # at the end of this block. Use a larger batch on an actual GPU only
    # after confirming available VRAM empirically.
    batch_size = 2
    x = torch.randn(batch_size, len(PROJECT_EEG_CHANNELS), PROJECT_N_TIMES)
    probs = model(x, return_probs=True)
    assert probs.shape == (batch_size, 3), f"expected ({batch_size}, 3), got {probs.shape}"
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(batch_size), atol=1e-5), (
        f"expected each row to sum to 1 (softmax output), got {row_sums.tolist()}"
    )
    assert (probs >= 0).all(), "probabilities must be non-negative"
    print(f"[PASS] check 1: output shape {tuple(probs.shape)} is a valid "
          f"probability distribution (row sums: {row_sums.tolist()}).")

    # ---- Check 2: backbone is fully frozen; only head has requires_grad. ----
    backbone_trainable = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_trainable = [n for n, p in model.head.named_parameters() if p.requires_grad]
    assert len(backbone_trainable) == 0, (
        f"expected ZERO trainable backbone parameters, found: {backbone_trainable}"
    )
    assert len(head_trainable) == 2, (  # weight + bias
        f"expected exactly 2 trainable head parameters (weight, bias), "
        f"found {len(head_trainable)}: {head_trainable}"
    )
    n_backbone_params = sum(p.numel() for p in model.backbone.parameters())
    n_head_params = sum(p.numel() for p in model.head.parameters())
    print(f"[PASS] check 2: backbone fully frozen (0/{n_backbone_params:,} "
          f"parameters trainable); head fully trainable "
          f"({n_head_params:,} parameters: {head_trainable}).")

    # ---- Check 3: gradient flows to the head after backward(), and NOT
    # into the backbone (confirmed both by requires_grad=False AND by the
    # no_grad() context inside forward() -- belt and suspenders). ----
    model.train()  # should NOT re-enable backbone dropout, per check 4 below
    x2 = torch.randn(batch_size, len(PROJECT_EEG_CHANNELS), PROJECT_N_TIMES)
    target = torch.tensor([0, 2])
    logits = model(x2, return_probs=False)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    assert model.head.weight.grad is not None and model.head.weight.grad.abs().sum() > 0, (
        "expected a nonzero gradient on the head's weight"
    )
    backbone_grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert len(backbone_grads) == 0, (
        f"expected NO gradients in the backbone, found {len(backbone_grads)} "
        f"non-None .grad tensors -- freezing failed"
    )
    print("[PASS] check 3: gradient flows to the head only; the frozen "
          "backbone received zero gradients.")

    # ---- Check 4: train()/eval() toggling never re-enables backbone
    # stochasticity (the standard frozen-backbone bug this override guards
    # against). ----
    model.train()
    backbone_training_flags = {m.training for m in model.backbone.modules()}
    assert backbone_training_flags == {False}, (
        f"expected every backbone submodule to remain in eval mode after "
        f"model.train(), found training flags: {backbone_training_flags}"
    )
    assert model.head.training is True, "expected the head to respect model.train()"
    model.eval()
    assert model.head.training is False, "expected the head to respect model.eval()"
    print("[PASS] check 4: model.train() leaves every backbone submodule in "
          "eval() (frozen dropout/drop-path); only the head's mode toggles.")

    # ---- Check 5: determinism -- two forward passes on the same input
    # produce identical output (confirms no residual stochasticity leaking
    # from the nominally-frozen backbone, e.g. via a dropout layer whose
    # .training flag check 4 might have missed). ----
    model.eval()
    with torch.no_grad():
        out_a = model(x, return_probs=True)
        out_b = model(x, return_probs=True)
    assert torch.allclose(out_a, out_b, atol=1e-7), (
        "expected identical output for two forward passes on the same "
        "input in eval mode -- found nondeterminism"
    )
    print("[PASS] check 5: two forward passes on identical input produce "
          "identical output (no residual stochasticity).")

    # ---- Check 6: channel-mapping diagnostic (informational, already
    # printed above at model construction, re-asserted here as a hard
    # check so a future braindecode version silently changing its
    # canonical channel list is caught immediately). ----
    _, diag = _build_chs_info(PROJECT_EEG_CHANNELS, PROJECT_SFREQ)
    assert diag["n_exact_name_match"] == 26, (
        f"expected 26/28 channels to exact-name-match LaBraM's canonical "
        f"list, got {diag['n_exact_name_match']} -- if this changed, check "
        f"whether braindecode's LABRAM_CHANNEL_ORDER was updated"
    )
    assert set(diag["interpolated_channels"]) == {"AFF5h", "AFF6h"}, (
        f"expected exactly AFF5h/AFF6h to require spline interpolation, "
        f"got {diag['interpolated_channels']}"
    )
    print(f"[PASS] check 6: channel mapping confirmed stable "
          f"(26/28 exact name match, {diag['interpolated_channels']} "
          f"spline-interpolated).")

    print("\nALL CHECKS PASSED.")
    print(
        "\n[MEMORY NOTE] LaBraM's attention operates over "
        "128 channels x 10 one-second patches = 1280 tokens per sample. "
        "Empirically (CPU, this sandbox), a forward+backward at "
        "batch_size=2 used ~2.2 GB beyond the frozen backbone's own "
        "footprint. This does NOT grow with the backbone being frozen "
        "(no_grad skips backbone activations, but the trainable head's "
        "backward is negligible) -- the memory driver is the frozen "
        "backbone's OWN forward activations inside no_grad, which still "
        "materializes the full 12-layer x 1280-token attention stack. "
        "Determine the actual usable batch size empirically on the real "
        "GPU before the accept/reject run; do not assume the project's "
        "usual batch size transfers unchanged from TSFNet's own (much "
        "smaller) attention footprint."
    )


# ============================================================================
# INTEGRATION NOTE (for wiring into tsfnet_model_notebook.py / run_all_subjects.py)
#
# 1. In run_all_subjects.py, add a new CLI flag:
#
#        --eeg-backbone {native, labram_frozen}   (default: native, EXACT NO-OP)
#
#    threaded through exactly like --domain-adapt-lambda, and gate the
#    import of this file behind `if args.eeg_backbone == 'labram_frozen':`
#    so `native` runs never pay LaBraM's import/memory cost.
#
# 2. Build the head ONCE per fold (it does not need to be rebuilt per
#    epoch), right alongside where the fold's TSFNet model is constructed:
#
#        labram_head = None
#        if eeg_backbone == 'labram_frozen':
#            from labram_eeg_branch import LaBraMFrozenEEGHead
#            labram_head = LaBraMFrozenEEGHead(
#                pretrained=True,
#                hub_repo_id="braindecode/labram-pretrained",
#                num_classes=3,
#            ).to(device)
#
#    (Requires `pip install "braindecode[hub]"` and network access to the
#    Hugging Face Hub at construction time -- do NOT manually download
#    "braindecode/Labram-Braindecode"'s .pt file and pass it as a raw
#    state_dict; that community checkpoint predates braindecode's current
#    128-channel canonical Labram and will raise a shape mismatch on
#    position_embedding/temporal_embedding. `from_pretrained` against
#    "braindecode/labram-pretrained" is the correct, currently-maintained
#    path -- see this file's module docstring for the full story of how
#    this was caught.)
#
# 3. Data: this file expects the SAME `(N, 28, 2000)` raw channel-space
#    array already on disk from priority-G4
#    (`{subject_id}_eeg_channel_windows.npy`) -- no new preprocessing run
#    is required, matching the roadmap's own "کم‌ریسک‌ترین نسخه" design.
#
# 4. Inside the fold's training loop, wherever `y_eeg` (the EEG-only head's
#    probability output, one of the three inputs to the paper_mean fusion
#    alongside y_fusion/y_fnirs) is currently computed from TSFNet's native
#    EEG-only head:
#
#        if labram_head is not None:
#            y_eeg = labram_head(x_eeg_channels_raw)   # replaces the native y_eeg
#        else:
#            y_eeg = <existing native EEG-only head computation>   # unchanged
#
#    `x_eeg_channels_raw` must be the RAW channel-space tensor (28, 2000),
#    not TSFNet's internal `eeg_s1`/`eeg_s2` grid-mapped features -- those
#    two are DIFFERENT tensors already computed at DIFFERENT points in
#    TSFNet's forward pass; do not conflate them.
#
# 5. Optimizer: when eeg_backbone=='labram_frozen', add ONLY
#    `labram_head.head.parameters()` to the optimizer's parameter group
#    (never `labram_head.backbone.parameters()` -- they have
#    requires_grad=False already, but an optimizer holding zero-grad
#    parameters wastes memory in its state and risks a future accidental
#    unfreeze going unnoticed).
#
# 6. Per this project's standing rule #11 (check the [ABLATION] banner
#    before interpreting any result): confirm the banner shows
#    `eeg_backbone='labram_frozen'` and that every OTHER flag is at its
#    E3+E4-accepted value (adabn=True, plus whatever E3/E4 already set),
#    with domain_adapt_lambda=0.0 (C-1 was rejected -- do not accidentally
#    combine a rejected intervention with C-2's own run).
#
# 7. Acceptance/rejection: identical protocol to every other intervention
#    in this project -- 3 independent seeds, isolated variable, improvement
#    in BOTH accuracy AND kappa in >=2/3 seeds vs. the E3+E4 baseline
#    (61.20%, kappa=0.4179), with recall-per-class and confusion matrix
#    checked alongside overall accuracy (B-5's own lesson: overall accuracy
#    alone can mislead).
#
# 8. Reminder (from this file's own docstring): do NOT combine
#    `--eeg-backbone labram_frozen` with a band-decomposed EEG input
#    (D-1) in the same run -- LaBraM was pretrained on broadband raw EEG.
# ============================================================================
