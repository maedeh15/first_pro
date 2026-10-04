"""
cbsi_transform.py -- Priority B-5: CBSI (Correlation-Based Signal
Improvement) for fNIRS, REPLACING the raw HbO/HbR channels rather than
adding a new one (per the roadmap's explicit design: "ترکیب همبستگی‌محور
HbO/HbR، جایگزینی نه افزودن").

REFERENCE: Cui, X., Bray, S., & Reiss, A. L. (2010). "Functional near
infrared spectroscopy (NIRS) signal improvement based on negative
correlation between oxygenated and deoxygenated hemoglobin dynamics."
NeuroImage, 49(4), 3039-3046.

WHY CBSI: real hemodynamic activity (the actual cognitive-load signal this
project cares about) produces NEGATIVELY correlated HbO/HbR fluctuations
(oxygenated hemoglobin rises as deoxygenated falls, and vice versa).
Motion artifacts and other shared instrumental noise, in contrast, tend to
move HbO and HbR in the SAME direction (positively correlated). CBSI
exploits this asymmetry directly in the signal domain:

    alpha = std_t(HbO) / std_t(HbR)
    C(t)  = (HbO(t) - alpha * HbR(t)) / 2

Subtracting alpha*HbR from HbO exactly reconstructs the true anti-
correlated hemodynamic component (since for genuine activity HbR ~
-HbO/alpha, so HbO - alpha*HbR ~ 2*HbO, recovered here on HbO's original
scale by the final /2), while any shared, positively-correlated noise
component (HbO and HbR moving together, e.g. motion) approximately
CANCELS in the same subtraction, since alpha*(+HbR) and HbO then partially
overlap with the SAME sign and are subtracted rather than added. alpha
rescales HbR onto HbO's amplitude scale before combining them, using each
location's own local standard deviation (computed along the time axis)
rather than a fixed global constant, so the correction adapts to each
pixel/segment/sample's own signal-to-noise regime.

REPLACEMENT, NOT ADDITION: the corrected signal is written into channel 0
(HbO's slot) and every other chromophore channel is zeroed, so the
resulting tensor has the EXACT SAME SHAPE as the raw input and needs NO
architecture change (FNIRSBranch's conv1 in_channels stays untouched) --
this mirrors the existing hbo_only/hbr_only masking pattern already used
for the fnirs_chroma_mode ablation (priority-A4), just with a data-
DERIVED channel instead of a raw one.

USAGE (inside TSFNet.forward, see integration note at the bottom of this
file for the exact insertion point):

    from cbsi_transform import apply_cbsi_transform

    if self.fnirs_chroma_mode == "cbsi":
        x_fnirs = apply_cbsi_transform(x_fnirs)
    else:
        x_fnirs = x_fnirs * self._fnirs_chroma_mask
"""

import torch


def apply_cbsi_transform(x_fnirs, eps=1e-6):
    """
    x_fnirs : (..., T, C) tensor with C >= 2; channel 0 = HbO, channel 1 =
              HbR (this project's channel convention, priority-G2a). Any
              leading dimensions (batch, segment, height, width) are
              treated identically -- alpha and the correction are computed
              independently per leading-dimension index, along the time
              axis T (second-to-last).
    eps     : numerical floor added to std_t(HbR) to avoid division by
              zero for a (near-)constant HbR trace.

    Returns a tensor of the SAME shape as x_fnirs: channel 0 holds the
    CBSI-corrected signal, every other channel is zero.
    """
    assert x_fnirs.shape[-1] >= 2, (
        f"CBSI needs at least 2 chromophore channels (HbO, HbR), got "
        f"shape {tuple(x_fnirs.shape)}"
    )
    hbo = x_fnirs[..., 0]
    hbr = x_fnirs[..., 1]

    std_hbo = hbo.std(dim=-1, keepdim=True)
    std_hbr = hbr.std(dim=-1, keepdim=True)
    alpha = std_hbo / (std_hbr + eps)

    corrected = (hbo - alpha * hbr) / 2.0

    out = torch.zeros_like(x_fnirs)
    out[..., 0] = corrected
    return out


# ============================================================================
# Numeric debug / verification script -- run this file directly
# (`python cbsi_transform.py`) before wiring it into the model, per this
# project's standing methodology rule.
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- Check 1: output shape matches input shape exactly, and every
    # channel beyond 0 is exactly zero (the "replacement" contract). ----
    x = torch.randn(4, 21, 16, 16, 30, 2)
    out = apply_cbsi_transform(x)
    assert out.shape == x.shape, f"shape changed: {x.shape} -> {out.shape}"
    assert torch.all(out[..., 1:] == 0.0), "non-HbO channels are not all zero"
    print(f"[PASS] check 1: output shape matches input ({tuple(out.shape)}), "
          f"and every channel beyond index 0 is exactly zero.")

    # ---- Check 2: for a genuine anti-correlated hemodynamic-like signal
    # (HbR = -k * HbO exactly, k>0), CBSI's alpha-scaling exactly recovers
    # the original clean waveform (same sign, same shape, and -- since
    # alpha = std(HbO)/std(HbR) exactly compensates the k scaling here --
    # the same amplitude as HbO alone). This confirms the formula correctly
    # reconstructs a perfectly anti-correlated true signal rather than
    # distorting or canceling it. ----
    T = 100
    t = torch.linspace(0, 4 * 3.14159, T)
    true_signal = torch.sin(t)  # a clean "hemodynamic-like" oscillation
    k = 0.6
    hbo_true = true_signal.clone()
    hbr_true = -k * true_signal  # perfectly anti-correlated, as real fNIRS is
    x_true = torch.stack([hbo_true, hbr_true], dim=-1).unsqueeze(0)  # (1, T, 2)
    corrected_true = apply_cbsi_transform(x_true)[..., 0].squeeze(0)
    corr = torch.corrcoef(torch.stack([corrected_true, true_signal]))[0, 1].item()
    assert corr > 0.99, f"expected near-perfect positive correlation with the true signal, got {corr:.4f}"
    assert torch.allclose(corrected_true, hbo_true, atol=1e-4), (
        f"for a perfectly anti-correlated pair, CBSI should exactly reconstruct "
        f"the clean waveform (alpha compensates the HbR scale exactly here) -- "
        f"max abs diff = {(corrected_true - hbo_true).abs().max().item():.4f}"
    )
    print(f"[PASS] check 2: for a genuine anti-correlated (hemodynamic-like) signal, "
          f"CBSI output correlates {corr:.4f} with and exactly reconstructs the "
          f"original clean waveform (alpha correctly compensates the HbR/HbO scale ratio).")

    # ---- Check 3: for a motion-artifact-like signal where HbO and HbR move
    # in the SAME direction (positively correlated, the classic motion-
    # artifact signature), CBSI should substantially SUPPRESS the shared
    # component relative to either raw channel's own amplitude. ----
    motion = torch.randn(T) * 3.0  # a large shared spike/drift
    hbo_motion = motion + 0.05 * torch.randn(T)
    hbr_motion = motion + 0.05 * torch.randn(T)  # same-direction, i.e. positively correlated
    x_motion = torch.stack([hbo_motion, hbr_motion], dim=-1).unsqueeze(0)
    corrected_motion = apply_cbsi_transform(x_motion)[..., 0].squeeze(0)
    assert corrected_motion.std().item() < 0.5 * hbo_motion.std().item(), (
        f"CBSI should strongly suppress positively-correlated (motion-like) "
        f"shared noise; got corrected std={corrected_motion.std().item():.3f} vs "
        f"raw HbO std={hbo_motion.std().item():.3f} (expected corrected << raw)."
    )
    print(f"[PASS] check 3: for a motion-artifact-like positively-correlated signal, "
          f"CBSI suppresses the shared component (corrected std="
          f"{corrected_motion.std().item():.3f} vs raw HbO std={hbo_motion.std().item():.3f}).")

    # ---- Check 4: gradient flows correctly back to both input channels. ----
    x_grad = torch.randn(2, 5, 4, 4, 10, 2, requires_grad=True)
    out_grad = apply_cbsi_transform(x_grad)
    out_grad.sum().backward()
    assert x_grad.grad is not None
    assert x_grad.grad[..., 0].abs().sum() > 0, "no gradient reached the HbO input channel"
    assert x_grad.grad[..., 1].abs().sum() > 0, "no gradient reached the HbR input channel"
    print("[PASS] check 4: gradients flow correctly back to both the HbO and HbR "
          "input channels (both contribute to the corrected output).")

    # ---- Check 5: division-by-zero guard -- a constant (zero-variance) HbR
    # trace must not produce NaN/Inf in the output. ----
    x_const_hbr = torch.randn(1, 1, 1, 1, 10, 2)
    x_const_hbr[..., 1] = 5.0  # perfectly constant HbR -> std = 0
    out_const = apply_cbsi_transform(x_const_hbr)
    assert torch.isfinite(out_const).all(), "NaN/Inf produced for a zero-variance HbR trace"
    print("[PASS] check 5: a zero-variance HbR trace does not produce NaN/Inf "
          "(division-by-zero guard via eps works correctly).")

    print("\nALL CHECKS PASSED.")


# ============================================================================
# INTEGRATION NOTE (for wiring into tsfnet_model_notebook.py's TSFNet):
#
# 1. In TSFNet.__init__, extend the _KEEP dict validation to also accept
#    "cbsi" as a valid fnirs_chroma_mode value (it does not need an entry
#    in _KEEP itself, since it never uses the static mask-multiply path --
#    just make sure the ValueError check doesn't reject it):
#
#        _KEEP = {"both": None, "hbo_only": (0,), "hbr_only": (1,),
#                 "hbt_only": (2,), "hbo_hbr": (0, 1), "cbsi": None}
#        # NOTE: "cbsi" reuses `None` (all-ones mask) as a placeholder only
#        # so it passes existing validation; forward() below branches BEFORE
#        # the mask is ever applied for this mode, so the mask itself is
#        # never actually used when fnirs_chroma_mode == "cbsi".
#
# 2. In TSFNet.forward, where the existing mask multiply happens:
#
#        # before:
#        x_fnirs = x_fnirs * self._fnirs_chroma_mask
#
#        # after:
#        if self.fnirs_chroma_mode == "cbsi":
#            from cbsi_transform import apply_cbsi_transform
#            x_fnirs = apply_cbsi_transform(x_fnirs)
#        else:
#            x_fnirs = x_fnirs * self._fnirs_chroma_mask
#
# 3. In run_all_subjects.py, "cbsi" simply becomes a new valid value for the
#    existing --fnirs-chroma-mode argument -- no new CLI flag needed, since
#    this ablation lives on the SAME axis (which chromophore information
#    reaches the network) as the already-existing hbo_only/hbr_only modes.
# ============================================================================
