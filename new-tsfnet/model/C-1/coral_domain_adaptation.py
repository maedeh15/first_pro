"""
coral_domain_adaptation.py -- Priority C-1: deepening domain adaptation
beyond AdaBN, via a Deep CORAL feature-alignment loss (Sun & Saenko, 2016,
"Deep CORAL: Correlation Alignment for Deep Domain Adaptation").

WHY THIS IS THE NATURAL NEXT STEP AFTER E3/E4 (see also E4's own docstring
in tsfnet_model_notebook.py, which this directly extends):
- E3 (per-session normalization) fixed session shift at the INPUT level.
- E4 (AdaBN) fixed the part of it surviving into BatchNorm's internal
  running statistics -- but ONLY the first and second marginal moments per
  channel, and only AFTER training finishes (a single forward pass, no
  gradients).
- C-1 goes one step further: it aligns the full second-order feature
  STATISTICS (covariance structure, not just per-channel variance) of the
  512-dim `enhanced_fnirs` representation between the training sessions
  (source) and the held-out session's unlabeled inputs (target), and does
  so DURING training itself, as an additional loss term the optimizer
  actively minimizes -- not a post-hoc statistic swap.

WHY fNIRS SPECIFICALLY (matching the roadmap's "خصوصا شاخه‌ی fNIRS"):
priority-G3 already established, on this project's own data, that fNIRS's
session-to-session distributional shift is roughly 4x larger than EEG's.
Aligning `enhanced_fnirs` (the FNIRSBranch's own 512-dim pooled feature,
already computed by TSFNet.forward and available in `aux`) is therefore
where this technique's cost (one extra forward pass per training batch,
plus the CORAL loss term) buys the most plausible benefit, precisely the
same targeting logic already used for E3/E4's isolated-variable design.

CORAL FORMULA (Sun & Saenko, 2016, Eq. 1-2):
    C_s = covariance(source_features)     # (d, d)
    C_t = covariance(target_features)     # (d, d)
    L_CORAL = 1 / (4*d^2) * || C_s - C_t ||_F^2
where d is the feature dimensionality (512 here) and ||.||_F is the
Frobenius norm. This is added to the existing tsfnet_loss total with a
weight `domain_adapt_lambda` (0.0 = exact no-op, reproducing prior
behavior bit-for-bit).

THIS IS TRANSDUCTIVE IN EXACTLY THE SAME SENSE AS E3/E4 (must be reported
alongside them in the thesis): it uses the held-out session's UNLABELED
inputs during training, never its labels. This is standard in the
BCI/domain-adaptation literature (Sun & Saenko's own setting is
unsupervised domain adaptation) and is a direct, explicit deepening of the
same transductive-access pattern this project already uses and documents
for E3/E4 -- not a new category of assumption.

USAGE (see the bottom of this file for the exact train_one_fold
integration point):

    from coral_domain_adaptation import coral_loss, InfiniteTargetIterator

    target_iter = InfiniteTargetIterator(target_loader)   # built once, outside the epoch loop
    ...
    x_eeg_t, x_fnirs_t, _, x_conn_t, x_chan_t = target_iter.next_batch()
    with torch.no_grad():
        pass  # (target forward pass below is NOT no_grad -- see integration note)
    _, _, _, _, aux_t = model(x_eeg_t.to(device), x_fnirs_t.to(device), x_conn_t, x_chan_t)
    coral = coral_loss(aux["enhanced_fnirs"], aux_t["enhanced_fnirs"])
    loss = loss + domain_adapt_lambda * coral
"""

import torch


def coral_loss(source_feat, target_feat, eps=1e-6):
    """
    source_feat : (N_s, d) tensor of source-domain (training-session)
                  features -- e.g. aux['enhanced_fnirs'] from a training
                  minibatch.
    target_feat : (N_t, d) tensor of target-domain (held-out session)
                  features from the SAME feature extractor -- e.g.
                  aux['enhanced_fnirs'] from an unlabeled minibatch drawn
                  from the test session.
    eps         : numerical floor added to the (N-1) divisor guard for a
                  degenerate single-sample batch (see check 4 below).

    Returns a scalar, differentiable CORAL loss (Sun & Saenko 2016, Eq. 1-2):
    the squared Frobenius norm of the difference between the source and
    target feature covariance matrices, normalized by 4*d^2 so the loss
    scale is roughly independent of feature dimensionality (as in the
    original paper).
    """
    assert source_feat.dim() == 2 and target_feat.dim() == 2, (
        f"expected 2D (N, d) feature tensors, got source={tuple(source_feat.shape)}, "
        f"target={tuple(target_feat.shape)}"
    )
    assert source_feat.shape[1] == target_feat.shape[1], (
        f"feature dimensionality mismatch: source has {source_feat.shape[1]}, "
        f"target has {target_feat.shape[1]}"
    )
    d = source_feat.shape[1]

    def _covariance(x):
        n = x.shape[0]
        x_centered = x - x.mean(dim=0, keepdim=True)
        # (N-1) denominator = the standard unbiased covariance estimator;
        # clamped at 1 so a degenerate single-sample batch returns an
        # all-zero (not NaN/Inf) covariance matrix instead of dividing by 0.
        denom = max(n - 1, 1)
        return (x_centered.t() @ x_centered) / denom

    c_s = _covariance(source_feat)
    c_t = _covariance(target_feat)
    return ((c_s - c_t) ** 2).sum() / (4 * d * d)


class InfiniteTargetIterator:
    """
    Cycles a DataLoader indefinitely, so a training loop can pull one
    unlabeled target-session minibatch per source minibatch even though
    the target (held-out session) loader has a different number of
    batches than the source (training sessions) loader -- exactly the
    same kind of loader-length mismatch AdaBN's target_loader already
    tolerates, just consumed per-batch here instead of once at the end.
    """

    def __init__(self, loader):
        self.loader = loader
        self._iterator = iter(self.loader)

    def next_batch(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            return next(self._iterator)


# ============================================================================
# Numeric debug / verification script -- run this file directly
# (`python coral_domain_adaptation.py`) before wiring it into training, per
# this project's standing methodology rule.
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- Check 1: CORAL loss is exactly zero for two batches drawn from
    # the SAME distribution with the SAME covariance (identical source and
    # target feature sets) -- the "no domain gap" sanity case. ----
    feat = torch.randn(64, 512)
    loss_same = coral_loss(feat, feat.clone())
    assert torch.allclose(loss_same, torch.tensor(0.0), atol=1e-6), (
        f"expected exactly 0 for identical source/target features, got {loss_same.item():.6f}"
    )
    print(f"[PASS] check 1: CORAL loss is exactly 0 for identical source/target "
          f"feature sets ({loss_same.item():.8f}).")

    # ---- Check 2: CORAL loss is strictly positive and grows with an
    # increasingly different covariance structure (a "domain gap" that
    # genuinely widens should be penalized more, not less or the same). ----
    d = 32
    source = torch.randn(100, d)
    small_shift = source + 0.1 * torch.randn(100, d)  # small extra scatter
    large_shift = source * 3.0 + 2.0 * torch.randn(100, d)  # much larger covariance change
    loss_small = coral_loss(source, small_shift)
    loss_large = coral_loss(source, large_shift)
    assert loss_small.item() > 0, "expected strictly positive loss for a shifted covariance"
    assert loss_large.item() > loss_small.item(), (
        f"a larger covariance shift should produce a larger CORAL loss, got "
        f"small={loss_small.item():.4f} >= large={loss_large.item():.4f}"
    )
    print(f"[PASS] check 2: CORAL loss is positive and monotonically increases with "
          f"the size of the covariance shift (small={loss_small.item():.4f} < "
          f"large={loss_large.item():.4f}).")

    # ---- Check 3: gradient flows correctly to BOTH source and target
    # features (both are being actively pulled toward each other). ----
    src = torch.randn(50, 16, requires_grad=True)
    tgt = torch.randn(50, 16, requires_grad=True)
    loss = coral_loss(src, tgt)
    loss.backward()
    assert src.grad is not None and src.grad.abs().sum() > 0
    assert tgt.grad is not None and tgt.grad.abs().sum() > 0
    print("[PASS] check 3: gradients flow correctly to both source and target features.")

    # ---- Check 4: a degenerate single-sample batch (N=1, undefined sample
    # covariance) does not produce NaN/Inf -- the (N-1) denominator guard. ----
    single_source = torch.randn(1, 8)
    single_target = torch.randn(1, 8)
    loss_single = coral_loss(single_source, single_target)
    assert torch.isfinite(loss_single), f"NaN/Inf for a single-sample batch: {loss_single}"
    print(f"[PASS] check 4: a degenerate single-sample batch does not produce "
          f"NaN/Inf ({loss_single.item():.6f}).")

    # ---- Check 5: InfiniteTargetIterator correctly cycles a short loader
    # past its natural end, without ever raising StopIteration. ----
    from torch.utils.data import DataLoader, TensorDataset
    tiny_ds = TensorDataset(torch.arange(6).float())
    tiny_loader = DataLoader(tiny_ds, batch_size=4)  # only 2 batches per real epoch
    target_iter = InfiniteTargetIterator(tiny_loader)
    seen = []
    for _ in range(5):  # pull more batches than one real epoch contains
        seen.append(target_iter.next_batch()[0].tolist())
    assert len(seen) == 5, "InfiniteTargetIterator did not yield the requested number of batches"
    print(f"[PASS] check 5: InfiniteTargetIterator correctly cycles past the loader's "
          f"natural end without raising StopIteration (batches seen: {seen}).")

    print("\nALL CHECKS PASSED.")


# ============================================================================
# INTEGRATION NOTE (for wiring into tsfnet_model_notebook.py's train_one_fold)
#
# 1. Add two new parameters to train_one_fold's signature:
#
#        domain_adapt_lambda=0.0, target_loader=None
#
#    (0.0 / None is an exact no-op -- see check below once wired in.)
#
# 2. Just before the "for epoch in range(1, stage2_max_epochs + 1):" loop
#    (stage 2 only -- matching the roadmap's cost-conscious targeting, and
#    because stage 2 is the fold's final convergence phase), build the
#    cycling target iterator ONCE:
#
#        from coral_domain_adaptation import coral_loss, InfiniteTargetIterator
#        if domain_adapt_lambda > 0.0:
#            assert target_loader is not None, (
#                "domain_adapt_lambda > 0.0 requires target_loader (the held-out "
#                "session's unlabeled DataLoader) to be passed in."
#            )
#            _target_iter = InfiniteTargetIterator(target_loader)
#
# 3. Inside stage 2's inner batch loop, right after the existing forward
#    pass and BEFORE tsfnet_loss is called, add:
#
#        if domain_adapt_lambda > 0.0:
#            xt_eeg, xt_fnirs, _, xt_conn, xt_chan = _unpack_batch(_target_iter.next_batch())
#            xt_eeg, xt_fnirs = xt_eeg.to(device), xt_fnirs.to(device)
#            xt_conn = xt_conn.to(device) if xt_conn is not None else None
#            xt_chan = xt_chan.to(device) if xt_chan is not None else None
#            _, _, _, _, aux_t = model(xt_eeg, xt_fnirs, xt_conn, xt_chan)
#            # NOTE: deliberately NOT wrapped in torch.no_grad() -- CORAL needs
#            # gradients to flow back through the target forward pass too, so
#            # the model actively learns to produce ALIGNED features for both
#            # domains, not just features that happen to already match.
#
#    then, right after the existing tsfnet_loss(...) call:
#
#        if domain_adapt_lambda > 0.0:
#            coral = coral_loss(aux["enhanced_fnirs"], aux_t["enhanced_fnirs"])
#            loss = loss + domain_adapt_lambda * coral
#
#    (`loss.backward()` immediately below already covers this, unchanged.)
#
# 4. In run_all_subjects.py, add matching --domain-adapt-lambda CLI flag
#    (float, default 0.0) threaded through exactly like --mhs-lambda /
#    --label-smoothing, AND pass the held-out session's own test_loader
#    (already built for evaluate()/AdaBN) as target_loader to train_one_fold
#    -- no new data loader needs to be constructed, since AdaBN already
#    proves this exact loader is available and appropriate at this point in
#    the per-fold loop.
# ============================================================================
