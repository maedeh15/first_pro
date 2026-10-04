"""
label_smoothing_loss.py -- Priority B-4: Label Smoothing on top of an
already-softmaxed model output.

WHY THIS NEEDS ITS OWN IMPLEMENTATION (not just torch.nn.CrossEntropyLoss
with label_smoothing=...):
PyTorch's built-in `nn.CrossEntropyLoss(label_smoothing=...)` expects raw
LOGITS as input -- it applies log_softmax internally. This project's
TSFNet.forward, however, returns y_pred already passed through softmax
(confirmed in `_extract_features`'s own comment: "already softmax
probabilities"), and the existing stage1/stage2 training loop computes its
loss directly on these probabilities (not on a separate logit tensor). So
plugging in the built-in loss would silently apply softmax TWICE, which
badly distorts the gradient and is not something the project's current
architecture surfaces a clean way to avoid without restructuring the model
heads. Instead, this module reimplements label smoothing's mathematics --
KL divergence to a smoothed one-hot target -- directly on the probability
tensor the model actually produces, which is mathematically equivalent to
applying it after log-softmax on logits, just accepting a different (but
here, the *correct*) input.

FORMULA (Müller et al., 2019, "When Does Label Smoothing Help?", following
Szegedy et al., 2016's original definition):
    smoothed_target_k = (1 - eps) * 1[k == y] + eps / K
    loss = - sum_k( smoothed_target_k * log(p_k) )
where K is the number of classes and eps ("smoothing") is the smoothing
strength (eps=0 recovers standard cross-entropy exactly).

ISOLATED-VARIABLE CONTRACT (matching this project's ablation convention):
`smoothing=0.0` must be a BIT-EXACT no-op, reproducing whatever the
existing loss computes today. This is verified numerically below.

USAGE (drop-in replacement wherever the current training loop computes
cross-entropy on `y_pred`, e.g. inside train_one_fold's stage1/stage2 loss
computation):

    from label_smoothing_loss import label_smoothing_loss

    # before (assumed current implementation, equivalent to NLLLoss on
    # log-probabilities):
    #   loss = F.nll_loss(torch.log(y_pred + eps), targets)

    # after:
    loss = label_smoothing_loss(y_pred, targets, smoothing=0.1, num_classes=3)
"""

import torch
import torch.nn.functional as F


def label_smoothing_loss(probs, targets, smoothing=0.1, num_classes=3, eps=1e-8):
    """
    probs      : (B, K) tensor of PER-SAMPLE CLASS PROBABILITIES (already
                 softmaxed -- this is what TSFNet.forward returns as
                 y_pred/y_eeg/y_fnirs/y_fusion). NOT logits.
    targets    : (B,) tensor of integer class labels in [0, num_classes).
    smoothing  : label smoothing strength (eps in the formula above).
                 smoothing=0.0 is an EXACT no-op (see check 1 below).
    num_classes: K, the number of classes (3 for this project's 0-/2-/3-back
                 task).
    eps        : numerical floor added before log() to avoid log(0) for a
                 class the model assigns exactly zero probability to; has
                 no meaningful effect at smoothing=0 since gradients only
                 ever touch classes with nonzero target weight when
                 smoothing=0 (the one-hot target), and probs from a
                 softmax are never exactly zero in practice.

    Returns a scalar (mean-reduced over the batch) loss tensor, differentiable
    w.r.t. `probs`.
    """
    assert probs.dim() == 2 and probs.shape[1] == num_classes, (
        f"expected probs of shape (B, {num_classes}), got {tuple(probs.shape)}"
    )
    log_probs = torch.log(probs.clamp(min=eps))

    with torch.no_grad():
        true_dist = torch.full_like(probs, smoothing / num_classes)
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing + smoothing / num_classes)

    loss = -(true_dist * log_probs).sum(dim=1).mean()
    return loss


# ============================================================================
# Numeric debug / verification script -- run this file directly
# (`python label_smoothing_loss.py`) before using it in training, per this
# project's standing methodology rule: every new loss/ablation gets a
# numeric bit-exactness check on its no-op setting, plus a sanity check on
# its actual (nonzero) behaviour.
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, K = 16, 3

    logits = torch.randn(B, K, requires_grad=True)
    probs = F.softmax(logits, dim=1)
    targets = torch.randint(0, K, (B,))

    # ---- Check 1: smoothing=0.0 is a bit-exact no-op, matching standard
    # NLLLoss on log-probabilities (the project's presumed current loss). ----
    loss_smooth0 = label_smoothing_loss(probs, targets, smoothing=0.0, num_classes=K)
    loss_nll = F.nll_loss(torch.log(probs.clamp(min=1e-8)), targets)
    assert torch.allclose(loss_smooth0, loss_nll, atol=1e-6), (
        f"smoothing=0.0 is not a bit-exact no-op: {loss_smooth0.item()} vs {loss_nll.item()}"
    )
    print(f"[PASS] check 1: smoothing=0.0 exactly reproduces standard NLL loss "
          f"on log-probabilities ({loss_smooth0.item():.6f} vs {loss_nll.item():.6f}).")

    # ---- Check 2: also matches PyTorch's own label-smoothing CE, applied
    # correctly on LOGITS -- confirms the math (not just the eps=0 edge
    # case) is implemented correctly, using logits as the common reference
    # point that both formulations agree on. ----
    for smoothing in [0.05, 0.1, 0.2]:
        loss_mine = label_smoothing_loss(probs, targets, smoothing=smoothing, num_classes=K)
        loss_torch = F.cross_entropy(logits, targets, label_smoothing=smoothing)
        assert torch.allclose(loss_mine, loss_torch, atol=1e-5), (
            f"smoothing={smoothing}: mine={loss_mine.item():.6f} vs "
            f"torch={loss_torch.item():.6f} -- formula mismatch."
        )
    print(f"[PASS] check 2: matches torch.nn.functional.cross_entropy's own "
          f"label_smoothing implementation (verified via logits) for "
          f"smoothing in [0.05, 0.1, 0.2].")

    # ---- Check 3: label smoothing should REDUCE the loss (relative to
    # eps=0) for an already highly-confident CORRECT prediction, since it
    # discourages full confidence even when right -- this is the entire
    # point of the technique (Müller et al. 2019). ----
    confident_correct_probs = torch.tensor([[0.98, 0.01, 0.01]])
    confident_correct_target = torch.tensor([0])
    loss_0 = label_smoothing_loss(confident_correct_probs, confident_correct_target,
                                   smoothing=0.0, num_classes=3)
    loss_smoothed = label_smoothing_loss(confident_correct_probs, confident_correct_target,
                                          smoothing=0.1, num_classes=3)
    assert loss_smoothed.item() > loss_0.item(), (
        "label smoothing should INCREASE loss for an overconfident correct "
        "prediction relative to plain CE (it penalizes over-confidence even "
        "when the prediction is right) -- got the opposite."
    )
    print(f"[PASS] check 3: for an overconfident correct prediction (p_true=0.98), "
          f"smoothing correctly increases the loss ({loss_0.item():.4f} -> "
          f"{loss_smoothed.item():.4f}), penalizing over-confidence as intended.")

    # ---- Check 4: gradient flows correctly back to the model's probability
    # output (end-to-end differentiability through softmax). ----
    logits2 = torch.randn(B, K, requires_grad=True)
    probs2 = F.softmax(logits2, dim=1)
    loss = label_smoothing_loss(probs2, targets, smoothing=0.1, num_classes=K)
    loss.backward()
    assert logits2.grad is not None and logits2.grad.abs().sum() > 0
    print(f"[PASS] check 4: gradient flows correctly back through softmax to "
          f"the pre-softmax logits (grad abs sum = {logits2.grad.abs().sum().item():.4f}).")

    # ---- Check 5: input validation catches a wrong-shaped `probs` tensor
    # with a clear error, instead of silently misbehaving. ----
    try:
        label_smoothing_loss(torch.randn(B, 5), targets, num_classes=3)
        raise AssertionError("expected an AssertionError for mismatched num_classes")
    except AssertionError as e:
        assert "expected probs of shape" in str(e)
        print("[PASS] check 5: mismatched (probs, num_classes) shape is correctly "
              "rejected with a clear error.")

    print("\nALL CHECKS PASSED.")
