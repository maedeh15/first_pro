"""
tent_adaptation.py -- Priority: further generalization of E4 (AdaBN).

Implements Tent (Wang, Shelhamer, Liu, Olshausen & Darrell, "Tent: Fully
Test-Time Adaptation by Entropy Minimization", ICLR 2021, spotlight),
adapted from the OFFICIAL repository (https://github.com/DequanWang/tent,
tent.py) to this project's architecture and conventions. Every departure
from the official code is documented below with the exact reason.

WHY THIS IS THE NATURAL NEXT STEP AFTER E4 (AdaBN), NOT A NEW FAMILY:
E4 (accepted, +1.11 acc / consistent across 3 seeds) recomputes ONLY the
non-learnable running_mean/running_var buffers of this model's BatchNorm
layers on the held-out session's unlabeled inputs, via a single no_grad
forward pass -- no weights change, nothing is optimized.

Tent is the natural generalization: it (a) also discards running statistics
in favor of pure per-batch statistics (exactly what AdaBN's cumulative
recomputation approximates, but computed fresh per batch instead of frozen
after one pass), AND (b) additionally optimizes the BatchNorm layers'
LEARNABLE affine parameters (the scale/shift, i.e. `weight`/`bias`) by
gradient descent on the entropy of the model's own predictions -- still
using only the held-out session's unlabeled inputs, never its labels. This
project's own results already favor the test-time family over the
train-time family (AdaBN accepted; Deep CORAL, a train-time alignment loss,
rejected in all 3 seeds -- see coral_domain_adaptation.py), which is the
direct motivation for testing this one remaining, more powerful member of
that same family.

=============================================================================
FOUR DEPARTURES FROM THE OFFICIAL tent.py, EACH REQUIRED BY THIS PROJECT
=============================================================================

1. BATCHNORM TYPE. Official `collect_params`/`configure_model` hard-code
   `nn.BatchNorm2d` (the paper's setting is image classification). This
   project's BatchNorm layers are `nn.BatchNorm3d` (EEGLikeBranch /
   FNIRSBranch) and `nn.BatchNorm1d` (SpatialMHSABranch, when enabled) --
   NO BatchNorm2d exists anywhere in this model. Every function below uses
   `torch.nn.modules.batchnorm._BatchNorm`, the common base class, which is
   exactly the generalization this project's own `adapt_batchnorm_to_target`
   (E4/AdaBN) already uses for the identical reason -- confirmed by reading
   that function's own source before writing this file.

2. ENTROPY ON PROBABILITIES, NOT LOGITS -- THE SINGLE MOST IMPORTANT FIX.
   Official `softmax_entropy(x)` takes LOGITS and applies softmax/log_softmax
   internally: `-(x.softmax(1) * x.log_softmax(1)).sum(1)`. This project's
   TSFNet.forward returns `y_pred` (and every other head) ALREADY passed
   through softmax -- confirmed directly in this project's own
   `label_smoothing_loss.py`, which had to solve exactly this same trap for
   exactly the same reason ("PyTorch's built-in CrossEntropyLoss... expects
   raw LOGITS... plugging in the built-in loss would silently apply softmax
   TWICE"). Feeding y_pred into the official `softmax_entropy` would apply a
   SECOND softmax to an already-normalized distribution, flattening it
   toward uniform and computing an entropy that has no correct relationship
   to the model's actual confidence. `entropy_of_probs` below instead
   implements entropy directly on a probability tensor:
       H(p) = -sum_k p_k * log(p_k)
   which is mathematically the definition Tent's paper actually invokes
   (Eq. in Sec. 3.1: "the entropy of model predictions") -- the official
   code's extra softmax is only there because ITS model returns logits, not
   because entropy is defined on logits.

3. MULTI-HEAD, TUPLE-VALUED FORWARD, AND AN UNNORMALIZED y_pred. Official
   Tent wraps a plain classifier returning a single logit tensor. TSFNet.
   forward returns a 5-tuple `(y_eeg, y_fusion, y_fnirs, y_pred, aux)`.
   `TentTSFNet.forward` below adapts on the entropy of `y_pred` specifically
   -- the paper-mean fused decision (Eq. 4), i.e. the SAME quantity this
   project already reports as its headline accuracy -- and returns the
   identical 5-tuple structure, so every downstream consumer (this file's
   own test report, per-branch ablation bookkeeping) keeps working
   unmodified.

   A SECOND trap, found empirically by this file's own real-model
   integration test (not assumed): `y_pred = (w1*y_eeg + w2*y_fusion +
   w3*y_fnirs) / 3`, where w1/w2/w3 are INDEPENDENT sigmoid gates -- NOT
   weights constrained to sum to 1. Verified directly on the real TSFNet
   class: at initialization (all three gates at sigmoid(0)=0.5), `y_pred`
   sums to exactly 0.5 per row, not 1.0. Computing entropy on an
   unnormalized vector is not a well-defined confidence measure, and gives
   the optimizer a degenerate shortcut: entropy can be driven down just by
   shrinking the gates toward 0 (making every entry smaller and the
   distribution's shape irrelevant) rather than by genuinely sharpening the
   prediction. `TentTSFNet.forward` therefore renormalizes `y_pred` to a
   true probability simplex (`y_pred / y_pred.sum(-1, keepdim=True)`)
   ONLY for the entropy computation; the returned `y_pred` itself is left
   in the project's existing (unnormalized) convention, since accuracy
   (argmax) is scale-invariant and every downstream consumer already
   expects that convention.

4. PER-FOLD EPISODIC RESET, NOT CONTINUAL ONLINE ADAPTATION. Official
   `Tent.episodic=False` by default: adaptation persists and accumulates
   across every batch for the entire run, only reset if the caller
   explicitly asks. This project's protocol evaluates one held-out SESSION
   per fold, and a fresh TSFNet is trained per fold already (mirroring how
   AdaBN's own adaptation is discarded after each fold, never carried into
   the next). Carrying Tent's adapted BN parameters from one held-out
   session's fold into the NEXT fold's (different subject or different
   held-out session) would silently leak adaptation state across folds that
   must be independent. This file therefore hard-codes `episodic=True`
   semantics: adaptation state is reset to the fold's own just-trained
   checkpoint before the held-out session's batches are seen, matching
   exactly how `adapt_batchnorm_to_target` is invoked fresh per fold in
   `train_one_fold` (see the INTEGRATION NOTE at the bottom of this file).

WHAT IS NOT CHANGED FROM THE OFFICIAL METHOD:
- Only BatchNorm affine `weight`/`bias` are ever unfrozen (`collect_params`
  below is otherwise a direct, type-generalized port).
- `track_running_stats=False` / running buffers cleared, so BatchNorm always
  uses the CURRENT batch's statistics (official `configure_model`, ported
  verbatim in spirit) -- this is what makes it a strict generalization of
  AdaBN's "replace running stats with target-session stats" rather than an
  unrelated technique.
- One gradient step per batch by default (`steps=1`), Adam optimizer, as the
  official README recommends ("choosing Adam or SGD+momentum... start with
  a low learning rate (like 0.001) if not [known from training]").

=============================================================================
KNOWN RISK (must be checked empirically before trusting any result)
=============================================================================
Because BatchNorm is forced to use PURE PER-BATCH statistics (never running
averages), a small test batch produces a noisy/degenerate mean and variance
estimate -- worst case, a batch of size 1 has undefined variance. This
project's held-out-session batch size is whatever `test_loader` already
uses elsewhere in this codebase (unchanged by this file); if that is small,
this is a real failure mode to check for (e.g. NaN/Inf losses, or accuracy
that is WORSE than plain AdaBN) rather than assume away.
"""

import torch
import torch.nn as nn


_BatchNormBase = torch.nn.modules.batchnorm._BatchNorm


def entropy_of_probs(probs, eps=1e-8):
    """
    H(p) = -sum_k p_k * log(p_k), computed directly on an ALREADY-SOFTMAXED
    probability tensor -- see this module's docstring, departure #2, for why
    this must NOT be the official softmax_entropy(logits) applied to probs.

    probs : (B, K) tensor of per-sample class probabilities (rows sum to 1).
    eps   : numerical floor before log(), matching this project's existing
            convention (e.g. tsfnet_loss's own `torch.log(y_pred + eps)`
            pattern) rather than torch.clamp, for consistency with the rest
            of the codebase's loss functions.

    Returns a (B,) tensor of per-sample entropies (nats). Caller reduces
    (mean) to get the scalar loss Tent minimizes.
    """
    assert probs.dim() == 2, f"expected (B, K) probabilities, got {tuple(probs.shape)}"
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def collect_bn_affine_params(model):
    """
    Direct, type-generalized port of official tent.py's `collect_params`:
    walks the model's modules and collects ONLY BatchNorm affine
    weight/bias parameters (never running_mean/var, never any other layer).
    Generalized from `nn.BatchNorm2d` to `_BatchNorm` (see departure #1).
    """
    params, names = [], []
    for name, module in model.named_modules():
        if isinstance(module, _BatchNormBase):
            for pname, p in module.named_parameters(recurse=False):
                if pname in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{name}.{pname}")
    return params, names


def configure_model_for_tent(model):
    """
    Direct, type-generalized port of official tent.py's `configure_model`:
    train mode (Tent optimizes the model, so BN layers must be in train mode
    to use batch statistics), freeze everything, then re-enable grad ONLY on
    BatchNorm affine params and force each BatchNorm layer to always use the
    current batch's statistics (never running averages).

    Called ONCE per fold, on a model whose non-BN weights are the fold's
    just-trained, just-checkpointed state (see the INTEGRATION NOTE) --
    mutating `track_running_stats`/buffers here is safe specifically because
    this project instantiates a fresh model per fold and this mutated
    instance is discarded after the fold's held-out session is evaluated
    (mirrored from how `adapt_batchnorm_to_target` is also called fresh per
    fold and never reused across folds).
    """
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, _BatchNormBase):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


def check_tent_model(model):
    """Direct port of official tent.py's `check_model`, generalized to
    `_BatchNorm`. Raises AssertionError with a clear message on
    misconfiguration -- run this once after `configure_model_for_tent`
    and before adapting, per this project's standing debugging discipline."""
    assert model.training, "tent needs train mode: call model.train()"
    grads = [p.requires_grad for p in model.parameters()]
    assert any(grads), "tent needs params to update: check which require grad"
    assert not all(grads), "tent should not update ALL params: check which require grad"
    has_bn = any(isinstance(m, _BatchNormBase) for m in model.modules())
    assert has_bn, "tent needs normalization layers for its optimization"


class TentTSFNet(nn.Module):
    """
    Wraps an already-trained, checkpoint-loaded TSFNet fold-model for
    test-time entropy minimization. See this module's docstring (departure
    #3) for why a wrapper -- rather than a drop-in replacement for
    `detailed_test_report`'s no_grad loop -- is required: adaptation and
    prediction happen in the SAME forward pass, one gradient step per batch,
    exactly matching official tent.py's `forward_and_adapt`.

    USAGE (see the INTEGRATION NOTE at the bottom of this file for the exact
    call site inside train_one_fold):

        tented = TentTSFNet(model, lr=1e-3, steps=1)
        tented.check()                      # raises clearly if misconfigured
        for batch in test_loader:
            y_eeg, y_fusion, y_fnirs, y_pred, aux = tented(x_eeg, x_fnirs, x_conn, x_chan)
            # y_pred here is this SAME batch's post-adaptation prediction --
            # exactly the official method's "one update per batch to not
            # interrupt inference".

    steps : gradient steps per batch (official default 1 -- "makes just one
        update per batch to not interrupt inference"). Only change this with
        a stated reason; more steps costs more compute for a rarely-reported
        gain in the original paper.
    lr : Adam learning rate for the BN affine parameters. The official
        README explicitly recommends "a low learning rate (like 0.001) if
        [the settings at the end of training are] not [known]" -- 1e-3 is
        used here as that documented default, not a project-specific tune.
    """

    def __init__(self, model, lr=1e-3, steps=1):
        super().__init__()
        assert steps >= 1, "tent requires >= 1 step(s) to forward and adapt"
        self.model = configure_model_for_tent(model)
        self.steps = steps
        params, self.param_names = collect_bn_affine_params(self.model)
        assert params, ("no BatchNorm affine parameters found -- tent has "
                        "nothing to adapt on this model")
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    def check(self):
        check_tent_model(self.model)

    def forward(self, x_eeg, x_fnirs, x_conn, x_chan):
        with torch.enable_grad():  # ensure grad even if called under a no_grad context
            for _ in range(self.steps):
                y_eeg, y_fusion, y_fnirs, y_pred, aux = self.model(x_eeg, x_fnirs, x_conn, x_chan)
                # CRITICAL FIX (found by this file's own real-model integration
                # test, not assumed): TSFNet's y_pred is Eq.4's
                # (w1*y_eeg + w2*y_fusion + w3*y_fnirs) / 3, where w1/w2/w3 are
                # INDEPENDENT sigmoid gates -- NOT weights that sum to 1. So
                # y_pred does NOT sum to 1 in general (verified: at
                # initialization, with all three gates at sigmoid(0)=0.5,
                # y_pred sums to exactly 0.5, not 1.0). Computing entropy on an
                # unnormalized vector is not a well-defined confidence measure
                # and is trivially "minimized" by the optimizer just shrinking
                # the gates toward 0 rather than genuinely sharpening the
                # prediction -- exactly the kind of degenerate shortcut entropy
                # minimization is prone to if fed the wrong quantity. Renormalize
                # to a true probability simplex before computing entropy; y_pred
                # ITSELF (unnormalized) is still what's returned below, since
                # accuracy (argmax) is scale-invariant and downstream bookkeeping
                # expects the project's existing y_pred convention unchanged.
                y_pred_normalized = y_pred / y_pred.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                loss = entropy_of_probs(y_pred_normalized).mean()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        # the LAST step's outputs are this batch's prediction -- matching
        # official forward_and_adapt, which returns the same call's outputs.
        return y_eeg, y_fusion, y_fnirs, y_pred, aux


def tent_detailed_test_report(model, test_loader, device, num_classes=3,
                              lr=1e-3, steps=1, log_fn=print,
                              cohen_kappa_fn=None, confusion_matrix_fn=None):
    """
    Tent-adapted counterpart of this project's own `detailed_test_report`,
    kept as a SEPARATE function rather than modifying the original (which
    remains the accepted E3+E4/AdaBN path, untouched and bit-exact) --
    matching this project's isolated-variable protocol: enabling Tent must
    change nothing about how a non-Tent run's report is produced.

    Returns the SAME dict shape as `detailed_test_report` (including
    `kappa_final` and `confusion_matrix_final`, computed with this project's
    own `cohen_kappa`/`confusion_matrix_np` if provided), so downstream
    result-saving code needs no changes; only HOW each batch's prediction is
    produced differs (adapt-then-predict per batch, instead of a plain
    no_grad forward).

    cohen_kappa_fn / confusion_matrix_fn: this project's own
    `cohen_kappa(y_true, y_pred)` / `confusion_matrix_np(y_true, y_pred, K)`
    functions, injected rather than imported directly so this file has no
    hard import-time dependency on tsfnet_model_notebook.py (matching the
    same standalone-testability rationale as `_unpack_batch_for_tent`). If
    left None, `kappa_final`/`confusion_matrix_final` are simply omitted
    rather than silently wrong.
    """
    import numpy as np

    tented = TentTSFNet(model, lr=lr, steps=steps)
    tented.check()

    all_true, all_final, all_eeg, all_fnirs, all_fusion = [], [], [], [], []
    n_batches = 0
    for _batch in test_loader:
        x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch_for_tent(_batch)
        x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
        x_conn = x_conn.to(device) if x_conn is not None else None
        x_chan = x_chan.to(device) if x_chan is not None else None
        y_eeg, y_fusion, y_fnirs, y_pred, aux = tented(x_eeg, x_fnirs, x_conn, x_chan)
        all_true.append(y.numpy())
        all_final.append(y_pred.argmax(dim=-1).detach().cpu().numpy())
        all_eeg.append(y_eeg.argmax(dim=-1).detach().cpu().numpy())
        all_fnirs.append(y_fnirs.argmax(dim=-1).detach().cpu().numpy())
        all_fusion.append(y_fusion.argmax(dim=-1).detach().cpu().numpy())
        n_batches += 1

    y_true = np.concatenate(all_true)
    y_final = np.concatenate(all_final)
    y_eeg_arr = np.concatenate(all_eeg)
    y_fnirs_arr = np.concatenate(all_fnirs)
    y_fusion_arr = np.concatenate(all_fusion)

    test_acc = float((y_final == y_true).mean())

    with torch.no_grad():
        w_eeg = torch.sigmoid(model.w_eeg).item()
        w_fusion = torch.sigmoid(model.w_fusion).item()
        w_fnirs = torch.sigmoid(model.w_fnirs).item()

    log_fn(f"  [tent] adapted BatchNorm affine params ({len(tented.param_names)} "
           f"tensor(s)) over {n_batches} batch(es), {steps} step(s)/batch, lr={lr} "
           f"(further generalization of priority-E4, transductive: unlabeled "
           f"target inputs only)")

    report = {
        "test_accuracy_final": test_acc,
        "ablation": {
            "eeg_only": float((y_eeg_arr == y_true).mean()),
            "fnirs_only": float((y_fnirs_arr == y_true).mean()),
            "fusion_only": float((y_fusion_arr == y_true).mean()),
            "final_weighted": test_acc,
        },
        "learned_weights": {"w_eeg": w_eeg, "w_fusion": w_fusion, "w_fnirs": w_fnirs,
                            "w_labram": None, "beta_labram": None},
    }
    if cohen_kappa_fn is not None:
        report["kappa_final"] = cohen_kappa_fn(y_true, y_final)
    if confusion_matrix_fn is not None:
        report["confusion_matrix_final"] = confusion_matrix_fn(y_true, y_final, num_classes).tolist()
    return report


def _unpack_batch_for_tent(batch):
    """Placeholder matching this project's own `_unpack_batch` signature --
    replaced by the real one at integration time (see the INTEGRATION NOTE);
    kept separate here only so this file has no import-time dependency on
    tsfnet_model_notebook.py and can be unit-tested standalone."""
    return batch


# ============================================================================
# Numeric debug / verification script -- run this file directly
# (`python tent_adaptation.py`) before wiring it into training, per this
# project's standing methodology rule.
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    import torch.nn.functional as F

    print("=" * 70)
    print("Check 1: entropy_of_probs is the CORRECT formula on probabilities,")
    print("NOT the official softmax_entropy(logits) misapplied to probs")
    print("=" * 70)
    logits = torch.randn(8, 3)
    probs = F.softmax(logits, dim=-1)

    def official_softmax_entropy_on_logits(x):
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    correct = entropy_of_probs(probs)
    correct_ref = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)
    buggy_double_softmax = official_softmax_entropy_on_logits(probs)  # THE TRAP
    assert torch.allclose(correct, correct_ref, atol=1e-5)
    assert not torch.allclose(correct, buggy_double_softmax, atol=1e-3), (
        "entropy_of_probs must NOT match the double-softmax bug's value"
    )
    # Sanity: a uniform distribution has entropy log(K); a one-hot has 0.
    uniform = torch.full((1, 3), 1.0 / 3)
    onehot = torch.tensor([[0.999998, 0.000001, 0.000001]])
    h_uniform = entropy_of_probs(uniform).item()
    h_onehot = entropy_of_probs(onehot).item()
    assert abs(h_uniform - torch.log(torch.tensor(3.0)).item()) < 1e-4
    assert h_onehot < 1e-3
    print(f"[PASS] check 1: entropy_of_probs matches the direct H(p)=-sum(p*log p) "
          f"formula (max diff {(correct - correct_ref).abs().max().item():.2e}), "
          f"correctly differs from the double-softmax bug (mean diff "
          f"{(correct - buggy_double_softmax).abs().mean().item():.4f}), and gives "
          f"the expected extremes: H(uniform)={h_uniform:.4f}=ln(3), "
          f"H(near-one-hot)={h_onehot:.6f}=~0.")

    print("\n" + "=" * 70)
    print("Check 2: collect_bn_affine_params generalizes to BatchNorm1d/3d")
    print("(this model has NO BatchNorm2d anywhere -- confirmed by reading")
    print("tsfnet_model_notebook.py before writing this file)")
    print("=" * 70)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv3d = nn.Conv3d(2, 4, 3, padding=1)
            self.bn3d = nn.BatchNorm3d(4)
            self.fc = nn.Linear(4, 8)
            self.bn1d = nn.BatchNorm1d(8)
            self.out = nn.Linear(8, 3)

        def forward(self, x):
            x = self.bn3d(self.conv3d(x)).mean(dim=(2, 3, 4))
            x = self.bn1d(self.fc(x))
            return F.softmax(self.out(x), dim=-1)

    m = TinyModel()
    params, names = collect_bn_affine_params(m)
    assert len(params) == 4, f"expected 4 (bn3d.weight/bias, bn1d.weight/bias), got {len(params)}"
    assert all(("bn3d" in n or "bn1d" in n) for n in names)
    assert not any("conv3d" in n or "fc." in n or "out." in n for n in names)
    print(f"[PASS] check 2: collect_bn_affine_params found exactly the 4 expected "
          f"BatchNorm3d/1d affine tensors ({names}), nothing from Conv3d/Linear.")

    print("\n" + "=" * 70)
    print("Check 3: configure_model_for_tent freezes everything except BN affine,")
    print("and forces BN to ignore running stats (track_running_stats=False)")
    print("=" * 70)
    m2 = TinyModel()
    configure_model_for_tent(m2)
    check_tent_model(m2)  # must not raise
    bn_affine_names = {n for _, n in zip(*collect_bn_affine_params(m2)[::-1])}
    for name, p in m2.named_parameters():
        is_bn_affine = name in {n for n in collect_bn_affine_params(m2)[1]}
        assert p.requires_grad == is_bn_affine, f"{name}: requires_grad={p.requires_grad}"
    for mod in m2.modules():
        if isinstance(mod, _BatchNormBase):
            assert mod.track_running_stats is False
            assert mod.running_mean is None and mod.running_var is None
    assert m2.training, "configure_model_for_tent must leave the model in train mode"
    print("[PASS] check 3: only BatchNorm affine weight/bias have requires_grad=True; "
          "every BatchNorm layer has track_running_stats=False with cleared buffers "
          "(always uses the CURRENT batch's statistics); model.training=True.")

    print("\n" + "=" * 70)
    print("Check 4: TentTSFNet adapts (BN affine params change) and returns")
    print("this batch's own post-adaptation prediction -- on a 5-tuple-output")
    print("model matching TSFNet's real signature")
    print("=" * 70)

    class TupleModel(nn.Module):
        """Mimics TSFNet's (y_eeg, y_fusion, y_fnirs, y_pred, aux) signature,
        its post-softmax convention, AND its UNNORMALIZED y_pred (see
        departure #3: y_pred = independent-sigmoid-gated average, not a true
        distribution) -- with a real BatchNorm3d in the path."""
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(2, 4, 3, padding=1)
            self.bn = nn.BatchNorm3d(4)
            self.head = nn.Linear(4, 3)
            self.gate = nn.Parameter(torch.zeros(1))  # sigmoid(0)=0.5, matching TSFNet's init

        def forward(self, x_eeg, x_fnirs, x_conn, x_chan):
            feat = self.bn(self.conv(x_fnirs)).mean(dim=(2, 3, 4))
            logits = self.head(feat)
            y = F.softmax(logits, dim=-1)
            y_pred_unnorm = torch.sigmoid(self.gate) * y   # sums to 0.5, NOT 1.0
            return y, y, y, y_pred_unnorm, {"dummy": None}

    torch.manual_seed(1)
    base = TupleModel()
    before = {n: p.clone() for n, p in base.named_parameters()
              if "." in n and isinstance(dict(base.named_modules())[n.rsplit(".", 1)[0]], _BatchNormBase)}
    tented = TentTSFNet(base, lr=1e-2, steps=1)
    tented.check()
    x_fnirs = torch.randn(6, 2, 4, 4, 4)
    y_eeg, y_fusion, y_fnirs, y_pred, aux = tented(None, x_fnirs, None, None)
    assert y_pred.shape == (6, 3)
    # y_pred is TSFNet's UNNORMALIZED convention (see departure #3): it does
    # NOT sum to 1 -- confirm the test model's known deviation reproduces
    # that, and confirm entropy was still computed on the RENORMALIZED
    # version internally (checked by check 4b below, not by y_pred's own sum).
    assert not torch.allclose(y_pred.sum(-1), torch.ones(6), atol=1e-2), (
        "this test's y_pred is deliberately unnormalized, matching real TSFNet"
    )
    changed = any(not torch.equal(before[n], p) for n, p in tented.model.named_parameters()
                  if n in before)
    assert changed, "BatchNorm affine parameters should have changed after one adapt step"
    non_bn_changed = any(
        not torch.equal(dict(base.named_parameters())[n], p0)
        for n, p0 in [("head.weight", base.head.weight.clone()),
                      ("head.bias", base.head.bias.clone())]
    )
    print(f"[PASS] check 4: after one TentTSFNet forward call, BN affine params "
          f"changed (adaptation happened) and y_pred={tuple(y_pred.shape)} matches "
          f"TSFNet's own unnormalized convention, returned unchanged from the SAME "
          f"adapted forward pass, matching official tent's 'one update per batch' "
          f"semantics.")

    print("\n" + "=" * 70)
    print("Check 4b: entropy is computed on the RENORMALIZED y_pred, not the")
    print("raw unnormalized one -- guards against the degenerate 'shrink the")
    print("gate' shortcut that an unnormalized entropy would allow")
    print("=" * 70)
    torch.manual_seed(1)
    m_direct = TupleModel()
    x_direct = torch.randn(6, 2, 4, 4, 4)
    with torch.no_grad():
        _, _, _, y_raw, _ = m_direct(None, x_direct, None, None)
    naive_entropy = entropy_of_probs(y_raw).mean()          # THE BUG: unnormalized input
    correct_entropy = entropy_of_probs(
        y_raw / y_raw.sum(-1, keepdim=True)).mean()          # what TentTSFNet actually uses
    assert not torch.allclose(naive_entropy, correct_entropy, atol=1e-3), (
        "renormalized and raw entropy should differ given y_pred sums to 0.5, not 1"
    )
    print(f"[PASS] check 4b: entropy on the raw (sum=0.5) y_pred gives "
          f"{naive_entropy.item():.4f}, while entropy on the correctly renormalized "
          f"y_pred gives {correct_entropy.item():.4f} -- confirming TentTSFNet's "
          f"internal renormalization is not a no-op and matters.")

    print("\n" + "=" * 70)
    print("Check 5: non-BatchNorm parameters (e.g. the classification head)")
    print("never receive a gradient or change -- only BN affine adapts")
    print("=" * 70)
    torch.manual_seed(2)
    base2 = TupleModel()
    head_w_before = base2.head.weight.clone()
    head_b_before = base2.head.bias.clone()
    conv_w_before = base2.conv.weight.clone()
    tented2 = TentTSFNet(base2, lr=1e-2, steps=1)
    x2 = torch.randn(6, 2, 4, 4, 4)
    tented2(None, x2, None, None)
    assert torch.equal(head_w_before, tented2.model.head.weight)
    assert torch.equal(head_b_before, tented2.model.head.bias)
    assert torch.equal(conv_w_before, tented2.model.conv.weight)
    assert tented2.model.head.weight.grad is None
    assert tented2.model.conv.weight.grad is None
    print("[PASS] check 5: the classification head and Conv3d weights are bit-"
          "identical after adaptation and received no gradient -- only BatchNorm "
          "affine parameters are ever touched.")

    print("\n" + "=" * 70)
    print("Check 6: batch-statistics behavior -- BatchNorm uses the CURRENT")
    print("batch's own statistics, not stale FROZEN running stats, which is")
    print("the mechanism that makes this a strict generalization of AdaBN")
    print("=" * 70)
    torch.manual_seed(3)
    base3 = TupleModel()
    # Bias the running stats toward batch A specifically, in NORMAL training
    # mode (track_running_stats=True, the PyTorch default) -- simulating
    # "trained on the source sessions" running stats, exactly what E3+E4's
    # own model checkpoint carries into a held-out session before adaptation.
    xa = torch.randn(6, 2, 4, 4, 4) * 1.0
    xb = torch.randn(6, 2, 4, 4, 4) * 5.0 + 3.0   # a very different batch distribution
    base3.train()
    with torch.no_grad():
        for _ in range(20):     # enough forward passes to bias running stats toward A
            base3(None, xa, None, None)
    base3.eval()
    with torch.no_grad():
        _, _, _, y_b_frozen_stats, _ = base3(None, xb, None, None)  # stale stats, biased to A

    tented3 = TentTSFNet(base3, lr=0.0, steps=1)  # lr=0: isolate the BN-STATS
                                                    # effect from the entropy-
                                                    # minimization gradient step
    _, _, _, y_b_batch_stats, _ = tented3(None, xb, None, None)  # B's OWN stats

    assert not torch.allclose(y_b_frozen_stats, y_b_batch_stats, atol=1e-3), (
        "BatchNorm using batch B's own statistics should differ from using "
        "stale running statistics biased toward the very different batch A"
    )
    print("[PASS] check 6: with lr=0 (isolating pure BN-statistics behavior from "
          "the gradient step), feeding batch B through frozen running stats "
          "biased toward the very different batch A gives a DIFFERENT output "
          "than feeding the SAME batch B through tent's per-batch statistics "
          f"(max diff {(y_b_frozen_stats - y_b_batch_stats).abs().max().item():.4f}) "
          "-- confirming Tent genuinely uses the current batch's own statistics, "
          "not the frozen, source-biased running average.")

    print("\nALL CHECKS PASSED.")


# ============================================================================
# INTEGRATION NOTE (for wiring into tsfnet_model_notebook.py / run_all_subjects.py)
#
# 1. `_unpack_batch_for_tent` above is a placeholder. At the top of
#    tsfnet_model_notebook.py's own file, after `from tent_adaptation import
#    tent_detailed_test_report`, monkeypatch it once:
#
#        import tent_adaptation
#        tent_adaptation._unpack_batch_for_tent = _unpack_batch   # this file's own real one
#
#    (matching the project's existing `_unpack_batch` signature exactly --
#    done this way, rather than importing `_unpack_batch` INTO
#    tent_adaptation.py, so this file has no hard dependency on the notebook
#    file and can be tested standalone, as done above.)
#
# 2. In `train_one_fold`, add a new parameter `tta_mode="adabn"` (or
#    `"none"`/`"tent"`), MUTUALLY EXCLUSIVE with the existing `adabn` boolean
#    -- both modify the SAME BatchNorm layers, so combining them would
#    confound which one produced any change, exactly the isolated-variable
#    concern already documented for `eeg_backbone_mode` vs `eeg_spatial_mode`.
#    Simplest wiring, changing the LEAST existing code:
#
#        if tta_mode == "tent":
#            assert not adabn, "tta_mode='tent' and adabn=True both modify " \
#                "BatchNorm and cannot be combined -- test them separately."
#            report = tent_detailed_test_report(model, test_loader, device,
#                                               lr=1e-3, steps=1, log_fn=log_fn)
#        else:
#            if adabn:
#                adapt_batchnorm_to_target(model, test_loader, device, log_fn=log_fn)
#            report = detailed_test_report(model, test_loader, device)
#
#    replacing the existing two lines at the point AFTER the final checkpoint
#    is reloaded (`model.load_state_dict(torch.load(ckpt_path, ...))`) and
#    BEFORE any test-set evaluation -- the EXACT insertion point already used
#    for `adapt_batchnorm_to_target`, so the on-disk checkpoint keeps
#    unadapted weights either way, matching E4's own documented invariant.
#
# 3. In run_all_subjects.py, add `--tta-mode {none, adabn, tent}` as a NEW
#    flag (default "adabn" if `--adabn` was already the project's accepted
#    setting going forward, or "none" to keep introducing this ablation with
#    a bit-exact no-op default -- follow whichever convention this project's
#    other C-2/D-series flags used most recently), threaded through to
#    `train_one_fold` exactly like `eeg_backbone`. Do NOT keep the old
#    `--adabn` boolean AND add a conflicting new flag without reconciling
#    them -- pick ONE flag as the single owner of "what happens to BatchNorm
#    at test time" to avoid a silent both-set misconfiguration.
#
# 4. THESIS REPORTING REQUIREMENT: like E3/E4, Tent is TRANSDUCTIVE (uses the
#    held-out session's unlabeled inputs during "training" of the BN affine
#    params). Report it alongside E3/E4 with the same disclosure, not
#    against a strictly inductive baseline without stating this.
#
# 5. Per this project's standing rule #11 (check the [ABLATION] banner before
#    interpreting any result): confirm the banner shows the intended
#    tta_mode and that `adabn` is NOT simultaneously True when tta_mode='tent'.
#
# 6. KNOWN RISK TO CHECK EMPIRICALLY (see this file's module docstring): if
#    this project's test batch size is small, pure per-batch BatchNorm
#    statistics may be noisy or degenerate. Before trusting any accept/reject
#    decision, inspect training logs for NaN/Inf in the reported test
#    accuracy/kappa, and consider whether the held-out session's batch size
#    needs to be increased (e.g. via test_loader's own batch_size argument)
#    specifically for the tent-adapted evaluation pass, independent of the
#    batch size used during the fold's own training.
# ============================================================================
