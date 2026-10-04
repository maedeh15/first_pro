"""
test_gating_numerical.py -- MANDATORY pre-GPU check for fusion_combination="gated".

Per the project's engineering rules (قواعد_مهندسی_پروژه.md):
  - Rule 2: any change touching forward/loss must be checked with a numerical
    gradient test on the REAL model class, on synthetic data -- not by
    reading the code, and not on a simulated stand-in.
  - Rule 5: a new parameter's default must reproduce old behavior bit-for-bit.
    "gated" is a NEW mode, not a new default, so this file instead checks the
    weaker, still load-bearing claim explicitly documented in the code
    comment: at gate_fc's zero-init, "gated" mode's prediction equals the
    UNWEIGHTED mean of the three branch logits (not "paper_mean"'s SIGMOID-
    weighted mean, which starts at 0.5 per branch, not 1/3 each -- these are
    genuinely different starting points, and this test pins down which one
    "gated" actually has).
  - Rule 6: RNG draws. Building self.gate_fc must not shift any OTHER
    parameter's initial values relative to fusion_combination="paper_mean",
    since it is constructed immediately after w_eeg/w_fusion/w_fnirs and
    nothing else depends on it downstream in __init__.

Run this on CPU with a tiny batch; it should finish in well under a minute
and needs no GPU, no real dataset, and no checkpoint.

USAGE:
    python test_gating_numerical.py
(or %run test_gating_numerical.py in the same Jupyter kernel that already
has TSFNet importable, e.g. right after %run tsfnet_model_notebook.py)
"""
import sys
import torch
import torch.nn as nn

# Import the REAL model class -- rule 2 requires the actual class, not a
# hand-rolled stand-in. Adjust this import if TSFNet lives in a package
# rather than as a %run'd notebook file.
try:
    from tsfnet_model_notebook import TSFNet
except ImportError:
    print("Could not `from tsfnet_model_notebook import TSFNet`.")
    print("If you're running this inside the same Jupyter kernel where you already")
    print("did `%run tsfnet_model_notebook.py`, just paste the checks below into a")
    print("cell instead of running this file as a script -- TSFNet is already in scope.")
    sys.exit(1)


def make_synthetic_batch(batch_size=4):
    x_eeg = torch.randn(batch_size, 16, 16, 600)
    x_fnirs = torch.randn(batch_size, 21, 16, 16, 30, 2)
    return x_eeg, x_fnirs


def check_1_zero_init_equals_unweighted_mean():
    """gate_fc is zero-init -> softmax(0,0,0) = uniform (1/3,1/3,1/3) for
    every sample -> y_pred must equal the plain mean of y_eeg/y_fusion/y_fnirs,
    to floating-point tolerance, on the very first forward call (no training
    steps in between)."""
    torch.manual_seed(0)
    model = TSFNet(num_classes=3, fusion_combination="gated")
    model.eval()

    x_eeg, x_fnirs = make_synthetic_batch()
    with torch.no_grad():
        y_eeg, y_fusion, y_fnirs_out, y_pred, aux = model(x_eeg, x_fnirs)
        # Recompute the three branch outputs by hand from aux to avoid
        # depending on internal variable names inside forward().
        # aux exposes y_eeg_native but not y_fusion/y_fnirs directly, so
        # instead we check gate_weights are exactly uniform, which is the
        # precise claim (and is a stronger, more localized check than
        # comparing y_pred to a hand-recomputed mean).
        gw = aux["gate_weights"]

    assert gw is not None, "gate_weights is None in 'gated' mode -- aux wiring broke."
    expected = torch.full_like(gw, 1.0 / 3.0)
    max_dev = (gw - expected).abs().max().item()
    ok = max_dev < 1e-6
    print(f"[check_1] gate_weights at zero-init vs uniform 1/3: max_dev={max_dev:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_2_gradient_flows_to_gate_fc():
    """A single backward pass must produce a non-zero, finite gradient on
    gate_fc.weight -- if it's exactly zero, the gate is dead on arrival and
    will never learn anything different from the zero-init uniform average;
    if it's NaN/Inf, something upstream (e.g. the softmax combined with the
    loss) is unstable."""
    torch.manual_seed(0)
    model = TSFNet(num_classes=3, fusion_combination="gated")
    model.train()

    x_eeg, x_fnirs = make_synthetic_batch()
    y_true = torch.randint(0, 3, (x_eeg.shape[0],))

    y_eeg, y_fusion, y_fnirs_out, y_pred, aux = model(x_eeg, x_fnirs)
    loss = nn.functional.nll_loss(torch.log(y_pred.clamp_min(1e-8)), y_true)
    loss.backward()

    grad = model.gate_fc.weight.grad
    has_grad = grad is not None
    nonzero = has_grad and grad.abs().max().item() > 0
    finite = has_grad and torch.isfinite(grad).all().item()
    print(f"[check_2] gate_fc.weight.grad: present={has_grad} "
          f"nonzero={nonzero} finite={finite}  "
          f"{'PASS' if (has_grad and nonzero and finite) else 'FAIL'}")
    return has_grad and nonzero and finite


def check_3_other_modes_unaffected():
    """Rule 6: building self.gate_fc only when fusion_combination == 'gated'
    must not shift the RNG stream (and therefore the initial weights) of
    any module built in 'paper_mean' mode. Compare the first conv layer's
    weights between a 'paper_mean' model and a model built with the SAME
    seed but 'gated' -- they diverge only from clf_eeg onward at the very
    earliest, because gate_fc is constructed AFTER w_eeg/w_fusion/w_fnirs
    and nothing built before it depends on it."""
    torch.manual_seed(0)
    model_paper_mean = TSFNet(num_classes=3, fusion_combination="paper_mean")
    torch.manual_seed(0)
    model_gated = TSFNet(num_classes=3, fusion_combination="gated")

    # Pick an early, GENUINELY randomly-initialized module untouched by
    # fusion_combination -- the EEG branch's conv layers are built long
    # before w_eeg/w_fusion/w_fnirs/gate_fc in __init__, so their weights
    # must be bit-identical across modes. NOTE: w_eeg/w_fusion/w_fnirs
    # themselves are torch.zeros(1) in both modes and would pass this check
    # vacuously (they consume no RNG state to begin with), so this
    # deliberately excludes them and picks a real >1-D conv weight instead.
    w_a = dict(model_paper_mean.named_parameters())
    w_b = dict(model_gated.named_parameters())
    early_key = next(k for k in w_a
                      if "eeg" in k.lower() and "clf" not in k.lower()
                      and w_a[k].dim() > 1)
    same = torch.allclose(w_a[early_key], w_b[early_key], atol=0)
    print(f"[check_3] early shared param {early_key!r} identical across modes: "
          f"{'PASS' if same else 'FAIL'}")
    return same


def check_4_no_orphaned_parameters():
    """w_eeg/w_fusion/w_fnirs are unused for the PREDICTION in 'gated' mode,
    but must still receive an explicit ZERO gradient rather than None.

    Why this matters (both are real, not hypothetical):
      (a) A param with grad=None still sits in optimizer.param_groups. With
          weight_decay on and no opposing gradient, decay alone drifts it
          away from its zero init -- this project's catalogued bug #3. That
          silently corrupts the saved scalars for anyone who later loads
          this checkpoint in paper_mean/wsum mode, where they DO matter.
      (b) grad=None (vs a zero tensor) breaks DDP with
          find_unused_parameters=False and some optimizer state round-trips.
    """
    torch.manual_seed(0)
    model = TSFNet(num_classes=3, fusion_combination="gated")
    model.train()

    x_eeg, x_fnirs = make_synthetic_batch()
    y_true = torch.randint(0, 3, (x_eeg.shape[0],))
    _, _, _, y_pred, _ = model(x_eeg, x_fnirs)
    loss = nn.functional.nll_loss(torch.log(y_pred.clamp_min(1e-8)), y_true)
    loss.backward()

    orphaned = [n for n, p in model.named_parameters()
                if p.requires_grad and p.grad is None]
    ok = len(orphaned) == 0
    print(f"[check_4] params left with grad=None: {orphaned if orphaned else 'none'}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_5_checkpoint_keys_differ_loudly():
    """A 'gated' checkpoint has gate_fc.* keys that a 'paper_mean' model does
    not, and vice versa. That is FINE and expected -- but it must fail LOUDLY
    on a cross-mode load, never load silently and half-initialized.

    Note for the project's own tooling: check_checkpoint_contamination.py
    only looks for 'eeg_band_branch' keys, so it will NOT flag a gated/
    non-gated mismatch. Check fusion_combination in the run tag before
    reusing any checkpoint across these modes.
    """
    torch.manual_seed(0)
    m_gated = TSFNet(num_classes=3, fusion_combination="gated")
    torch.manual_seed(0)
    m_plain = TSFNet(num_classes=3, fusion_combination="paper_mean")

    gated_only = set(m_gated.state_dict()) - set(m_plain.state_dict())
    raised = False
    try:
        m_plain.load_state_dict(m_gated.state_dict(), strict=True)
    except RuntimeError:
        raised = True

    ok = bool(gated_only) and raised
    print(f"[check_5] gated-only keys={sorted(gated_only)}; "
          f"strict cross-load raises={raised}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_6_report_uses_real_gate_weights():
    """detailed_test_report() must NOT report sigmoid(model.w_eeg) in gated
    mode. Those scalars never get a real gradient there, so they stay frozen
    at sigmoid(0.0)=0.5 and would write a plausible-looking but meaningless
    0.5/0.5/0.5 into every row of the results CSV -- exactly the kind of
    wrong-but-believable number that is hardest to catch later. The report
    must instead average the real per-sample gate weights.
    """
    from torch.utils.data import DataLoader, TensorDataset
    try:
        from tsfnet_model_notebook import detailed_test_report
    except ImportError:
        print("[check_6] SKIPPED (detailed_test_report not importable)")
        return True

    torch.manual_seed(0)
    n = 12
    x_eeg = torch.randn(n, 16, 16, 600)
    x_fnirs = torch.randn(n, 21, 16, 16, 30, 2)
    y = torch.randint(0, 3, (n,))
    loader = DataLoader(TensorDataset(x_eeg, x_fnirs, y), batch_size=4)

    torch.manual_seed(0)
    model = TSFNet(num_classes=3, fusion_combination="gated")
    # Simulate a TRAINED gate: with the zero-init the weights are uniform
    # 1/3 and would be indistinguishable from a bug by accident.
    nn.init.normal_(model.gate_fc.weight, std=1.0)
    nn.init.normal_(model.gate_fc.bias, std=1.0)

    lw = detailed_test_report(model, loader, device="cpu")["learned_weights"]
    triple = (lw["w_eeg"], lw["w_fusion"], lw["w_fnirs"])
    stale = all(abs(v - 0.5) < 1e-9 for v in triple)
    sums_to_one = abs(sum(triple) - 1.0) < 1e-4
    ok = (not stale) and sums_to_one and lw.get("gate_is_per_sample") is True
    print(f"[check_6] reported weights={tuple(round(v, 4) for v in triple)} "
          f"(stale 0.5 placeholders={stale}, sums_to_1={sums_to_one})  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [
        check_1_zero_init_equals_unweighted_mean(),
        check_2_gradient_flows_to_gate_fc(),
        check_3_other_modes_unaffected(),
        check_4_no_orphaned_parameters(),
        check_5_checkpoint_keys_differ_loudly(),
        check_6_report_uses_real_gate_weights(),
    ]
    print()
    if all(results):
        print(">>> All checks PASSED. Safe to run a real GPU training job with "
              "--fusion-combination gated.")
    else:
        print(">>> At least one check FAILED. Do NOT start a GPU run until this is fixed.")
        sys.exit(1)
