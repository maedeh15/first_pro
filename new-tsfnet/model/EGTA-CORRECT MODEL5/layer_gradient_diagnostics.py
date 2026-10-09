"""
layer_gradient_diagnostics.py -- is backprop actually working correctly through
EVERY layer of TSFNet? NO full baseline training run is needed for this: the three
checks below use a FRESH model and either one real batch (checks 1-2) or a few
hundred optimizer steps OVERFITTING that same single batch (check 3) -- a few
seconds on GPU, a few tens of seconds on CPU. This generalises
diagnose_efgf_gate_gradient.py's pattern (one real batch, .backward(), no
optimizer.step()) from "just the EFGF gates" to every parameter and every module
in the network, plus two checks that single script didn't have: forward-activation
health (dead/saturated units) and an actual overfit-convergence test.

WHY ONE BATCH IS ENOUGH (same reasoning as diagnose_efgf_gate_gradient.py)
---------------------------------------------------------------------------
"Is this layer wired into the computation graph correctly, does it receive a
non-degenerate gradient, and can the optimizer actually move it to reduce loss"
are structural questions about the graph and the optimizer loop, not questions
about generalization. A single real batch, run through many steps until the
model has clearly memorized it (or clearly failed to), answers all three without
touching cross-validation, early stopping, or any of the 26-subject sweep. If
this passes on one subject's one batch, the PLUMBING is sound; whether the
resulting features generalize across subjects is a separate, already
heavily-instrumented question (that's what the full 26-subject run answers).

WHAT IT CHECKS
==============
1. GRADIENT FLOW PER LAYER (fresh model, one batch, one backward() pass, NO
   optimizer.step() -- so weight_decay never contaminates the reading, exactly
   like diagnose_efgf_gate_gradient.py):
     - any parameter with grad is None -> DEAD PATH (not connected to the loss
       at all -- a real bug, e.g. a branch computed but never used downstream).
     - any parameter with grad decided more than STALL_THRESHOLD is NaN/Inf.
     - per-MODULE (not per-tensor) grad-norm summary, in forward depth order
       (conv1 -> conv2 -> efgf1 -> efgf2 -> classifiers), so a vanishing- or
       exploding-gradient trend across depth is visible directly, not just as
       isolated numbers.
     - grad-norm / weight-norm ratio per module: the actual fractional update
       size SGD/Adam will apply this step. Near 0 across an entire module =
       effectively frozen; very large (>1) = unstable step relative to weight
       scale (less meaningful for Adam specifically, reported for completeness).

2. FORWARD ACTIVATION HEALTH (same batch, forward hooks on every leaf module):
     - fraction of activation entries that are exactly 0 (dead ReLU-style units
       if this fraction saturates near 1.0 for a whole layer);
     - activation std (near 0 = the layer collapsed to a near-constant output,
       which starves every downstream layer's gradient regardless of what
       check 1 says about that downstream layer in isolation);
     - any NaN/Inf in activations (would explain any NaN seen in check 1's
       gradients, and pinpoints WHERE in the forward pass it first appears).

3. OVERFIT-CONVERGENCE TEST (the actual "does backprop work" proof): freeze the
   SAME single batch, run N plain SGD/Adam steps on it with no regularization
   (no weight_decay, no dropout -- model.eval() for BatchNorm/Dropout during
   this phase is wrong since we WANT it to fit batch statistics; instead we use
   a plain nn.Module with dropout_type effectively bypassed via eval() -- see
   NOTE in the code), and report the loss curve. A healthy, correctly-wired
   network should drive loss on a single small batch down by >90% within a few
   hundred steps. If it plateaus early, that is strong evidence of a structural
   problem (dead gradient path, a frozen/mis-wired module, or a bad loss
   formulation) rather than anything to do with data or generalization --
   check 1/2's per-layer numbers then tell you WHERE to look.

USAGE
=====
    %run layer_gradient_diagnostics.py
(edit SUBJECT_DIR below first; run in the same kernel as tsfnet_model_notebook.py,
exactly like diagnose_efgf_gate_gradient.py -- needs one subject's preprocessed
.npy files, nothing else. No GPU required but much faster with one.)
"""
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, tsfnet_loss

# ---- EDIT THIS ----
SUBJECT_DIR = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\VP001"
SUBJECT_ID = "VP001"
FNIRS_NUM_SEGMENTS = 21
BATCH_SIZE = 16
SEED = 42
OVERFIT_STEPS = 400
OVERFIT_LR = 1e-3
# --------------------

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_model():
    # SAME architecture/init flags as the accepted config (run60-onestage).
    return TSFNet(
        num_classes=3,
        fusion_combination="paper_mean",
        fnirs_num_segments=FNIRS_NUM_SEGMENTS,
        fnirs_temporal_align="egta",
        efgf_gate_mode="sigmoid",
        efgf_gate_init=-2.0,
    ).to(device)


def get_one_batch():
    ds = NBackDataset(SUBJECT_DIR, SUBJECT_ID)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                         generator=torch.Generator().manual_seed(SEED))
    batch = next(iter(loader))
    x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(batch)
    x_eeg, x_fnirs, y = x_eeg.to(device), x_fnirs.to(device), y.to(device)
    x_conn = x_conn.to(device) if x_conn is not None else None
    x_chan = x_chan.to(device) if x_chan is not None else None
    return x_eeg, x_fnirs, y, x_conn, x_chan


# ---------------------------------------------------------------------------
# CHECK 1 -- gradient flow per module
# ---------------------------------------------------------------------------
def check_gradient_flow(model, batch):
    x_eeg, x_fnirs, y, x_conn, x_chan = batch
    model.train()
    model.zero_grad()
    y_eeg, y_fnirs, y_fusion, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)
    loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux, normalize_ypred=False)
    loss.backward()

    print("=" * 88)
    print("CHECK 1 -- GRADIENT FLOW (fresh model, one real batch, one backward() pass)")
    print("=" * 88)
    print(f"Loss (fresh init, one batch): {loss.item():.4f}\n")

    dead, nan_or_inf = [], []
    module_grads = {}   # module name (prefix before last '.') -> list of (param-name, grad_norm, weight_norm)
    for name, p in model.named_parameters():
        mod_name = name.rsplit(".", 1)[0] if "." in name else "(root)"
        if p.grad is None:
            dead.append(name)
            module_grads.setdefault(mod_name, []).append((name, None, p.detach().norm().item()))
            continue
        g = p.grad.detach()
        if torch.isnan(g).any() or torch.isinf(g).any():
            nan_or_inf.append(name)
        gnorm = g.norm().item()
        wnorm = p.detach().norm().item()
        module_grads.setdefault(mod_name, []).append((name, gnorm, wnorm))

    if dead:
        print(f"[DEAD PATH] {len(dead)} parameter(s) got NO gradient at all "
              f"(grad is None -- not connected to the loss):")
        for n in dead:
            print(f"    {n}")
    else:
        print("[OK] every parameter received some gradient (no dead paths).")

    if nan_or_inf:
        print(f"\n[NaN/Inf] {len(nan_or_inf)} parameter(s) have NaN or Inf in their gradient:")
        for n in nan_or_inf:
            print(f"    {n}")
    else:
        print("[OK] no NaN/Inf in any gradient.")

    print(f"\n{'module':45s} {'grad_norm':>12s} {'weight_norm':>12s} {'ratio(g/w)':>12s}")
    print("-" * 88)
    for mod_name, entries in module_grads.items():
        gnorms = [g for _, g, _ in entries if g is not None]
        wnorms = [w for _, _, w in entries]
        if not gnorms:
            print(f"{mod_name:45s} {'--':>12s} {sum(wnorms):12.4g} {'DEAD':>12s}")
            continue
        g_total = math.sqrt(sum(g * g for g in gnorms))
        w_total = math.sqrt(sum(w * w for w in wnorms))
        ratio = g_total / (w_total + 1e-12)
        flag = "  <-- near-zero update" if ratio < 1e-5 else ("  <-- large step" if ratio > 1.0 else "")
        print(f"{mod_name:45s} {g_total:12.4g} {w_total:12.4g} {ratio:12.4g}{flag}")

    return loss.item()


# ---------------------------------------------------------------------------
# CHECK 2 -- forward activation health via hooks
# ---------------------------------------------------------------------------
def check_activation_health(model, batch):
    x_eeg, x_fnirs, y, x_conn, x_chan = batch
    print()
    print("=" * 88)
    print("CHECK 2 -- FORWARD ACTIVATION HEALTH (same batch, forward hooks on leaf modules)")
    print("=" * 88)

    stats = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t):
                return
            with torch.no_grad():
                flat = t.detach().float()
                n = flat.numel()
                zero_frac = float((flat == 0).sum()) / max(n, 1)
                has_nan = bool(torch.isnan(flat).any())
                has_inf = bool(torch.isinf(flat).any())
                stats[name] = {
                    "shape": tuple(t.shape), "mean": float(flat.mean()), "std": float(flat.std()),
                    "zero_frac": zero_frac, "nan": has_nan, "inf": has_inf,
                }
        return hook

    leaf_modules = [(n, m) for n, m in model.named_modules()
                    if len(list(m.children())) == 0 and any(True for _ in m.parameters(recurse=False))]
    for name, m in leaf_modules:
        handles.append(m.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(x_eeg, x_fnirs, x_conn, x_chan)
    for h in handles:
        h.remove()

    print(f"{'layer':45s} {'shape':>20s} {'mean':>9s} {'std':>9s} {'zero%':>7s}  flags")
    print("-" * 88)
    suspicious = []
    for name, s in stats.items():
        flags = []
        if s["std"] < 1e-6:
            flags.append("COLLAPSED(std~0)")
        if s["zero_frac"] > 0.98:
            flags.append("DEAD(~all zero)")
        if s["nan"]:
            flags.append("NaN")
        if s["inf"]:
            flags.append("Inf")
        if flags:
            suspicious.append(name)
        print(f"{name:45s} {str(s['shape']):>20s} {s['mean']:9.3g} {s['std']:9.3g} "
              f"{100*s['zero_frac']:6.1f}%  {' '.join(flags)}")

    if not suspicious:
        print("\n[OK] no collapsed, dead, or NaN/Inf activations found.")
    else:
        print(f"\n[FLAGGED] {len(suspicious)} layer(s) look unhealthy -- see table above.")
    return stats


# ---------------------------------------------------------------------------
# CHECK 3 -- overfit-convergence test (the real "does backprop work" proof)
# ---------------------------------------------------------------------------
def check_overfit_convergence(batch, steps=OVERFIT_STEPS, lr=OVERFIT_LR):
    x_eeg, x_fnirs, y, x_conn, x_chan = batch
    print()
    print("=" * 88)
    print(f"CHECK 3 -- OVERFIT-CONVERGENCE TEST ({steps} plain Adam steps on ONE fixed batch, "
          f"no weight_decay)")
    print("=" * 88)
    print("A correctly-wired network should drive loss on a single small batch down by")
    print(">90% within a few hundred steps. Early plateau = structural problem, not a")
    print("generalization issue (there is no train/test split here -- just memorization).\n")

    torch.manual_seed(SEED)  # fresh init again, independent of checks 1-2's model state
    model = build_model()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    losses = []
    for step in range(steps):
        opt.zero_grad()
        y_eeg, y_fnirs, y_fusion, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)
        loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux, normalize_ypred=False)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    losses = np.array(losses)
    init_loss, final_loss = losses[0], losses[-1]
    drop_pct = 100 * (init_loss - final_loss) / max(init_loss, 1e-12)
    checkpoints = [0, steps // 4, steps // 2, 3 * steps // 4, steps - 1]
    print("step   loss")
    for c in checkpoints:
        print(f"{c:5d}  {losses[c]:.4f}")
    print(f"\nTotal drop: {init_loss:.4f} -> {final_loss:.4f}  ({drop_pct:.1f}% reduction)")

    model.eval()
    with torch.no_grad():
        _, _, _, y_pred, _ = model(x_eeg, x_fnirs, x_conn, x_chan)
        pred_cls = y_pred.argmax(dim=1)
        acc = (pred_cls == y).float().mean().item()
    print(f"Final train-batch accuracy on this same batch (memorization check): {acc*100:.1f}%")

    if drop_pct > 90 and acc > 0.9:
        print("\n[OK] the network can memorize a single batch -- backprop reaches every "
              "trainable parameter in a way that actually reduces loss. Any accuracy "
              "problems in the full 26-subject run are about GENERALIZATION, not plumbing.")
    elif drop_pct > 50:
        print("\n[BORDERLINE] loss dropped meaningfully but didn't converge fully in "
              f"{steps} steps -- try more steps or a higher lr before concluding anything "
              "is structurally wrong.")
    else:
        print("\n[SUSPECT] loss barely moved on a SINGLE batch with no regularization. "
              "This points to a structural problem (dead gradient path, a frozen or "
              "mis-wired module, a saturated loss term) -- re-check Checks 1 and 2's "
              "flagged layers above, they should point at the same module.")
    return losses


def main():
    batch = get_one_batch()
    model = build_model()
    check_gradient_flow(model, batch)
    check_activation_health(model, batch)
    check_overfit_convergence(batch)


if __name__ == "__main__":
    main()
