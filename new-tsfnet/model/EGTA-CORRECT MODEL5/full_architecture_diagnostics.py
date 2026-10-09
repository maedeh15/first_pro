"""
full_architecture_diagnostics.py -- a single-file battery covering the 8
categories of model-behaviour diagnostics discussed for this project, beyond
plain gradient-flow (layer_gradient_diagnostics.py covers category 1 only).

NO FULL 26-SUBJECT TRAINING IS NEEDED. Every section below uses either:
  (a) a FRESH model + one or a few real batches of ONE subject (cheap, seconds), or
  (b) a SHORT local training run (a few hundred/thousand steps on that same
      subject's real batches -- SHORT_TRAIN_STEPS below) as a fast PROXY for
      full training dynamics. This is NOT a substitute for the real 26-subject/
      3-seed sweep -- it answers "is there a structural problem visible early"
      the same way layer_gradient_diagnostics.py's overfit test did, not "what
      is the final generalization accuracy".

SECTIONS
========
1. GRADIENT FLOW -- same as layer_gradient_diagnostics.py (kept here too so
   this file is self-contained); see that file's docstring for the full
   rationale.
2. WEIGHT HEALTH OVER TRAINING -- per-module weight_norm and grad_norm logged
   every few steps of the short training run. Flags: runaway growth
   (explosion), a module whose norm stops changing after early steps (frozen/
   dead), bimodal/near-zero weight distributions at the end (dead units).
3. REPRESENTATION QUALITY -- linear CKA between consecutive leaf layers'
   activations (does depth add anything beyond a linear transform of the
   previous layer?) and effective rank (participation-ratio) of the final
   fusion embedding (collapse detector).
4. SENSITIVITY / ROBUSTNESS -- input-gradient saliency per input modality
   (EEG vs fNIRS, and per EEG frequency band if eeg_filterbank is on),
   output stability under small Gaussian input noise (local Lipschitz
   estimate), and EFGF/EGTA gate-entropy trajectory over the short training
   run (does gating move away from its near-uniform/near-closed init in a
   way that is MEASURED over many steps, not just step 0 as in
   layer_gradient_diagnostics.py).
5. LOSS LANDSCAPE -- sharpness around the short-run's final weights (random
   weight perturbation -> loss increase; SAM-style, cheap filter-normalized
   version) and, OPTIONALLY, if you pass two real checkpoint paths (e.g. two
   different seeds of the full run), linear-interpolation loss barrier
   between them on this one subject's data (are they in the same basin?).
6. INTER-BRANCH INFORMATION FLOW -- single-branch ablation at inference
   (zero out EEG / fNIRS / fusion branch's contribution to y_pred, measure
   accuracy/logit drift) and a histogram-based mutual-information ESTIMATE
   between pooled EEG and fNIRS branch embeddings (complementary vs
   redundant).
7. OPTIMIZATION DYNAMICS -- per-step (not per-epoch) loss curve variance
   over the short training run, and a simple gradient-noise-scale estimate
   (McCandlish et al. 2018-style: compare a big-batch vs small-batch
   gradient norm to see whether batch_size=16 is noise-dominated for this
   model).
8. CAPACITY / COMPLEXITY -- trainable-parameter count vs. this subject's
   number of training windows (sanity ratio), and a random-label
   memorization test (Zhang et al. 2017): shuffle this subject's own labels
   and run the SAME short training loop -- if the network memorizes shuffled
   labels about as easily as real ones, its capacity is well above what the
   task needs (not inherently bad, but tells you regularization/early
   stopping is doing the real generalization work, not architecture limits).

USAGE
=====
    %run full_architecture_diagnostics.py
(edit SUBJECT_DIR below; run in the same kernel as tsfnet_model_notebook.py.
Needs only one subject's preprocessed .npy files -- no GPU required, much
faster with one. Expect 1-3 minutes total on CPU, seconds on GPU, dominated
by SHORT_TRAIN_STEPS.)

Set CKPT_A / CKPT_B below (optional) to run section 5's two-checkpoint
basin check with real trained weights instead of the short run's own.
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, tsfnet_loss

# ---- EDIT THESE ----
SUBJECT_DIR = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\VP001"
SUBJECT_ID = "VP001"
FNIRS_NUM_SEGMENTS = 21
BATCH_SIZE = 16
SEED = 42
SHORT_TRAIN_STEPS = 500     # proxy training run length for sections 2/4/7/8
SHORT_TRAIN_LR = 1e-3
LOG_EVERY = 25
CKPT_A = None   # e.g. r"...\results_40b7d8ea53_seed42\VP001_session1_best.pt"
CKPT_B = None   # e.g. r"...\results_40b7d8ea53_seed999\VP001_session1_best.pt"
RUN_SECTIONS = {1, 2, 3, 4, 5, 6, 7, 8}   # comment out any you want to skip
# ---------------------

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_model():
    return TSFNet(
        num_classes=3, fusion_combination="paper_mean",
        fnirs_num_segments=FNIRS_NUM_SEGMENTS, fnirs_temporal_align="egta",
        efgf_gate_mode="sigmoid", efgf_gate_init=-2.0,
    ).to(device)


def get_loader():
    ds = NBackDataset(SUBJECT_DIR, SUBJECT_ID)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                       generator=torch.Generator().manual_seed(SEED)), ds


def to_device(batch):
    x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(batch)
    x_eeg, x_fnirs, y = x_eeg.to(device), x_fnirs.to(device), y.to(device)
    x_conn = x_conn.to(device) if x_conn is not None else None
    x_chan = x_chan.to(device) if x_chan is not None else None
    return x_eeg, x_fnirs, y, x_conn, x_chan


def forward_loss(model, batch):
    x_eeg, x_fnirs, y, x_conn, x_chan = batch
    y_eeg, y_fnirs, y_fusion, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)
    loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux, normalize_ypred=False)
    return loss, y_pred


def hline(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


# ===========================================================================
# SECTION 1 -- gradient flow (see layer_gradient_diagnostics.py for rationale)
# ===========================================================================
def section1_gradient_flow(model, batch):
    hline("SECTION 1 -- GRADIENT FLOW (fresh model, one batch, one backward pass)")
    model.train()
    model.zero_grad()
    loss, _ = forward_loss(model, batch)
    loss.backward()
    print(f"Loss (fresh init, one batch): {loss.item():.4f}")

    dead, nan_or_inf, mod_grads = [], [], {}
    for name, p in model.named_parameters():
        mod_name = name.rsplit(".", 1)[0] if "." in name else "(root)"
        if p.grad is None:
            dead.append(name)
            continue
        g = p.grad.detach()
        if torch.isnan(g).any() or torch.isinf(g).any():
            nan_or_inf.append(name)
        mod_grads.setdefault(mod_name, []).append(g.norm().item())

    print(f"[dead paths] {len(dead)}" + (f": {dead}" if dead else " -- none"))
    print(f"[NaN/Inf grads] {len(nan_or_inf)}" + (f": {nan_or_inf}" if nan_or_inf else " -- none"))
    weak = [(m, np.sqrt(sum(g * g for g in gs))) for m, gs in mod_grads.items()]
    weak.sort(key=lambda t: t[1])
    print("Weakest 5 modules by grad-norm (candidates for cold-start issues):")
    for m, g in weak[:5]:
        print(f"    {m:50s} grad_norm={g:.4g}")


# ===========================================================================
# SECTION 2 -- weight health over a short training run
# ===========================================================================
def short_train_run(model, loader, steps, lr, shuffle_labels=False, log_every=LOG_EVERY,
                     track_modules=True, track_gates=True):
    """Shared engine for sections 2, 4(gate part), 7, 8. Runs `steps` plain
    Adam steps cycling through this subject's real batches (no weight_decay,
    no val split -- this is a diagnostic proxy, not the real training loop).
    Returns a dict of logged time series."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    data_iter = iter(loader)
    log = {"step": [], "loss": [], "module_wnorm": {}, "module_gnorm": {},
           "gate_vals": {}, "grad_norms_all_steps": []}

    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        x_eeg, x_fnirs, y, x_conn, x_chan = to_device(batch)
        if shuffle_labels:
            y = y[torch.randperm(y.shape[0], device=device)]
        opt.zero_grad()
        loss, _ = forward_loss(model, (x_eeg, x_fnirs, y, x_conn, x_chan))
        loss.backward()

        total_g = math.sqrt(sum(p.grad.detach().norm().item() ** 2
                                 for p in model.parameters() if p.grad is not None))
        log["grad_norms_all_steps"].append(total_g)
        opt.step()
        log["step"].append(step)
        log["loss"].append(loss.item())

        if step % log_every == 0 or step == steps - 1:
            if track_modules:
                for name, p in model.named_parameters():
                    mod = name.rsplit(".", 1)[0] if "." in name else "(root)"
                    log["module_wnorm"].setdefault(mod, []).append(p.detach().norm().item())
                    g = p.grad.detach().norm().item() if p.grad is not None else 0.0
                    log["module_gnorm"].setdefault(mod, []).append(g)
            if track_gates:
                with torch.no_grad():
                    for gate_name in ("efgf1.gamma_raw", "efgf1.alpha_raw",
                                      "efgf2.gamma_raw", "efgf2.alpha_raw"):
                        p = dict(model.named_parameters()).get(gate_name)
                        if p is not None:
                            log["gate_vals"].setdefault(gate_name, []).append(
                                float(torch.sigmoid(p).item()))
    return log


def section2_weight_health(log):
    hline("SECTION 2 -- WEIGHT HEALTH OVER A SHORT TRAINING RUN "
          f"({SHORT_TRAIN_STEPS} steps, logged every {LOG_EVERY})")
    print(f"{'module':50s} {'w_start':>10s} {'w_end':>10s} {'w_growth_x':>11s} "
          f"{'g_start':>10s} {'g_end':>10s}  flags")
    print("-" * 92)
    flagged = []
    for mod in log["module_wnorm"]:
        w = np.array(log["module_wnorm"][mod])
        g = np.array(log["module_gnorm"][mod])
        if len(w) < 2 or w[0] < 1e-9:
            growth = float("nan")
        else:
            growth = w[-1] / w[0]
        flags = []
        if growth > 5:
            flags.append("GROWTH>5x(watch for explosion)")
        if g[-5:].mean() < 1e-6 and w[-1] > 1e-6:
            flags.append("GRAD_VANISHED_LATE")
        if flags:
            flagged.append(mod)
        print(f"{mod:50s} {w[0]:10.4g} {w[-1]:10.4g} {growth:11.3g} "
              f"{g[0]:10.4g} {g[-1]:10.4g}  {' '.join(flags)}")
    if not flagged:
        print("\n[OK] no module shows runaway growth or a late-training vanished gradient.")
    else:
        print(f"\n[FLAGGED] {len(flagged)} module(s) -- see table above.")


# ===========================================================================
# SECTION 3 -- representation quality (CKA between layers, effective rank)
# ===========================================================================
def _linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X.T @ Y).norm() ** 2
    norm_x = (X.T @ X).norm()
    norm_y = (Y.T @ Y).norm()
    return float(hsic / (norm_x * norm_y + 1e-12))


def section3_representation_quality(model, batch):
    hline("SECTION 3 -- REPRESENTATION QUALITY (linear CKA between layers, effective rank)")
    x_eeg, x_fnirs, y, x_conn, x_chan = batch
    acts = {}
    handles = []

    def hook(name):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t):
                acts[name] = t.detach().reshape(t.shape[0], -1).float()
        return fn

    targets = ["eeg_branch.conv1.conv.conv", "eeg_branch.conv2.conv.conv",
               "fnirs_branch.conv1.conv.conv", "fnirs_branch.conv2.conv.conv",
               "cafe.fusion_proj", "cafe.fusion_out_proj", "clf_fusion.net.0"]
    for n, m in model.named_modules():
        if n in targets:
            handles.append(m.register_forward_hook(hook(n)))
    model.eval()
    with torch.no_grad():
        model(x_eeg, x_fnirs, x_conn, x_chan)
    for h in handles:
        h.remove()

    print("Linear CKA between CONSECUTIVE layers in each branch (1.0 = layer 2 is just a")
    print("linear function of layer 1 -- depth adds nothing; near 0 = genuinely new structure):")
    pairs = [("eeg_branch.conv1.conv.conv", "eeg_branch.conv2.conv.conv"),
             ("fnirs_branch.conv1.conv.conv", "fnirs_branch.conv2.conv.conv"),
             ("cafe.fusion_proj", "cafe.fusion_out_proj"),
             ("cafe.fusion_out_proj", "clf_fusion.net.0")]
    for a, b in pairs:
        if a in acts and b in acts:
            cka = _linear_cka(acts[a], acts[b])
            flag = "  <-- near-redundant layers" if cka > 0.9 else ""
            print(f"    {a:35s} vs {b:35s}  CKA={cka:.3f}{flag}")

    print("\nEffective rank (participation ratio) of final embeddings (collapse detector):")
    for name in ("cafe.fusion_out_proj", "clf_fusion.net.0"):
        if name not in acts:
            continue
        X = acts[name]
        X = X - X.mean(0, keepdim=True)
        s = torch.linalg.svdvals(X)
        eig = s ** 2
        eff_rank = float((eig.sum() ** 2) / (eig ** 2).sum() + 1e-12)
        dim = X.shape[1]
        flag = "  <-- possible representation collapse" if eff_rank < 0.1 * dim else ""
        print(f"    {name:35s} dim={dim:4d}  effective_rank={eff_rank:.1f}{flag}")


# ===========================================================================
# SECTION 4 -- sensitivity / robustness
# ===========================================================================
def section4_robustness(model, batch, train_log):
    hline("SECTION 4 -- SENSITIVITY / ROBUSTNESS")
    x_eeg, x_fnirs, y, x_conn, x_chan = batch
    model.eval()

    # --- 4a: input-gradient saliency per modality ---
    x_eeg_g = x_eeg.clone().requires_grad_(True)
    x_fnirs_g = x_fnirs.clone().requires_grad_(True)
    y_eeg_o, y_fnirs_o, y_fusion_o, y_pred, aux = model(x_eeg_g, x_fnirs_g, x_conn, x_chan)
    target = y_pred.gather(1, y.view(-1, 1)).sum()
    g_eeg, g_fnirs = torch.autograd.grad(target, [x_eeg_g, x_fnirs_g], retain_graph=False)
    sal_eeg = g_eeg.detach().abs().mean().item()
    sal_fnirs = g_fnirs.detach().abs().mean().item()
    print("4a. Input-gradient saliency (mean |d(true-class prob)/d(input)|):")
    print(f"    EEG input:   {sal_eeg:.4g}")
    print(f"    fNIRS input: {sal_fnirs:.4g}")
    ratio = sal_eeg / (sal_fnirs + 1e-12)
    print(f"    ratio EEG/fNIRS = {ratio:.3g} "
          f"({'EEG dominates' if ratio > 3 else 'fNIRS dominates' if ratio < 1/3 else 'balanced'})")

    if hasattr(model, "eeg_filterbank"):
        print("    (eeg_filterbank present -- per-band saliency needs a band-wise Jacobian; "
              "use analyze_fixed_filterbank_usage.py for band-level analysis instead)")

    # --- 4b: local Lipschitz / noise stability ---
    with torch.no_grad():
        _, _, _, y_pred_clean, _ = model(x_eeg, x_fnirs, x_conn, x_chan)
        deltas = []
        for noise_std in (0.01, 0.05, 0.1):
            x_eeg_n = x_eeg + torch.randn_like(x_eeg) * noise_std * x_eeg.std()
            x_fnirs_n = x_fnirs + torch.randn_like(x_fnirs) * noise_std * x_fnirs.std()
            _, _, _, y_pred_n, _ = model(x_eeg_n, x_fnirs_n, x_conn, x_chan)
            d = (y_pred_n - y_pred_clean).abs().mean().item()
            deltas.append((noise_std, d))
    print("\n4b. Output stability under input Gaussian noise (local Lipschitz proxy):")
    print("    noise_std(relative)   mean |delta y_pred|")
    for ns, d in deltas:
        print(f"    {ns:18.2f}   {d:.4g}")
    slope = (deltas[-1][1] - deltas[0][1]) / (deltas[-1][0] - deltas[0][0] + 1e-12)
    print(f"    slope ~ {slope:.3g}  (large slope = output is sensitive to small input "
          "perturbations, worth checking against real sensor noise levels)")

    # --- 4c: gate-entropy trajectory over the short training run ---
    print("\n4c. EFGF gate trajectory over the short training run "
          f"(logged every {LOG_EVERY} steps, init=sigmoid(-2.0)={torch.sigmoid(torch.tensor(-2.0)).item():.3f}):")
    for gate_name, vals in train_log["gate_vals"].items():
        v = np.array(vals)
        moved = abs(v[-1] - v[0])
        flag = "  <-- barely moved from init" if moved < 0.02 else ""
        print(f"    {gate_name:20s} start={v[0]:.4f} end={v[-1]:.4f} "
              f"max={v.max():.4f} min={v.min():.4f}{flag}")


# ===========================================================================
# SECTION 5 -- loss landscape
# ===========================================================================
def section5_loss_landscape(model, batch, n_dirs=5, scale=0.02):
    hline("SECTION 5 -- LOSS LANDSCAPE (sharpness around current weights)")
    model.eval()
    with torch.no_grad():
        base_loss, _ = forward_loss(model, batch)
    base_loss = base_loss.item()
    print(f"Loss at current weights: {base_loss:.4f}")
    print(f"Filter-normalized random perturbations (scale={scale}), {n_dirs} directions:")
    increases = []
    orig_state = copy.deepcopy(model.state_dict())
    for i in range(n_dirs):
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() == 0:
                    continue
                noise = torch.randn_like(p)
                noise = noise * (p.norm() / (noise.norm() + 1e-12)) * scale
                p.add_(noise)
            pert_loss, _ = forward_loss(model, batch)
        model.load_state_dict(orig_state)
        increases.append(pert_loss.item() - base_loss)
    increases = np.array(increases)
    print(f"  mean loss increase: {increases.mean():.4f}  (SD {increases.std(ddof=1):.4f})")
    print("  Large, consistent increase = SHARP minimum (small weight changes hurt a lot --")
    print("  often associated with worse generalization). Small/near-zero = FLAT minimum.")

    if CKPT_A and CKPT_B:
        print(f"\nTwo-checkpoint linear-interpolation barrier: {CKPT_A} <-> {CKPT_B}")
        try:
            sd_a = torch.load(CKPT_A, map_location=device)
            sd_b = torch.load(CKPT_B, map_location=device)
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
            losses = []
            for a in alphas:
                interp = {k: (1 - a) * sd_a[k].float() + a * sd_b[k].float()
                          for k in sd_a if sd_a[k].dtype.is_floating_point}
                model.load_state_dict(interp, strict=False)
                model.eval()
                with torch.no_grad():
                    l, _ = forward_loss(model, batch)
                losses.append(l.item())
            print("  alpha   loss")
            for a, l in zip(alphas, losses):
                print(f"  {a:5.2f}   {l:.4f}")
            mid_bump = losses[2] - 0.5 * (losses[0] + losses[-1])
            print(f"  mid-point bump vs endpoint average: {mid_bump:+.4f} "
                  f"({'barrier -- different basins' if mid_bump > 0.1 else 'no strong barrier -- same basin'})")
        except Exception as e:
            print(f"  [skipped -- could not load/interpolate checkpoints: {e}]")
        finally:
            model.load_state_dict(orig_state)
    else:
        print("\n[skip] CKPT_A/CKPT_B not set -- set both to real .pt paths to run the "
              "two-checkpoint basin check.")


# ===========================================================================
# SECTION 6 -- inter-branch information flow
# ===========================================================================
def section6_branch_info_flow(model, batch):
    hline("SECTION 6 -- INTER-BRANCH INFORMATION FLOW")
    x_eeg, x_fnirs, y, x_conn, x_chan = batch
    model.eval()
    with torch.no_grad():
        _, _, _, y_pred_full, _ = model(x_eeg, x_fnirs, x_conn, x_chan)
        acc_full = (y_pred_full.argmax(1) == y).float().mean().item()

        print("6a. Single-branch ablation at inference (zero that branch's INPUT, "
              "re-run full forward):")
        for name, zero_fn in [
            ("EEG",   lambda: (torch.zeros_like(x_eeg), x_fnirs)),
            ("fNIRS", lambda: (x_eeg, torch.zeros_like(x_fnirs))),
        ]:
            xe, xf = zero_fn()
            _, _, _, y_pred_abl, _ = model(xe, xf, x_conn, x_chan)
            acc_abl = (y_pred_abl.argmax(1) == y).float().mean().item()
            drift = (y_pred_abl - y_pred_full).abs().mean().item()
            print(f"    zero {name:6s} input -> acc {acc_full:.3f} -> {acc_abl:.3f} "
                  f"(drop {acc_full-acc_abl:+.3f}), mean |pred drift|={drift:.4g}")

    print("\n6b. Mutual-information ESTIMATE between pooled EEG and fNIRS branch embeddings")
    print("    (histogram-based MI on the top-2 PCA components of each pooled embedding;")
    print("    coarse but needs no extra dependencies -- low MI = complementary branches,")
    print("    high MI = redundant):")
    acts = {}
    handles = []

    def hook(name):
        def fn(_m, _i, out):
            # EEGLikeBranch/FNIRSBranch.forward returns (s1, s2, pooled) --
            # we want the 3rd element (the final pooled F_eeg/F_fnirs vector).
            pooled = out[2] if isinstance(out, (tuple, list)) else out
            acts[name] = pooled.detach()
        return fn
    h1 = model.eeg_branch.register_forward_hook(hook("eeg_pooled"))
    h2 = model.fnirs_branch.register_forward_hook(hook("fnirs_pooled"))
    with torch.no_grad():
        model(x_eeg, x_fnirs, x_conn, x_chan)
    h1.remove(); h2.remove()
    if "eeg_pooled" in acts and "fnirs_pooled" in acts:
        e = acts["eeg_pooled"].reshape(acts["eeg_pooled"].shape[0], -1).cpu().numpy()
        f = acts["fnirs_pooled"]
        f = f.reshape(f.shape[0], -1).cpu().numpy()
        f = f.reshape(f.shape[0], -1).cpu().numpy()

        def top_pc(x):
            x = x - x.mean(0, keepdims=True)
            u, s, vt = np.linalg.svd(x, full_matrices=False)
            return (u[:, 0] * s[0])

        try:
            e1, f1 = top_pc(e), top_pc(f)
            bins = max(3, int(np.sqrt(len(e1))))
            joint, _, _ = np.histogram2d(e1, f1, bins=bins)
            joint = joint / joint.sum() + 1e-12
            pe = joint.sum(axis=1, keepdims=True)
            pf = joint.sum(axis=0, keepdims=True)
            mi = float((joint * np.log(joint / (pe * pf))).sum())
            print(f"    MI estimate (nats, top PCA component, batch_size={e.shape[0]}): {mi:.3f}")
            print("    (small batch -> noisy estimate; trust the SIGN/magnitude trend across "
                  "multiple batches more than this one number)")
        except Exception as ex:
            print(f"    [MI estimate failed: {ex}]")
    else:
        print("    [could not capture branch pooled outputs -- check hook names against "
              "EEGLikeBranch.forward's return signature]")


# ===========================================================================
# SECTION 7 -- optimization dynamics
# ===========================================================================
def section7_optimization_dynamics(train_log):
    hline("SECTION 7 -- OPTIMIZATION DYNAMICS (per-step loss variance, gradient noise scale)")
    losses = np.array(train_log["loss"])
    grad_norms = np.array(train_log["grad_norms_all_steps"])
    window = min(50, len(losses))
    tail = losses[-window:]
    print(f"Per-step loss, last {window} steps: mean={tail.mean():.4f} SD={tail.std(ddof=1):.4f} "
          f"CV={tail.std(ddof=1)/(tail.mean()+1e-9):.3f}")
    print("High CV late in training = noisy/unstable steps (consider smaller lr or bigger batch).")

    g_tail = grad_norms[-window:]
    noise_scale = (g_tail.var(ddof=1)) / (g_tail.mean() ** 2 + 1e-12) * BATCH_SIZE
    print(f"\nGradient-noise-scale estimate (McCandlish-style, batch_size={BATCH_SIZE}): "
          f"{noise_scale:.3g}")
    print("Rough reading: if this is >> batch_size, individual batches are dominated by noise")
    print("and a bigger batch size would likely give cleaner, more consistent gradient steps;")
    print("if << batch_size, batch_size=16 is already more than this model/data needs.")


# ===========================================================================
# SECTION 8 -- capacity / complexity
# ===========================================================================
def section8_capacity(model, ds, random_label_log):
    hline("SECTION 8 -- CAPACITY / COMPLEXITY")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_windows = len(ds)
    print(f"Trainable parameters: {n_params:,}")
    print(f"This subject's training windows (all 3 sessions pooled): {n_windows:,}")
    print(f"Parameters per window: {n_params/max(n_windows,1):,.0f}  "
          "(very high ratios are common and not inherently bad for CNNs with heavy weight "
          "sharing, but worth keeping in mind given the per-subject training protocol)")

    print(f"\nRandom-label memorization test ({SHORT_TRAIN_STEPS} steps, same short-run engine, "
          "labels shuffled per batch):")
    losses = np.array(random_label_log["loss"])
    init_l, final_l = losses[0], losses[-1]
    drop_pct = 100 * (init_l - final_l) / max(init_l, 1e-12)
    print(f"  loss: {init_l:.4f} -> {final_l:.4f}  ({drop_pct:.1f}% reduction on SHUFFLED labels)")
    if drop_pct > 50:
        print("  [capacity far exceeds this task] the network can substantially fit random "
              "labels too -- generalization on real labels relies on regularization/early "
              "stopping, not on the architecture lacking capacity. Expected for a modern CNN; "
              "flags that more capacity is not what would improve accuracy here.")
    else:
        print("  the network struggled even to fit random labels as well as it fit real ones "
              "in the comparable real-label overfit test -- capacity may be closer to a real "
              "constraint than expected (cross-check against layer_gradient_diagnostics.py's "
              "Check 3 real-label result).")


def main():
    loader, ds = get_loader()
    batch = to_device(next(iter(loader)))

    if 1 in RUN_SECTIONS:
        section1_gradient_flow(build_model(), batch)

    train_log = None
    if {2, 4, 7} & RUN_SECTIONS:
        torch.manual_seed(SEED)
        model_short = build_model()
        train_log = short_train_run(model_short, loader, SHORT_TRAIN_STEPS, SHORT_TRAIN_LR)
        if 2 in RUN_SECTIONS:
            section2_weight_health(train_log)

    if 3 in RUN_SECTIONS:
        section3_representation_quality(build_model(), batch)

    if 4 in RUN_SECTIONS:
        section4_robustness(train_log and model_short or build_model(), batch,
                             train_log or {"gate_vals": {}})

    if 5 in RUN_SECTIONS:
        model5 = train_log and model_short or build_model()
        section5_loss_landscape(model5, batch)

    if 6 in RUN_SECTIONS:
        section6_branch_info_flow(train_log and model_short or build_model(), batch)

    if 7 in RUN_SECTIONS and train_log is not None:
        section7_optimization_dynamics(train_log)

    if 8 in RUN_SECTIONS:
        torch.manual_seed(SEED + 1)
        model_rand = build_model()
        rand_log = short_train_run(model_rand, loader, SHORT_TRAIN_STEPS, SHORT_TRAIN_LR,
                                    shuffle_labels=True, track_modules=False, track_gates=False)
        section8_capacity(build_model(), ds, rand_log)

    hline("DONE")
    print("Reminder: sections 2/4c/7/8 use a SHORT local training run on ONE subject as a fast")
    print("proxy -- treat findings as 'worth investigating further', not as the final word on")
    print("the real 26-subject/3-seed training dynamics.")


if __name__ == "__main__":
    main()
