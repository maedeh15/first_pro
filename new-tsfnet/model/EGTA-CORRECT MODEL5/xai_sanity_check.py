"""
xai_sanity_check.py -- model-parameter randomisation test for the trunk's explanations (Adebayo et al. 2018, "Sanity checks for saliency
maps"; applied to EEG with ground truth by Ravindran & Contreras-Vidal 2023, Sci Rep 13:18110).

Question. An explanation that does not change when the network's weights are destroyed is describing the INPUT (or the method), not the
model. Agreement between methods (xai_multimethod_consensus.py) cannot detect that, because two methods can agree for the same wrong
reason (for example both follow the input's magnitude). This script randomises the trained network progressively, from the output
backwards, and measures how fast each explanation stops resembling the trained one.

Method. For every held-out fold and window, per-fNIRS-segment importance (21 values) from
  attention             the EGTA softmax weights                        (a learned parameter of the network)
  input_gradient        |d target / d x_fnirs| averaged per segment
  integrated_gradients  Sundararajan et al. 2017, zero baseline
  input_magnitude       mean |x_fnirs| per segment -- a CONTROL that does not depend on the weights at all
is computed with the trained model and again after cumulatively re-drawing the parameters of
  heads+gates -> +CAFE -> +EFGF -> +fNIRS branch -> +fusion/EEG branches (= everything)
Spearman rho between the trained and the randomised explanation of the same window is averaged over windows, then over subjects.

Reading (fixed before running; --pass-rho sets the bar, default 0.3):
  * the control must stay at rho = 1.000 -- if it does not, the pipeline is broken, not the method;
  * a method PASSES when the mean rho after full randomisation is below --pass-rho: its output depends on what was learned;
  * a method that keeps rho high after full randomisation is not explaining the model, whatever it agrees with.
Not covered: the data-randomisation (shuffled-label) test, which needs a model retrained on permuted labels (a GPU run per configuration);
parameters here are re-drawn from N(0, 1/fan_in) (>=2-D) or N(0, 0.1^2) (1-D); batch-norm statistics are re-estimated on the test windows
when AdaBN is on, as in every other evaluation here. Operates on the TRUNK (member-branch checkpoints are loaded without their head).

USAGE
  %run xai_sanity_check.py --results-dir "<...\\results_cf64a71127_seed42>" --data-root "<DATA_ROOT>" --subjects 1-26
"""
import argparse, copy, os, sys
import numpy as np
from scipy.stats import spearmanr

import torch
from torch.utils.data import DataLoader, Subset

METHODS = ("attention", "input_gradient", "integrated_gradients", "input_magnitude")
LEVELS = ("heads+gates", "+CAFE", "+EFGF", "+fNIRS branch", "+all (fusion, EEG)")
GROUPS = [("clf_eeg", "clf_fusion", "clf_fnirs", "w_eeg", "w_fusion", "w_fnirs"),
          ("cafe",), ("efgf1", "efgf2"), ("fnirs_branch",), ("fusion_branch", "eeg_branch")]


def redraw(tensor_param, gen):
    with torch.no_grad():
        if tensor_param.dim() >= 2:
            fan_in = int(np.prod(tensor_param.shape[1:]))
            tensor_param.copy_(torch.randn(tensor_param.shape, generator=gen) / np.sqrt(fan_in))
        else:
            tensor_param.copy_(torch.randn(tensor_param.shape, generator=gen) * 0.1)


def randomise_upto(model, level, seed):
    """cumulatively re-draw parameters of GROUPS[0..level]; returns a NEW model, the input is untouched"""
    # a forward pass leaves non-leaf tensors on some modules, also inside tuples/lists (for example the stored EGTA weights
    # `cafe._last_egta_pair`); deepcopy refuses them, so they are replaced by detached copies first
    def _det(v):
        if torch.is_tensor(v):
            return v.detach() if v.grad_fn is not None else v
        if isinstance(v, tuple):
            return tuple(_det(x) for x in v)
        if isinstance(v, list):
            return [_det(x) for x in v]
        if type(v) is dict:                    # exactly dict: torch keeps its hook registries in OrderedDicts, which must stay OrderedDicts
            return {k: _det(x) for k, x in v.items()}
        return v
    for mod in model.modules():
        for k, v in list(vars(mod).items()):
            if k not in ("_parameters", "_buffers", "_modules"):
                nv = _det(v)
                if nv is not v:
                    setattr(mod, k, nv)
    m = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(int(seed))
    touched = set()
    for names in GROUPS[:level + 1]:
        for nm in names:
            obj = getattr(m, nm, None)
            if obj is None:
                continue
            ps = [obj] if isinstance(obj, torch.nn.Parameter) else list(obj.parameters())
            for p in ps:
                redraw(p, gen); touched.add(id(p))
    return m, touched


def per_segment_mean(vol):
    return vol.abs().mean(dim=tuple(range(2, vol.dim())))


def target_value(y_pred, cls, objective):
    p = y_pred.gather(1, cls.view(-1, 1)).squeeze(1)
    return p if objective == "prob" else torch.log(p.clamp_min(1e-8))


def method_list(deeplift=False):
    """the three explanation methods (+ DeepLift when requested) and, always last, the weight-independent control"""
    return METHODS[:3] + (("deeplift",) if deeplift else ()) + METHODS[3:]


def captum_available():
    """True if captum imports. Checked ONCE at start-up so a missing package costs seconds, not the minutes of attributions before it."""
    try:
        import captum  # noqa: F401
        return True
    except ImportError:
        print("  [xai] --deeplift needs captum, which is not installed: continuing WITHOUT DeepLift. Install it with  pip install captum  (in this "
              "same Python environment) and run again to include it.")
        return False


class _FnirsOnly(torch.nn.Module):
    """captum wants a module whose first argument is the attributed input; the other model inputs travel as additional args"""
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, x_fnirs, x_eeg, x_conn=None, x_chan=None):
        return self.model(x_eeg, x_fnirs, x_conn, x_chan)[3]


def deeplift_vector(model, x_eeg, x_fnirs, cls, x_conn, x_chan):
    """DeepLift (Shrikumar et al. 2017, via captum) per fNIRS segment, zero baseline, with its COMPLETENESS check.

    DeepLift's rules are applied by captum to nn.Module non-linearities only; TSFNet also uses functional sigmoid/softmax inside its
    attention and gates, which fall back to the gradient rule. Completeness (the attributions of a sample sum to f(x) - f(baseline)) is
    the check that this approximation is acceptable: returned as the ratio mean|delta| / mean|f(x)-f(baseline)|; a ratio above ~0.1 means
    DeepLift is NOT trustworthy for this model and must not be reported as DeepLift."""
    from captum.attr import DeepLift
    base = torch.zeros_like(x_fnirs)
    fo = _FnirsOnly(model)
    attr, delta = DeepLift(fo).attribute(x_fnirs, baselines=base, target=cls, additional_forward_args=(x_eeg, x_conn, x_chan),
                                         return_convergence_delta=True)
    with torch.no_grad():
        diff = (fo(x_fnirs, x_eeg, x_conn, x_chan) - fo(base, x_eeg, x_conn, x_chan)).gather(1, cls.view(-1, 1)).squeeze(1)
    ratio = float(delta.abs().mean() / (diff.abs().mean() + 1e-12))
    return per_segment_mean(attr.detach()).cpu().numpy(), ratio


def attributions(model, x_eeg, x_fnirs, y, x_conn, x_chan, ig_steps, target="true", objective="prob", deeplift=False):
    with torch.no_grad():
        y0 = model(x_eeg, x_fnirs, x_conn, x_chan)[3]
        attn = model.cafe._last_egta_weights.mean(dim=1)
    cls = y if target == "true" else y0.argmax(1)
    xg = x_fnirs.clone().requires_grad_(True)
    yp = model(x_eeg, xg, x_conn, x_chan)[3]
    sal = per_segment_mean(torch.autograd.grad(target_value(yp, cls, objective).sum(), xg)[0].detach())
    base = torch.zeros_like(x_fnirs); tot = torch.zeros_like(x_fnirs)
    for k in range(ig_steps):
        xi = (base + (k + 0.5) / ig_steps * (x_fnirs - base)).clone().requires_grad_(True)
        yi = model(x_eeg, xi, x_conn, x_chan)[3]
        tot = tot + torch.autograd.grad(target_value(yi, cls, objective).sum(), xi)[0].detach()
    ig = per_segment_mean((x_fnirs - base) * tot / ig_steps)
    mag = per_segment_mean(x_fnirs.detach())
    out = {"attention": attn.detach().cpu().numpy(), "input_gradient": sal.cpu().numpy(),
           "integrated_gradients": ig.cpu().numpy(), "input_magnitude": mag.cpu().numpy()}
    if deeplift:
        out["deeplift"], out["deeplift_delta_ratio"] = deeplift_vector(model, x_eeg, x_fnirs, cls, x_conn, x_chan)
    return out


def mean_rho(A, B):
    r = []
    for a, b in zip(A, B):
        # scale-free "constant vector" test: np.allclose's absolute tolerance would discard gradients of a probability (~1e-8)
        if np.ptp(a) <= 1e-9 * (np.abs(a).max() + 1e-300) or np.ptp(b) <= 1e-9 * (np.abs(b).max() + 1e-300):
            continue
        v = spearmanr(a, b)[0]
        if not np.isnan(v):
            r.append(v)
    return float(np.mean(r)) if r else float("nan")


def batch_report(model, batches, ig_steps, seed, target, objective, readapt=None, deeplift=False):
    """batches: list of (x_eeg, x_fnirs, y, x_conn, x_chan). Returns {level: {method: rho}}"""
    ml = method_list(deeplift)
    # copy BEFORE any captum call: captum leaves hook attributes on the modules that make a later deepcopy fail
    rms = [randomise_upto(model, lvl, seed + lvl)[0] for lvl in range(len(LEVELS))]
    trained = {m: [] for m in ml}
    ratios = []
    for bt in batches:
        v = attributions(model, *bt, ig_steps, target, objective, deeplift)
        ratios.append(v.get("deeplift_delta_ratio", np.nan))
        for m in ml:
            trained[m].append(v[m])
    trained = {m: np.concatenate(v) for m, v in trained.items()}
    out = {}
    for lvl in range(len(LEVELS)):
        rm = rms[lvl]
        rm.eval()
        if readapt is not None:
            readapt(rm)
        rnd = {m: [] for m in ml}
        for bt in batches:
            v = attributions(rm, *bt, ig_steps, target, objective, deeplift)
            for m in ml:
                rnd[m].append(v[m])
        rnd = {m: np.concatenate(v) for m, v in rnd.items()}
        out[lvl] = {m: mean_rho(trained[m], rnd[m]) for m in ml}
    if deeplift:
        out["deeplift_delta_ratio"] = float(np.nanmean(ratios))
    return out


def self_test():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tsfnet_model_notebook as M
    torch.manual_seed(0)
    kw = dict(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=21, fnirs_temporal_align="egta",
              efgf_gate_mode="sigmoid", efgf_gate_init=-2.0)
    model = M.TSFNet(**kw)
    with torch.no_grad():                      # EGTA keys are zero-initialised (uniform attention); give the test model non-trivial ones
        for n_, p_ in model.named_parameters():
            if "egta" in n_ and p_.dim() >= 1:
                p_.copy_(torch.randn_like(p_) * 0.5)
    model.eval()
    # 1) the randomisation touches every parameter at the last level and only the claimed groups at the first
    m0, t0 = randomise_upto(model, 0, 1)
    mA, tA = randomise_upto(model, len(GROUPS) - 1, 1)
    allp = {id(p) for p in mA.parameters()}
    assert tA == allp, f"the last level must re-draw every parameter ({len(tA)} of {len(allp)})"
    same_eeg = all(torch.equal(a, b) for a, b in zip(model.eeg_branch.parameters(), m0.eeg_branch.parameters()))
    changed_clf = not all(torch.equal(a, b) for a, b in zip(model.clf_eeg.parameters(), m0.clf_eeg.parameters()))
    assert same_eeg and changed_clf, "level 0 must change the heads and leave the EEG branch alone"
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), copy.deepcopy(model).parameters())), "the input model must be untouched"
    print(f"[1] randomisation order: level 0 changes only heads+gates; the last level re-draws all {len(allp)} parameters; the original is untouched")
    # 2) the control must be exactly invariant, and the learned-parameter methods must move
    xe, xf = torch.randn(4, 16, 16, 600), torch.randn(4, 21, 16, 16, 30, 2)
    y = torch.tensor([0, 1, 2, 0])
    rep = batch_report(model, [(xe, xf, y, None, None)], ig_steps=2, seed=5, target="true", objective="prob")
    last = rep[len(LEVELS) - 1]
    assert abs(last["input_magnitude"] - 1.0) < 1e-9, last["input_magnitude"]
    assert not any(np.isnan(last[m]) for m in METHODS), f"a method produced no valid windows: {last}"
    assert last["attention"] < 0.999 and last["input_gradient"] < 0.999 and last["integrated_gradients"] < 0.999, last
    print("[2] control (input magnitude) stays at rho = 1.000 under full randomisation; learned-parameter methods change: "
          + ", ".join(f"{m} {last[m]:+.2f}" for m in METHODS[:3]))
    # 3) the target options run
    for tg in ("true", "pred"):
        for ob in ("prob", "logprob"):
            v = attributions(model, xe, xf, y, None, None, 2, tg, ob)
            assert all(np.isfinite(x).all() for x in v.values())
    print("[3] target true/pred x objective prob/logprob all finite -> OK")
    print("SELFTEST PASSED")


def parse_subjects(spec):
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return [f"VP{i:03d}" for i in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("USAGE")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir"); p.add_argument("--data-root")
    p.add_argument("--subjects", default="1-26"); p.add_argument("--out-dir", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=3, help="batches per fold (cost: (1 + 5 levels) x (2 + ig-steps) passes each)")
    p.add_argument("--random-windows", action="store_true", help="take the --max-batches batches from a seeded random permutation of the session instead of its first windows (the first windows are one contiguous stretch of one class order)")
    p.add_argument("--ig-steps", type=int, default=10)
    p.add_argument("--target", choices=["true", "pred"], default="pred")
    p.add_argument("--objective", choices=["prob", "logprob"], default="logprob")
    p.add_argument("--pass-rho", type=float, default=0.3)
    p.add_argument("--deeplift", action="store_true", help="add DeepLift (captum) with its completeness gate; pip install captum")
    p.add_argument("--max-delta-ratio", type=float, default=0.1, help="DeepLift is reported only if mean|delta| / mean|f(x)-f(baseline)| is below this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adabn", action="store_true", default=True); p.add_argument("--no-adabn", dest="adabn", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not (a.results_dir and a.data_root):
        sys.exit("need --results-dir and --data-root")
    if a.deeplift and not captum_available():
        a.deeplift = False
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
    from xai_trunk import trunk_state, load_checkpoint
    out_dir = a.out_dir or os.path.join(a.results_dir, "xai_sanity_check"); os.makedirs(out_dir, exist_ok=True)
    per, ratios = {}, {}
    ML = method_list(a.deeplift)
    for sid in parse_subjects(a.subjects):
        full = NBackDataset(os.path.join(a.data_root, sid), sid)
        nseg = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
        rows = []
        for ts in (1, 2, 3):
            ck = os.path.join(a.results_dir, f"{sid}_session{ts}_best.pt")
            if not os.path.isfile(ck):
                print(f"[{sid} s{ts}] SKIP -- no checkpoint"); continue
            model = TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=nseg, fnirs_temporal_align="egta",
                           efgf_gate_mode="sigmoid", efgf_gate_init=-2.0).to(a.device)
            model.load_state_dict(trunk_state(load_checkpoint(ck, map_location=a.device)), strict=True); model.eval()
            idx = np.flatnonzero(full.session_ids == ts)
            if a.random_windows:
                idx = np.random.RandomState(1000 * int(sid[2:]) + ts).permutation(idx)
            loader = DataLoader(Subset(full, idx), batch_size=a.batch_size, shuffle=False)
            readapt = (lambda m, loader=loader: adapt_batchnorm_to_target(m, loader, a.device, log_fn=lambda *x, **k: None)) if a.adabn else None
            if readapt:
                readapt(model)
            batches = []
            for k, b in enumerate(loader):
                if k >= a.max_batches:
                    break
                xe, xf, y, xc, xch = _unpack_batch(b)
                batches.append((xe.to(a.device), xf.to(a.device), y.to(a.device),
                                xc.to(a.device) if xc is not None else None, xch.to(a.device) if xch is not None else None))
            rows.append(batch_report(model, batches, a.ig_steps, a.seed + 1000 * ts, a.target, a.objective, readapt, a.deeplift))
            del model
            if str(a.device).startswith("cuda"):
                torch.cuda.empty_cache()
        if rows:
            per[sid] = {l: {m: float(np.nanmean([r[l][m] for r in rows])) for m in ML} for l in range(len(LEVELS))}
            if a.deeplift:
                ratios.setdefault(sid, float(np.nanmean([r["deeplift_delta_ratio"] for r in rows])))
            print(f"[{sid}] done: after full randomisation " + ", ".join(f"{m} {per[sid][len(LEVELS) - 1][m]:+.2f}" for m in ML), flush=True)
    subs = sorted(per)
    print(f"\n{len(subs)} subjects; mean Spearman rho between the TRAINED and the randomised explanation of the same window "
          f"(target={a.target}, objective={a.objective})\n")
    print(f"  {'randomised up to':22}" + "".join(f"{m:>22}" for m in ML))
    for l, nm in enumerate(LEVELS):
        print(f"  {nm:22}" + "".join(f"{np.nanmean([per[s][l][m] for s in subs]):22.3f}" for m in ML))
    ok_ctrl = abs(np.nanmean([per[s][len(LEVELS) - 1]['input_magnitude'] for s in subs]) - 1.0) < 1e-6
    print(f"\n  control (input magnitude) invariant: {'YES' if ok_ctrl else 'NO -- the pipeline is broken, ignore everything below'}")
    if a.deeplift:
        rr = float(np.nanmean(list(ratios.values())))
        print(f"  DeepLift completeness: mean|delta| / mean|f(x)-f(baseline)| = {rr:.3f} "
              f"({'acceptable' if rr < a.max_delta_ratio else 'ABOVE ' + str(a.max_delta_ratio) + ': DeepLift is NOT trustworthy for this model, do not report it as DeepLift'})")
    for m in ML[:-1]:
        last = np.array([per[s][len(LEVELS) - 1][m] for s in subs])
        print(f"  {m:22} full-randomisation rho {np.nanmean(last):+.3f}; below {a.pass_rho} in {int(np.sum(last < a.pass_rho))}/{len(subs)} subjects"
              f"  -> {'PASS (depends on what was learned)' if np.nanmean(last) < a.pass_rho else 'FAIL (does not depend on the learned weights)'}")
    np.savez(os.path.join(out_dir, "sanity_check.npz"), subjects=np.array(subs),
             rho=np.array([[[per[s][l][m] for m in ML] for l in range(len(LEVELS))] for s in subs]), methods=np.array(ML), levels=np.array(LEVELS))
    print("\n  Not tested here: the shuffled-label (data randomisation) variant, which needs a retrained model.")


if __name__ == "__main__":
    main()
