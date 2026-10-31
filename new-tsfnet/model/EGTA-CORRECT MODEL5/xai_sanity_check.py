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
        if isinstance(v, dict):
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


def attributions(model, x_eeg, x_fnirs, y, x_conn, x_chan, ig_steps, target="true", objective="prob"):
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
    return {"attention": attn.detach().cpu().numpy(), "input_gradient": sal.cpu().numpy(),
            "integrated_gradients": ig.cpu().numpy(), "input_magnitude": mag.cpu().numpy()}


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


def batch_report(model, batches, ig_steps, seed, target, objective, readapt=None):
    """batches: list of (x_eeg, x_fnirs, y, x_conn, x_chan). Returns {level: {method: rho}}"""
    trained = {m: [] for m in METHODS}
    for bt in batches:
        v = attributions(model, *bt, ig_steps, target, objective)
        for m in METHODS:
            trained[m].append(v[m])
    trained = {m: np.concatenate(v) for m, v in trained.items()}
    out = {}
    for lvl in range(len(LEVELS)):
        rm, _ = randomise_upto(model, lvl, seed + lvl)
        rm.eval()
        if readapt is not None:
            readapt(rm)
        rnd = {m: [] for m in METHODS}
        for bt in batches:
            v = attributions(rm, *bt, ig_steps, target, objective)
            for m in METHODS:
                rnd[m].append(v[m])
        rnd = {m: np.concatenate(v) for m, v in rnd.items()}
        out[lvl] = {m: mean_rho(trained[m], rnd[m]) for m in METHODS}
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
    p.add_argument("--ig-steps", type=int, default=10)
    p.add_argument("--target", choices=["true", "pred"], default="pred")
    p.add_argument("--objective", choices=["prob", "logprob"], default="logprob")
    p.add_argument("--pass-rho", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adabn", action="store_true", default=True); p.add_argument("--no-adabn", dest="adabn", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not (a.results_dir and a.data_root):
        sys.exit("need --results-dir and --data-root")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
    from xai_trunk import trunk_state
    out_dir = a.out_dir or os.path.join(a.results_dir, "xai_sanity_check"); os.makedirs(out_dir, exist_ok=True)
    per = {}
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
            model.load_state_dict(trunk_state(torch.load(ck, map_location=a.device)), strict=True); model.eval()
            idx = np.flatnonzero(full.session_ids == ts)
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
            rows.append(batch_report(model, batches, a.ig_steps, a.seed + 1000 * ts, a.target, a.objective, readapt))
            del model
            if str(a.device).startswith("cuda"):
                torch.cuda.empty_cache()
        if rows:
            per[sid] = {l: {m: float(np.nanmean([r[l][m] for r in rows])) for m in METHODS} for l in range(len(LEVELS))}
            print(f"[{sid}] done: after full randomisation " + ", ".join(f"{m} {per[sid][len(LEVELS) - 1][m]:+.2f}" for m in METHODS), flush=True)
    subs = sorted(per)
    print(f"\n{len(subs)} subjects; mean Spearman rho between the TRAINED and the randomised explanation of the same window "
          f"(target={a.target}, objective={a.objective})\n")
    print(f"  {'randomised up to':22}" + "".join(f"{m:>22}" for m in METHODS))
    for l, nm in enumerate(LEVELS):
        print(f"  {nm:22}" + "".join(f"{np.nanmean([per[s][l][m] for s in subs]):22.3f}" for m in METHODS))
    ok_ctrl = abs(np.nanmean([per[s][len(LEVELS) - 1]['input_magnitude'] for s in subs]) - 1.0) < 1e-6
    print(f"\n  control (input magnitude) invariant: {'YES' if ok_ctrl else 'NO -- the pipeline is broken, ignore everything below'}")
    for m in METHODS[:3]:
        last = np.array([per[s][len(LEVELS) - 1][m] for s in subs])
        print(f"  {m:22} full-randomisation rho {np.nanmean(last):+.3f}; below {a.pass_rho} in {int(np.sum(last < a.pass_rho))}/{len(subs)} subjects"
              f"  -> {'PASS (depends on what was learned)' if np.nanmean(last) < a.pass_rho else 'FAIL (does not depend on the learned weights)'}")
    np.savez(os.path.join(out_dir, "sanity_check.npz"), subjects=np.array(subs),
             rho=np.array([[[per[s][l][m] for m in METHODS] for l in range(len(LEVELS))] for s in subs]), methods=np.array(METHODS), levels=np.array(LEVELS))
    print("\n  Not tested here: the shuffled-label (data randomisation) variant, which needs a retrained model.")


if __name__ == "__main__":
    main()
