"""
xai_label_randomisation.py -- DATA-randomisation sanity check (Adebayo et al. 2018): do the explanations of a network trained on the true labels
differ from the explanations of a network trained on PERMUTED labels?

Why. xai_sanity_check.py destroys the trained weights. This is the other half: keep the architecture and the training procedure, but train it on
labels that carry no information, so it can only memorise. An explanation that looks the same for both networks is describing the input, not
what was learned about the task.

Design. Train once on shuffled labels (GPU; ~6 min per subject), with block-level permutation of the TRAINING sessions' labels (windows of one
block share one random label; class counts preserved; the held-out session keeps its true labels):
  %run run_all_subjects.py --subjects 1-3 --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --shuffle-train-labels 7
then compare it with the true-label run of the same subjects and seed on the SAME held-out windows:
  %run xai_label_randomisation.py --true-dir "<results_<hash>_seed42>" --shuffled-dir "<results_<hash2>_seed42>" --data-root "<DATA_ROOT>" --subjects 1-3

Reading (fixed before running; --pass-rho, default 0.3):
  * the shuffled network's accuracy on the true-labelled held-out windows must be near chance (<= 0.45); if not, the shuffle did not remove the
    signal and nothing below means anything (a leak or a too-weak permutation);
  * the weight-independent control (input magnitude) must be exactly 1.000;
  * a method PASSES when the mean Spearman rho between the two networks' per-segment explanations of the same window is below --pass-rho.
Operates on the TRUNK (member-branch checkpoints are loaded without their head).
"""
import argparse, os, sys
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai_sanity_check import attributions, method_list, mean_rho, parse_subjects       # noqa: E402


def compare_models(model_a, model_b, batches, ig_steps, target, objective, deeplift=False):
    """-> ({method: mean rho between A's and B's explanation of the same window}, acc_a, acc_b)"""
    ml = method_list(deeplift)
    va, vb = {m: [] for m in ml}, {m: [] for m in ml}
    ok_a = ok_b = n = 0
    for bt in batches:
        a = attributions(model_a, *bt, ig_steps, target, objective, deeplift)
        b = attributions(model_b, *bt, ig_steps, target, objective, deeplift)
        for m in ml:
            va[m].append(a[m]); vb[m].append(b[m])
        with torch.no_grad():
            y = bt[2]
            ok_a += (model_a(*[bt[0], bt[1], bt[3], bt[4]])[3].argmax(1) == y).sum().item()
            ok_b += (model_b(*[bt[0], bt[1], bt[3], bt[4]])[3].argmax(1) == y).sum().item()
            n += len(y)
    return {m: mean_rho(np.concatenate(va[m]), np.concatenate(vb[m])) for m in ml}, ok_a / n, ok_b / n


def self_test():
    import tsfnet_model_notebook as M
    kw = dict(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=21, fnirs_temporal_align="egta",
              efgf_gate_mode="sigmoid", efgf_gate_init=-2.0)

    def mk(seed):
        torch.manual_seed(seed)
        m = M.TSFNet(**kw)
        with torch.no_grad():
            for n_, p_ in m.named_parameters():
                if "egta" in n_ and p_.dim() >= 1:
                    p_.copy_(torch.randn_like(p_) * 0.5)
        return m.eval()
    a, b = mk(1), mk(2)
    xe, xf = torch.randn(4, 16, 16, 600), torch.randn(4, 21, 16, 16, 30, 2); y = torch.tensor([0, 1, 2, 0])
    bt = [(xe, xf, y, None, None)]
    same, accA, accA2 = compare_models(a, mk(1), bt, 2, "pred", "logprob")
    assert all(abs(v - 1.0) < 1e-6 for m_, v in same.items() if m_ in ("attention", "input_gradient", "integrated_gradients", "input_magnitude")), same
    diff, _, _ = compare_models(a, b, bt, 2, "pred", "logprob")
    assert abs(diff["input_magnitude"] - 1.0) < 1e-9
    assert diff["input_gradient"] < 0.999 and diff["attention"] < 0.999
    print(f"[1] identical networks give rho = 1.000 for every method; two different networks differ: "
          + ", ".join(f"{m} {diff[m]:+.2f}" for m in ("attention", "input_gradient", "integrated_gradients")) + "; the weight-free control stays at 1.000")
    print("SELFTEST PASSED")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("Design.")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--true-dir"); p.add_argument("--shuffled-dir"); p.add_argument("--data-root")
    p.add_argument("--subjects", default="1-3"); p.add_argument("--out-dir", default=None)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--max-batches", type=int, default=3)
    p.add_argument("--ig-steps", type=int, default=10)
    p.add_argument("--target", choices=["true", "pred"], default="pred"); p.add_argument("--objective", choices=["prob", "logprob"], default="logprob")
    p.add_argument("--pass-rho", type=float, default=0.3)
    p.add_argument("--deeplift", action="store_true")
    p.add_argument("--adabn", action="store_true", default=True); p.add_argument("--no-adabn", dest="adabn", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not (a.true_dir and a.shuffled_dir and a.data_root):
        sys.exit("need --true-dir, --shuffled-dir and --data-root")
    os.environ.pop("TSFNET_PLANT", None)
    from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
    from xai_trunk import trunk_state
    ml = method_list(a.deeplift)
    out_dir = a.out_dir or os.path.join(a.shuffled_dir, "xai_label_randomisation"); os.makedirs(out_dir, exist_ok=True)
    per, accs = {}, {}

    def load(d, sid, ts, nseg, loader):
        ck = os.path.join(d, f"{sid}_session{ts}_best.pt")
        if not os.path.isfile(ck):
            return None
        m = TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=nseg, fnirs_temporal_align="egta",
                   efgf_gate_mode="sigmoid", efgf_gate_init=-2.0).to(a.device)
        m.load_state_dict(trunk_state(torch.load(ck, map_location=a.device)), strict=True); m.eval()
        if a.adabn:
            adapt_batchnorm_to_target(m, loader, a.device, log_fn=lambda *x, **k: None)
        return m
    for sid in parse_subjects(a.subjects):
        full = NBackDataset(os.path.join(a.data_root, sid), sid)
        nseg = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
        rows, accA, accB = [], [], []
        for ts in (1, 2, 3):
            idx = np.flatnonzero(full.session_ids == ts)
            loader = DataLoader(Subset(full, idx), batch_size=a.batch_size, shuffle=False)
            ma, mb = load(a.true_dir, sid, ts, nseg, loader), load(a.shuffled_dir, sid, ts, nseg, loader)
            if ma is None or mb is None:
                print(f"[{sid} s{ts}] SKIP -- a checkpoint is missing"); continue
            batches = []
            for k, b in enumerate(loader):
                if k >= a.max_batches:
                    break
                xe, xf, y, xc, xch = _unpack_batch(b)
                batches.append((xe.to(a.device), xf.to(a.device), y.to(a.device), xc.to(a.device) if xc is not None else None,
                                xch.to(a.device) if xch is not None else None))
            r, aa, ab = compare_models(ma, mb, batches, a.ig_steps, a.target, a.objective, a.deeplift)
            rows.append(r); accA.append(aa); accB.append(ab)
            del ma, mb
        if rows:
            per[sid] = {m: float(np.nanmean([r[m] for r in rows])) for m in ml}
            accs[sid] = (float(np.mean(accA)), float(np.mean(accB)))
            print(f"[{sid}] true-label net acc {accs[sid][0]:.3f} | shuffled-label net acc {accs[sid][1]:.3f} | " +
                  ", ".join(f"{m} {per[sid][m]:+.2f}" for m in ml), flush=True)
    subs = sorted(per)
    ta, sa = np.mean([accs[s][0] for s in subs]), np.mean([accs[s][1] for s in subs])
    print(f"\n{len(subs)} subjects. Accuracy on the TRUE-labelled held-out windows: true-label network {ta:.3f}, shuffled-label network {sa:.3f}")
    if sa > 0.45:
        print("  !! the shuffled-label network is clearly above chance: the permutation did not remove the signal. Nothing below is interpretable.")
    print(f"\n  mean Spearman rho between the two networks' explanations of the same window (target={a.target}, objective={a.objective})")
    for m in ml:
        v = np.array([per[s][m] for s in subs])
        if m == "input_magnitude":
            print(f"  {m:22} {v.mean():+.3f}   control: must be 1.000 -> {'OK' if abs(v.mean() - 1) < 1e-6 else 'PIPELINE BROKEN'}")
        else:
            print(f"  {m:22} {v.mean():+.3f}   below {a.pass_rho} in {int(np.sum(v < a.pass_rho))}/{len(subs)} subjects -> "
                  f"{'PASS (differs from a network trained on noise)' if v.mean() < a.pass_rho else 'FAIL (looks the same as a network trained on noise)'}")
    np.savez(os.path.join(out_dir, "label_randomisation.npz"), subjects=np.array(subs), methods=np.array(ml),
             rho=np.array([[per[s][m] for m in ml] for s in subs]), acc=np.array([accs[s] for s in subs]))


if __name__ == "__main__":
    main()
