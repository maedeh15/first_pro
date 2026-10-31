"""
xai_plant_recovery.py -- GROUND-TRUTH benchmark for the trunk's explanations: does each method point at a signal we know the network can use?

Design. A known load-dependent hemodynamic response is planted in ONE fNIRS segment of EVERY window (training and test), with
run_all_subjects.py --plant-fnirs "seg=K,strength=S" (see plant_fnirs in tsfnet_model_notebook.py: a 4x4 spatial patch, HRF time course, HbO +1 /
HbR -0.5, scaled -1 / 0 / +1 for 0- / 2- / 3-back). A baseline network is trained on that data. The truth is then known by construction:
the explanation of a 0-back or 3-back window should concentrate on segment K. For each method (attention, input gradient, integrated gradients,
optionally DeepLift) and a weight-independent control (input magnitude) the script reports, over the planted windows (0- and 3-back; 2-back has
no planted response):
  * rank percentile of segment K among the S segments (1.0 = the method's top segment, 0.5 = chance)
  * hit@1 and hit@3 (chance 1/S and 3/S)
and, because "the network could use the signal" is itself a claim, the CAUSAL check: accuracy lost when segment K is zeroed, against the mean loss
when any other single segment is zeroed. If zeroing K does not hurt more than zeroing the others, the network did not rely on the planted signal
and the benchmark is INCONCLUSIVE for that subject-fold set (reported, not hidden).

Pre-registered reading (change with --recover-percentile before running): a method RECOVERS the plant when its mean rank percentile >= 0.90
and hit@3 is above chance with Wilcoxon p < 0.05 across subjects. A method that does not recover a planted signal the network demonstrably
uses cannot be trusted to find a real one.

USAGE
  1. train (GPU, ~6 min per subject), e.g. three subjects:
     %run run_all_subjects.py --subjects 1-3 --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --plant-fnirs "seg=3,strength=1.0"
  2. explain: %run xai_plant_recovery.py --results-dir "<results_<hash>_seed42>" --data-root "<DATA_ROOT>" --subjects 1-3 --plant "seg=3,strength=1.0"
Strength 1.0 is a strong cue by design; run a weaker one (0.3) as well to see where each method stops finding it.
"""
import argparse, os, sys
import numpy as np
from scipy.stats import wilcoxon

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai_sanity_check import attributions, method_list, parse_subjects        # noqa: E402


def rank_percentile(v, k):
    """share of the OTHER segments that segment k beats: 1.0 = k is the top segment, 0.0 = the bottom, 0.5 = chance"""
    v = np.asarray(v, dtype=np.float64)
    others = np.delete(v, k)
    return float((v[k] > others).mean() + 0.5 * (v[k] == others).mean())


def recovery_stats(vecs, k):
    """vecs: (n_windows, S) importance per segment -> (mean rank percentile, hit@1, hit@3)"""
    pct = [rank_percentile(v, k) for v in vecs]
    top = np.argsort(-vecs, axis=1)
    return float(np.mean(pct)), float(np.mean(top[:, 0] == k)), float(np.mean((top[:, :3] == k).any(1)))


def causal_loss(model, bt, k, device):
    """accuracy lost when segment k is zeroed, and the mean loss when each other single segment is zeroed (same windows)"""
    x_eeg, x_fnirs, y, x_conn, x_chan = bt
    with torch.no_grad():
        acc0 = (model(x_eeg, x_fnirs, x_conn, x_chan)[3].argmax(1) == y).float().mean().item()
        S = x_fnirs.shape[1]
        losses = []
        for s in range(S):
            xa = x_fnirs.clone(); xa[:, s] = 0
            losses.append(acc0 - (model(x_eeg, xa, x_conn, x_chan)[3].argmax(1) == y).float().mean().item())
    return losses[k], float(np.mean([l for s, l in enumerate(losses) if s != k]))


def self_test():
    import tsfnet_model_notebook as M
    # 1) metric definitions
    v = np.zeros(21); v[3] = 5.0
    assert rank_percentile(v, 3) == 1.0 and rank_percentile(-v, 3) == 0.0
    assert abs(rank_percentile(np.ones(21), 3) - 0.5) < 1e-12
    vecs = np.tile(v, (10, 1)); assert recovery_stats(vecs, 3) == (1.0, 1.0, 1.0)
    rng = np.random.RandomState(0)
    r = recovery_stats(rng.rand(4000, 21), 3)
    assert abs(r[0] - 0.5) < 0.02 and abs(r[1] - 1 / 21) < 0.02 and abs(r[2] - 3 / 21) < 0.03, r
    print(f"[1] rank percentile / hit@k: perfect = 1/1/1, random vectors = {r[0]:.2f} / {r[1]:.3f} / {r[2]:.3f} (chance 0.5 / {1 / 21:.3f} / {3 / 21:.3f})")
    # 2) a model that is NOT trained cannot be said to recover the plant; the pipeline runs end to end and the control behaves
    torch.manual_seed(0)
    model = M.TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=21, fnirs_temporal_align="egta",
                     efgf_gate_mode="sigmoid", efgf_gate_init=-2.0).eval()
    xe, xf = torch.randn(6, 16, 16, 600), torch.randn(6, 21, 16, 16, 30, 2)
    y = torch.tensor([0, 2, 0, 2, 0, 2])
    xf = M.plant_fnirs(xf.numpy().copy(), y.numpy(), seg=3, strength=3.0); xf = torch.from_numpy(xf)
    v = attributions(model, xe, xf, y, None, None, 2, "pred", "logprob")
    mag = recovery_stats(v["input_magnitude"], 3)
    assert mag[0] > 0.8, f"a strongly planted segment must stand out in raw input magnitude, got {mag}"
    for m in ("attention", "input_gradient", "integrated_gradients"):
        assert np.isfinite(v[m]).all()
    cl = causal_loss(model, (xe, xf, y, None, None), 3, "cpu")
    assert all(np.isfinite(cl))
    print(f"[2] pipeline runs end to end; the weight-free control finds a strong plant (percentile {mag[0]:.2f}); "
          f"an untrained model has no causal reliance on it (loss {cl[0]:+.2f} vs {cl[1]:+.2f})")
    print("NOTE: this self-test does NOT show that a TRAINED network's methods recover the plant -- that is what the benchmark is for.")
    print("SELFTEST PASSED")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("USAGE")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir"); p.add_argument("--data-root")
    p.add_argument("--plant", default="seg=3,strength=1.0", help="MUST equal the spec the network was trained with")
    p.add_argument("--subjects", default="1-3"); p.add_argument("--out-dir", default=None)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--ig-steps", type=int, default=10)
    p.add_argument("--target", choices=["true", "pred"], default="pred"); p.add_argument("--objective", choices=["prob", "logprob"], default="logprob")
    p.add_argument("--recover-percentile", type=float, default=0.90)
    p.add_argument("--deeplift", action="store_true")
    p.add_argument("--adabn", action="store_true", default=True); p.add_argument("--no-adabn", dest="adabn", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not (a.results_dir and a.data_root):
        sys.exit("need --results-dir and --data-root")
    os.environ["TSFNET_PLANT"] = a.plant                 # the dataset must contain exactly the planted data the network saw
    from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target, parse_plant_spec
    from xai_trunk import trunk_state
    K = parse_plant_spec(a.plant)["seg"]
    ml = method_list(a.deeplift)
    out_dir = a.out_dir or os.path.join(a.results_dir, "xai_plant_recovery"); os.makedirs(out_dir, exist_ok=True)
    res, caus = {}, {}
    for sid in parse_subjects(a.subjects):
        full = NBackDataset(os.path.join(a.data_root, sid), sid)
        nseg = full.fnirs.shape[1]
        vecs = {m: [] for m in ml}; cK, cO = [], []
        for ts in (1, 2, 3):
            ck = os.path.join(a.results_dir, f"{sid}_session{ts}_best.pt")
            if not os.path.isfile(ck):
                print(f"[{sid} s{ts}] SKIP -- no checkpoint"); continue
            model = TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=nseg, fnirs_temporal_align="egta",
                           efgf_gate_mode="sigmoid", efgf_gate_init=-2.0).to(a.device)
            model.load_state_dict(trunk_state(torch.load(ck, map_location=a.device)), strict=True); model.eval()
            idx = np.flatnonzero(full.session_ids == ts)
            loader = DataLoader(Subset(full, idx), batch_size=a.batch_size, shuffle=False)
            if a.adabn:
                adapt_batchnorm_to_target(model, loader, a.device, log_fn=lambda *x, **k: None)
            for n_b, b in enumerate(loader):
                if n_b >= a.max_batches:
                    break
                xe, xf, y, xc, xch = _unpack_batch(b)
                bt = (xe.to(a.device), xf.to(a.device), y.to(a.device), xc.to(a.device) if xc is not None else None,
                      xch.to(a.device) if xch is not None else None)
                keep = (bt[2] != 1).cpu().numpy()                    # 0- and 3-back carry the planted response
                if not keep.any():
                    continue
                v = attributions(model, *bt, a.ig_steps, a.target, a.objective, a.deeplift)
                for m in ml:
                    vecs[m].append(v[m][keep])
                kk, oo = causal_loss(model, bt, K, a.device); cK.append(kk); cO.append(oo)
            del model
        if cK:
            res[sid] = {m: recovery_stats(np.concatenate(vecs[m]), K) for m in ml}
            caus[sid] = (float(np.mean(cK)), float(np.mean(cO)))
            print(f"[{sid}] " + ", ".join(f"{m} pct {res[sid][m][0]:.2f}" for m in ml) + f" | zeroing K costs {caus[sid][0]:+.3f} vs others {caus[sid][1]:+.3f}", flush=True)
    subs = sorted(res)
    print(f"\n{len(subs)} subjects; planted segment K={K}, spec '{a.plant}'; windows: 0- and 3-back\n")
    print(f"  {'method':22} {'rank percentile':>16} {'hit@1':>8} {'hit@3':>8}   {'p (pct>0.5)':>12}  verdict")
    S = full.fnirs.shape[1]
    for m in ml:
        pct = np.array([res[s][m][0] for s in subs]); h1 = np.array([res[s][m][1] for s in subs]); h3 = np.array([res[s][m][2] for s in subs])
        pv = wilcoxon(pct - 0.5, alternative="greater").pvalue if len(subs) >= 6 and np.any(pct != 0.5) else float("nan")
        ph3 = wilcoxon(h3 - 3 / S, alternative="greater").pvalue if len(subs) >= 6 and np.any(h3 != 3 / S) else float("nan")
        ok = pct.mean() >= a.recover_percentile and (np.isnan(ph3) or ph3 < 0.05)
        tag = "CONTROL (weight-free)" if m == "input_magnitude" else ("RECOVERS" if ok else "does not recover")
        print(f"  {m:22} {pct.mean():16.3f} {h1.mean():8.3f} {h3.mean():8.3f}   {pv:12.4g}  {tag}")
    ck_, co_ = np.array([caus[s][0] for s in subs]), np.array([caus[s][1] for s in subs])
    d = ck_ - co_
    pc = wilcoxon(ck_, co_, alternative="greater").pvalue if len(subs) >= 6 and np.any(d != 0) else float("nan")
    print(f"\n  CAUSAL: accuracy lost by zeroing segment K {ck_.mean():+.3f} vs any other segment {co_.mean():+.3f}  (K larger in {int((d > 0).sum())}/{len(d)} subjects, p={pc:.4g})")
    print("  -> " + ("the network relies on the planted signal: the benchmark is informative." if ck_.mean() > co_.mean() + 0.02 else
                     "the network does NOT clearly rely on the planted signal: the benchmark is INCONCLUSIVE, raise --strength / retrain."))
    print("  Chance: percentile 0.5, hit@1 %.3f, hit@3 %.3f." % (1 / S, 3 / S))
    np.savez(os.path.join(out_dir, "plant_recovery.npz"), subjects=np.array(subs), methods=np.array(ml),
             stats=np.array([[res[s][m] for m in ml] for s in subs]), causal=np.array([caus[s] for s in subs]))


if __name__ == "__main__":
    main()
