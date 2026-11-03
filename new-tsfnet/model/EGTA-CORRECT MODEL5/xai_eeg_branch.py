"""
xai_eeg_branch.py -- validated explanation of the DEEP EEG branch in TIME (10 s window = 10 blocks of 1 s). No channel/region claim (channel order is checked separately
by check_channel_order.py; this script does not use it). Same validation chain as xai_validated_maps.py, with EEG time blocks instead of fNIRS segments:
  explanations per window (target = predicted class of the chosen output):
     occlusion      replace one 1 s block (all 16x16 cells) by the window-batch mean block; impact = drop in P(class)          (causal, the ceiling)
     input_gradient mean |dlogP/dx| per block;   grad_x_input mean |x * dlogP/dx| per block
  V1 model randomisation (trained vs fully randomised model; rho < 0.3 in >= 2/3 of subjects), V2 deletion (top-k vs random-k vs bottom-k blocks replaced by the
  batch mean, k = 2 of 10; top > random > bottom, one-sided Wilcoxon p < 0.05), V3 rank agreement of the gradient methods with the occlusion impact.
  --output eeg (default) explains the EEG head (it does not see fNIRS, so this is a clean test of the EEG branch alone); --output pred explains the fused output.
Also saves the occlusion profile by predicted class (10 values per class and subject) and a figure (profile with 95% CI over subjects + validation dots).
The EEG branch is weak (about 0.49 head accuracy): the result describes how that head behaves, not a discovery about cognitive load. Windows are random (seeded), AdaBN as everywhere.
  python xai_eeg_branch.py --results-dir "...\\results_cf64a71127_seed42" --data-root "..." --subjects 1-26 --out-dir "...\\eeg_branch_xai"
"""
import argparse, json, os, sys
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai_sanity_check import randomise_upto, GROUPS
from xai_validated_maps import pc, const, rho_rows, summarise

METHODS_E = ("occlusion", "input_gradient", "grad_x_input")


def head(out, which):
    y = out[0] if which == "eeg" else out[3]
    return y / (y.sum(dim=-1, keepdim=True) + 1e-8)


def block_slices(T, nb):
    w = T // nb; return [slice(k * w, (k + 1) * w) for k in range(nb)]


def explain_eeg(model, xe, xf, xc, xh, which="eeg", nb=10, cls=None):
    """returns dict method -> (B, nb), p0 (B,), predicted class c (B,)"""
    sl = block_slices(xe.shape[-1], nb); B = xe.shape[0]
    with torch.no_grad():
        p_all = head(model(xe, xf, xc, xh), which); c = p_all.argmax(1) if cls is None else cls; p0 = pc(p_all, c)
    xg = xe.clone().requires_grad_(True); pg = head(model(xg, xf, xc, xh), which)
    g, = torch.autograd.grad(torch.log(pc(pg, c).clamp_min(1e-8)).sum(), [xg]); g = g.detach()
    blk = lambda t: torch.stack([t[..., s].abs().mean(dim=(1, 2, 3)) for s in sl], 1).cpu().numpy()
    out = {"input_gradient": blk(g), "grad_x_input": blk(g * xe)}
    imp = np.zeros((B, nb)); rep = xe.mean(dim=0, keepdim=True).expand_as(xe)
    with torch.no_grad():
        for k, s in enumerate(sl):
            xa = xe.clone(); xa[..., s] = rep[..., s]; imp[:, k] = (p0 - pc(head(model(xa, xf, xc, xh), which), c)).cpu().numpy()
    out["occlusion"] = imp
    return out, p0.detach(), c


def deletion_eeg(model, xe, xf, xc, xh, c, p0, scores, k, rng, which, nb=10):
    sl = block_slices(xe.shape[-1], nb); B = xe.shape[0]; rep = xe.mean(dim=0, keepdim=True).expand_as(xe); res = {}
    with torch.no_grad():
        for m, sc in scores.items():
            order = np.argsort(-sc, axis=1)
            sets = {"top": order[:, :k], "bottom": order[:, -k:], "random": np.stack([rng.choice(nb, k, replace=False) for _ in range(B)])}
            for nm, idx in sets.items():
                xa = xe.clone()
                for b in range(B):
                    for j in idx[b]: xa[b, ..., sl[int(j)]] = rep[b, ..., sl[int(j)]]
                res.setdefault(m, {})[nm] = (p0 - pc(head(model(xa, xf, xc, xh), which), c)).cpu().numpy()
    return res


def process_batch(model, rmodel, xe, xf, xc, xh, a, rng):
    sc, p0, c = explain_eeg(model, xe, xf, xc, xh, a.output, a.blocks); rs, _, _ = explain_eeg(rmodel, xe, xf, xc, xh, a.output, a.blocks, cls=c)   # randomised model explains the SAME class
    dele = deletion_eeg(model, xe, xf, xc, xh, c, p0, sc, a.k, rng, a.output, a.blocks)
    rec = {m: {"rand_rho": rho_rows(sc[m], rs[m])} for m in sc}
    for m in rec:
        if m != "occlusion": rec[m]["impact_rho"] = rho_rows(sc[m], sc["occlusion"])
        for nm in ("top", "random", "bottom"): rec[m][nm] = list(dele[m][nm])
    cn = c.cpu().numpy(); prof, cnt = {}, {}
    for cl in np.unique(cn):
        o = sc["occlusion"][cn == cl]; z = (o - o.mean(1, keepdims=True)) / (o.std(1, keepdims=True) + 1e-12); prof[int(cl)] = z.sum(0); cnt[int(cl)] = int((cn == cl).sum())
    return rec, prof, cnt


def run_subject(sid, a, rng):
    from torch.utils.data import DataLoader, Subset
    from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
    from xai_trunk import trunk_state, load_checkpoint, set_xai_output
    full = NBackDataset(os.path.join(a.data_root, sid), sid); S = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21; acc, prof, cnt = {}, {}, {}
    for ts in (1, 2, 3):
        ck = os.path.join(a.results_dir, f"{sid}_session{ts}_best.pt")
        if not os.path.isfile(ck): continue
        model = TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=S, fnirs_temporal_align="egta", efgf_gate_mode="sigmoid", efgf_gate_init=-2.0).to(a.device)
        model.load_state_dict(trunk_state(load_checkpoint(ck, map_location=a.device)), strict=True); set_xai_output(model); model.eval()
        idx = np.random.RandomState(1000 * int(sid[2:]) + ts).permutation(np.flatnonzero(full.session_ids == ts)); dl = DataLoader(Subset(full, idx), batch_size=a.batch_size, shuffle=False)
        adapt_batchnorm_to_target(model, dl, a.device, log_fn=lambda *x, **k: None)
        rmodel, _ = randomise_upto(model, len(GROUPS) - 1, a.seed); rmodel = rmodel.to(a.device).eval(); adapt_batchnorm_to_target(rmodel, dl, a.device, log_fn=lambda *x, **k: None)
        for bi, batch in enumerate(dl):
            if bi >= a.max_batches: break
            xe, xf, y, xc, xh = _unpack_batch(batch); xe, xf = xe.to(a.device), xf.to(a.device)
            xc = xc.to(a.device) if xc is not None else None; xh = xh.to(a.device) if xh is not None else None
            rec, pr, ct = process_batch(model, rmodel, xe, xf, xc, xh, a, rng)
            for m, d in rec.items():
                for k_, v in d.items(): acc.setdefault(m, {}).setdefault(k_, []).extend(v)
            for cl in pr: prof[cl] = prof.get(cl, 0) + pr[cl]; cnt[cl] = cnt.get(cl, 0) + ct[cl]
        del model, rmodel
    return {m: {k_: float(np.mean(v)) if len(v) else float("nan") for k_, v in d.items()} for m, d in acc.items()}, {str(cl): (prof[cl] / max(cnt[cl], 1)).tolist() for cl in prof}


def figure(per, prof, out_dir):
    import fig_style as fs; fs.apply(); import matplotlib.pyplot as plt
    ids = list(prof); cc = {"0": fs.C["blue"], "1": fs.C["orange"], "2": fs.C["red"]}; names = {"0": "0-back", "1": "2-back", "2": "3-back"}
    fig, ax = plt.subplots(1, 2, figsize=(fs.W2, 2.8), gridspec_kw={"width_ratios": [1.4, 1]})
    for cl in "012":
        A = np.array([prof[s][cl] for s in ids if cl in prof[s]]); x = np.arange(1, A.shape[1] + 1); lo, hi = np.array([fs.boot_ci(A[:, k], n=1000) for k in range(A.shape[1])]).T
        ax[0].fill_between(x, lo, hi, color=cc[cl], alpha=0.2, lw=0); ax[0].plot(x, A.mean(0), color=cc[cl], label=f"predicted {names[cl]}")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xlabel("Block of the 10 s window (1 s each)"); ax[0].set_ylabel("Occlusion impact (within-window z)"); ax[0].legend(frameon=False, fontsize=6); fs.panel(ax[0], "a")
    lab = {"occlusion": "Occlusion", "input_gradient": "Input gradient", "grad_x_input": "Gradient x input"}; col = {"occlusion": fs.C["green"], "input_gradient": fs.C["sky"], "grad_x_input": fs.C["purple"]}
    for j, m in enumerate(METHODS_E):
        v = np.array([per[s][m]["top"] - per[s][m]["random"] for s in per]); fs.strip(ax[1], j, v, color=col[m], width=0.15); fs.mean_ci(ax[1], j, v, color="k")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xticks(range(3)); ax[1].set_xticklabels([lab[m] for m in METHODS_E], rotation=20, fontsize=6.5); ax[1].set_ylabel("Deletion: top minus random drop in P"); fs.panel(ax[1], "b")
    fig.suptitle(f"Deep EEG branch, explanation in time (n = {len(ids)} subjects)", fontsize=8); fig.tight_layout(); return fs.save(fig, out_dir, "fig_eeg_branch")


def self_test():
    # needs torch + the model: a tiny stand-in checks the bookkeeping only
    class Fake(torch.nn.Module):
        def forward(self, xe, xf, xc, xh):
            s = xe[..., 600:800].mean(dim=(1, 2, 3)); y = torch.softmax(torch.stack([s, -s, 0 * s], 1), 1); return y, y, y, y, {}
    m = Fake().eval(); xe = torch.randn(8, 16, 16, 2000); xe[:, :, :, 600:800] += torch.randn(8, 1, 1, 1) * 2
    sc, p0, c = explain_eeg(m, xe, None, None, None, "eeg", 10)
    assert sc["occlusion"].shape == (8, 10) and int(np.argmax(sc["occlusion"].mean(0))) == 3, "the informative block (600-800 = block 4) must carry the largest occlusion impact"
    assert int(np.argmax(sc["input_gradient"].mean(0))) == 3
    d = deletion_eeg(m, xe, None, None, None, c, p0, sc, 2, np.random.RandomState(0), "eeg", 10); assert d["occlusion"]["top"].mean() > d["occlusion"]["random"].mean() > d["occlusion"]["bottom"].mean() - 1e-9
    print("[1] on a fake head with one informative block, occlusion and gradient find block 4 and top-k deletion hurts most\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir"); ap.add_argument("--data-root"); ap.add_argument("--subjects", default="1-26"); ap.add_argument("--out-dir", default=None)
    ap.add_argument("--output", choices=("eeg", "pred"), default="eeg"); ap.add_argument("--blocks", type=int, default=10); ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--max-batches", type=int, default=3); ap.add_argument("--pass-rho", type=float, default=0.3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adabn", action="store_true", default=True); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    if not (a.results_dir and a.data_root): sys.exit("need --results-dir and --data-root")
    out = a.out_dir or os.path.join(a.results_dir, "eeg_branch_xai"); os.makedirs(out, exist_ok=True); rng = np.random.RandomState(a.seed); per, prof = {}, {}
    lo, hi = a.subjects.split("-") if "-" in a.subjects else (a.subjects, a.subjects)
    for i in range(int(lo), int(hi) + 1):
        sid = f"VP{i:03d}"
        try:
            per[sid], prof[sid] = run_subject(sid, a, rng); print(f"[{sid}] " + " | ".join(f"{m} {per[sid][m]['top'] - per[sid][m]['random']:+.4f}" for m in per[sid]) + "   (top - random deletion drop)", flush=True)
        except Exception as e: print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
    if len(per) < 6: sys.exit(f"{len(per)} usable subjects; group tests need >= 6")
    print(f"\n(EEG TIME BLOCKS of the {a.output} output; 'fNIRS segment' wording in the header below is inherited from the shared table code)")
    res = summarise(per, a.pass_rho)
    cfg = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    json.dump({"per_subject": per, "group": res, "config": cfg}, open(os.path.join(out, "eeg_branch_summary.json"), "w"), indent=2, default=float); json.dump(prof, open(os.path.join(out, "eeg_block_profiles_by_class.json"), "w"))
    print("figure:", figure(per, prof, out)); print("[written]", out)


if __name__ == "__main__":
    main()
