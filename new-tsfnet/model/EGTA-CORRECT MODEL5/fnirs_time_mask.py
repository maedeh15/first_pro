"""
fnirs_time_mask.py -- LAYER A: a faithful, sparse, instance-wise TIME explanation of the FINAL model's fNIRS use (no retraining).
Dynamask-style (Crabbe & van der Schaar, ICML 2021; mask idea: Fong & Vedaldi 2017), adapted to this model:
  * The 21 overlapping 3 s segments are rebuilt into ONE continuous 23 s trace (same reconstruction as fnirs_band_deletion.py), because
    overlapping segments share raw samples and EFGF also reads the early-stage fNIRS maps -- masking per segment would be inconsistent.
  * A mask m(t) in [0,1] with 1 value per second (23 values, linearly upsampled) multiplies the trace; reference = 0 (fNIRS is baseline-
    corrected + z-scored, so 0 is the resting level), then the trace is cut back into the same 21 segments.
  * For each window the mask is OPTIMISED to preserve the model's own prediction (KL to the intact posterior) under a fixed AREA budget.
    Faithfulness is judged against baselines, never by the mask itself:
        KL of the optimised mask  vs  random contiguous masks of the same area  vs  uniform scaling m=area.
  * Output: per-class mean time profile of the optimised mask (which seconds the model keeps) + group tests (paired Wilcoxon, uncorrected, exploratory).
LIMITS (say them): window-relative seconds, not seconds from block start; zero-reference is out-of-distribution for the model (controlled by the random-mask
baseline); the explanation concerns fNIRS only; EEG is untouched.

  python fnirs_time_mask.py --self-test
  python fnirs_time_mask.py --results-dir "...\\results_cf64a71127_seed999" --data-root "..." --subjects 1-26 --stride 3 [--fnirs-head linear]
"""
import argparse, csv, os, sys
import numpy as np
import torch
import torch.nn.functional as F

STEP, SEG = 10, 30
AREAS = (0.3, 0.5)


def to_trace(x):
    """(B,S,H,W,T,C) -> (B,H,W,L,C); overlapping samples are identical, so overwriting reconstructs the continuous trace"""
    B, S, H, W, T, C = x.shape
    tr = x.new_zeros(B, H, W, (S - 1) * STEP + T, C)
    for k in range(S):
        tr[:, :, :, k * STEP:k * STEP + T, :] = x[:, k]
    return tr


def to_segments(tr, S):
    """(B,H,W,L,C) -> (B,S,H,W,SEG,C), differentiable"""
    return tr.unfold(3, SEG, STEP).permute(0, 3, 1, 2, 5, 4)[:, :S]


def moving_average(tr, sec):
    """centred moving average along time (reflect padding), window 2W+1 samples, W = round(sec * 10 Hz): the 'fade to moving average' reference of Dynamask (pi^m)"""
    B, H, W_, L, C = tr.shape; W = int(round(sec * 10)); x = tr.permute(0, 1, 2, 4, 3).reshape(-1, 1, L)
    x = F.avg_pool1d(F.pad(x, (W, W), mode="reflect"), kernel_size=2 * W + 1, stride=1)
    return x.reshape(B, H, W_, C, L).permute(0, 1, 2, 4, 3)


def apply_mask(tr, m, ref=None):
    """m (B,Tm) in [0,1], one value per second. Perturbed trace = ref + m * (trace - ref). ref = None -> 0 (the STATIC Hadamard case, out of distribution);
    ref = moving average -> Dynamask's dynamic fade-out (Crabbe & van der Schaar 2021, Sec. 2.2, pi^m): m = 1 keeps the value, m = 0 replaces it by its local mean."""
    up = F.interpolate(m[:, None, :], size=tr.shape[3], mode="linear", align_corners=False).view(m.shape[0], 1, 1, tr.shape[3], 1)
    return tr * up if ref is None else ref + up * (tr - ref)


def posterior(model, xe, xf, xc, xh):
    yp = model(xe, xf, xc, xh)[3]
    return yp / (yp.sum(-1, keepdim=True) + 1e-8)


def kl(p0, p):
    return (p0 * (torch.log(p0 + 1e-8) - torch.log(p + 1e-8))).sum(-1)


def area_loss(m, area):
    """Dynamask L_a = || vecsort(M) - r_a ||^2 with r_a = ((1-a)T zeros, then aT ones): pushes the mask to be BINARY with a fraction `area` of ones.
    (A penalty on the mean alone, |mean(m) - a|, is satisfied by the flat mask m = a, i.e. by uniform scaling, and explains nothing.)"""
    Tm = m.shape[1]; k = int(round(area * Tm)); r = torch.cat([torch.zeros(Tm - k), torch.ones(k)]).to(m.device)
    return ((torch.sort(m, dim=1).values - r) ** 2).mean(1)


def fit_mask(model, xe, tr, p0, S, area, steps, lr, lam, xc=None, xh=None, ref=None, lam_c=0.0):
    Tm = tr.shape[3] // STEP
    a = torch.full((tr.shape[0], Tm), float(np.log(area / (1 - area))), device=tr.device, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    for i in range(steps):
        m = torch.sigmoid(a); r = i / max(steps - 1, 1)                          # continuation: start with the mean-area penalty, end with the binarising vecsort penalty
        reg = (1 - r) * (m.mean(1) - area).abs() + r * area_loss(m, area)        # (vecsort from step 0 locked arbitrary seconds in the stub test: share inside the informative window 0.37, chance 0.30)
        loss = (kl(p0, posterior(model, xe, to_segments(apply_mask(tr, m, ref), S), xc, xh)) + lam * reg + lam_c * (m[:, 1:] - m[:, :-1]).abs().sum(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return torch.sigmoid(a).detach()


def hard_topk(m, area):
    k = max(1, int(round(area * m.shape[1])))
    out = torch.zeros_like(m)
    out.scatter_(1, m.topk(k, dim=1).indices, 1.0)
    return out


def random_subset(B, Tm, area, gen, device):
    """random NON-contiguous seconds of the same area: the fair comparison for the optimised (non-contiguous) hard mask"""
    k = max(1, int(round(area * Tm))); out = torch.zeros(B, Tm, device=device)
    for b in range(B): out[b, torch.randperm(Tm, generator=gen)[:k].to(device)] = 1.0
    return out


def random_block(B, Tm, area, gen, device):
    k = max(1, int(round(area * Tm))); out = torch.zeros(B, Tm, device=device)
    for b in range(B):
        s = int(torch.randint(0, Tm - k + 1, (1,), generator=gen)); out[b, s:s + k] = 1.0
    return out


@torch.no_grad()
def _kl_agree(model, xe, tr, p0, S, m, xc, xh, ref=None):
    p = posterior(model, xe, to_segments(apply_mask(tr, m, ref), S), xc, xh)
    return kl(p0, p), (p.argmax(1) == p0.argmax(1)).float()


def explain_batch(model, xe, xf, y, xc, xh, a, gen):
    S = xf.shape[1]; tr = to_trace(xf)
    mode = getattr(a, "perturb", "batchmean")
    ref = (moving_average(tr, getattr(a, "ma_sec", 2.0)) if mode == "ma" else tr.mean(dim=0, keepdim=True).expand_as(tr) if mode == "batchmean" else None)
    with torch.no_grad():
        p0 = posterior(model, xe, xf, xc, xh)
    res = {}; prof = None
    for area in AREAS:
        with torch.enable_grad():
            m = fit_mask(model, xe, tr, p0, S, area, a.steps, a.lr, a.lam, xc, xh, ref, getattr(a, "lam_c", 0.0))
        hard = hard_topk(m, area); Tm = m.shape[1]
        blk = [random_block(tr.shape[0], Tm, area, gen, tr.device) for _ in range(5)]; sub = [random_subset(tr.shape[0], Tm, area, gen, tr.device) for _ in range(5)]
        k_soft, g_soft = _kl_agree(model, xe, tr, p0, S, m, xc, xh, ref)
        k_hard, g_hard = _kl_agree(model, xe, tr, p0, S, hard, xc, xh, ref)
        k_rnd = torch.stack([_kl_agree(model, xe, tr, p0, S, r, xc, xh, ref)[0] for r in blk]).mean(0)
        k_sub = torch.stack([_kl_agree(model, xe, tr, p0, S, r, xc, xh, ref)[0] for r in sub]).mean(0)
        k_uni, _ = _kl_agree(model, xe, tr, p0, S, torch.full_like(m, area), xc, xh, ref)
        res[area] = dict(soft=k_soft, hard=k_hard, rand=k_rnd, subset=k_sub, uniform=k_uni, agree_hard=g_hard)
        if area == AREAS[-1]:
            prof = m
    return res, prof


def build_model(S, a):
    from tsfnet_model_notebook import TSFNet
    kw = dict(fnirs_head=a.fnirs_head) if a.fnirs_head != "mlp" else {}
    return TSFNet(num_classes=3, fusion_combination="paper_mean", fnirs_num_segments=S, fnirs_temporal_align="egta",
                  efgf_gate_mode="sigmoid", efgf_gate_init=-2.0, **kw).to(a.device)


def run_subject(sid, a, gen):
    from torch.utils.data import DataLoader, Subset
    from tsfnet_model_notebook import NBackDataset, _unpack_batch, adapt_batchnorm_to_target
    from xai_trunk import trunk_state, load_checkpoint
    full = NBackDataset(os.path.join(a.data_root, sid), sid); S = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    acc = {ar: {k: [] for k in ("soft", "hard", "rand", "subset", "uniform", "agree_hard")} for ar in AREAS}
    prof = {c: [] for c in range(3)}
    for ts in (1, 2, 3):
        ck = os.path.join(a.results_dir, f"{sid}_session{ts}_best.pt")
        if not os.path.isfile(ck): continue
        m = build_model(S, a); m.load_state_dict(trunk_state(load_checkpoint(ck, map_location=a.device)), strict=True); m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        idx = np.flatnonzero(full.session_ids == ts)
        if a.adabn:
            adapt_batchnorm_to_target(m, DataLoader(Subset(full, idx), batch_size=a.batch_size, shuffle=False), a.device, log_fn=lambda *x, **k: None)
        for batch in DataLoader(Subset(full, idx[::a.stride]), batch_size=a.batch_size, shuffle=False):
            xe, xf, y, xc, xh = _unpack_batch(batch)
            xe, xf = xe.to(a.device), xf.to(a.device)
            xc = xc.to(a.device) if xc is not None else None; xh = xh.to(a.device) if xh is not None else None
            res, pm = explain_batch(m, xe, xf, y, xc, xh, a, gen)
            for ar in AREAS:
                for k in acc[ar]: acc[ar][k] += res[ar][k].cpu().tolist()
            for c in range(3):
                sel = (y == c)
                if sel.any(): prof[c].append(pm[sel.to(pm.device)].cpu().numpy())
        del m
    summ = {ar: {k: float(np.mean(v)) for k, v in acc[ar].items() if v} for ar in AREAS}
    profs = {c: np.concatenate(v).mean(0) for c, v in prof.items() if v}
    return summ, profs


def self_test():
    """stub head whose prediction depends ONLY on seconds 8-15 of the trace (strong gain): the optimised mask must be near-binary, concentrate there,
    beat random SUBSETS of the same area, and the static-zero mode must also run"""
    torch.manual_seed(0); S = 21; L = (S - 1) * STEP + SEG
    tr0 = torch.randn(8, 4, 4, L, 2); x = to_segments(tr0, S)
    assert torch.allclose(to_trace(x), tr0, atol=1e-6), "trace <-> segments round trip"
    ref = moving_average(tr0, 2.0); assert ref.shape == tr0.shape and ref.std() < tr0.std(), "moving average must smooth"
    m1 = torch.ones(8, L // STEP); assert torch.allclose(apply_mask(tr0, m1, ref), tr0, atol=1e-6), "mask 1 must leave the trace unchanged"
    assert torch.allclose(apply_mask(tr0, torch.zeros_like(m1), ref), ref, atol=1e-5), "mask 0 must give the reference"
    print("[1] round trip exact; mask 1 keeps the trace, mask 0 gives the moving average")

    class Stub(torch.nn.Module):
        def forward(self, xe, xf, xc, xh):
            t = to_trace(xf)[:, :, :, 80:150, :].mean((1, 2, 3))
            return None, None, None, torch.softmax(torch.stack([t[:, 0] * 30, t[:, 1] * 30, 0 * t[:, 0]], -1), -1), None
    mdl = Stub(); ar = argparse.Namespace(steps=200, lr=0.3, lam=1.0, lam_c=0.0, perturb="batchmean", ma_sec=2.0)
    x = to_segments(torch.randn(16, 4, 4, L, 2), S)
    res, prof = explain_batch(mdl, None, x, torch.zeros(16, dtype=torch.long), None, None, ar, torch.Generator().manual_seed(0))
    hard = hard_topk(prof, AREAS[-1]); inf = set(range(8, 15)); k = int(hard[0].sum().item())
    share = float(np.mean([len(set(np.flatnonzero(hard[b].numpy())) & inf) / len(inf) for b in range(16)])); chance = AREAS[-1]   # recall of the informative seconds by the hard mask (the returned profile is the LAST area, 0.5)
    inside = float(prof[:, 8:15].mean()); outside = float(torch.cat([prof[:, :8], prof[:, 15:]], 1).mean()); r = res[0.3]
    print(f"[2] KL(area 0.3): hard {r['hard'].mean():.4f} | random subset {r['subset'].mean():.4f} | random block {r['rand'].mean():.4f} | uniform {r['uniform'].mean():.4f}; "
          f"soft mask inside the informative 7 s {inside:.2f} vs outside {outside:.2f}; recall of the informative seconds by the hard mask (area {AREAS[-1]}) {share:.2f} (chance {chance:.2f})")
    assert r["hard"].mean() < r["subset"].mean() * 0.7, "optimised mask must beat random subsets of the same area"
    assert share > 0.65 and inside > outside + 0.15, (share, inside, outside)
    for md in ("zero", "ma"):
        ar0 = argparse.Namespace(steps=60, lr=0.3, lam=1.0, lam_c=0.0, perturb=md, ma_sec=2.0); explain_batch(mdl, None, x, torch.zeros(16, dtype=torch.long), None, None, ar0, torch.Generator().manual_seed(0))
    # a moving-average reference keeps slow content: a stub that reads the 7 s MEAN level is nearly blind to a mask under 'ma'
    ar1 = argparse.Namespace(steps=120, lr=0.3, lam=1.0, lam_c=0.0, perturb="ma", ma_sec=2.0); r1, _ = explain_batch(mdl, None, x, torch.zeros(16, dtype=torch.long), None, None, ar1, torch.Generator().manual_seed(0))
    print(f"[3] zero and ma modes run; with 'ma' the level-reading stub is nearly insensitive to masking (uniform-mask KL {r1[0.3]['uniform'].mean():.4f} vs {r['uniform'].mean():.4f} with batchmean) -- the reason batchmean is the default\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir"); ap.add_argument("--data-root"); ap.add_argument("--subjects", default="1-26"); ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--steps", type=int, default=150); ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--lam-c", type=float, default=0.0); ap.add_argument("--perturb", choices=("batchmean", "ma", "zero"), default="batchmean", help="reference the masked seconds fade to: batchmean = mean trace of the window batch (default; removes the slow content the model uses, in distribution, same as the V2 deletion); ma = short moving average (Dynamask pi^m: KEEPS slow content, so it cannot detect reliance on it); zero = static zero (out of distribution)"); ap.add_argument("--ma-sec", type=float, default=2.0); ap.add_argument("--fnirs-head", default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--adabn", action="store_true", default=True); ap.add_argument("--no-adabn", dest="adabn", action="store_false")
    ap.add_argument("--device", default=None); ap.add_argument("--out", default="fnirs_time_mask"); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    if not (a.results_dir and a.data_root): sys.exit("need --results-dir and --data-root")
    a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu"); gen = torch.Generator().manual_seed(0)
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2] if "-" in a.subjects else (a.subjects, a.subjects); per, profs = {}, {}
    for i in range(int(lo), int(hi) + 1):
        sid = f"VP{i:03d}"
        try:
            per[sid], profs[sid] = run_subject(sid, a, gen)
            print(f"[{sid}] " + " | ".join(f"a={ar}: opt-hard {per[sid][ar]['hard']:.3f} rand {per[sid][ar]['rand']:.3f} unif {per[sid][ar]['uniform']:.3f}" for ar in AREAS), flush=True)
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
    if len(per) < 6: sys.exit(f"{len(per)} subjects; group tests need >= 6")
    from scipy.stats import wilcoxon
    ids = list(per)
    print(f"\nTIME-MASK FIDELITY, {len(ids)} subjects (KL to the intact posterior, lower = prediction better preserved; paired Wilcoxon, exploratory, uncorrected)")
    for ar in AREAS:
        opt = np.array([per[s][ar]["hard"] for s in ids]); rnd = np.array([per[s][ar]["subset"] for s in ids]); blk = np.array([per[s][ar]["rand"] for s in ids]); uni = np.array([per[s][ar]["uniform"] for s in ids])
        ag = np.mean([per[s][ar]["agree_hard"] for s in ids])
        print(f"  area {ar}: optimised(hard) {opt.mean():.3f} | random SUBSET (primary) {rnd.mean():.3f} | random contiguous block {blk.mean():.3f} | uniform {uni.mean():.3f} | argmax kept {ag:.3f} | "
              f"better than random subset in {(opt < rnd).sum()}/{len(ids)} subjects, p={wilcoxon(opt - rnd).pvalue:.3g}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "time_mask_profiles.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subject", "class(0/2/3-back)"] + [f"sec{t}" for t in range(len(next(iter(next(iter(profs.values())).values()))))])
        for s in ids:
            for c, v in profs[s].items(): w.writerow([s, c] + [f"{x:.4f}" for x in v])
    g = {c: np.mean([profs[s][c] for s in ids if c in profs[s]], 0) for c in range(3)}
    print("  group mean kept-time profile (seconds 0..22 of the window), per class (0/2/3-back):")
    for c in range(3): print(f"   class {c}: " + " ".join(f"{x:.2f}" for x in g[c]))
    print("figure data ->", os.path.join(a.out, "time_mask_profiles.csv"))


if __name__ == "__main__":
    main()
