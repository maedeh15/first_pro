"""
fusion_controls.py -- attribution controls + per-class confusion for the late fusion
"TSFNet + Riemannian EEG member" (see late_fusion_riemann.py). CPU only, no training of any network.

QUESTION 1 (attribution): the fused system gains ~+6 pp (window) / ~+8 pp (block) over the deep
ensemble. Is that because of the Riemannian multiband tangent-space member specifically, or would ANY
cheap classical EEG member added to the deep model give the same? Same fusion rule, same folds, same
weight (w = 0.5, pre-registered), only the EEG member changes:
    logvar          log-variance per channel (the classical feature, broadband)
    raw|broad       tangent vector of the broadband covariance, no domain handling
    raw|multi       4 bands (broadband, theta 4-8, alpha 8-13, beta 13-30), no domain handling
    EA|multi        + per-session arithmetic-mean whitening (Euclidean Alignment)
    riem|multi      + per-session Frechet-mean re-centring
    riem_disp|broad broadband only, Frechet re-centring + dispersion normalisation
    riem_disp|multi THE MAIN MEMBER of late_fusion_riemann.py
Reading rule (fixed before looking): the claim "the Riemannian/multiband member is what matters" is
supported only if fused(riem_disp|multi) beats fused(logvar) by >= 1.0 pp at window level with
Holm-adjusted p < 0.05 AND beats fused(riem_disp|broad) (so the multiband part matters). If fused(logvar)
is within 1 pp of the main member, say so: the gain would then come from adding any non-deep EEG member.

QUESTION 2 (where does the gain come from?): pooled 3x3 confusion matrices (rows = true class) and
recall per class, window level and block level, for: deep ensemble, EEG member alone, fused (main),
fused (logvar control); plus per-subject recall differences (Wilcoxon) and the share of errors that
are 2-back <-> 3-back.

Alignment guard as in late_fusion_riemann.py: saved labels / trial ids of a fold must equal the
recomputed ones, otherwise the fold is refused.

HONEST CAVEATS (printed again at the end): same 26 subjects and same 3 deep seeds for everything; the
member variants were all evaluated on those folds (a family of 7 -> Holm is reported); transductive
per-session centring; this is an offline fusion, not a retrained model.

usage
  python fusion_controls.py --self-test
  python fusion_controls.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback" ^
        --posteriors-dir "E:\\...\\ensemble_pure_baseline_40b7d8ea53\\posteriors" [--out fusion_controls_folds.csv]
"""
import os
import re
import csv
import glob
import argparse
import tempfile
import numpy as np
from scipy.stats import wilcoxon

import check_riemannian_eeg as R
import late_fusion_riemann as L

W = 0.5
VARIANTS = ("logvar", "raw|broad", "raw|multi", "EA|multi", "riem|multi", "riem_disp|broad", "riem_disp|multi")
MAIN = "riem_disp|multi"          # member whose confusion matrices / per-class recall are printed (--main)
PREREG = "riem_disp|multi"        # reference of the PRE-REGISTERED attribution reading -- never follows --main
CONTROL = "logvar"
CLASSES = (0, 1, 2)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def wil(d):
    d = np.asarray(d)
    return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0


def variant_features(x, ses):
    """all member variants for one subject (each session re-centred independently)"""
    xc = x - x.mean(-1, keepdims=True)
    feats = {"logvar": np.log(np.einsum("nct,nct->nc", xc, xc) / x.shape[-1] + 1e-8)}
    bands = R.build_covs(x, True)

    def multi(mode):
        return np.hstack([R.domain_features(bands[b], ses, mode) for b in bands])
    feats["raw|broad"] = R.domain_features(bands["broad"], ses, "raw")
    feats["raw|multi"] = multi("raw")
    feats["EA|multi"] = multi("EA")
    feats["riem|multi"] = multi("riem")
    feats["riem_disp|broad"] = R.domain_features(bands["broad"], ses, "riem_disp")
    feats["riem_disp|multi"] = multi("riem_disp")
    return feats


def conf(P, y):
    M = np.zeros((3, 3), int)
    for t, p in zip(y, P.argmax(-1)):
        M[int(t), int(p)] += 1
    return M


def conf_block(P, y, tid):
    M = np.zeros((3, 3), int)
    for t in np.unique(tid):
        m = tid == t
        M[int(y[m][0]), int(P[m].mean(0).argmax())] += 1
    return M


def run(data, post_dir, subjects=None, verbose=True):
    files = sorted(glob.glob(os.path.join(post_dir, "*_session*.npz")))
    if not files:
        raise SystemExit("no *_session*.npz in --posteriors-dir")
    by_sid = {}
    for f in files:
        m = re.match(r"(VP\d+)_session(\d+)\.npz$", os.path.basename(f))
        if m:
            by_sid.setdefault(m.group(1), {})[int(m.group(2))] = f
    rows, skipped = [], []
    systems = ("deep", "eeg", "fused", "fused_ctrl")
    CM = {(s, lvl): np.zeros((3, 3), int) for s in systems for lvl in ("win", "blk")}
    subj_cnt = {}          # (sid, system) -> (correct[3], total[3]) window level
    for sid in sorted(by_sid):
        if subjects and sid not in subjects:
            continue
        d = os.path.join(data, sid)
        pe = os.path.join(d, f"{sid}_eeg_channel_windows.npy")
        if not os.path.exists(pe):
            skipped.append((sid, "no eeg_channel_windows.npy"))
            continue
        x = np.asarray(np.load(pe, mmap_mode="r"), dtype=np.float64)
        y = np.load(os.path.join(d, f"{sid}_labels.npy"))
        ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        tid = np.load(os.path.join(d, f"{sid}_trial_ids.npy"))
        feats = variant_features(x, ses)
        for s, f in sorted(by_sid[sid].items()):
            z = np.load(f)
            te = ses == s
            if not (np.array_equal(z["y"], y[te]) and np.array_equal(z["trial_ids"], tid[te])):
                skipped.append((sid, f"session {s}: labels/trial ids differ from the saved fold -- refused"))
                continue
            yt, tt = y[te], tid[te]
            Pk = L.norm_rows(z["P_final"].astype(np.float64))        # (K,N,3)
            Pd = Pk.mean(0)
            deep_win, deep_tri = L.acc_window(Pd, yt), L.acc_trial(Pd, yt, tt)
            deep_single = float(np.mean([L.acc_window(Pk[k], yt) for k in range(len(Pk))]))
            Pmem = {v: L.eeg_posteriors(feats[v], y, ses, s) for v in VARIANTS}
            for v in VARIANTS:
                Pf = L.fuse(Pd, Pmem[v], W)
                rows.append(dict(subject=sid, session=s, variant=v,
                                 member_win=L.acc_window(Pmem[v], yt),
                                 fused_win=L.acc_window(Pf, yt), fused_tri=L.acc_trial(Pf, yt, tt),
                                 fused_single_win=float(np.mean([L.acc_window(L.fuse(Pk[k], Pmem[v], W), yt)
                                                                 for k in range(len(Pk))])),
                                 deep_win=deep_win, deep_tri=deep_tri, deep_single_win=deep_single,
                                 n=len(yt)))
            sysP = {"deep": Pd, "eeg": Pmem[MAIN], "fused": L.fuse(Pd, Pmem[MAIN], W),
                    "fused_ctrl": L.fuse(Pd, Pmem[CONTROL], W)}
            for name, P in sysP.items():
                CM[(name, "win")] += conf(P, yt)
                CM[(name, "blk")] += conf_block(P, yt, tt)
                key = (sid, name)
                c, t = subj_cnt.get(key, (np.zeros(3), np.zeros(3)))
                pred = P.argmax(-1)
                for k in CLASSES:
                    m = yt == k
                    c[k] += (pred[m] == k).sum()
                    t[k] += m.sum()
                subj_cnt[key] = (c, t)
        if verbose:
            print(f"[{sid}] done", flush=True)
    return rows, skipped, CM, subj_cnt


def recall_from(M):
    return np.diag(M) / np.clip(M.sum(1), 1, None)


def print_conf(name, M):
    r = recall_from(M)
    err = M.sum() - np.trace(M)
    e23 = M[1, 2] + M[2, 1]
    print(f"  {name}")
    print("               pred 0   pred 2   pred 3")
    for i, lab in enumerate(("true 0-back", "true 2-back", "true 3-back")):
        print(f"    {lab}  {M[i,0]:7d}  {M[i,1]:7d}  {M[i,2]:7d}   | recall {100*r[i]:5.1f}%")
    print(f"    accuracy {100*np.trace(M)/M.sum():.2f}% | 2<->3 confusions {e23}/{err} errors = {100*e23/max(err,1):.1f}% of all errors")


def summarise(rows, CM, subj_cnt):
    subs = sorted(set(r["subject"] for r in rows))

    def ps(variant, key):
        return np.array([np.mean([r[key] for r in rows if r["subject"] == s and r["variant"] == variant]) for s in subs])

    deep_w, deep_t = ps(MAIN, "deep_win"), ps(MAIN, "deep_tri")
    deep_single = ps(MAIN, "deep_single_win")
    print(f"\n{len(subs)} subjects, fusion weight w = {W} (pre-registered), deep member = 3-seed ensemble of the 'final' head")
    print(f"deep ensemble alone: window {deep_w.mean():.4f}  block {deep_t.mean():.4f}   |  deep single seeds: window {deep_single.mean():.4f}")
    print("\n" + "=" * 118)
    print("QUESTION 1 -- attribution: same fusion, different EEG member")
    print("=" * 118)
    print(f"{'EEG member':17} {'alone':>7} | {'fused win':>9} {'fused blk':>9} {'single':>7} | {'dWin vs deep':>12} {'p':>7} {'p_holm':>7} | {'dWin vs PREREG':>14} {'p':>7} {'p_holm':>7}")
    pd_raw, pm_raw, store = [], [], {}
    for v in VARIANTS:
        fw, ft = ps(v, "fused_win"), ps(v, "fused_tri")
        d = fw - deep_w
        pd_raw.append(wil(d))
        store[v] = (ps(v, "member_win"), fw, ft, ps(v, "fused_single_win"), d)
    main_fw = store[PREREG][1]
    others = [v for v in VARIANTS if v != PREREG]
    for v in others:
        pm_raw.append(wil(store[v][1] - main_fw))
    adj_d = dict(zip(VARIANTS, holm(pd_raw)))
    adj_m = dict(zip(others, holm(pm_raw)))
    pm = dict(zip(others, pm_raw))
    for i, v in enumerate(VARIANTS):
        mem, fw, ft, fs, d = store[v]
        if v == PREREG:
            tail = f"{'-':>14} {'-':>7} {'-':>7}"
        else:
            dm = fw - main_fw
            tail = f"{dm.mean():+14.4f} {pm[v]:7.4f} {adj_m[v]:7.4f}"
        print(f"{v:17} {mem.mean():7.4f} | {fw.mean():9.4f} {ft.mean():9.4f} {fs.mean():7.4f} | {d.mean():+12.4f} {pd_raw[i]:7.4f} {adj_d[v]:7.4f} | {tail}")
    # pre-registered reading
    dl = main_fw - store[CONTROL][1]
    pl = wil(dl)
    db = main_fw - store["riem_disp|broad"][1]
    pb = wil(db)
    ph = holm([pl, pb])
    ok = dl.mean() >= 0.01 and ph[0] < 0.05 and db.mean() > 0 and ph[1] < 0.05
    print(f"\nPre-registered reading (reference member pinned to {PREREG}, NOT affected by --main):")
    print(f"  fused({PREREG}) - fused(logvar)          = {dl.mean():+.4f}  p={pl:.4f}  Holm p={ph[0]:.4f}")
    print(f"  fused({PREREG}) - fused(riem_disp|broad) = {db.mean():+.4f}  p={pb:.4f}  Holm p={ph[1]:.4f}")
    print(f"  -> claim 'the Riemannian MULTIBAND member is what matters' is "
          f"{'SUPPORTED' if ok else 'NOT supported'} under the rule (>= +1.0 pp over logvar, Holm p<0.05, and > broadband-only).")

    print("\n" + "=" * 118)
    print("QUESTION 2 -- where does the gain come from? pooled confusion matrices (window and block), rows = true class")
    print("=" * 118)
    names = {"deep": "deep ensemble alone", "eeg": f"EEG member alone ({MAIN})",
             "fused": f"FUSED ({MAIN}, w={W})", "fused_ctrl": f"control: FUSED ({CONTROL}, w={W})"}
    for lvl, lab in (("win", "WINDOW level"), ("blk", "BLOCK level")):
        print(f"\n--- {lab} ---")
        for s in ("deep", "eeg", "fused", "fused_ctrl"):
            print_conf(names[s], CM[(s, lvl)])
    print("\nper-subject recall (mean over subjects) and paired difference vs deep ensemble (window level):")
    print(f"  {'system':32} {'rec 0-back':>11} {'rec 2-back':>11} {'rec 3-back':>11}")
    rec = {}
    for s in ("deep", "eeg", "fused", "fused_ctrl"):
        rec[s] = np.array([[subj_cnt[(sid, s)][0][k] / max(subj_cnt[(sid, s)][1][k], 1) for k in CLASSES] for sid in subs])
        print(f"  {names[s]:32} " + " ".join(f"{100*rec[s][:, k].mean():10.1f}%" for k in CLASSES))
    ps_ = []
    for s in ("eeg", "fused", "fused_ctrl"):
        for k, kn in zip(CLASSES, ("0-back", "2-back", "3-back")):
            d = rec[s][:, k] - rec["deep"][:, k]
            ps_.append((s, kn, d.mean(), int((d > 0).sum()), wil(d)))
    adj = holm([x[4] for x in ps_])
    print(f"  {'system - deep':22} {'class':>7} {'diff':>8} {'wins':>7} {'p':>8} {'p_holm(9)':>10}")
    for x, a in zip(ps_, adj):
        print(f"  {names[x[0]][:22]:22} {x[1]:>7} {x[2]:+8.4f} {x[3]:3d}/{len(subs):<2} {x[4]:8.4f} {a:10.4f}")
    print("\nCaveats: same 26 subjects and same 3 deep seeds for every row; 7 member variants were evaluated on "
          "these folds (Holm is reported); transductive per-session centring of every domain-aware member; offline "
          "fusion, TSFNet not retrained; block metrics rest on 9 blocks per fold; recall differences are "
          "descriptive unless Holm p < 0.05.")


# ----------------------------------------------------------------------------- self-test
def self_test():
    """planted: per-session per-source gains break logvar but not the centred Riemannian members;
    'deep' posteriors weak and independent. Expect fused(riem) >> fused(logvar); confusion matrices must
    sum to the number of windows / blocks; the control must actually be weaker."""
    rng = np.random.RandomState(0)
    C, T = 8, 400
    ok = True
    with tempfile.TemporaryDirectory() as td:
        dd, pdir = os.path.join(td, "data"), os.path.join(td, "post")
        os.makedirs(pdir)
        for sidn in (1, 2):
            sid = f"VP{sidn:03d}"
            os.makedirs(os.path.join(dd, sid))
            Lm = rng.randn(C, C) * 0.3 + np.eye(C)
            X, y, ses, tid = [], [], [], []
            t = 0
            for s in (1, 2, 3):
                A = np.diag(np.exp(0.9 * rng.randn(C)))
                for k in range(12):
                    cls = k % 3
                    for _ in range(3):
                        z = rng.randn(C, T)
                        z[cls] *= 1.8
                        X.append(A @ Lm @ z)
                        y.append(cls); ses.append(s); tid.append(t)
                    t += 1
            X, y, ses, tid = np.array(X), np.array(y), np.array(ses), np.array(tid)
            np.save(os.path.join(dd, sid, f"{sid}_eeg_channel_windows.npy"), X.astype(np.float32))
            for nm, a in (("labels", y), ("session_ids", ses), ("trial_ids", tid)):
                np.save(os.path.join(dd, sid, f"{sid}_{nm}.npy"), a)
            for s in (1, 2, 3):
                m = ses == s
                P = np.empty((3, m.sum(), 3))
                for k in range(3):
                    Lg = rng.randn(m.sum(), 3)
                    Lg[np.arange(m.sum()), y[m]] += 0.4
                    e = np.exp(Lg)
                    P[k] = e / e.sum(1, keepdims=True)
                np.savez(os.path.join(pdir, f"{sid}_session{s}.npz"), seeds=np.array([1, 2, 3]), y=y[m],
                         trial_ids=tid[m], pos_in_trial=np.zeros(m.sum(), np.int32),
                         P_final=P, P_eeg=P, P_fusion=P, P_fnirs=P)
        rows, skipped, CM, subj_cnt = run(dd, pdir, verbose=False)
        ok &= not skipped
        mean = lambda v, k: np.mean([r[k] for r in rows if r["variant"] == v])
        fr, fl, deep = mean(MAIN, "fused_win"), mean(CONTROL, "fused_win"), mean(MAIN, "deep_win")
        print(f"[attribution] deep {deep:.3f} | fused(logvar) {fl:.3f} | fused(riem_disp|multi) {fr:.3f}")
        good = fr > fl + 0.08 and fr > deep + 0.08
        print(f"  riemannian member clearly beats the logvar control and the deep member -> {'OK' if good else 'FAIL'}")
        ok &= good
        n_win = sum(r["n"] for r in rows if r["variant"] == MAIN)
        good = CM[("fused", "win")].sum() == n_win and CM[("deep", "win")].sum() == n_win
        print(f"[confusion] window confusion sums to {CM[('fused', 'win')].sum()} == {n_win} windows -> {'OK' if good else 'FAIL'}")
        ok &= good
        good = CM[("fused", "blk")].sum() == 2 * 3 * 12
        print(f"[confusion] block confusion sums to {CM[('fused', 'blk')].sum()} == {2 * 3 * 12} blocks -> {'OK' if good else 'FAIL'}")
        ok &= good
        # recall consistency: pooled recall vs per-subject counts
        c = sum(subj_cnt[(s, 'fused')][0] for s in ("VP001", "VP002"))
        good = np.array_equal(c, np.diag(CM[("fused", "win")]))
        print(f"[recall] per-subject correct counts == pooled diagonal -> {'OK' if good else 'FAIL'}")
        ok &= good
        # the full report must render without error on this synthetic run
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                summarise(rows, CM, subj_cnt)
            good = "QUESTION 1" in buf.getvalue() and "QUESTION 2" in buf.getvalue() and "Pre-registered reading" in buf.getvalue()
        except Exception as exc:                    # noqa: BLE001
            print("  summarise raised:", repr(exc))
            good = False
        print(f"[report] full report renders -> {'OK' if good else 'FAIL'}")
        ok &= good
        # holm sanity
        good = np.allclose(holm([0.001, 0.04, 0.5, 0.03]), [0.004, 0.09, 0.5, 0.09])
        print(f"[holm] -> {'OK' if good else 'FAIL'}")
        ok &= good
        # guard
        z = dict(np.load(os.path.join(pdir, "VP001_session2.npz")))
        z["y"] = (z["y"] + 1) % 3
        np.savez(os.path.join(pdir, "VP001_session2.npz"), **z)
        _r, sk, _c, _s = run(dd, pdir, verbose=False)
        good = len(sk) == 1 and "refused" in sk[0][1]
        print(f"[guard] misaligned fold refused -> {'OK' if good else 'FAIL'}")
        ok &= good
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    global MAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--posteriors-dir")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--subjects", nargs="+")
    ap.add_argument("--out", help="per-fold rows csv")
    ap.add_argument("--main", default=MAIN, choices=list(VARIANTS),
                    help="member used for the QUESTION 2 confusion matrices and per-class recall. The\nPRE-REGISTERED attribution reading of QUESTION 1 always stays on riem_disp|multi, whatever this is set to.")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    MAIN = a.main
    if not (a.data and a.posteriors_dir):
        ap.error("--data and --posteriors-dir are required unless --self-test")
    rows, skipped, CM, subj_cnt = run(a.data, a.posteriors_dir, subjects=a.subjects)
    for sid, why in skipped:
        print(f"[skipped] {sid}: {why}")
    if not rows:
        raise SystemExit("no fold could be evaluated")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print("per-fold rows ->", a.out)
    summarise(rows, CM, subj_cnt)


if __name__ == "__main__":
    main()
