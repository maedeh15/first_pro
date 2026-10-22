"""
block_member_contrast.py -- the one paired test the late-fusion report is missing: at BLOCK level,
does the fused system beat the EEG member ALONE?

Why it is needed: in fusion_controls the block-level numbers are 0.6382 (deep alone), 0.7108 (EA
member alone) and 0.7179 (fused). The fused-vs-deep contrast is tested; the fused-vs-member contrast
is not, and at block level it is only +0.7 points -- small enough that "the two members are
complementary" cannot be asserted at block level without a test. The window-level version of the
same contrast IS reported by late_fusion_riemann.py (+3.11, p=0.0036 for EA).

It needs nothing new to be computed: late_fusion_riemann.py already writes per-fold accuracies for
every weight on the grid, so w=1.0 is the member alone and w=0.5 is the fused system. This script
only re-reads that csv and runs the paired Wilcoxon over subjects that the report skips.

usage:
  python block_member_contrast.py --self-test
  python block_member_contrast.py --folds late_fusion_EA_folds.csv
  python block_member_contrast.py --folds late_fusion_EA_folds.csv late_fusion_riem_disp_folds.csv
"""
import csv
import argparse
import tempfile
import os
import numpy as np
from scipy.stats import wilcoxon


def load(path, deep="ensemble"):
    """-> {w: {subject: mean over folds}} for the window, block and 2v3 columns"""
    out = {"win": {}, "tri": {}, "v23": {}}
    acc = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["deep"] != deep:
                continue
            key = (round(float(r["w"]), 6), r["subject"])
            acc.setdefault(key, []).append((float(r["win"]), float(r["tri"]), float(r["v23"])))
    for (w, sid), vals in acc.items():
        m = np.mean(vals, axis=0)
        for j, k in enumerate(("win", "tri", "v23")):
            out[k].setdefault(w, {})[sid] = m[j]
    return out


def paired(a, b):
    """a, b: {subject: value}. returns diff, wins, n, p over the shared subjects"""
    subs = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in subs])
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return d.mean(), int((d > 0).sum()), len(d), float(p)


def report(path, deep="ensemble", w_fused=0.5):
    tab = load(path, deep)
    for lvl in ("win", "tri", "v23"):
        if w_fused not in tab[lvl] or 1.0 not in tab[lvl] or 0.0 not in tab[lvl]:
            raise SystemExit(f"{path}: need w=0.0, w={w_fused} and w=1.0 rows for deep='{deep}'")
    print(f"\n{os.path.basename(path)}  (deep member = {deep}, fused at w={w_fused})")
    print(f"  {'level':8} {'deep':>7} {'member':>7} {'fused':>7} | {'fused-deep':>10} {'wins':>7} {'p':>8} "
          f"| {'fused-member':>12} {'wins':>7} {'p':>8}")
    for lvl, lab in (("win", "window"), ("tri", "block"), ("v23", "2-vs-3")):
        deep_a, mem, fus = tab[lvl][0.0], tab[lvl][1.0], tab[lvl][w_fused]
        dd, wd, n, pd_ = paired(fus, deep_a)
        dm, wm, _, pm = paired(fus, mem)
        g = lambda d: np.mean([d[s] for s in sorted(d)])
        print(f"  {lab:8} {g(deep_a):7.4f} {g(mem):7.4f} {g(fus):7.4f} | {dd:+10.4f} {wd:3d}/{n:<3} {pd_:8.4f} "
              f"| {dm:+12.4f} {wm:3d}/{n:<3} {pm:8.4f}")
    print("  note: only the WINDOW-level 'fused - deep' contrast at w=0.5 was pre-registered; the rest, "
          "including every 'fused - member' cell, is descriptive and uncorrected.")


def self_test():
    ok = True
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "folds.csv")
        rng = np.random.RandomState(0)
        # planted: fused sits ABOVE both ends at window level, and EQUAL to the member at block level
        rows = []
        for sid in [f"VP{i:03d}" for i in range(1, 13)]:
            base = 0.55 + 0.05 * rng.rand()
            for s in (1, 2, 3):
                for w, dw, dt in ((0.0, 0.00, 0.00), (0.5, 0.08, 0.05), (1.0, 0.04, 0.05)):
                    for deep in ("ensemble", "single"):
                        rows.append(dict(subject=sid, session=s, deep=deep, w=w,
                                         win=base + dw + 0.01 * rng.randn(),
                                         tri=base + dt + 0.01 * rng.randn(),
                                         v23=0.5 + 0.01 * rng.randn()))
        with open(p, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        tab = load(p, "ensemble")
        good = sorted(tab["win"]) == [0.0, 0.5, 1.0] and len(tab["win"][0.5]) == 12
        print(f"[load] 3 weights x 12 subjects, folds averaged -> {'OK' if good else 'FAIL'}")
        ok &= good
        dm_win, _, _, p_win = paired(tab["win"][0.5], tab["win"][1.0])
        dm_tri, _, _, p_tri = paired(tab["tri"][0.5], tab["tri"][1.0])
        good = dm_win > 0.03 and p_win < 0.01
        print(f"[window] planted fused>member recovered: {dm_win:+.4f} p={p_win:.4f} -> {'OK' if good else 'FAIL'}")
        ok &= good
        good = abs(dm_tri) < 0.01 and p_tri > 0.05
        print(f"[block] planted fused==member not called a win: {dm_tri:+.4f} p={p_tri:.4f} -> "
              f"{'OK' if good else 'FAIL'}")
        ok &= good
        report(p)
        print("[report] renders -> OK")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", help="per-fold csv written by late_fusion_riemann.py --out")
    ap.add_argument("--deep", default="ensemble", choices=["ensemble", "single"])
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if not a.folds:
        ap.error("--folds is required unless --self-test")
    for p in a.folds:
        report(p, a.deep, a.w)


if __name__ == "__main__":
    main()
