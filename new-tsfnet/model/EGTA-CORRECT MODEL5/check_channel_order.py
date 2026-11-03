"""
check_channel_order.py -- verify the EEG channel order of the (N, 28, 2000) windows, so that channel/region claims become possible (or stay forbidden).
The preprocessing builds the channel axis of {sid}_eeg_channel_windows.npy with get_data(picks="eeg"), i.e. in the order of the EEG-TYPE channels of the cleaned raw object,
which is saved as {sid}_eeg_clean_raw.fif.  Two independent checks:
  (1) NAMES  : read the EEG-type channel names, in order, from the .fif of each subject; they must be 28 names, identical across subjects. Writes channels.txt
               (one name per line; use it as --channel-names in xai_member_shap.py).
  (2) DATA   : neighbouring scalp electrodes record correlated signals. With positions from the standard 10-05 montage for those names, the Spearman correlation between
               pairwise channel correlation and inter-electrode distance must be clearly NEGATIVE, and clearly beyond a permutation null in which the names are shuffled over the
               data channels. A wrong order would give a value near the null. (Common-reference leakage makes all correlations positive but keeps this ordering effect.)
CAVEAT: the data check assumes the saved channel windows are NOT covariance-aligned (run preprocessing without --eeg-cov-align, the default); whitened windows would destroy the neighbour structure
and make the check fail for a reason unrelated to the order.
Decision (stated before running): the order is CONFIRMED if the names are identical across the subjects checked AND, in at least 90% of them, rho(corr, distance) < 0 with
permutation p < 0.01. Otherwise channel/region claims stay forbidden.
  python check_channel_order.py --data-root "..." --subjects 1-26 --out-dir "...\\channel_order"
  python check_channel_order.py --self-test
"""
import argparse, os, sys, json
import numpy as np
from scipy.stats import spearmanr


def montage_positions(names):
    import mne, warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try: m = mne.channels.make_standard_montage("standard_1005")
        except Exception: m = mne.channels.make_standard_montage("colin27_1005")
    pos = m.get_positions()["ch_pos"]; low = {k.lower(): v for k, v in pos.items()}
    P, missing = [], []
    for n in names:
        v = low.get(n.lower())
        if v is None: missing.append(n); P.append([np.nan] * 3)
        else: P.append(v)
    return np.array(P, float), missing


def order_check(corr, pos, n_perm=2000, seed=0):
    """corr (C,C) channel correlation, pos (C,3) -> (rho, p_perm, null_mean). rho = Spearman(corr, distance) over channel pairs"""
    C = corr.shape[0]; iu = np.triu_indices(C, 1); ok = np.isfinite(pos).all(1)
    D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    def rho_of(perm):
        c = corr[np.ix_(perm, perm)]; m = ok[:, None] & ok[None]; sel = m[iu]
        return spearmanr(c[iu][sel], D[iu][sel])[0]
    rho = rho_of(np.arange(C)); rng = np.random.RandomState(seed); null = np.array([rho_of(rng.permutation(C)) for _ in range(n_perm)])
    return float(rho), float((1 + (null <= rho).sum()) / (1 + n_perm)), float(null.mean())


def eeg_names_from_fif(path):
    import mne
    raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
    return [n for n, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]


def channel_corr(x):
    """x (N, C, T) -> (C, C) correlation of the standardised, window-concatenated broadband signals (mean over windows of the per-window correlation)"""
    z = x - x.mean(-1, keepdims=True); z = z / (z.std(-1, keepdims=True) + 1e-12)
    return np.mean(np.einsum("nct,ndt->ncd", z, z) / z.shape[-1], axis=0)


def self_test():
    rng = np.random.RandomState(0); C = 28; pos = rng.randn(C, 3); pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    D = np.linalg.norm(pos[:, None] - pos[None], axis=-1); K = np.exp(-D / 0.7) + 1e-3 * np.eye(C); L = np.linalg.cholesky(K)
    x = np.einsum("cd,ndt->nct", L, rng.randn(20, C, 400)); corr = channel_corr(x)
    rho, p, nm = order_check(corr, pos, 500); assert rho < -0.5 and p < 0.01, (rho, p)
    perm = rng.permutation(C); rho2, p2, _ = order_check(corr[np.ix_(perm, perm)], pos, 500); assert rho2 > -0.3 and p2 > 0.05, (rho2, p2)
    print(f"[1] correct order: rho {rho:+.2f} (p {p:.3f}); shuffled names: rho {rho2:+.2f} (p {p2:.2f})")
    try:
        import mne, tempfile
        names = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2", "CCP3"]
        info = mne.create_info(names + ["VEOG"], 200.0, ["eeg"] * len(names) + ["eog"]); raw = mne.io.RawArray(rng.randn(len(names) + 1, 2000) * 1e-6, info, verbose=False)
        f = os.path.join(tempfile.mkdtemp(), "VP_eeg_clean_raw.fif"); raw.save(f, overwrite=True, verbose=False)
        got = eeg_names_from_fif(f); assert got == names, got
        P, miss = montage_positions(names); assert not miss and np.isfinite(P).all()
        print("[2] fif reading keeps the EEG-type order and drops the EOG channel; standard-montage positions found for all names")
    except ImportError: print("[2] mne missing: the .fif part was skipped (it runs on your machine)")
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root"); ap.add_argument("--subjects", default="1-26"); ap.add_argument("--out-dir", default="channel_order")
    ap.add_argument("--max-windows", type=int, default=200); ap.add_argument("--perm", type=int, default=2000); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    if not a.data_root: sys.exit("need --data-root")
    lo, hi = a.subjects.split("-") if "-" in a.subjects else (a.subjects, a.subjects); res, names0, bad = {}, None, []
    for i in range(int(lo), int(hi) + 1):
        sid = f"VP{i:03d}"; d = os.path.join(a.data_root, sid); fif = os.path.join(d, f"{sid}_eeg_clean_raw.fif"); npy = os.path.join(d, f"{sid}_eeg_channel_windows.npy")
        if not (os.path.isfile(fif) and os.path.isfile(npy)): print(f"[{sid}] missing .fif or windows file -- skipped"); continue
        names = eeg_names_from_fif(fif)
        if len(names) != 28: bad.append((sid, f"{len(names)} EEG channels")); continue
        names0 = names0 or names
        if names != names0: bad.append((sid, "channel names/order differ from the first subject"))
        x = np.load(npy, mmap_mode="r"); idx = np.random.RandomState(0).choice(len(x), min(a.max_windows, len(x)), replace=False); idx.sort()
        pos, miss = montage_positions(names); rho, p, nm = order_check(channel_corr(np.asarray(x[idx], dtype=np.float64)), pos, a.perm)
        res[sid] = dict(rho=rho, p=p, null_mean=nm, missing_positions=miss); print(f"[{sid}] rho(corr, distance) {rho:+.3f} | permutation p {p:.4f} | null mean {nm:+.3f}" + (f" | no montage position for {miss}" if miss else ""), flush=True)
    if not res: sys.exit("no subject could be checked")
    ok = [s for s, r in res.items() if r["rho"] < 0 and r["p"] < 0.01]; frac = len(ok) / len(res); names_same = not bad
    verdict = "CONFIRMED" if (names_same and frac >= 0.9) else "NOT CONFIRMED"
    print(f"\nnames identical across subjects: {names_same}{'' if names_same else ' -- ' + str(bad)}")
    print(f"data check passed in {len(ok)}/{len(res)} subjects ({100 * frac:.0f}%; rule: >= 90%)")
    print(f"VERDICT: channel order {verdict}" + ("" if verdict == "CONFIRMED" else "; channel/region claims stay forbidden"))
    os.makedirs(a.out_dir, exist_ok=True)
    if names0: open(os.path.join(a.out_dir, "channels.txt"), "w", encoding="utf-8").write("\n".join(names0) + "\n"); print("names, in data order:", ", ".join(names0))
    json.dump(dict(verdict=verdict, names=names0, names_identical=names_same, problems=bad, per_subject=res), open(os.path.join(a.out_dir, "channel_order_check.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
