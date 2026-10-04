"""
build_erp_windows.py -- priority-ERP: per-window stimulus-locked ERP tensors.

WHAT IT PRODUCES (per subject, written NEXT TO the existing preprocessed files;
nothing existing is modified):
    {sid}_erp_windows.npy   (N, 28, 200) float32 -- for every saved window, the
                            average of the digit-locked epochs [-0.2, +0.8) s
                            that fit entirely inside that window (~4-5), each
                            baseline-corrected on [-0.2, 0) s
    {sid}_erp_counts.npy    (N,) number of epochs averaged per window
    {sid}_window_offsets.npy(N,) the window's offset (s) from its block onset
    {sid}_erp_meta.json     provenance + a fingerprint of the window set, which
                            NBackDataset(load_erp=True) checks before using it

WHY A SEPARATE BUILDER (no preprocessing re-run)
The saved windows do not record where in the recording they start, and some are
dropped (|z| > 8, fNIRS bounds), so their positions cannot be inferred from
their order alone. But every saved channel-space window is an exact slice
eeg_full[:, w_start : w_start + 2000] of the cleaned continuous EEG that
preprocessing also saved ({sid}_eeg_clean_raw.fif), with
w_start = block_onset_sample + 200 * offset, offset in {-2, -1, ..., 32}. So
each window's offset is RECOVERED by matching it against those candidate
slices (per-channel correlation; per-session normalization is a per-channel
affine map and does not change correlation). Every window must match with
r >= --min-r (default 0.999) or the subject is rejected -- nothing is guessed.

The ERP itself is cut from the SAVED window (not from the .fif), so it has
exactly the normalization the model sees.

LEAKAGE: stimulus marker codes encode the class (16 = 0-back, 48/64 = 2-back,
80/96 = 3-back). They are used ONLY for onset times; digits appear on the same
2 s grid in every class, so the epoch positions carry no class information,
and a stimulus's code never enters the tensor.

USAGE
    python build_erp_windows.py
    python build_erp_windows.py --data-root "<folder with VPxxx\\*.npy and *_eeg_clean_raw.fif>" ^
                                --raw-root  "<folder with VPxxx-EEG>"
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import scipy.io as sio

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--data-root", default=r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback")
p.add_argument("--raw-root", default=r"E:\kntu\Thesis\codes\DATA")
p.add_argument("--subjects", default="1-26")
p.add_argument("--min-r", type=float, default=0.999)
p.add_argument("--max-lag", type=int, default=200,
               help="samples searched for a constant marker-vs-fif clock lag (first block)")
p.add_argument("--overwrite", action="store_true")
ARGS = p.parse_args()

FS = 200
WIN = 2000
OFFSETS = np.arange(-2, 33)                     # = np.arange(SLIDE_TMIN, SLIDE_TMAX-10+1e-6, 1)
EP_PRE, EP_POST = 40, 160                       # -0.2 s .. +0.8 s at 200 Hz
BLOCK_CODES = (112, 128, 144)
STIM_CODES = (16, 48, 64, 80, 96)               # timing only


def parse_subjects(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [f"VP{i:03d}" for i in out]


def _first_struct(path):
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    return m[[k for k in m if not k.startswith("__")][0]]


def load_markers(eeg_folder):
    """Same as Run D0.1b / diag_2v3_separability: (sample @200 Hz, code)."""
    cnt = _first_struct(os.path.join(eeg_folder, "cnt_nback.mat"))
    fs = FS if float(cnt.fs) != FS else float(cnt.fs)
    mrk = _first_struct(os.path.join(eeg_folder, "mrk_nback.mat"))
    if hasattr(mrk, "time"):
        pos = np.round(np.asarray(mrk.time, dtype=float) * fs / 1000).astype(int)
    else:
        pos = np.asarray(mrk.pos, dtype=int) - 1
    return pos, np.asarray(mrk.event.desc, dtype=int).flatten()


def _z(x):
    x = x - x.mean(axis=-1, keepdims=True)
    return x / (x.std(axis=-1, keepdims=True) + 1e-12)


def corr(a, b):
    """mean over channels of the per-channel Pearson r of two (C, T) arrays."""
    return float((_z(a) * _z(b)).mean())


def fingerprint(labels, trial_ids, session_ids):
    h = hashlib.sha1()
    for a in (labels, trial_ids, session_ids):
        h.update(np.ascontiguousarray(a).astype(np.int64).tobytes())
    return h.hexdigest()


def build_subject(sid):
    import mne
    d = os.path.join(ARGS.data_root, sid)
    out = os.path.join(d, f"{sid}_erp_windows.npy")
    if os.path.exists(out) and not ARGS.overwrite:
        return "exists (use --overwrite)"
    chan = np.load(os.path.join(d, f"{sid}_eeg_channel_windows.npy"), mmap_mode="r")
    labels = np.load(os.path.join(d, f"{sid}_labels.npy"))
    trial_ids = np.load(os.path.join(d, f"{sid}_trial_ids.npy"))
    session_ids = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
    if chan.shape[1:] != (28, WIN):
        raise ValueError(f"unexpected channel-window shape {chan.shape}")
    raw = mne.io.read_raw_fif(os.path.join(d, f"{sid}_eeg_clean_raw.fif"), preload=True, verbose=False)
    full = raw.get_data(picks="eeg")                       # same picks as extract_windows
    if full.shape[0] != 28:
        raise ValueError(f".fif has {full.shape[0]} EEG channels, expected 28")
    pos, codes = load_markers(os.path.join(ARGS.raw_root, f"{sid}-EEG"))
    blocks = pos[np.isin(codes, BLOCK_CODES)]
    stims = np.sort(pos[np.isin(codes, STIM_CODES)])

    # constant clock lag between the marker file and the .fif, from the first saved window
    first = np.flatnonzero(trial_ids == trial_ids.min())[0]
    t0 = int(trial_ids[first])
    best = (-2.0, None)
    for o in OFFSETS:
        for lag in range(-ARGS.max_lag, ARGS.max_lag + 1):
            s = blocks[t0] + FS * o + lag
            if s < 0 or s + WIN > full.shape[1]:
                continue
            r = corr(np.asarray(chan[first]), full[:, s:s + WIN])
            if r > best[0]:
                best = (r, lag)
        if best[0] >= ARGS.min_r:
            break
    if best[1] is None or best[0] < ARGS.min_r:
        raise ValueError(f"could not locate the first window in the .fif (best r={best[0]:.4f}); "
                         f"is the .fif from the same preprocessing run as the .npy files?")
    lag = best[1]
    stims = stims + lag        # stimulus times into the .fif / window clock, like the window starts

    N = len(labels)
    offsets = np.full(N, np.nan)
    rs = np.zeros(N)
    erp = np.zeros((N, 28, EP_PRE + EP_POST), dtype=np.float32)
    counts = np.zeros(N, dtype=np.int16)
    for t in np.unique(trial_ids):
        idx = np.flatnonzero(trial_ids == t)            # saved order == increasing offset
        k = 0
        for i in idx:
            w = np.asarray(chan[i], dtype=np.float64)
            while k < len(OFFSETS):
                s = blocks[t] + FS * OFFSETS[k] + lag
                r = corr(w, full[:, s:s + WIN]) if 0 <= s and s + WIN <= full.shape[1] else -1.0
                k += 1
                if r >= ARGS.min_r:
                    offsets[i], rs[i] = OFFSETS[k - 1], r
                    break
            if np.isnan(offsets[i]):
                raise ValueError(f"window {i} (block {t}) matched no candidate offset at r >= "
                                 f"{ARGS.min_r}")
            start = s
            rel = stims[(stims - EP_PRE >= start) & (stims + EP_POST <= start + WIN)] - start
            if len(rel):
                eps = np.stack([w[:, r0 - EP_PRE:r0 + EP_POST] for r0 in rel])
                eps -= eps[:, :, :EP_PRE].mean(axis=2, keepdims=True)
                erp[i] = eps.mean(axis=0)
                counts[i] = len(rel)

    np.save(out, erp)
    np.save(os.path.join(d, f"{sid}_erp_counts.npy"), counts)
    np.save(os.path.join(d, f"{sid}_window_offsets.npy"), offsets)
    meta = {"n_windows": int(N), "fingerprint": fingerprint(labels, trial_ids, session_ids),
            "clock_lag_samples": int(lag), "min_match_r": float(rs.min()),
            "epoch_s": [-EP_PRE / FS, EP_POST / FS], "baseline_s": [-EP_PRE / FS, 0.0],
            "mean_epochs_per_window": float(counts.mean()),
            "windows_with_no_epoch": int((counts == 0).sum()),
            "stimulus_codes_used_for_timing_only": list(STIM_CODES)}
    with open(os.path.join(d, f"{sid}_erp_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return (f"{N} windows | min r {rs.min():.5f} | lag {lag} | "
            f"{counts.mean():.2f} epochs/window ({(counts == 0).sum()} with none)")


def main():
    failed = []
    for sid in parse_subjects(ARGS.subjects):
        if not os.path.isdir(os.path.join(ARGS.data_root, sid)):
            continue
        try:
            print(f"[{sid}] {build_subject(sid)}")
        except Exception as e:
            print(f"[{sid}] FAILED -- {type(e).__name__}: {e}")
            failed.append(sid)
    if failed:
        sys.exit(f"\nFAILED for {failed}: do not train with --erp-branch until these are fixed.")
    print("\nDONE. Train with --erp-branch eeg (or eeg_fusion).")


if __name__ == "__main__":
    main()
