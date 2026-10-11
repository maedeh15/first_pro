# ============================================================================
# MGFNet input, TSFNet-MATCHED preprocessing -- n-back (Shin et al., 2018)
#
# PURPOSE: a fair MGFNet-vs-TSFNet comparison. Every preprocessing step and
# the window length are IDENTICAL to Preproc_tsfnet_nback_badch.py; only the
# final tensor FORMAT differs, because MGFNet consumes channel-space signals
# (EEG 28 x T, HbO / HbR 36 x T) instead of TSFNet's 16x16 interpolated grid.
#
# HOW "IDENTICAL" IS GUARANTEED: the EEG chain is not re-implemented. This
# script imports Preproc_tsfnet_nback_badch.py (must sit in the same folder)
# and calls ITS load_eeg_data() + preprocess_eeg() directly, so EEG gets the
# exact same code path (0.5-50 Hz order-6 Butterworth -> PyPREP bad channels
# -> CAR -> ICA [EOG correlation with the per-subject thresholds file +
# ICLabel + MANUAL_IC_OVERRIDES] -> per-channel z-score). Its constants (EEG
# band, order, notch flag, baseline span, window/step/slide, segment count,
# rejection rule, per-session flag) are read from that module, so any change
# you make there propagates here.
#
# fNIRS and windowing are re-implemented in channel space with the same
# constants and the same order of operations as the TSFNet file:
#   fNIRS: (already Beer-Lambert) -> bandpass only if ENABLE_FNIRS_BANDPASS
#          (False in your file) -> resample to 10 Hz -> per-channel z-score
#          -> per-trial baseline subtraction (mean of -5..-2 s, fNIRS clock)
#   windows: EEG WINDOW_LEN_EEG = 10 s, step 1 s, slide -2..42 s, fNIRS timed
#          on its own clock, |z| > 8 window rejection, then per-session
#          normalisation (ENABLE_PER_SESSION_NORM) as the very last step.
#
# SAME WINDOWS AS TSFNet: a window is kept only if TSFNet would also keep it
# (including TSFNet's requirement that its 21 fNIRS segments, i.e. 23 s of
# fNIRS after the window start, exist). Each window's (trial, offset) key is
# saved, and --check-against <TSFNet subject folder root> verifies that the
# windows, labels and sessions line up 1:1 with the TSFNet .npy files, so the
# two models are evaluated on exactly the same samples.
#
# THE ONE UNAVOIDABLE INPUT DIFFERENCE: MGFNet's fNIRS input covers the SAME
# 10 s as its EEG window (Sec. 3.2: fNIRS is interpolated onto the EEG time
# axis), while TSFNet additionally sees 13 s of fNIRS after the window
# (21 x 3 s segments spanning 23 s) to absorb the hemodynamic lag. Feeding
# MGFNet that extra span would break its EEG-fNIRS temporal alignment.
#
# OUTPUTS (per subject) -- same file names as Preproc_mgfnet_nback.py, so
# mgfnet_dataset / run_all_subjects_MGFNET.py work unchanged (point
# --data-root at OUTPUT_PATH):
#   {subject}_mgf_eeg.npy      (N, 28, 2000) float32
#   {subject}_mgf_hbo.npy      (N, 36,  100) float32
#   {subject}_mgf_hbr.npy      (N, 36,  100) float32
#   {subject}_labels.npy / _session_ids.npy / _trial_ids.npy
#   {subject}_window_offsets.npy (N,) float32   window start rel. to trial onset (s)
#   {subject}_mgf_metadata.json
# ============================================================================
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
import time
import traceback

import numpy as np
from scipy.signal import resample

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The TSFNet module parses its own CLI flags with parse_known_args at import
# time; hide ours from it so they cannot be misread.
_argv, sys.argv = sys.argv, sys.argv[:1]
try:
    import Preproc_tsfnet_nback_badch as T
finally:
    sys.argv = _argv

# ============================================================================
# CONFIGURATION (paths only -- every preprocessing constant comes from T)
# ============================================================================
BASE_DATASET_PATH = T.BASE_DATASET_PATH
OUTPUT_PATH = r"E:\kntu\Thesis\codes\Results\MGFNET_TSFNETMATCHED_nback"
TSFNET_OUTPUT_PATH = T.OUTPUT_PATH        # only used by --check-against default
SUBJECTS = list(range(1, 27))
SAVE_QC = True                            # TSFNet's own QC plots/.fif (same as its run)

N_FNIRS_CHANNELS = 36


def log(msg):
    T.log_message(msg)


# ============================================================================
# fNIRS (channel space, same steps/order as T.preprocess_fnirs)
# ============================================================================
def preprocess_fnirs_channel_space(subject_id):
    oxy_raw, deoxy_raw, sfreq_raw, _mnt, ch_names, ev_time, ev_code = T.load_fnirs_data(subject_id)

    if T.DATASET_ALREADY_BEER_LAMBERT:
        hbo, hbr = oxy_raw, deoxy_raw
    else:
        hbo, hbr = T.modified_beer_lambert(oxy_raw, deoxy_raw)

    if T.ENABLE_FNIRS_BANDPASS:
        hbo, hbr = T.fnirs_bandpass(hbo, sfreq_raw), T.fnirs_bandpass(hbr, sfreq_raw)

    n_out = int(hbo.shape[0] * T.SFREQ_FNIRS / sfreq_raw)
    hbo_rs = resample(hbo, n_out, axis=0)
    hbr_rs = resample(hbr, n_out, axis=0)

    # per-channel, per-subject z-score (identical formula, incl. the +1e-8)
    hbo_n = (hbo_rs - hbo_rs.mean(axis=0, keepdims=True)) / (hbo_rs.std(axis=0, keepdims=True) + 1e-8)
    hbr_n = (hbr_rs - hbr_rs.mean(axis=0, keepdims=True)) / (hbr_rs.std(axis=0, keepdims=True) + 1e-8)
    log(f"  {subject_id}: fNIRS {'bandpass ON' if T.ENABLE_FNIRS_BANDPASS else 'NO bandpass'}, "
        f"resampled {sfreq_raw:g}->{T.SFREQ_FNIRS:g} Hz, per-channel z-score "
        f"(HbO std={float(np.mean(hbo_n.std(axis=0))):.4f}, HbR std={float(np.mean(hbr_n.std(axis=0))):.4f})")
    # (n_samples, 36, 2): chromophore last, like TSFNet's grid
    return np.stack([hbo_n, hbr_n], axis=-1), ev_time, ev_code, ch_names


# ============================================================================
# WINDOWING (same rules as T.extract_windows, channel space)
# ============================================================================
def extract_windows_channel_space(eeg_full, sfreq_eeg, events, fnirs_ch, fnirs_ev_time,
                                  fnirs_ev_code, subject_id):
    """eeg_full: (28, n_times) z-scored EEG; fnirs_ch: (n_samples, 36, 2)."""
    empty = (None,) * 7
    target = events[np.isin(events[:, 2], list(T.EEG_TRIAL_LABEL_MAP.keys()))]
    if len(target) == 0:
        log(f"  {subject_id}: no trial markers found (codes seen: {np.unique(events[:, 2])})")
        return empty

    # fNIRS clock sync -- identical logic to T.extract_windows
    sync_ok, fnirs_times = False, None
    if fnirs_ev_time is not None and fnirs_ev_code is not None:
        mask = np.isin(fnirs_ev_code, list(T.FNIRS_TRIAL_LABEL_MAP.keys()))
        times_ms = np.asarray(fnirs_ev_time)[mask].astype(float)
        codes = np.asarray(fnirs_ev_code)[mask]
        order = np.argsort(times_ms)
        fnirs_times = times_ms[order] / 1000.0
        f_cls = np.array([T.FNIRS_TRIAL_LABEL_MAP[c] for c in codes[order]])
        e_cls = np.array([T.EEG_TRIAL_LABEL_MAP.get(int(ev[2]), -1) for ev in target])
        n_common = min(len(f_cls), len(e_cls))
        if len(fnirs_times) == len(target) and np.array_equal(f_cls[:n_common], e_cls[:n_common]):
            sync_ok = True
        else:
            log(f"  {subject_id}: WARNING - fNIRS/EEG trial markers mismatch, "
                f"falling back to EEG-clock timing for fNIRS windows.")
    else:
        log(f"  {subject_id}: WARNING - fNIRS trial markers unavailable, "
            f"falling back to EEG-clock timing for fNIRS windows.")

    win_eeg = int(T.WINDOW_LEN_EEG * sfreq_eeg)                    # 2000
    win_nir = int(round(T.WINDOW_LEN_EEG * T.SFREQ_FNIRS))         # 100 (same 10 s as EEG)
    seg_nir = int(T.WINDOW_LEN_FNIRS_SEGMENT * T.SFREQ_FNIRS)      # 30 (TSFNet segment)
    offsets = np.arange(T.SLIDE_TMIN, T.SLIDE_TMAX - T.WINDOW_LEN_EEG + 1e-6, T.WINDOW_STEP)

    E, O, R, lab, sess, trl, offs = [], [], [], [], [], [], []
    for trial_idx, ev in enumerate(target):
        label = T.EEG_TRIAL_LABEL_MAP.get(int(ev[2]), -1)
        if label == -1:
            continue
        session_id = trial_idx // T.TRIALS_PER_SESSION + 1
        t_eeg = ev[0] / sfreq_eeg
        t_nir = fnirs_times[trial_idx] if sync_ok else t_eeg

        # per-trial baseline: mean of [-5, -2] s before onset (fNIRS clock)
        b0 = int(np.round((t_nir + T.BASELINE_TMIN) * T.SFREQ_FNIRS))
        b1 = int(np.round((t_nir + T.BASELINE_TMAX) * T.SFREQ_FNIRS))
        baseline = fnirs_ch[b0:b1].mean(axis=0, keepdims=True) if (b0 >= 0 and b1 > b0) else 0.0

        for off in offsets:
            s_e = int(np.round((t_eeg + off) * sfreq_eeg))
            if s_e < 0 or s_e + win_eeg > eeg_full.shape[1]:
                continue
            eeg_slice = eeg_full[:, s_e:s_e + win_eeg]
            if np.max(np.abs(eeg_slice)) > 8.0:                 # TSFNet's z-score rejection
                continue
            # TSFNet keeps a window only if ALL its fNIRS segments fit -- the
            # last one starts at off + (S-1)*step. Same test here, so the two
            # pipelines keep exactly the same windows.
            ok = True
            for seg in range(T.FNIRS_NUM_SEGMENTS):
                st = int(np.round((t_nir + off + seg * T.WINDOW_STEP) * T.SFREQ_FNIRS))
                if st < 0 or st + seg_nir > fnirs_ch.shape[0]:
                    ok = False
                    break
            s_f = int(np.round((t_nir + off) * T.SFREQ_FNIRS))
            if not ok or s_f < 0 or s_f + win_nir > fnirs_ch.shape[0]:
                continue
            f_slice = fnirs_ch[s_f:s_f + win_nir] - baseline          # (100, 36, 2)
            E.append(eeg_slice)
            O.append(f_slice[..., 0].T)                               # (36, 100)
            R.append(f_slice[..., 1].T)
            lab.append(label); sess.append(session_id); trl.append(trial_idx); offs.append(off)

    if not E:
        return empty
    E = np.asarray(E, dtype=np.float32)
    O = np.asarray(O, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    lab = np.asarray(lab, dtype=np.int64)
    sess = np.asarray(sess, dtype=np.int64)
    trl = np.asarray(trl, dtype=np.int64)
    offs = np.asarray(offs, dtype=np.float32)
    extract_windows_channel_space.last_sync_ok = sync_ok

    if T.ENABLE_PER_SESSION_NORM:
        E, O, R = per_session_normalization(E, O, R, sess, subject_id)
    return E, O, R, lab, sess, trl, offs


def per_session_normalization(E, O, R, sess, subject_id):
    """Channel-space equivalent of T.apply_per_session_normalization: within
    each session, per channel (and per chromophore for fNIRS), over all
    windows and time points of that session. Like TSFNet's, it uses the
    held-out session's own UNLABELLED statistics (transductive) -- keep it
    identical to TSFNet for the comparison, and disclose it in the thesis."""
    E, O, R = E.copy(), O.copy(), R.copy()
    for s in np.unique(sess):
        m = sess == s
        if m.sum() < 2:
            log(f"  {subject_id}: per-session norm SKIPPED for session {s} (only {int(m.sum())} window)")
            continue
        for arr in (E, O, R):
            x = arr[m]
            arr[m] = (x - x.mean(axis=(0, 2), keepdims=True)) / (x.std(axis=(0, 2), keepdims=True) + 1e-8)
    log(f"  {subject_id}: applied PER-SESSION normalization over {len(np.unique(sess))} session(s) "
        f"(EEG std={float(E.std()):.4f}, HbO std={float(O.std()):.4f}, HbR std={float(R.std()):.4f})")
    return E, O, R


# ============================================================================
# CHECK AGAINST THE TSFNet OUTPUT
# ============================================================================
def check_against_tsfnet(subject_id, out_dir, tsfnet_root):
    """Verifies the windows line up 1:1 with the TSFNet .npy files: same N,
    same labels/sessions/trials in the same order, and (if TSFNet saved its
    channel-space EEG windows, ENABLE_EEG_CHANNEL_SPACE) the same EEG values --
    expected max |diff| = 0.0, since both arrays go through the same
    per-session normalisation."""
    tdir = os.path.join(tsfnet_root, subject_id)
    res = {"tsfnet_dir": tdir}
    try:
        for name in ("labels", "session_ids", "trial_ids"):
            a = np.load(os.path.join(out_dir, f"{subject_id}_{name}.npy"))
            b = np.load(os.path.join(tdir, f"{subject_id}_{name}.npy"))
            res[name] = bool(a.shape == b.shape and np.array_equal(a, b))
        ch_path = os.path.join(tdir, f"{subject_id}_eeg_channel_windows.npy")
        if os.path.exists(ch_path):
            a = np.load(os.path.join(out_dir, f"{subject_id}_mgf_eeg.npy"), mmap_mode="r")
            b = np.load(ch_path, mmap_mode="r")
            res["eeg_values_max_abs_diff"] = (float(np.max(np.abs(np.asarray(a[:50]) - np.asarray(b[:50]))))
                                             if a.shape == b.shape else f"shape {a.shape} vs {b.shape}")
        res["ok"] = all(res[k] for k in ("labels", "session_ids", "trial_ids"))
    except Exception as e:
        res.update({"ok": False, "error": str(e)})
    log(f"  {subject_id}: check vs TSFNet output -> {res}")
    return res


# ============================================================================
# PER-SUBJECT DRIVER
# ============================================================================
def process_subject(subject_num, check_root=None):
    subject_id = f"VP{subject_num:03d}"
    t0 = time.time()
    log(f"===== START {subject_id} (MGFNet input, TSFNet-matched preprocessing) =====")
    out_dir = os.path.join(OUTPUT_PATH, subject_id)
    qc_dir = os.path.join(out_dir, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    raw, events, sfreq, _mnt = T.load_eeg_data(os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG"))
    # EXACT TSFNet EEG chain (filter -> PyPREP -> CAR -> ICA/ICLabel/overrides -> z-score).
    filtered = T.preprocess_eeg(raw, subject_id, qc_dir if SAVE_QC else os.path.join(out_dir, "_tmp"))
    eeg_names = [filtered.ch_names[i] for i in T.mne.pick_types(filtered.info, eeg=True, eog=False)]
    eeg_full = filtered.get_data(picks="eeg")                       # (28, n_times), EOG excluded

    fnirs_ch, ev_t, ev_c, nirs_names = preprocess_fnirs_channel_space(subject_id)
    extract_windows_channel_space.last_sync_ok = False
    E, O, R, lab, sess, trl, offs = extract_windows_channel_space(
        eeg_full, sfreq, events, fnirs_ch, ev_t, ev_c, subject_id)
    if E is None:
        log(f"  {subject_id}: no windows extracted, nothing saved.")
        return {"subject_id": subject_id, "status": "no_windows"}
    sync_ok = bool(extract_windows_channel_space.last_sync_ok)

    np.save(os.path.join(out_dir, f"{subject_id}_mgf_eeg.npy"), E)
    np.save(os.path.join(out_dir, f"{subject_id}_mgf_hbo.npy"), O)
    np.save(os.path.join(out_dir, f"{subject_id}_mgf_hbr.npy"), R)
    np.save(os.path.join(out_dir, f"{subject_id}_labels.npy"), lab)
    np.save(os.path.join(out_dir, f"{subject_id}_session_ids.npy"), sess)
    np.save(os.path.join(out_dir, f"{subject_id}_trial_ids.npy"), trl)
    np.save(os.path.join(out_dir, f"{subject_id}_window_offsets.npy"), offs)

    meta = {
        "subject_id": subject_id,
        "pipeline": "MGFNet channel-space input, preprocessing identical to Preproc_tsfnet_nback_badch.py",
        "n_windows": int(E.shape[0]),
        "eeg_shape": list(E.shape), "hbo_shape": list(O.shape), "hbr_shape": list(R.shape),
        "eeg_channels": eeg_names, "eeg_includes_eog": False, "fnirs_channels": nirs_names,
        "windows_per_session": {int(s): int((sess == s).sum()) for s in np.unique(sess)},
        "windows_per_class": {int(c): int((lab == c).sum()) for c in np.unique(lab)},
        "window_len_s": T.WINDOW_LEN_EEG, "window_step_s": T.WINDOW_STEP,
        "slide_s": [T.SLIDE_TMIN, T.SLIDE_TMAX],
        "fnirs_window": f"same {T.WINDOW_LEN_EEG:g} s as EEG (TSFNet: {T.FNIRS_NUM_SEGMENTS} x "
                        f"{T.WINDOW_LEN_FNIRS_SEGMENT:g} s segments)",
        "window_kept_iff_tsfnet_keeps_it": True,
        "eeg_bandpass_hz": [T.EEG_BP_LOW, T.EEG_BP_HIGH], "eeg_bandpass_order": T.EEG_BP_ORDER,
        "eeg_notch": T.APPLY_NOTCH, "eeg_steps": "bandpass -> PyPREP -> CAR -> ICA(EOG corr + ICLabel "
                                                 "+ manual overrides) -> per-channel z-score",
        "fnirs_bandpass_enabled": T.ENABLE_FNIRS_BANDPASS,
        "baseline_window_s": [T.BASELINE_TMIN, T.BASELINE_TMAX],
        "artifact_rejection": "window dropped if max |z| > 8 (EEG)",
        "per_session_norm": T.ENABLE_PER_SESSION_NORM,
        "fnirs_clock_sync": sync_ok,
        "label_map": {"0-back": 0, "2-back": 1, "3-back": 2},
    }
    if check_root:
        meta["tsfnet_check"] = check_against_tsfnet(subject_id, out_dir, check_root)
    with open(os.path.join(out_dir, f"{subject_id}_mgf_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"DONE {subject_id} in {(time.time() - t0) / 60:.1f} min | EEG {E.shape} | HbO {O.shape} | "
        f"labels {np.bincount(lab).tolist()} | sessions {meta['windows_per_session']}")
    return {"status": "ok", **meta}


def main():
    global BASE_DATASET_PATH, OUTPUT_PATH, SUBJECTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default=None, help="e.g. 1 or 1,2,3")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--check-against", type=str, nargs="?", const=TSFNET_OUTPUT_PATH, default=None,
                    help="TSFNet output root (folder holding VPxxx/): verify the windows match "
                         "1:1. Without a value, uses the TSFNet script's OUTPUT_PATH.")
    a = ap.parse_args()
    if a.data_root:
        BASE_DATASET_PATH = a.data_root
        T.BASE_DATASET_PATH = a.data_root        # T.load_fnirs_data reads its own global
    if a.out:
        OUTPUT_PATH = a.out
    if a.subjects:
        SUBJECTS = [int(s) for s in a.subjects.split(",")]
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    T.LOG_PATH = os.path.join(OUTPUT_PATH, "preprocessing_log.txt")   # don't write into TSFNet's log

    log(f"MGFNet TSFNet-matched preprocessing | subjects={SUBJECTS} | out={OUTPUT_PATH}")
    log(f"EEG {T.EEG_BP_LOW}-{T.EEG_BP_HIGH} Hz order {T.EEG_BP_ORDER} + PyPREP + CAR + ICA | "
        f"fNIRS bandpass={T.ENABLE_FNIRS_BANDPASS}, baseline [{T.BASELINE_TMIN},{T.BASELINE_TMAX}] s | "
        f"window {T.WINDOW_LEN_EEG} s step {T.WINDOW_STEP} s slide [{T.SLIDE_TMIN},{T.SLIDE_TMAX}] s | "
        f"per-session norm={T.ENABLE_PER_SESSION_NORM}")
    results = []
    for n in SUBJECTS:
        try:
            results.append(process_subject(n, a.check_against))
        except Exception:
            log(f"  VP{n:03d}: FAILED\n{traceback.format_exc()}")
            results.append({"subject_id": f"VP{n:03d}", "status": "failed"})
    ok = [r for r in results if r.get("status") == "ok"]
    log(f"===== DONE: {len(ok)}/{len(results)} subjects =====")
    bad_sync = [r["subject_id"] for r in ok if not r["fnirs_clock_sync"]]
    if bad_sync:
        log(f"*** fNIRS clock NOT synced for: {bad_sync}")
    if a.check_against:
        mism = [r["subject_id"] for r in ok if not r.get("tsfnet_check", {}).get("ok")]
        log(f"TSFNet window check: {len(ok) - len(mism)}/{len(ok)} match" +
            (f"; MISMATCH: {mism}" if mism else ""))
    with open(os.path.join(OUTPUT_PATH, "preprocessing_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
