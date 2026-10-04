"""
MSTFDN -- EEG/fNIRS Sync Verification Script
==============================================
Re-checks (empirically, on the real .mat files) the claim that EEG and
fNIRS device clocks drift by ~46-93s across sessions for the N-back
(Shin et al. 2018) dataset, and that trial markers must be read from each
modality's OWN clock rather than derived from the other's.

For every subject and every trial, this script computes:

  delta_i = eeg_trial_onset_time_i  -  fnirs_trial_onset_time_i

where both times are measured from each device's own marker stream
(EEG markers 112/128/144, fNIRS markers 7/8/9), independently.

If EEG and fNIRS clocks were perfectly aligned (delta constant, e.g. 0s
if both start recording at the same instant), delta_i should be roughly
CONSTANT across all trials of a session. What the earlier note claims is:

  - delta jumps by ~46-93s somewhere across sessions (a session-to-session
    clock-start offset, not necessarily a within-session drift)
  - trial CLASS ORDER (0-back/2-back/3-back sequence) must still match
    between EEG-marker-order and fNIRS-marker-order 1:1, even if the
    absolute time delta differs

This script checks BOTH things directly and reports, per subject:
  - the class-order match (sync_ok, same check used in preprocessing)
  - min/max/mean of delta_i within each session
  - the jump in delta_i between consecutive sessions
  - a saved plot of delta_i vs. trial index (visual confirmation)

Usage:
    Set BASE_DATASET_PATH below, then run:
        python check_eeg_fnirs_sync.py
    or in Jupyter:
        %run check_eeg_fnirs_sync.py
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
#  CONFIGURATION -- تنظیم کن قبل از اجرا
# ============================================================
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"
OUTPUT_PATH        = r"E:\kntu\Thesis\codes\Results\SYNC_CHECK"

SFREQ_EEG = 200.0   # Hz, after any resampling step (same as preprocessing)

EEG_TRIAL_LABEL_MAP   = {112: 0, 128: 1, 144: 2}   # 0-back, 2-back, 3-back
FNIRS_TRIAL_LABEL_MAP = {7: 0,   8: 1,   9: 2}
TRIALS_PER_SESSION = 9

ALL_SUBJECTS = [f"VP{i:03d}" for i in range(1, 27)]
# برای تست سریع فقط چند subject:
# ALL_SUBJECTS = ["VP001", "VP002", "VP003"]


# ============================================================
#  LOADERS (minimal, matching MSTFDN_preprocessing_v6.py logic)
# ============================================================
def load_eeg_trial_times(eeg_folder):
    """
    Returns (times_sec, classes) for EEG markers 112/128/144, in trial order.
    """
    cnt_path = os.path.join(eeg_folder, 'cnt_nback.mat')
    mrk_path = os.path.join(eeg_folder, 'mrk_nback.mat')

    mat_cnt = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
    cnt_key = [k for k in mat_cnt if not k.startswith('__')][0]
    cnt     = mat_cnt[cnt_key]
    sfreq   = float(cnt.fs)

    mat_mrk = sio.loadmat(mrk_path, struct_as_record=False, squeeze_me=True)
    mrk_key = [k for k in mat_mrk if not k.startswith('__')][0]
    mrk     = mat_mrk[mrk_key]

    if hasattr(mrk, 'time'):
        ev_pos = np.round(mrk.time * sfreq / 1000).astype(int)
    elif hasattr(mrk, 'pos'):
        ev_pos = np.array(mrk.pos, dtype=int) - 1
    else:
        raise ValueError("No 'time'/'pos' in EEG marker struct")

    if hasattr(mrk, 'event') and hasattr(mrk.event, 'desc'):
        desc = np.asarray(mrk.event.desc)
        if desc.ndim > 1:
            desc = desc.squeeze()
        ev_codes = desc.flatten().astype(int)
    else:
        ev_codes = ev_pos.copy()

    valid = np.isin(ev_codes, list(EEG_TRIAL_LABEL_MAP.keys()))
    pos_valid   = ev_pos[valid]
    codes_valid = ev_codes[valid]

    order = np.argsort(pos_valid)
    times_sec = pos_valid[order] / sfreq   # EEG's own clock, seconds since recording start
    classes   = np.array([EEG_TRIAL_LABEL_MAP[c] for c in codes_valid[order]])

    return times_sec, classes


def load_fnirs_trial_times(fnirs_folder):
    """
    Returns (times_sec, classes) for fNIRS markers 7/8/9, in trial order.
    """
    mrk_path = os.path.join(fnirs_folder, 'mrk_nback.mat')
    if not os.path.exists(mrk_path):
        return None, None

    mat_fmrk = sio.loadmat(mrk_path, struct_as_record=False, squeeze_me=True)
    fmrk_key = [k for k in mat_fmrk if not k.startswith('__')][0]
    fmrk     = mat_fmrk[fmrk_key]

    if not (hasattr(fmrk, 'time') and hasattr(fmrk, 'event') and hasattr(fmrk.event, 'desc')):
        return None, None

    times_ms = np.asarray(fmrk.time).flatten().astype(float)
    codes    = np.asarray(fmrk.event.desc).flatten()

    valid = np.isin(codes, list(FNIRS_TRIAL_LABEL_MAP.keys()))
    times_valid = times_ms[valid]
    codes_valid = codes[valid]

    order = np.argsort(times_valid)
    times_sec = times_valid[order] / 1000.0   # fNIRS's own clock, seconds since recording start
    classes   = np.array([FNIRS_TRIAL_LABEL_MAP[c] for c in codes_valid[order]])

    return times_sec, classes


# ============================================================
#  PER-SUBJECT CHECK
# ============================================================
def check_subject(subject_id):
    eeg_folder   = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    fnirs_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-NIRS")

    if not os.path.exists(eeg_folder) or not os.path.exists(fnirs_folder):
        print(f"[{subject_id}] SKIPPED -- folder not found")
        return None

    eeg_times, eeg_classes     = load_eeg_trial_times(eeg_folder)
    fnirs_times, fnirs_classes = load_fnirs_trial_times(fnirs_folder)

    if fnirs_times is None:
        print(f"[{subject_id}] SKIPPED -- no fNIRS markers found")
        return None

    n = min(len(eeg_times), len(fnirs_times))
    if len(eeg_times) != len(fnirs_times):
        print(f"[{subject_id}] WARNING: trial count mismatch "
              f"(EEG={len(eeg_times)}, fNIRS={len(fnirs_times)}) -- "
              f"comparing first {n} common trials only")

    class_order_matches = bool(np.array_equal(eeg_classes[:n], fnirs_classes[:n]))

    delta = eeg_times[:n] - fnirs_times[:n]   # seconds, per trial

    # Session boundaries (every TRIALS_PER_SESSION trials = 1 session)
    session_id = np.arange(n) // TRIALS_PER_SESSION + 1
    n_sessions = int(session_id.max()) if n > 0 else 0

    session_stats = []
    for s in range(1, n_sessions + 1):
        mask = session_id == s
        if mask.sum() == 0:
            continue
        d = delta[mask]
        session_stats.append({
            'session': s,
            'n_trials': int(mask.sum()),
            'delta_min': float(d.min()),
            'delta_max': float(d.max()),
            'delta_mean': float(d.mean()),
            'delta_range': float(d.max() - d.min()),   # within-session drift
        })

    # Jump between consecutive sessions (mean delta of session s+1 minus session s)
    session_jumps = []
    for i in range(len(session_stats) - 1):
        jump = session_stats[i + 1]['delta_mean'] - session_stats[i]['delta_mean']
        session_jumps.append({
            'from_session': session_stats[i]['session'],
            'to_session':   session_stats[i + 1]['session'],
            'jump_sec':     jump,
        })

    result = {
        'subject_id': subject_id,
        'n_trials_common': n,
        'class_order_matches': class_order_matches,
        'session_stats': session_stats,
        'session_jumps': session_jumps,
        'delta_all': delta,
    }

    # --- console report ---
    tag = "OK" if class_order_matches else "MISMATCH"
    print(f"\n[{subject_id}] class order match: {tag}  ({n} common trials)")
    for st in session_stats:
        print(f"    Session {st['session']}: delta range "
              f"[{st['delta_min']:.2f}, {st['delta_max']:.2f}]s "
              f"(mean={st['delta_mean']:.2f}s, within-session spread={st['delta_range']:.2f}s)")
    for j in session_jumps:
        flag = " <-- large jump (matches 46-93s claim)" if 40 <= abs(j['jump_sec']) <= 100 else ""
        print(f"    Jump S{j['from_session']}->S{j['to_session']}: "
              f"{j['jump_sec']:+.2f}s{flag}")

    return result


def plot_subject_delta(result, output_dir):
    subject_id = result['subject_id']
    delta = result['delta_all']
    n = len(delta)
    trial_idx = np.arange(n)
    session_id = trial_idx // TRIALS_PER_SESSION + 1

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, session_id.max()))
    for s in range(1, session_id.max() + 1):
        mask = session_id == s
        ax.plot(trial_idx[mask], delta[mask], 'o-', color=colors[s - 1],
                label=f"Session {s}", markersize=5)

    ax.set_xlabel("Trial index (across all sessions)")
    ax.set_ylabel("delta = EEG_trial_time - fNIRS_trial_time (s)")
    ax.set_title(f"{subject_id} -- EEG vs fNIRS clock offset per trial\n"
                 f"(jumps between session colors = cross-device clock drift)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{subject_id}_sync_delta.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ============================================================
#  MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    all_results = []

    print("=" * 60)
    print("  EEG-fNIRS Sync Verification -- N-back dataset")
    print(f"  Checking claim: session-to-session clock jump ~46-93s")
    print("=" * 60)

    for subject_id in ALL_SUBJECTS:
        try:
            result = check_subject(subject_id)
            if result is not None:
                all_results.append(result)
                plot_subject_delta(result, OUTPUT_PATH)
        except Exception as e:
            print(f"[{subject_id}] ERROR: {e}")

    # ---- summary across all subjects ----
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    n_mismatch = sum(1 for r in all_results if not r['class_order_matches'])
    print(f"Subjects checked: {len(all_results)}")
    print(f"Subjects with class-order MISMATCH: {n_mismatch}")

    all_jumps = []
    for r in all_results:
        for j in r['session_jumps']:
            all_jumps.append(j['jump_sec'])

    if all_jumps:
        all_jumps = np.array(all_jumps)
        print(f"\nSession-to-session jump statistics (all subjects, all session pairs):")
        print(f"  min    = {all_jumps.min():.2f}s")
        print(f"  max    = {all_jumps.max():.2f}s")
        print(f"  mean   = {all_jumps.mean():.2f}s")
        print(f"  median = {np.median(all_jumps):.2f}s")

        in_claimed_range = np.sum((np.abs(all_jumps) >= 40) & (np.abs(all_jumps) <= 100))
        print(f"\n  Jumps within [40,100]s (matches the '46-93s' claim range): "
              f"{in_claimed_range}/{len(all_jumps)}")

    # ---- save a combined CSV ----
    import csv
    csv_path = os.path.join(OUTPUT_PATH, "sync_check_summary.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['subject_id', 'class_order_matches', 'session',
                         'n_trials', 'delta_min', 'delta_max', 'delta_mean',
                         'delta_range_within_session'])
        for r in all_results:
            for st in r['session_stats']:
                writer.writerow([r['subject_id'], r['class_order_matches'],
                                 st['session'], st['n_trials'],
                                 f"{st['delta_min']:.3f}", f"{st['delta_max']:.3f}",
                                 f"{st['delta_mean']:.3f}", f"{st['delta_range']:.3f}"])
    print(f"\nSaved: {csv_path}")
    print(f"Saved per-subject plots in: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
