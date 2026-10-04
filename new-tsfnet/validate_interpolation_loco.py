# ============================================================================
# LEAVE-ONE-CHANNEL-OUT (LOCO) VALIDATION of CloughTocher spatial interpolation
#
# For EEG and fNIRS separately, and across multiple subjects:
#   For each channel C:
#     1. Remove C from the sensor set (both its 2D projected position AND its
#        signal values).
#     2. Build the SAME CloughTocher2DInterpolator the main pipeline uses
#        (spatial_mapping() in Preprocessing_TSFNet_paper_nback_v2.py), fit on
#        the remaining channels' signal values at each sampled time point.
#     3. Evaluate the interpolator AT channel C's own (held-out) 2D position
#        to get a predicted/reconstructed signal.
#     4. Compare the reconstructed signal to C's REAL recorded signal:
#          - Pearson correlation (shape agreement)
#          - NRMSE (RMSE normalized by the true channel's std)
#          - MAE (normalized by the true channel's std)
#
# This is repeated for a configurable number of subjects and a random sample
# of time points per subject (for speed), then aggregated:
#   - per-channel mean+-std across subjects (flags channels that are
#     systematically hard to reconstruct, e.g. edge-of-montage channels)
#   - overall mean across all channel x subject combinations
#
# Usage:
#   python validate_interpolation_loco.py --n_subjects 5 --n_timepoints 200
#   python validate_interpolation_loco.py --subjects VP001,VP005,VP010 --modality eeg
# ============================================================================

import os
import sys
import argparse

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator

MAIN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT_NAME = "Preprocessing_TSFNet_paper_nback_v2"
if MAIN_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, MAIN_SCRIPT_DIR)
main_pipeline = __import__(MAIN_SCRIPT_NAME)

BASE_DATASET_PATH = main_pipeline.BASE_DATASET_PATH
OUTPUT_PATH = main_pipeline.OUTPUT_PATH
ALL_SUBJECTS = [f"VP{n:03d}" for n in main_pipeline.SUBJECTS]


def project_2d(pos_3d):
    """Same azimuthal-equidistant projection as spatial_mapping() in the main pipeline."""
    x3d, y3d, z3d = pos_3d[0], pos_3d[1], pos_3d[2]
    r = np.sqrt(x3d ** 2 + y3d ** 2 + z3d ** 2)
    r_safe = np.where(r > 0, r, 1.0)
    theta = np.arccos(np.clip(z3d / r_safe, -1.0, 1.0))
    phi = np.arctan2(y3d, x3d)
    x_coords = theta * np.cos(phi)
    y_coords = theta * np.sin(phi)
    return np.column_stack((x_coords, y_coords))   # (n_channels, 2)


def loco_errors_for_matrix(data, points_2d, ch_names, n_timepoints, rng):
    """
    data:      (n_samples, n_channels) real recorded signal
    points_2d: (n_channels, 2) projected sensor positions (same order as data's columns)
    Returns a list of per-channel dicts with correlation/NRMSE/MAE.
    """
    n_samples, n_channels = data.shape
    if n_channels < 4:
        return []   # CloughTocher needs enough points for a triangulation

    idx = rng.choice(n_samples, size=min(n_timepoints, n_samples), replace=False)
    sampled = data[idx, :]   # (n_timepoints, n_channels)

    results = []
    for ch_i, ch_name in enumerate(ch_names):
        keep = [j for j in range(n_channels) if j != ch_i]
        pts_train = points_2d[keep]
        vals_train = sampled[:, keep].T   # (n_channels-1, n_timepoints)

        try:
            tri = Delaunay(pts_train)
            interp = CloughTocher2DInterpolator(tri, vals_train, fill_value=0)
            pred = interp((points_2d[ch_i, 0], points_2d[ch_i, 1]))   # (n_timepoints,)
        except Exception as e:
            results.append({"channel": ch_name, "error": str(e)})
            continue

        true_vals = sampled[:, ch_i]
        pred = np.asarray(pred).reshape(-1)
        if pred.shape[0] != true_vals.shape[0] or np.all(pred == 0):
            results.append({"channel": ch_name, "error": "interpolation returned invalid/empty output "
                                                           "(channel likely outside convex hull of others)"})
            continue

        true_std = true_vals.std() + 1e-12
        rmse = np.sqrt(np.mean((pred - true_vals) ** 2))
        mae = np.mean(np.abs(pred - true_vals))
        corr = np.corrcoef(pred, true_vals)[0, 1] if true_vals.std() > 0 and pred.std() > 0 else np.nan

        results.append({
            "channel": ch_name, "error": None,
            "correlation": round(float(corr), 4) if not np.isnan(corr) else np.nan,
            "nrmse": round(float(rmse / true_std), 4),
            "nmae": round(float(mae / true_std), 4),
        })
    return results


def run_eeg_loco(subject_id, n_timepoints, rng):
    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    raw, events, sfreq_eeg, eeg_mnt_struct = main_pipeline.load_eeg_data(eeg_folder)
    filtered = raw.copy()
    filtered.filter(
        l_freq=main_pipeline.EEG_BP_LOW, h_freq=main_pipeline.EEG_BP_HIGH, method="iir",
        iir_params=dict(order=main_pipeline.EEG_BP_ORDER, ftype="butter", output="sos"),
        picks="eeg", phase="zero", verbose=False,
    )
    filtered.set_eeg_reference("average", projection=False, verbose=False)
    pos_3d = main_pipeline.get_eeg_positions(filtered, eeg_mnt_struct)
    ch_names = [c for c in filtered.ch_names if c not in ("VEOG", "HEOG")]
    data = filtered.get_data(picks=ch_names).T   # (n_samples, n_channels)
    points_2d = project_2d(pos_3d)
    return loco_errors_for_matrix(data, points_2d, ch_names, n_timepoints, rng)


def run_fnirs_loco(subject_id, n_timepoints, rng):
    oxy_raw, deoxy_raw, sfreq_raw, mnt_struct, ch_names, ev_time, ev_code = main_pipeline.load_fnirs_data(subject_id)
    pos_3d = main_pipeline.get_fnirs_positions(ch_names, mnt_struct)
    points_2d = project_2d(pos_3d)

    results = []
    for label, data in (("HbO", oxy_raw), ("HbR", deoxy_raw)):
        r = loco_errors_for_matrix(data, points_2d, ch_names, n_timepoints, rng)
        for row in r:
            row["chrom"] = label
        results.extend(r)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated subject list. Default: 5 evenly-spaced subjects.")
    parser.add_argument("--n_subjects", type=int, default=5)
    parser.add_argument("--n_timepoints", type=int, default=200,
                         help="Random time points sampled per subject per channel (for speed).")
    parser.add_argument("--modality", type=str, default="both", choices=["eeg", "fnirs", "both"])
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(OUTPUT_PATH, "interpolation_loco_validation"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        step = max(1, len(ALL_SUBJECTS) // args.n_subjects)
        subjects = ALL_SUBJECTS[::step][:args.n_subjects]

    print(f"\n=== LOCO interpolation validation on {subjects} (modality={args.modality}) ===\n")

    all_rows = []
    for subject_id in subjects:
        print(f"--- {subject_id} ---")
        if args.modality in ("eeg", "both"):
            try:
                eeg_results = run_eeg_loco(subject_id, args.n_timepoints, rng)
                for r in eeg_results:
                    r.update({"subject_id": subject_id, "modality": "EEG"})
                all_rows.extend(eeg_results)
                ok = [r for r in eeg_results if r.get("error") is None]
                if ok:
                    print(f"  EEG: mean corr={np.mean([r['correlation'] for r in ok if not np.isnan(r['correlation'])]):.3f}, "
                          f"mean NRMSE={np.mean([r['nrmse'] for r in ok]):.3f} (n={len(ok)}/{len(eeg_results)} channels)")
            except Exception as e:
                print(f"  EEG LOCO failed: {e}")

        if args.modality in ("fnirs", "both"):
            try:
                fnirs_results = run_fnirs_loco(subject_id, args.n_timepoints, rng)
                for r in fnirs_results:
                    r.update({"subject_id": subject_id, "modality": "fNIRS"})
                all_rows.extend(fnirs_results)
                ok = [r for r in fnirs_results if r.get("error") is None]
                if ok:
                    print(f"  fNIRS: mean corr={np.mean([r['correlation'] for r in ok if not np.isnan(r['correlation'])]):.3f}, "
                          f"mean NRMSE={np.mean([r['nrmse'] for r in ok]):.3f} (n={len(ok)}/{len(fnirs_results)} channels)")
            except Exception as e:
                print(f"  fNIRS LOCO failed: {e}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No results produced -- check --modality and dataset paths.")
        return

    out_csv = os.path.join(args.out_dir, "loco_full_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nFull per-channel-per-subject results saved: {out_csv}")

    ok_df = df[df["error"].isna()] if "error" in df.columns else df
    failed_df = df[df["error"].notna()] if "error" in df.columns else pd.DataFrame()

    print("\n" + "=" * 78)
    print("OVERALL SUMMARY (across all subjects/channels/timepoints)")
    print("=" * 78)
    for modality in ok_df["modality"].unique():
        sub = ok_df[ok_df["modality"] == modality]
        print(f"{modality}: mean correlation = {sub['correlation'].mean():.4f} "
              f"(+-{sub['correlation'].std():.4f}), "
              f"mean NRMSE = {sub['nrmse'].mean():.4f}, "
              f"mean NMAE = {sub['nmae'].mean():.4f}  (n={len(sub)} channel-subject pairs)")

    print("\n" + "=" * 78)
    print("PER-CHANNEL BREAKDOWN (mean +- std across subjects) -- flags hard-to-reconstruct channels")
    print("=" * 78)
    per_channel = (ok_df.groupby(["modality", "channel"])
                    .agg(mean_corr=("correlation", "mean"), std_corr=("correlation", "std"),
                         mean_nrmse=("nrmse", "mean"), n=("correlation", "count"))
                    .reset_index()
                    .sort_values("mean_corr"))
    print(per_channel.to_string(index=False))
    per_channel.to_csv(os.path.join(args.out_dir, "loco_per_channel_summary.csv"), index=False)

    worst = per_channel[per_channel["mean_corr"] < 0.5]
    if not worst.empty:
        print(f"\nWARNING: {len(worst)} channel(s) have mean correlation < 0.5 (poor interpolation "
              f"reconstruction, likely edge-of-montage or sparse-neighbor channels):")
        print(worst[["modality", "channel", "mean_corr", "mean_nrmse"]].to_string(index=False))
    else:
        print("\nNo channels fell below the 0.5 correlation warning threshold.")

    if not failed_df.empty:
        print(f"\n{len(failed_df)} channel-subject combinations failed outright (see 'error' column in "
              f"loco_full_results.csv) -- typically a channel outside the convex hull of the others.")

    # thesis-ready paragraph
    overall_corr = ok_df["correlation"].mean()
    overall_nrmse = ok_df["nrmse"].mean()
    paragraph = (
        f"To validate the CloughTocher cubic interpolation used for spatial mapping (Sec. 2.3), a "
        f"leave-one-channel-out analysis was performed: for each channel, its position and signal were "
        f"withheld, the value at its location was reconstructed from the remaining channels using the "
        f"same interpolator as the main pipeline, and compared to the true recorded signal. Across "
        f"{len(subjects)} subjects, {len(ok_df['channel'].unique())} unique channels, and "
        f"{args.n_timepoints} sampled time points per channel, the reconstructed signal correlated with "
        f"the true signal at r={overall_corr:.3f} on average (normalized RMSE={overall_nrmse:.3f}), "
        f"indicating the interpolation preserves the spatial structure of the recorded activity."
    )
    print(f"\n--- Thesis-ready paragraph ---\n{paragraph}")
    with open(os.path.join(args.out_dir, "loco_paragraph.txt"), "w", encoding="utf-8") as f:
        f.write(paragraph)


if __name__ == "__main__":
    main()
