# ============================================================================
# check_grid_consistency.py
#
# DIAGNOSTIC SCRIPT -- does NOT touch or import the main preprocessing pipeline.
# Purpose: verify whether the azimuthal-equidistant projection's (min_x, max_x,
# min_y, max_y) extent -- which spatial_mapping() in the main pipeline uses to
# build the 16x16 np.mgrid -- is effectively IDENTICAL across subjects, for
# both the EEG montage (mnt_nback.mat under VPxxx-EEG) and the fNIRS montage
# (mnt_nback.mat under VPxxx-NIRS).
#
# If extents differ meaningfully between subjects, grid pixel (i, j) does NOT
# correspond to the same scalp/optode location across subjects, which quietly
# undermines the spatial-convolution assumption of the Conv3d branches.
#
# HOW TO USE:
#   1. Edit BASE_DATASET_PATH and SUBJECT_IDS below to match your setup.
#   2. Run: python check_grid_consistency.py
#   3. Read the printed table + saved plot (grid_consistency_report.png) and
#      the saved grid_consistency_report.csv.
#
# This script changes NOTHING in your training/preprocessing pipeline -- it
# only reads mnt_nback.mat files and reports numbers.
# ============================================================================

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIGURATION -- EDIT THESE TWO LINES
# ----------------------------------------------------------------------------
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"
SUBJECT_IDS = [f"VP{i:03d}" for i in range(1, 27)]   # VP001..VP026; trim if you only want to test a few first

# Set to True to also require an exact channel-NAME-order check (extra strictness)
CHECK_CHANNEL_ORDER = True


# ----------------------------------------------------------------------------
# Reproduces EXACTLY the projection math used in spatial_mapping() of the main
# pipeline (azimuthal equidistant projection from pos_3d), so the extents
# reported here are the same extents your real pipeline would compute.
# ----------------------------------------------------------------------------
def project_azimuthal_equidistant(pos_3d):
    """pos_3d: shape (3, n_channels) -> returns x_coords, y_coords, each (n_channels,)"""
    x3d, y3d, z3d = pos_3d[0, :], pos_3d[1, :], pos_3d[2, :]
    r = np.sqrt(x3d**2 + y3d**2 + z3d**2)
    r_safe = np.where(r > 0, r, 1.0)
    theta = np.arccos(np.clip(z3d / r_safe, -1.0, 1.0))
    phi = np.arctan2(y3d, x3d)
    x_coords = theta * np.cos(phi)
    y_coords = theta * np.sin(phi)
    return x_coords, y_coords


def load_mnt(mnt_path):
    """Loads a mnt_nback.mat file the same way the main pipeline does
    (struct_as_record=False, squeeze_me=True), returns (pos_3d (3,N), clab list(N,))."""
    if not os.path.exists(mnt_path):
        return None, None
    mat = sio.loadmat(mnt_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat.keys() if not k.startswith('__')]
    struct = mat[real_keys[0]]
    pos_3d = np.asarray(struct.pos_3d, dtype=np.float64)   # (3, N)
    clab = [str(c).strip() for c in np.asarray(struct.clab).flatten()]
    return pos_3d, clab


def analyze_montage(base_path, subject_ids, modality):
    """modality: 'EEG' or 'NIRS'. Returns a DataFrame with one row per subject."""
    rows = []
    reference_clab = None
    for sid in subject_ids:
        folder = os.path.join(base_path, f"{sid}-{modality}")
        mnt_path = os.path.join(folder, 'mnt_nback.mat')
        pos_3d, clab = load_mnt(mnt_path)
        if pos_3d is None:
            print(f"  [{modality}] {sid}: mnt_nback.mat not found at {mnt_path} -- skipping")
            continue

        x_coords, y_coords = project_azimuthal_equidistant(pos_3d)

        row = {
            'subject': sid,
            'modality': modality,
            'n_channels': pos_3d.shape[1],
            'min_x': x_coords.min(),
            'max_x': x_coords.max(),
            'min_y': y_coords.min(),
            'max_y': y_coords.max(),
            'range_x': x_coords.max() - x_coords.min(),
            'range_y': y_coords.max() - y_coords.min(),
            'clab_order_matches_first': None,  # filled below
        }

        if reference_clab is None:
            reference_clab = clab
            row['clab_order_matches_first'] = True
        else:
            row['clab_order_matches_first'] = (clab == reference_clab)

        rows.append(row)

    return pd.DataFrame(rows)


def report(df, modality):
    if df.empty:
        print(f"No data found for {modality}.")
        return

    print(f"\n{'='*70}")
    print(f"  {modality} montage -- grid extent summary across {len(df)} subjects")
    print(f"{'='*70}")
    print(df[['subject', 'n_channels', 'min_x', 'max_x', 'min_y', 'max_y',
              'range_x', 'range_y', 'clab_order_matches_first']].to_string(index=False))

    for col in ['min_x', 'max_x', 'min_y', 'max_y']:
        spread = df[col].max() - df[col].min()
        pct_of_typical_range = 100 * spread / df['range_x'].mean() if df['range_x'].mean() != 0 else float('nan')
        flag = "  <-- POTENTIAL PROBLEM" if pct_of_typical_range > 2.0 else ""
        print(f"  spread of {col} across subjects: {spread:.6f} "
              f"({pct_of_typical_range:.2f}% of mean grid range){flag}")

    n_chan_unique = df['n_channels'].nunique()
    if n_chan_unique > 1:
        print(f"  WARNING: n_channels is NOT constant across subjects "
              f"({sorted(df['n_channels'].unique())}) -- grid input size differs per subject.")
    else:
        print(f"  n_channels constant across subjects: {df['n_channels'].iloc[0]}")

    if CHECK_CHANNEL_ORDER and not df['clab_order_matches_first'].all():
        bad = df.loc[~df['clab_order_matches_first'], 'subject'].tolist()
        print(f"  WARNING: channel name ORDER in mnt_nback.mat's clab differs from the "
              f"first subject for: {bad}")
        print("  (spatial_mapping() indexes positions purely by array position, not by name, "
              "so a differing clab order means position[i] refers to a DIFFERENT physical "
              "channel for these subjects than for the reference subject.)")
    elif CHECK_CHANNEL_ORDER:
        print("  clab channel-name order is IDENTICAL across all subjects (good).")

    return df


def plot_extents(df_eeg, df_nirs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, df, title in [(axes[0], df_eeg, 'EEG'), (axes[1], df_nirs, 'fNIRS')]:
        if df.empty:
            ax.set_title(f"{title}: no data")
            continue
        for _, row in df.iterrows():
            ax.plot([row['min_x'], row['max_x'], row['max_x'], row['min_x'], row['min_x']],
                    [row['min_y'], row['min_y'], row['max_y'], row['max_y'], row['min_y']],
                    alpha=0.5, linewidth=1)
        ax.set_title(f"{title}: per-subject grid bounding boxes\n"
                     f"(should overlap almost exactly if grid is consistent)")
        ax.set_xlabel("x (azimuthal-equidistant)")
        ax.set_ylabel("y (azimuthal-equidistant)")
        ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved visual comparison to: {out_path}")


if __name__ == "__main__":
    print("Checking EEG montage (VPxxx-EEG/mnt_nback.mat) across subjects...")
    df_eeg = analyze_montage(BASE_DATASET_PATH, SUBJECT_IDS, 'EEG')
    report(df_eeg, 'EEG')

    print("\nChecking fNIRS montage (VPxxx-NIRS/mnt_nback.mat) across subjects...")
    df_nirs = analyze_montage(BASE_DATASET_PATH, SUBJECT_IDS, 'NIRS')
    report(df_nirs, 'fNIRS')

    out_dir = os.path.dirname(os.path.abspath(__file__))
    df_all = pd.concat([df_eeg, df_nirs], ignore_index=True)
    csv_path = os.path.join(out_dir, "grid_consistency_report.csv")
    df_all.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved full numeric report to: {csv_path}")

    plot_extents(df_eeg, df_nirs, os.path.join(out_dir, "grid_consistency_report.png"))

    print("\n" + "="*70)
    print("HOW TO READ THIS:")
    print("- If 'spread ... % of mean grid range' is small (e.g. <1-2%) for all four")
    print("  extent columns, AND n_channels is constant, AND clab order matches --")
    print("  then the grid IS effectively consistent across subjects, and pixel (i,j)")
    print("  refers to (approximately) the same scalp/optode location for everyone.")
    print("  In that case the concern raised earlier is NOT a practical problem here.")
    print("- If spreads are large, or clab order differs, or n_channels differs --")
    print("  the grid is NOT consistent across subjects, and this is worth fixing")
    print("  (e.g. by computing one FIXED global grid_x/grid_y from a reference")
    print("  montage or from the union/mean of all subjects' positions, then reusing")
    print("  those same fixed bounds for every subject's griddata() call).")
    print("="*70)
