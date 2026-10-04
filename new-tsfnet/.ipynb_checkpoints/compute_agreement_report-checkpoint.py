# ============================================================================
# ICLabel <-> Correlation AGREEMENT RATE REPORT
#
# Uses ONLY the data already produced by qc_eog_fast_full_check.py
# (full_check_summary.csv) -- no re-fitting of ICA, no re-running QC. Computes,
# per subject AND aggregated across all subjects, the standard agreement
# statistics between the two independent ocular-IC detectors used in the
# pipeline (correlation-based find_bads_eog vs. ICLabel eye-blink/eye-movement
# classification):
#
#   For each subject, every IC falls into exactly one of 4 categories:
#     BOTH        = flagged by correlation AND ICLabel        (n_both)
#     CORR_ONLY   = flagged by correlation only                (n_corr_only)
#     ICLABEL_ONLY= flagged by ICLabel only                     (n_iclabel_only)
#     NEITHER     = flagged by neither                          (n_ics - the above 3)
#
#   From this 2x2 contingency table we compute:
#     - Percent agreement   = (BOTH + NEITHER) / n_ics
#     - Jaccard index       = BOTH / (BOTH + CORR_ONLY + ICLABEL_ONLY)
#                             (agreement restricted to the "positive" / flagged class,
#                              standard choice when the negative class -- non-ocular
#                              ICs -- vastly outnumbers the positive class)
#     - Cohen's kappa       = chance-corrected agreement (accounts for the fact that
#                              with few excluded ICs per subject, high raw agreement
#                              can happen by chance alone)
#
#   Aggregation across all 26 subjects is reported TWO ways:
#     - Micro-average: pool raw IC counts across all subjects first, then compute
#       the statistics once on the pooled contingency table (weights subjects by
#       how many ICs they had -- a subject with 16 ICs counts more than one with 4).
#     - Macro-average: compute the statistic per subject first, then average those
#       per-subject numbers (each subject counts equally regardless of IC count).
#
# Usage:
#   python compute_agreement_report.py
#   python compute_agreement_report.py --csv "custom/path/full_check_summary.csv"
# ============================================================================

import os
import sys
import argparse

import numpy as np
import pandas as pd

MAIN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT_NAME = "Preprocessing_TSFNet_paper_nback_v2"
if MAIN_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, MAIN_SCRIPT_DIR)
try:
    main_pipeline = __import__(MAIN_SCRIPT_NAME)
    DEFAULT_OUTPUT_PATH = main_pipeline.OUTPUT_PATH
except Exception:
    DEFAULT_OUTPUT_PATH = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"

DEFAULT_CSV = os.path.join(DEFAULT_OUTPUT_PATH, "eog_fast_full_check", "full_check_summary.csv")


def cohens_kappa(both, corr_only, iclabel_only, neither):
    """Standard 2x2 Cohen's kappa from raw counts."""
    n = both + corr_only + iclabel_only + neither
    if n == 0:
        return np.nan
    po = (both + neither) / n   # observed agreement
    # expected agreement under independence: P(corr flags) * P(iclabel flags) + P(corr doesn't) * P(iclabel doesn't)
    p_corr_pos = (both + corr_only) / n
    p_iclabel_pos = (both + iclabel_only) / n
    pe = p_corr_pos * p_iclabel_pos + (1 - p_corr_pos) * (1 - p_iclabel_pos)
    if pe == 1:
        return 1.0 if po == 1 else np.nan
    return (po - pe) / (1 - pe)


def jaccard(both, corr_only, iclabel_only):
    denom = both + corr_only + iclabel_only
    return both / denom if denom > 0 else np.nan


def percent_agreement(both, corr_only, iclabel_only, neither):
    n = both + corr_only + iclabel_only + neither
    return (both + neither) / n if n > 0 else np.nan


def compute_subject_stats(row):
    n_ics = int(row["n_ics"])
    both = int(row["n_both"])
    corr_only = int(row["n_corr_only"])
    iclabel_only = int(row["n_iclabel_only"])
    neither = n_ics - both - corr_only - iclabel_only
    return {
        "subject_id": row["subject_id"],
        "n_ics": n_ics,
        "both": both,
        "corr_only": corr_only,
        "iclabel_only": iclabel_only,
        "neither": neither,
        "percent_agreement": round(percent_agreement(both, corr_only, iclabel_only, neither), 4),
        "jaccard": round(jaccard(both, corr_only, iclabel_only), 4) if (both + corr_only + iclabel_only) > 0 else np.nan,
        "cohens_kappa": round(cohens_kappa(both, corr_only, iclabel_only, neither), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                         help="Path to full_check_summary.csv from qc_eog_fast_full_check.py")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="Where to save the report (default: same folder as --csv)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found. Run qc_eog_fast_full_check.py first "
              f"(it produces full_check_summary.csv).")
        return

    out_dir = args.out_dir or os.path.dirname(args.csv)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    required_cols = {"subject_id", "n_ics", "n_both", "n_corr_only", "n_iclabel_only"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: {args.csv} is missing required column(s): {missing}. "
              f"Are you sure this is a full_check_summary.csv from qc_eog_fast_full_check.py?")
        return

    per_subject_rows = [compute_subject_stats(row) for _, row in df.iterrows()]
    per_subject_df = pd.DataFrame(per_subject_rows)

    # ----- micro-average: pool raw counts across all subjects, then compute once -----
    total_both = per_subject_df["both"].sum()
    total_corr_only = per_subject_df["corr_only"].sum()
    total_iclabel_only = per_subject_df["iclabel_only"].sum()
    total_neither = per_subject_df["neither"].sum()
    total_ics = per_subject_df["n_ics"].sum()

    micro_agreement = percent_agreement(total_both, total_corr_only, total_iclabel_only, total_neither)
    micro_jaccard = jaccard(total_both, total_corr_only, total_iclabel_only)
    micro_kappa = cohens_kappa(total_both, total_corr_only, total_iclabel_only, total_neither)

    # ----- macro-average: mean of per-subject stats (each subject weighted equally) -----
    macro_agreement = per_subject_df["percent_agreement"].mean()
    macro_jaccard = per_subject_df["jaccard"].mean(skipna=True)
    macro_kappa = per_subject_df["cohens_kappa"].mean(skipna=True)

    # ----- print report -----
    print("=" * 78)
    print("ICLabel <-> Correlation AGREEMENT RATE REPORT")
    print("=" * 78)
    print(f"Subjects analyzed: {len(per_subject_df)}")
    print(f"Total ICs across all subjects: {int(total_ics)}")
    print(f"  BOTH (agreed ocular):        {int(total_both)}")
    print(f"  CORR_ONLY (disagreement):    {int(total_corr_only)}")
    print(f"  ICLABEL_ONLY (disagreement): {int(total_iclabel_only)}")
    print(f"  NEITHER (agreed non-ocular): {int(total_neither)}")
    print()
    print(f"{'Metric':<20} {'Micro-avg (pooled)':<22} {'Macro-avg (per-subject mean)':<30}")
    print(f"{'-'*20} {'-'*22} {'-'*30}")
    print(f"{'Percent agreement':<20} {micro_agreement:<22.4f} {macro_agreement:<30.4f}")
    print(f"{'Jaccard index':<20} {micro_jaccard:<22.4f} {macro_jaccard:<30.4f}")
    print(f"{'Cohen kappa':<20} {micro_kappa:<22.4f} {macro_kappa:<30.4f}")
    print()
    print("Per-subject breakdown:")
    print(per_subject_df.to_string(index=False))

    # ----- save CSV: per-subject rows + a final aggregate row -----
    agg_row = pd.DataFrame([{
        "subject_id": "ALL_SUBJECTS_MICRO_AVG", "n_ics": int(total_ics),
        "both": int(total_both), "corr_only": int(total_corr_only),
        "iclabel_only": int(total_iclabel_only), "neither": int(total_neither),
        "percent_agreement": round(micro_agreement, 4),
        "jaccard": round(micro_jaccard, 4), "cohens_kappa": round(micro_kappa, 4),
    }, {
        "subject_id": "ALL_SUBJECTS_MACRO_AVG", "n_ics": "", "both": "", "corr_only": "",
        "iclabel_only": "", "neither": "",
        "percent_agreement": round(macro_agreement, 4),
        "jaccard": round(macro_jaccard, 4), "cohens_kappa": round(macro_kappa, 4),
    }])
    full_out = pd.concat([per_subject_df, agg_row], ignore_index=True)
    out_csv = os.path.join(out_dir, "iclabel_correlation_agreement_report.csv")
    full_out.to_csv(out_csv, index=False)
    print(f"\nFull report saved: {out_csv}")

    # ----- thesis-ready paragraph -----
    interpretation = (
        "poor" if micro_kappa < 0.20 else
        "fair" if micro_kappa < 0.40 else
        "moderate" if micro_kappa < 0.60 else
        "substantial" if micro_kappa < 0.80 else
        "almost perfect"
    )
    paragraph = (
        f"Across all {len(per_subject_df)} subjects and {int(total_ics)} total independent "
        f"components, the correlation-based (find_bads_eog) and ICLabel-based ocular-artifact "
        f"detectors agreed on {int(total_both)} components as ocular and {int(total_neither)} "
        f"components as non-ocular ({micro_agreement*100:.1f}% raw percent agreement). "
        f"Disagreements occurred for {int(total_corr_only)} components (correlation-only) and "
        f"{int(total_iclabel_only)} components (ICLabel-only). The chance-corrected agreement "
        f"(Cohen's kappa) was {micro_kappa:.3f}, indicating {interpretation} agreement between "
        f"the two independent detectors."
    )
    print(f"\n--- Thesis-ready paragraph ---\n{paragraph}")
    txt_path = os.path.join(out_dir, "iclabel_correlation_agreement_paragraph.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(paragraph)
    print(f"\nParagraph saved: {txt_path}")


if __name__ == "__main__":
    main()
