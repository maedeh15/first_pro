"""
audit_A1_trial_vs_window_counts.py
====================================
هدف: تعیین اینکه اگر class_counts در خط ۱۰۱۲ (به‌جای شمارش سطح پنجره روی
tv_labels) روی شمارش سطح trial واقعی (تعداد trialهای یکتای هر کلاس، از
tv_trial_ids) محاسبه می‌شد، وزن‌های effective_number با beta=0.9 چه می‌شدند.

این اسکریپت هیچ مدلی آموزش نمی‌دهد و به GPU نیاز ندارد -- فقط فایل‌های
{subject}_labels.npy و {subject}_trial_ids.npy موجود را می‌خواند و همان
منطق trial_level_train_val_split + session_holdout را برای بازسازی fold ها
تکرار می‌کند تا tv_labels/tv_trial_ids هر fold به‌دست بیاید.

خروجی: یک CSV با شمارش سطح پنجره (که کد فعلی واقعاً استفاده می‌کند)، شمارش
سطح trial (که در تحلیل اولیه‌ی گزارش فرض شده بود)، و وزن effective_number
هر دو حالت برای هر subject x session، تا A1.3 با شواهد واقعی -- نه تخمین --
تصمیم‌گیری شود.

نحوه‌ی اجرا:
    python audit_A1_trial_vs_window_counts.py --data-root <مسیر پوشه‌ی دیتا>

فرض‌های ساختاری (بر اساس کد دیده‌شده در tsfnet_model_notebook.py):
    <data_root>/<SUBJECT_ID>/<SUBJECT_ID>_labels.npy      -- برچسب هر پنجره
    <data_root>/<SUBJECT_ID>/<SUBJECT_ID>_session_ids.npy -- session هر پنجره (1/2/3)
    <data_root>/<SUBJECT_ID>/<SUBJECT_ID>_trial_ids.npy   -- شناسه‌ی trial هر پنجره

اگر مسیر یا نام فایل‌ها فرق دارد، مقادیر --labels-suffix/--session-suffix/
--trial-suffix را عوض کنید یا مستقیماً الگوی glob را در تابع find_subject_files
ویرایش کنید.
"""

import argparse
import glob
import os
import sys
import csv

import numpy as np


def effective_number_weights(class_counts, beta, num_classes):
    """دقیقاً همان فرمول خطوط ۱۰۴۸-۱۰۵۱ کد اصلی، برای بازتولید دقیق رفتار."""
    class_counts = np.asarray(class_counts, dtype=float)
    eff_num = np.where(
        class_counts > 0,
        (1.0 - beta ** np.maximum(class_counts, 1)) / (1.0 - beta),
        1.0,
    )
    raw_weight = np.where(class_counts > 0, 1.0 / eff_num, 0.0)
    # نرمال‌سازی هم‌تراز با چیزی که در ادامه‌ی کد اصلی (بعد از خط ۱۰۷۴) انجام می‌شود:
    # میانگین وزن موزون با شمارش کلاس = ۱ (حفظ مقیاس کلی loss)
    n_samples_present = class_counts.sum()
    weighted_mass = (class_counts * raw_weight).sum()
    if weighted_mass > 0:
        raw_weight = raw_weight * (n_samples_present / weighted_mass)
    return raw_weight


def find_subject_files(data_root, subject_id, labels_suffix, session_suffix, trial_suffix):
    subj_dir = os.path.join(data_root, subject_id)
    labels_path = os.path.join(subj_dir, f"{subject_id}{labels_suffix}")
    session_path = os.path.join(subj_dir, f"{subject_id}{session_suffix}")
    trial_path = os.path.join(subj_dir, f"{subject_id}{trial_suffix}")
    return labels_path, session_path, trial_path


def session_holdout_fold_indices(session_ids, test_session):
    """بازسازی ساده‌ی session_holdout_split: هر fold یک session را test می‌گیرد،
    بقیه train+val است. اگر منطق واقعی کد پیچیده‌تر است (مثلاً حذف نمونه‌های
    نامعتبر)، این تابع باید با تابع اصلی session_holdout_split جایگزین شود --
    نسخه‌ی فعلی صرفاً برای این audit ساده‌سازی شده و فرض می‌کند session_ids
    فقط شامل 1/2/3 معتبر است."""
    test_mask = session_ids == test_session
    train_val_mask = ~test_mask
    return np.where(train_val_mask)[0], np.where(test_mask)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="پوشه‌ی ریشه‌ی دیتای پیش‌پردازش‌شده (شامل زیرپوشه‌ی هر آزمودنی)")
    ap.add_argument("--subjects", nargs="*", default=None,
                     help="لیست subject id ها (پیش‌فرض: تمام زیرپوشه‌های VP*)")
    ap.add_argument("--labels-suffix", default="_labels.npy")
    ap.add_argument("--session-suffix", default="_session_ids.npy")
    ap.add_argument("--trial-suffix", default="_trial_ids.npy")
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--beta-window", type=float, default=0.9,
                     help="beta فعلی که کد با شمارش سطح پنجره استفاده می‌کند")
    ap.add_argument("--beta-trial", type=float, default=0.9,
                     help="beta برای مقایسه با شمارش سطح trial (پیش‌فرض همان 0.9، برای مقایسه‌ی مستقیم)")
    ap.add_argument("--out-csv", default="audit_A1_trial_vs_window_counts.csv")
    args = ap.parse_args()

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(args.data_root, "VP*"))
            if os.path.isdir(p)
        )

    if not subjects:
        print(f"هیچ زیرپوشه‌ی آزمودنی زیر {args.data_root} پیدا نشد. "
              f"--subjects را صریح بدهید یا مسیر --data-root را بررسی کنید.", file=sys.stderr)
        sys.exit(1)

    rows = []
    missing = []

    for subject_id in subjects:
        labels_path, session_path, trial_path = find_subject_files(
            args.data_root, subject_id, args.labels_suffix, args.session_suffix, args.trial_suffix
        )
        if not (os.path.exists(labels_path) and os.path.exists(session_path) and os.path.exists(trial_path)):
            missing.append((subject_id, labels_path, session_path, trial_path))
            continue

        labels = np.load(labels_path)
        session_ids = np.load(session_path)
        trial_ids = np.load(trial_path)

        if not (len(labels) == len(session_ids) == len(trial_ids)):
            print(f"[هشدار] {subject_id}: طول آرایه‌ها یکسان نیست "
                  f"(labels={len(labels)}, session_ids={len(session_ids)}, trial_ids={len(trial_ids)}) -- رد شد.",
                  file=sys.stderr)
            continue

        for test_session in sorted(np.unique(session_ids)):
            train_val_idx, _test_idx = session_holdout_fold_indices(session_ids, test_session)
            tv_labels = labels[train_val_idx]
            tv_trial_ids = trial_ids[train_val_idx]

            # شمارش سطح پنجره -- دقیقاً همان چیزی که کد فعلی در خط ۱۰۱۲ محاسبه می‌کند
            window_counts = np.bincount(tv_labels, minlength=args.num_classes)

            # شمارش سطح trial واقعی -- تعداد trial یکتا در هر کلاس
            # (هر trial دقیقاً یک برچسب دارد طبق فرض کد در تابع trial_level_train_val_split)
            trial_counts = np.zeros(args.num_classes, dtype=int)
            for cls in range(args.num_classes):
                cls_mask = tv_labels == cls
                trial_counts[cls] = len(np.unique(tv_trial_ids[cls_mask]))

            w_window = effective_number_weights(window_counts, args.beta_window, args.num_classes)
            w_trial = effective_number_weights(trial_counts, args.beta_trial, args.num_classes)

            rows.append({
                "subject_id": subject_id,
                "test_session": int(test_session),
                "window_counts": window_counts.tolist(),
                "window_weights": np.round(w_window, 4).tolist(),
                "window_ratio_max_min": round(float(w_window.max() / max(w_window.min(), 1e-9)), 4),
                "trial_counts": trial_counts.tolist(),
                "trial_weights": np.round(w_trial, 4).tolist(),
                "trial_ratio_max_min": round(float(w_trial.max() / max(w_trial.min(), 1e-9)), 4),
            })

    if missing:
        print(f"\n[هشدار] {len(missing)} آزمودنی فایل‌شان پیدا نشد (نمونه‌ی اول‌ها):", file=sys.stderr)
        for subj, lp, sp, tp in missing[:5]:
            print(f"  {subj}: {lp} / {sp} / {tp}", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... و {len(missing) - 5} مورد دیگر", file=sys.stderr)

    if not rows:
        print("هیچ fold ای پردازش نشد -- مسیرها یا نام فایل‌ها را بررسی کنید.", file=sys.stderr)
        sys.exit(1)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ratios_window = [r["window_ratio_max_min"] for r in rows]
    ratios_trial = [r["trial_ratio_max_min"] for r in rows]

    print(f"\n{len(rows)} fold پردازش شد از {len(set(r['subject_id'] for r in rows))} آزمودنی.")
    print(f"CSV کامل ذخیره شد در: {args.out_csv}\n")
    print("=" * 70)
    print(f"خلاصه (beta_window={args.beta_window}, beta_trial={args.beta_trial}):")
    print("=" * 70)
    print(f"  میانگین نسبت max/min وزن -- سطح پنجره (کد فعلی): {np.mean(ratios_window):.4f}")
    print(f"  میانگین نسبت max/min وزن -- سطح trial (واقعی):   {np.mean(ratios_trial):.4f}")
    print(f"  حداکثر نسبت -- سطح پنجره: {max(ratios_window):.4f}")
    print(f"  حداکثر نسبت -- سطح trial: {max(ratios_trial):.4f}")
    n_window_noop = sum(1 for r in ratios_window if abs(r - 1.0) < 1e-6)
    n_trial_noop = sum(1 for r in ratios_trial if abs(r - 1.0) < 1e-6)
    print(f"  تعداد fold با وزن دقیقاً یکسان [no-op] -- سطح پنجره: {n_window_noop} / {len(rows)}")
    print(f"  تعداد fold با وزن دقیقاً یکسان [no-op] -- سطح trial: {n_trial_noop} / {len(rows)}")


if __name__ == "__main__":
    main()
