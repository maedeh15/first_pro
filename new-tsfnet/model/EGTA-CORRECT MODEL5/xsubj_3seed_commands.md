# پیش‌آموزش میان‌آزمودنی — سه بذر، از صفر تا جدول

قاعده‌ی پیش‌ثبت‌شده (ثبت‌شده پیش از ران): مقایسه‌ی زوجی با پایه‌ی `a1621a1ecc`
روی **همان بذرها**، اختلاف میانگین ≥ 0.01 و p ≤ 0.05 و هم‌علامتی هر سه بذر.
هیچ‌کدام از `--n-groups 4`، `--n-val-subjects 4`، `--window-stride 3`,
`--max-epochs 30`، `--patience 6` پس از دیدن نتیجه عوض نمی‌شود.

بودجه: ~1.2 ساعت پیش‌آموزش + ~8.5 ساعت ارزیابی، در هر بذر → ~29 ساعت کل.

---

## گام ۰ — پایه‌ها را پیدا کن (صفر GPU، قبل از هر چیز)

```python
import glob, csv, statistics as st
ROOT = r"E:\kntu\Thesis\codes\Results"
for h in ("a1621a1ecc",):
    for s in (42, 999, 2024):
        g = glob.glob(rf"{ROOT}\**\baseline_all_subjects_*_seed{s}_{h}.csv", recursive=True)
        for f in g:
            ok = [r for r in csv.DictReader(open(f, newline="", encoding="utf-8"))
                  if r.get("status") == "ok"]
            m = st.mean(float(r["mean_acc"]) for r in ok) if ok else float("nan")
            print(f"{h} seed{s}  acc={m:.4f}  n={len(ok):2d}  {f[len(ROOT)+1:]}")
        if not g:
            print(f"{h} seed{s}  MISSING -- باید ران شود")
```

هر بذری که `MISSING` بود، پایه‌اش را اول بگیر وگرنه آن بذر از مقایسه می‌افتد:

```python
%run run_all_subjects.py --seed <بذر> --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member
```

---

## گام ۱ — پیش‌آموزش، یک بار برای هر بذر

گارد بذر در `run_all_subjects.py` اجازه نمی‌دهد مدل بذر ۴۲ برای بذر ۹۹۹
استفاده شود، پس سه پیش‌آموزش جدا لازم است. هر کدام ~1.2 ساعت.

```python
%run pretrain_cross_subject.py --seed 42   --n-groups 4 --n-val-subjects 4 --window-stride 3
%run pretrain_cross_subject.py --seed 999  --n-groups 4 --n-val-subjects 4 --window-stride 3
%run pretrain_cross_subject.py --seed 2024 --n-groups 4 --n-val-subjects 4 --window-stride 3
```

خط `out:` هر ران یک پوشه با `id` متفاوت می‌دهد. **هر سه را یادداشت کن** —
`id` تابع بذر است و حدس‌زدنی نیست:

```
pretrain_<id42>_seed42
pretrain_<id999>_seed999
pretrain_<id2024>_seed2024
```

پیش از گام ۲ مطمئن شو هر ران `groups [..] still to run` نگفته باشد؛ با
پیش‌آموزش ناقص، گارد پوشش ارزیابی را متوقف می‌کند (که درست است).

---

## گام ۲ — ارزیابی، یک بار برای هر بذر

```python
PRE = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"

%run run_all_subjects.py --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member --pretrained-dir "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\pretrain_<id42>_seed42"

%run run_all_subjects.py --seed 999 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member --pretrained-dir "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\pretrain_<id999>_seed999"

%run run_all_subjects.py --seed 2024 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member --pretrained-dir "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\pretrain_<id2024>_seed2024"
```

در خط اول هر خروجی باید ببینی:
```
[pretrained] 26 subjects mapped to 4 pretrained models from ... (id <...>)
```
اگر این خط نبود، `--pretrained-dir` نادیده گرفته شده و ران بی‌ارزش است.

نام پوشه‌ی نتایج زیربرچسب `_pt-<id>` می‌گیرد، پس روی نتایج پایه نوشته
نمی‌شود. **مسیر دقیق CSV را از خط آخر هر لاگ بردار**، حدس نزن.

---

## گام ۳ — آزمون پیش‌ثبت‌شده

```python
import subprocess, sys
A = [  # پایه a1621a1ecc، سه بذر — مسیرها را از گام ۰ بگذار
  r"...\results_a1621a1ecc_seed42\baseline_all_subjects_*_seed42_a1621a1ecc.csv",
  r"...\results_a1621a1ecc_seed999\baseline_all_subjects_*_seed999_a1621a1ecc.csv",
  r"...\results_a1621a1ecc_seed2024\baseline_all_subjects_*_seed2024_a1621a1ecc.csv",
]
B = [  # پیش‌آموزش‌دیده — مسیرها را از خط آخر لاگ‌های گام ۲ بگذار
  r"...\results_a1621a1ecc_pt-<id42>_seed42\baseline_all_subjects_*_seed42_*.csv",
  r"...\results_a1621a1ecc_pt-<id999>_seed999\baseline_all_subjects_*_seed999_*.csv",
  r"...\results_a1621a1ecc_pt-<id2024>_seed2024\baseline_all_subjects_*_seed2024_*.csv",
]
cmd = [sys.executable, "compare_runs.py", "--a", *A, "--b", *B,
       "--label-a", "riem member", "--label-b", "+xsubj pretrain",
       "--rule-min-diff", "0.01", "--rule-max-p", "0.05", "--rule-same-sign"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout or r.stderr)
```

کنترل درستی اجرا پیش از خواندن نتیجه: هر سمت **3 csv**، **26 آزمودنی**،
بدون `[PARTIAL]`، و سه خط `seed 42/999/2024: mean difference ...`.

---

## گام ۴ — گروهه‌ی سه‌بذری (اندازه‌گیری، نه آزمون)

اینجا بذر ۴۲ هم حاضر است، چون فرضیه‌ای آزمون نمی‌شود:

```python
%run ensemble_eval.py --seeds 42 999 2024 --results-dirs "...\results_a1621a1ecc_pt-<id42>_seed42" "...\results_a1621a1ecc_pt-<id999>_seed999" "...\results_a1621a1ecc_pt-<id2024>_seed2024" --riem-branch member --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --save-posteriors
```

`--results-dirs` لازم است و `--variant-hash` کافی نیست، چون پوشه‌ها
زیربرچسب `_pt-<id>` دارند و `id` در سه بذر متفاوت است.
