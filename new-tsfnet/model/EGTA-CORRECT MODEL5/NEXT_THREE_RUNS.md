# سه ران بعدی، به ترتیب

هر سه کد آماده است. ترتیب بر پایه‌ی ارزش انتظاری و وابستگی‌ها.

قاعده‌ی مشترک: هر مقایسه زوجی روی ۲۶ آزمودنی، و قاعده‌ی پیش‌ثبت‌شده اختلاف ≥ 0.01 و
p ≤ 0.05 نسبت به مرجع همان ران. هیچ ابرپارامتری بعد از دیدن نتیجه عوض نمی‌شود.

---

## ۱) پیش‌آموزش میان‌آزمودنی با پروتکل نشست (GPU، ~۳.۸ + ۸.۵ ساعت)

بالاترین اولویت: تنها خانواده‌ی مداخله‌ای که در این پروژه رد نشده، و رژیم داده را
عوض می‌کند نه ظرفیت مدل را.

```
%run pretrain_cross_subject.py --seed 42 --n-groups 4 --n-val-subjects 4 --window-stride 3 --session-protocol loso
```
۱۲ مدل (۴ گروه × ۳ نشست). آدرس پوشه را از آخر خروجی بردار، بعد:
```
%run run_all_subjects.py --seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none --riem-branch member --pretrained-dir "<پوشه>" --pretrained-new-modules riem_head
```
در خط اول باید `session protocol: loso` را ببینی، و برای هر آزمودنی
`starts from ...pretrain_g<g>_s<نشست>.pt`.

مرجع: پایه‌ی `a1621a1ecc` بذر ۴۲ = **0.6411**
`...\results_a1621a1ecc_seed42-riem-branch member\baseline_all_subjects_20261021_205046_seed42_a1621a1ecc.csv`

قاعده‌ی توقف: اگر اختلاف زیر +0.005 بود، بذرهای ۹۹۹ و ۲۰۲۴ را نزن.

---

## ۲) نیمه‌ی به‌روزرسانی T-TIME (GPU، اول روی ۲ آزمودنی زمان بگیر)

```
%run ensemble_eval.py --seeds 42 --results-dirs "...\results_84c18210ab_seed42" --subjects 1-2 --riem-branch member --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --ttime
```
زمان هر فولد × ۷۸ = زمان یک بذر. اگر زیاد بود `--ttime-stride 4` یا `--ttime-mode offline`.

سه بذر کامل:
```
%run ensemble_eval.py --seeds 42 999 2024 --results-dirs "...seed42" "...seed999" "...seed2024" --riem-branch member --adabn --fnirs-temporal-align egta --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --ttime
```
مرجع: همان ران بدون `--ttime` = پنجره **0.6562**، بلوک **0.6994**

کنترل درستی: خط `SELF-CHECK` باید عددی **متفاوت** از 0.6523/0.6535/0.6428 بدهد.
اگر عین رکورد را بازتولید کرد، به‌روزرسانی کار نکرده و ران بی‌ارزش است.

---

## ۳) FBC در مقیاس بلوک ۴۰ ثانیه (CPU، ~۴۰ دقیقه، می‌تواند هم‌زمان با ۱ یا ۲ اجرا شود)

نسخه‌ی پنجره‌ای منفی شد (بهترین 0.5449 در برابر شبکه‌ی عمیق 0.5996). این نسخه تنها
تفاوت باقی‌مانده با مقاله را می‌بندد: آن‌ها همه‌ی اعدادشان را با پنجره‌ی ۴۰ ثانیه گرفتند
(شکل ۳B خودشان)، ما با ۱۰ ثانیه.

```
%run check_eeg_fbc.py --data "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback" --level block
```
بلوک از پنجره‌های خودش بازسازی می‌شود (اولین ثانیه‌ی هر پنجره + کل پنجره‌ی آخر = ~۴۴ ثانیه).

مرجع: شبکه‌ی عمیق 0.5996 روی ۲-در-برابر-۳. باید از آن رد شود تا ران GPU توجیه شود.
نسخه‌ی پنجره‌ای برای مقایسه: pcc:theta 0.5449، psd:theta 0.5429.

**قید مهم:** در این حالت هر فولد فقط **۱۸ بلوک آموزش** دارد، پس انتخاب ANOVA روی ۱۸ نمونه
برازش می‌شود و هم انتخاب و هم صحت به‌مراتب نوفه‌ای‌ترند. این همان رژیمی است که مقاله دارد،
ولی یعنی یک عدد بالا در این حالت را نمی‌شود بی‌قید به حساب FBC گذاشت.
