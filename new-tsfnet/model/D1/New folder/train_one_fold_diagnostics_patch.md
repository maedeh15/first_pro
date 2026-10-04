# پچ مینیمال train_one_fold برای ثبت اطلاعات تشخیصی بیشتر
# (فقط افزودن -- هیچ رفتار آموزشی تغییر نمی‌کند، رعایت قاعده‌ی ۷: قرارداد خروجی)

## در تعریف history (نزدیک خط ۲۲۴۶)، جایگزین کن:

```python
    # قبل:
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_kappa": []}

    # بعد -- دو کلید جدید، بقیه دست‌نخورده:
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_kappa": [],
               "stage": [],       # <-- NEW: 1 یا 2، برای جدا کردن دو مرحله در نمودار
               "gates": []}       # <-- NEW: dict از {نام_گیت: مقدار} در پایان هر epoch
```

## بعد از `history["val_kappa"].append(val_kappa)` در STAGE 1 (خط ۲۳۰۶)، اضافه کن:

```python
        history["stage"].append(1)
        history["gates"].append(_collect_gate_values(model))   # <-- NEW
```

## دقیقاً همین دو خط را در حلقه‌ی STAGE 2 هم اضافه کن (جایی که همین ۶ خط `history[...].append(...)` برای stage 2 تکرار شده‌اند -- با `grep -n 'history\["val_kappa"\].append'` پیدایش کن، باید دو جا باشد)، فقط با `history["stage"].append(2)`.

## این تابع کمکی را هرجای مناسب فایل (مثلاً بالای `train_one_fold`) اضافه کن:

```python
def _collect_gate_values(model):
    """
    Generically finds every sigmoid-gated scalar parameter on the model
    (the pattern this project uses repeatedly: beta_labram, cafe.egta_q's
    effective gate, eeg_band_branch.gate, w_labram, ...) and returns their
    CURRENT sigmoid-activated values as a plain dict, for logging into
    history without needing to know in advance which gates a given run's
    configuration actually constructed. Pure read-only inspection -- no
    gradient, no side effect on training.
    """
    gates = {}
    with torch.no_grad():
        if hasattr(model, "beta_labram"):
            gates["beta_labram"] = torch.sigmoid(model.beta_labram).item()
        if hasattr(model, "w_labram"):
            gates["w_labram"] = torch.sigmoid(model.w_labram).item()
        if hasattr(model, "eeg_band_branch") and hasattr(model.eeg_band_branch, "gate"):
            gates["eeg_band_gate"] = torch.sigmoid(model.eeg_band_branch.gate).item()
        # Eq. 4 fusion weights -- not "gates" in the zero-init sense, but the
        # same diagnostic question applies ("did the model actually move
        # these, or did they stay near their init"), so tracked identically.
        if hasattr(model, "w_eeg"):
            gates["w_eeg_fusion"] = torch.sigmoid(model.w_eeg).item()
        if hasattr(model, "w_fusion"):
            gates["w_fusion_fusion"] = torch.sigmoid(model.w_fusion).item()
        if hasattr(model, "w_fnirs"):
            gates["w_fnirs_fusion"] = torch.sigmoid(model.w_fnirs).item()
    return gates
```

نتیجه: `report["history"]` از این به بعد شامل `stage` (۱/۲) و `gates` (لیست دیکشنری، یکی به‌ازای هر epoch) هم هست -- بدون این‌که هیچ کلید قبلی حذف یا تغییر معنا بدهد.
