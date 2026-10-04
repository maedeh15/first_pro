# پچ TSFNet برای D-1-safe

## در `TSFNet.__init__` (خط ~۸۳۹)، امضای تابع را اضافه کن:

```python
def __init__(self, num_classes=3, dropout=0.5, cafe_dropout=0.3,
             cafe_d_model=64, cafe_n_heads=4, fusion_combination="paper_mean",
             eeg_temporal_mode="gap", eeg_lstm_hidden=64, eeg_readout_mode="h_n",
             fnirs_num_segments=21, fnirs_chroma_mode="both",
             fnirs_num_chromophores=2,
             use_fc_features=False, fc_num_channels=28, fc_embed_dim=64,
             eeg_spatial_mode="none", eeg_spatial_channels=28,
             eeg_spatial_d_model=64, eeg_spatial_heads=4,
             eeg_spatial_layers=2, eeg_spatial_embed_dim=64,
             eeg_backbone_mode="native", fnirs_temporal_align="none",
             eeg_band_mode="none", eeg_band_channels=28,          # <-- NEW
             eeg_band_d_model=32, eeg_band_embed_dim=32):         # <-- NEW
```

## بعد از بلوک `eeg_spatial_mode` (بلافاصله بعد از خط `self.eeg_spatial_branch = SpatialMHSABranch(...)`)، این را اضافه کن:

```python
    # eeg_band_mode: ISOLATED ABLATION PARAMETER (priority-D-1-safe).
    # "none" (default) reproduces prior behavior EXACTLY -- SafeFiveBandAuxBranch
    # is never constructed, clf_eeg's input dim is completely unchanged, and
    # forward() never touches x_chan through this path (rule 5: bit-exact
    # no-op). "safe" concatenates a gated (sigmoid(-6) init) embedding of five
    # FIXED EEG frequency bands (delta/theta/alpha/beta/gamma, non-trainable
    # FIR filterbank) onto the EEG branch's pooled feature before clf_eeg --
    # see SafeFiveBandAuxBranch's docstring for the full "safe capacity"
    # design rationale (why this is expected to fail LESS often than the
    # project's prior 10+ "add capacity" rejections).
    #
    # MUTUALLY EXCLUSIVE with eeg_backbone_mode != 'native' for the same
    # reason eeg_spatial_mode='mhsa' is: both would modify the y_eeg decision
    # path and compete to explain the same signal, breaking the "one
    # variable per experiment" protocol (rule 3).
    # NOT mutually exclusive with eeg_spatial_mode='mhsa' -- they consume the
    # same x_chan tensor but write to DISJOINT parts of clf_eeg's input
    # (different concatenated segments), so both can in principle be
    # combined later; for now this project's protocol tests eeg_band_mode
    # alone (isolated variable), so run scripts should not set both at once
    # even though the class itself would not raise an error for it.
    self.eeg_band_mode = eeg_band_mode
    if eeg_band_mode not in ("none", "safe"):
        raise ValueError(f"eeg_band_mode must be 'none' or 'safe', got {eeg_band_mode!r}")
    if eeg_band_mode == "safe" and eeg_backbone_mode != "native":
        raise ValueError(
            f"eeg_band_mode='safe' and eeg_backbone_mode={eeg_backbone_mode!r} both modify "
            "the EEG decision path and cannot be combined -- this project's isolated-"
            "variable protocol requires testing them separately. Set eeg_backbone_mode="
            "'native' to use eeg_band_mode='safe'."
        )
    if eeg_band_mode == "safe":
        self.eeg_band_branch = SafeFiveBandAuxBranch(
            n_channels=eeg_band_channels, sfreq=200.0,
            d_model=eeg_band_d_model, embed_dim=eeg_band_embed_dim,
            dropout=cafe_dropout)
```

## در محاسبه‌ی `clf_eeg_in_dim` (خط ~۱۰۸۵)، این خط را جایگزین کن:

```python
    # قبل:
    clf_eeg_in_dim = eeg_feat_dim + (eeg_spatial_embed_dim
                                      if eeg_spatial_mode == "mhsa" else 0)
    # بعد:
    clf_eeg_in_dim = eeg_feat_dim + (eeg_spatial_embed_dim
                                      if eeg_spatial_mode == "mhsa" else 0) \
                                   + (eeg_band_embed_dim
                                      if eeg_band_mode == "safe" else 0)   # <-- NEW
```

## در `TSFNet.forward` (خط ۱۰۹۴)، جایی که `eeg_spatial_mode == "mhsa"` مسیرش را با `x_chan` می‌سازد (این بخش را پیدا کن -- معمولاً چیزی شبیه):

```python
    if self.eeg_spatial_mode == "mhsa":
        spatial_embed = self.eeg_spatial_branch(x_chan)
        eeg_pooled = torch.cat([eeg_pooled, spatial_embed], dim=1)
```

بلافاصله بعدش اضافه کن:

```python
    band_gate_value = None
    if self.eeg_band_mode == "safe":
        band_embed, band_gate_value = self.eeg_band_branch(x_chan)
        eeg_pooled = torch.cat([eeg_pooled, band_embed], dim=1)
```

## در دیکشنری `aux` که در انتهای `forward` برگردانده می‌شود، یک کلید جدید اضافه کن (رعایت قاعده‌ی ۷ -- فقط افزودن، نه حذف):

```python
    aux["eeg_band_gate"] = band_gate_value.detach() if band_gate_value is not None else None
```

## نکته‌ی مهم درباره‌ی `x_chan` وقتی `eeg_spatial_mode='none'` و `eeg_band_mode='safe'`

در حالت فعلی کد، احتمالاً `x_chan` فقط وقتی از `DataLoader` واقعاً به `forward` پاس داده می‌شود که `eeg_spatial_mode=='mhsa'` یا `eeg_backbone_mode!='native'` باشد (چون قبلاً فقط این دو مصرف‌کننده‌اش بودند). باید در `run_subject_cross_session` (یا هرجایی که تصمیم می‌گیرد `x_chan` را از دیتاست بخواند/پاس بدهد) شرط را گسترش بدهی:

```python
    NEEDS_X_CHAN = (EEG_SPATIAL_MODE == "mhsa") or (EEG_BACKBONE_MODE != "native") \
                   or (EEG_BAND_MODE == "safe")   # <-- NEW
```

این را با `grep -n "NEEDS_X_CHAN\|x_chan is not None\|if.*mhsa.*or.*labram"` در `run_all_subjects.py`/`tsfnet_model_notebook.py` پیدا کن و همین الگو را اعمال کن -- من دقیق جای آن را در فایل شما ندیدم چون این بخش در محدوده‌ی گرفتاری‌شده‌ی من نبود؛ لطفاً قبل از اجرا این را خودت چک کن (طبق قاعده‌ی ۴: هیچ فرضی را بدون تأیید از کد واقعی قبول نکن).
