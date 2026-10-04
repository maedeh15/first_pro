# پچ run_all_subjects.py برای D-1-safe

## بعد از بلوک `--eeg-spatial-mode` (نزدیک خط ۴۲۵-۴۴۳)، این آرگومان‌ها را اضافه کن:

```python
_arg_parser.add_argument("--eeg-band-mode", type=str, default="none",
                          choices=["none", "safe"],
                          help="priority-D-1-safe: 'none' (default) reproduces prior "
                               "behavior exactly. 'safe' adds a gated (sigmoid(-6) init) "
                               "embedding of five FIXED EEG frequency bands "
                               "(delta/theta/alpha/beta/gamma) via a non-trainable FIR "
                               "filterbank + shared-weight encoder, concatenated onto the "
                               "EEG branch. Reuses the SAME raw channel-space x_chan input "
                               "--eeg-spatial-mode mhsa already uses -- no new preprocessing "
                               "needed. Mutually exclusive with --eeg-backbone != native.")
_arg_parser.add_argument("--eeg-band-d-model", type=int, default=32,
                          help="priority-D-1-safe: per-band temporal pooling width.")
_arg_parser.add_argument("--eeg-band-embed-dim", type=int, default=32,
                          help="priority-D-1-safe: final band-branch embedding size.")
_arg_parser.add_argument("--eeg-band-weight-decay", type=float, default=1e-2,
                          help="priority-D-1-safe (Safe-Capacity Protocol): weight_decay "
                               "applied ONLY to eeg_band_branch's parameters, separate "
                               "from --weight-decay (which stays at its usual 1e-4 for "
                               "every other parameter). Default 1e-2 is 100x heavier, per "
                               "the project's diagnosis that unregularized new capacity in "
                               "this low-data (n=26) regime is what has driven 10+ prior "
                               "rejections, not capacity itself.")
```

## بعد از `EEG_SPATIAL_MODE = SEED_ARGS.eeg_spatial_mode` (خط ۵۱۷)، اضافه کن:

```python
EEG_BAND_MODE = SEED_ARGS.eeg_band_mode
EEG_BAND_D_MODEL = SEED_ARGS.eeg_band_d_model
EEG_BAND_EMBED_DIM = SEED_ARGS.eeg_band_embed_dim
EEG_BAND_WEIGHT_DECAY = SEED_ARGS.eeg_band_weight_decay
```

## در بنر لاگ (خط ~۷۳۸، کنار خط eeg_spatial_mode)، اضافه کن:

```python
          f"| eeg_band_mode={EEG_BAND_MODE!r} (d_model={EEG_BAND_D_MODEL}, "
          f"embed_dim={EEG_BAND_EMBED_DIM}, wd={EEG_BAND_WEIGHT_DECAY}, priority-D-1-safe) "
```

## در فراخوانی سازنده‌ی `TSFNet(...)` (خط ~۸۱۴-۸۲۲)، اضافه کن:

```python
                eeg_band_mode=EEG_BAND_MODE,               # <-- NEW: priority-D-1-safe
                eeg_band_d_model=EEG_BAND_D_MODEL,         # <-- NEW
                eeg_band_embed_dim=EEG_BAND_EMBED_DIM,     # <-- NEW
```

## بازسازی بهینه‌ساز با weight_decay نامتقارن (Safe-Capacity Protocol)

در `tsfnet_model_notebook.py`، خط ۲۲۵۳:

```python
    # قبل:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # بعد -- فقط وقتی eeg_band_mode=='safe' گروه دوم واقعاً چیزی دارد؛ در حالت
    # 'none' لیست band_params خالی است و رفتار دقیقاً مثل قبل می‌ماند:
    band_params = list(model.eeg_band_branch.parameters()) if hasattr(model, "eeg_band_branch") else []
    band_param_ids = {id(p) for p in band_params}
    other_params = [p for p in model.parameters() if id(p) not in band_param_ids]
    optimizer = torch.optim.Adam([
        {"params": other_params, "weight_decay": weight_decay},
        {"params": band_params, "weight_decay": eeg_band_weight_decay},  # آرگومان جدید تابع
    ], lr=lr)
```

`train_one_fold` باید یک آرگومان جدید `eeg_band_weight_decay=1e-2` هم در امضایش بگیرد و از `run_subject_cross_session` تا اینجا پاس داده شود -- دقیقاً مثل مسیری که `weight_decay` معمولی الان طی می‌کند (همان زنجیره‌ی فراخوانی را دنبال کن).

## دستور اجرای پیشنهادی (یک بذر، طبق پروتکل تک‌بذری آزمایش‌های ارزان این پروژه)

```
%run run_all_subjects.py --seed 999 --eeg-band-mode safe
```
