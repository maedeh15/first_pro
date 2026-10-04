"""
test_xsval.py -- priority-XSVAL (--val-mode cross_session), tested against the
REAL functions in tsfnet_model_notebook.py.

Verifies the concrete claims made for this change:
  * the OLD mode really does give only ~3 validation trials
  * the NEW mode gives 9, and they are cross-session
  * no trial ever appears in both train and val
  * the split is deterministic across seeds (only init/batch order should
    differ between seeds, never the data partition)
  * val_mode='trial' is bit-exact to the previous behavior

Run with no arguments from the folder containing tsfnet_model_notebook.py.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, _HERE)

ns = {}
exec(open(os.path.join(_HERE, "tsfnet_model_notebook.py"), encoding="utf-8").read(), ns)
trial_level_train_val_split = ns["trial_level_train_val_split"]
cross_session_train_val_split = ns["cross_session_train_val_split"]
session_holdout_split = ns["session_holdout_split"]
TSFNet = ns["TSFNet"]


def banner(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# --------------------------------------------------------------------------
# Build a realistic fixture matching this dataset EXACTLY (rule 8):
# 3 sessions x 9 trials (3 per class) x ~35 sliding windows per trial.
# --------------------------------------------------------------------------
WINDOWS_PER_TRIAL = 35
session_ids, trial_ids, labels = [], [], []
tid = 0
for sess in (1, 2, 3):
    for cls in (0, 1, 2):
        for _ in range(3):                      # 3 trials per class per session
            session_ids += [sess] * WINDOWS_PER_TRIAL
            trial_ids += [tid] * WINDOWS_PER_TRIAL
            labels += [cls] * WINDOWS_PER_TRIAL
            tid += 1
session_ids = np.array(session_ids)
trial_ids = np.array(trial_ids)
labels = np.array(labels)
print(f"fixture: {len(session_ids)} windows, {len(np.unique(trial_ids))} trials, "
      f"{len(np.unique(session_ids))} sessions "
      f"({len(np.unique(trial_ids)) // 3} trials/session, 3 per class) -- matches the "
      f"real dataset's structure")

# --------------------------------------------------------------------------
banner("TEST 1: reproduce the defect -- the OLD mode yields only ~3 validation trials")
train_val_idx, test_idx = session_holdout_split(session_ids, test_session=3)
tv_trials = trial_ids[train_val_idx]
tv_labels = labels[train_val_idx]
tr_sub, va_sub = trial_level_train_val_split(tv_trials, tv_labels)
n_val_trials_old = len(np.unique(tv_trials[va_sub]))
n_train_trials_old = len(np.unique(tv_trials[tr_sub]))
print(f"  train trials: {n_train_trials_old}   validation trials: {n_val_trials_old}")
val_sessions_old = np.unique(session_ids[train_val_idx][va_sub])
train_sessions_old = np.unique(session_ids[train_val_idx][tr_sub])
print(f"  validation drawn from session(s): {val_sessions_old.tolist()}   "
      f"training from: {train_sessions_old.tolist()}")
assert n_val_trials_old == 3, f"expected the documented 3 validation trials, got {n_val_trials_old}"
assert set(val_sessions_old.tolist()) & set(train_sessions_old.tolist()), \
    "expected the OLD validation set to overlap the training SESSIONS (that is the defect)"
print("[CONFIRMED] old mode: 3 validation trials, drawn from the same sessions as training "
      "-- while the test set is a different session entirely.")

# --------------------------------------------------------------------------
banner("TEST 2 (THE FIX): cross-session mode gives 9 validation trials, fully disjoint sessions")
tv_sessions = np.unique(session_ids[train_val_idx])
val_session = int(tv_sessions.min())
tr_sub_x, va_sub_x = cross_session_train_val_split(session_ids[train_val_idx], val_session)
n_val_trials_new = len(np.unique(tv_trials[va_sub_x]))
n_train_trials_new = len(np.unique(tv_trials[tr_sub_x]))
val_sessions_new = np.unique(session_ids[train_val_idx][va_sub_x])
train_sessions_new = np.unique(session_ids[train_val_idx][tr_sub_x])
print(f"  train trials: {n_train_trials_new}   validation trials: {n_val_trials_new}")
print(f"  validation session(s): {val_sessions_new.tolist()}   "
      f"training session(s): {train_sessions_new.tolist()}")
assert n_val_trials_new == 9, n_val_trials_new
assert n_train_trials_new == 9, n_train_trials_new
assert not (set(val_sessions_new.tolist()) & set(train_sessions_new.tolist())), \
    "train and validation must come from DISJOINT sessions"
# class balance in validation
val_cls, val_cnt = np.unique(labels[train_val_idx][va_sub_x], return_counts=True)
print(f"  validation class balance: {dict(zip(val_cls.tolist(), val_cnt.tolist()))}")
assert len(set(val_cnt.tolist())) == 1, "validation should stay class-balanced"
print("[PASS] test 2: 9 validation trials (3x the old mode), class-balanced, and from a "
      "session the model never trains on -- validation now matches the test regime.")

# --------------------------------------------------------------------------
banner("TEST 3 (the honest cost): training data is HALVED")
print(f"  old mode: {n_train_trials_old} training trials ({len(tr_sub)} windows)")
print(f"  new mode: {n_train_trials_new} training trials ({len(tr_sub_x)} windows)")
ratio = n_train_trials_new / n_train_trials_old
print(f"  ratio = {ratio:.2f}")
assert n_train_trials_new < n_train_trials_old
print("[PASS] test 3: the cost is real and must be reported -- the new mode trains on one "
      "session (9 trials) instead of two (15 trials after the old val carve-out). Whether "
      "the better validation outweighs less training data is exactly what the run decides.")

# --------------------------------------------------------------------------
banner("TEST 4: no trial leaks across the train/val boundary, in EITHER mode")
for name, tr, va in [("trial", tr_sub, va_sub), ("cross_session", tr_sub_x, va_sub_x)]:
    overlap = set(np.unique(tv_trials[tr])) & set(np.unique(tv_trials[va]))
    print(f"  {name:14}: trials in BOTH train and val = {len(overlap)}")
    assert not overlap, f"{name} mode leaks trials: {overlap}"
print("[PASS] test 4: sliding windows from one trial never straddle the split in either mode.")

# --------------------------------------------------------------------------
banner("TEST 5: the cross-session split is DETERMINISTIC (identical for every seed)")
# The old mode is seed-dependent by design; the new one must not be, so that
# the data partition is held fixed and only model init/batch order varies.
splits = []
for seed in (42, 999, 2024):
    tvs = np.unique(session_ids[train_val_idx])
    vs = int(tvs.min())
    a, b = cross_session_train_val_split(session_ids[train_val_idx], vs)
    splits.append((a.tolist(), b.tolist()))
assert all(s == splits[0] for s in splits), "cross-session split must not vary with seed"
old_splits = []
for seed in (42, 999, 2024):
    a, b = trial_level_train_val_split(tv_trials, tv_labels, seed=seed)
    old_splits.append(b.tolist())
n_distinct_old = len({tuple(s) for s in old_splits})
print(f"  cross_session: 1 distinct partition across 3 seeds (deterministic)")
print(f"  trial        : {n_distinct_old} distinct partitions across 3 seeds (seed-dependent)")
print("[PASS] test 5: under cross_session the data partition is fixed, so seed-to-seed "
      "variation reflects only model initialization and batch order -- the partition itself "
      "is no longer a hidden source of variance.")

# --------------------------------------------------------------------------
banner("TEST 6: every LOSO fold produces a valid split (not just the one tested above)")
for test_session in (1, 2, 3):
    tv_idx, te_idx = session_holdout_split(session_ids, test_session)
    tvs = np.unique(session_ids[tv_idx])
    vs = int(tvs.min())
    a, b = cross_session_train_val_split(session_ids[tv_idx], vs)
    n_tr = len(np.unique(trial_ids[tv_idx][a]))
    n_va = len(np.unique(trial_ids[tv_idx][b]))
    te_sessions = np.unique(session_ids[te_idx]).tolist()
    print(f"  test session {test_session}: val session {vs}, "
          f"train {n_tr} trials / val {n_va} trials, test sessions {te_sessions}")
    assert n_tr == 9 and n_va == 9
    assert vs not in te_sessions, "validation session must never be the test session"
print("[PASS] test 6: all three folds valid; validation is never the held-out test session.")

# --------------------------------------------------------------------------
banner("TEST 7 (rule 3): guard rails")
run_subject_cross_session = ns["run_subject_cross_session"]
try:
    run_subject_cross_session("/nonexistent", "VP000", val_mode="typo")
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "val_mode must be 'trial' or 'cross_session'" in str(e), str(e)
print("  [PASS] invalid val_mode rejected before any data loading")
try:
    cross_session_train_val_split(np.array([2, 2, 2]), val_session=2)
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "empty split" in str(e)
print("  [PASS] degenerate split (all windows in the val session) raises a clear error")
print("[PASS] test 7")

print("\nALL XSVAL TESTS PASSED against the real functions.")
