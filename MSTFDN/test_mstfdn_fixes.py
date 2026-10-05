"""
test_mstfdn_fixes.py -- verifies all five fixes applied to
MSTFDN_train_LOSO_3class.py, using the REAL functions from that file.

Run with no arguments from the folder containing MSTFDN_train_LOSO_3class.py.
"""
import os
import ast
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
SRC = os.path.join(_HERE, "MSTFDN_train_LOSO_3class.py")


def banner(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# Load ONLY the pure helper functions via AST, without importing torch or the
# model module (which this test environment doesn't need).
src_text = open(SRC, encoding="utf-8").read()
tree = ast.parse(src_text)
WANTED = {"trial_level_train_val_split", "cross_session_train_val_split", "cohen_kappa"}
ns = {"np": np}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANTED:
        exec(compile(ast.Module(body=[node], type_ignores=[]), SRC, "exec"), ns)
missing = WANTED - set(ns)
assert not missing, f"these functions were not found in the file: {missing}"
trial_level_train_val_split = ns["trial_level_train_val_split"]
cross_session_train_val_split = ns["cross_session_train_val_split"]
cohen_kappa = ns["cohen_kappa"]
print(f"loaded {len(WANTED)} helper functions from the real source file")

# Fixture matching the real dataset structure exactly:
# 3 sessions x 9 trials (3 per class) x 33 sliding windows per trial.
WINDOWS_PER_TRIAL = 33
session_ids, trial_ids, labels = [], [], []
tid = 0
for sess in (1, 2, 3):
    for cls in (0, 1, 2):
        for _ in range(3):
            session_ids += [sess] * WINDOWS_PER_TRIAL
            trial_ids += [tid] * WINDOWS_PER_TRIAL
            labels += [cls] * WINDOWS_PER_TRIAL
            tid += 1
session_ids = np.array(session_ids)
trial_ids = np.array(trial_ids)
labels = np.array(labels)
print(f"fixture: {len(session_ids)} windows, {len(np.unique(trial_ids))} trials "
      f"({len(np.unique(trial_ids)) // 3} per session, 3 per class), 3 sessions")

# ---------------------------------------------------------------------------
banner("FIX 1a: val_mode='trial' -- validation comes from TRAINING sessions only, "
      "never the test session")
test_session = 3
train_idx = np.where(session_ids != test_session)[0]
test_idx = np.where(session_ids == test_session)[0]
tv_trials, tv_labels = trial_ids[train_idx], labels[train_idx]
sub_tr, sub_va = trial_level_train_val_split(tv_trials, tv_labels, seed=42)
fit_idx, val_idx = train_idx[sub_tr], train_idx[sub_va]

print(f"  train windows={len(fit_idx)}  val windows={len(val_idx)}  test windows={len(test_idx)}")
print(f"  val trials={len(np.unique(trial_ids[val_idx]))}  "
      f"val sessions={np.unique(session_ids[val_idx]).tolist()}  test session={test_session}")
assert not (set(val_idx) & set(test_idx)), "LEAK: validation overlaps the test set"
assert not (set(fit_idx) & set(test_idx)), "LEAK: training overlaps the test set"
assert not (set(fit_idx) & set(val_idx)), "train and val must be disjoint"
assert test_session not in np.unique(session_ids[val_idx]), \
    "LEAK: the held-out test session is being used for validation"
print("[PASS] the critical leak is gone: val and test share ZERO windows, and the "
      "test session never appears in validation.")

# ---------------------------------------------------------------------------
banner("FIX 1b: no TRIAL straddles the train/val boundary (windows overlap ~90%)")
overlap = set(np.unique(trial_ids[fit_idx])) & set(np.unique(trial_ids[val_idx]))
print(f"  trials appearing in BOTH train and val: {len(overlap)}")
assert not overlap, f"trial-level leak: {overlap}"
print("[PASS] splitting at trial level (not window level) means near-duplicate "
      "overlapping windows can never appear on both sides.")

# ---------------------------------------------------------------------------
banner("FIX 1c: val_mode='cross_session' -- more validation trials, and validation "
      "is itself cross-session")
tv_sessions = np.unique(session_ids[train_idx])
val_session = int(tv_sessions.min())
sub_tr2, sub_va2 = cross_session_train_val_split(session_ids[train_idx], val_session)
fit2, val2 = train_idx[sub_tr2], train_idx[sub_va2]
n_val_trial_mode = len(np.unique(trial_ids[val_idx]))
n_val_xs_mode = len(np.unique(trial_ids[val2]))
print(f"  val trials -- 'trial' mode: {n_val_trial_mode}   'cross_session' mode: {n_val_xs_mode}")
print(f"  'cross_session': train sessions={np.unique(session_ids[fit2]).tolist()}  "
      f"val session={np.unique(session_ids[val2]).tolist()}  test session={test_session}")
assert not (set(val2) & set(test_idx)), "LEAK: cross_session val overlaps test"
assert test_session not in np.unique(session_ids[val2])
assert not set(np.unique(session_ids[fit2])) & set(np.unique(session_ids[val2])), \
    "train and val sessions must be disjoint in cross_session mode"
assert n_val_xs_mode > n_val_trial_mode
print(f"[PASS] cross_session gives {n_val_xs_mode} val trials vs {n_val_trial_mode}, from a "
      f"session the model never trains on -- at the cost of halved training data "
      f"({len(fit2)} vs {len(fit_idx)} windows), exactly the documented trade-off.")

# ---------------------------------------------------------------------------
banner("FIX 1d: cross_session split is DETERMINISTIC across seeds; trial mode is not")
xs_splits, tr_splits = set(), set()
for seed in (42, 999, 2024):
    _, b = cross_session_train_val_split(session_ids[train_idx], val_session)
    xs_splits.add(tuple(b.tolist()))
    _, bb = trial_level_train_val_split(tv_trials, tv_labels, seed=seed)
    tr_splits.add(tuple(bb.tolist()))
print(f"  distinct partitions across 3 seeds -- cross_session: {len(xs_splits)}  "
      f"trial: {len(tr_splits)}")
assert len(xs_splits) == 1, "cross_session partition must not vary with seed"
print("[PASS] under cross_session the data partition is fixed, so seed-to-seed variance "
      "reflects only model init / batch order -- the partition is not a hidden "
      "source of variance.")

# ---------------------------------------------------------------------------
banner("FIX 1e: validation stays class-balanced in both modes")
for name, vi in [("trial", val_idx), ("cross_session", val2)]:
    cls, cnt = np.unique(labels[vi], return_counts=True)
    print(f"  {name:14}: {dict(zip(cls.tolist(), cnt.tolist()))}")
    assert len(set(cnt.tolist())) == 1, f"{name} validation is class-imbalanced"
print("[PASS] both modes keep all three classes equally represented in validation.")

# ---------------------------------------------------------------------------
banner("FIX 1f: every LOSO fold produces a valid split, in both modes")
for ts in (1, 2, 3):
    tr_i = np.where(session_ids != ts)[0]
    te_i = np.where(session_ids == ts)[0]
    a, b = trial_level_train_val_split(trial_ids[tr_i], labels[tr_i], seed=42)
    vs = int(np.unique(session_ids[tr_i]).min())
    c2, d2 = cross_session_train_val_split(session_ids[tr_i], vs)
    assert not (set(tr_i[b]) & set(te_i)) and not (set(tr_i[d2]) & set(te_i))
    print(f"  test session {ts}: trial-mode val trials="
          f"{len(np.unique(trial_ids[tr_i[b]]))}  cross_session val session={vs} "
          f"(val trials={len(np.unique(trial_ids[tr_i[d2]]))})")
print("[PASS] all three folds valid in both modes; validation is never the test session.")

# ---------------------------------------------------------------------------
banner("FIX 2: cohen_kappa matches sklearn exactly (and is NOT the simplified "
      "(acc-p0)/(1-p0) formula some papers use)")
try:
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(0)
    for trial in range(5):
        yt = rng.integers(0, 3, 200)
        yp = np.where(rng.random(200) < 0.6, yt, rng.integers(0, 3, 200))
        mine = cohen_kappa(yt, yp, num_classes=3)
        theirs = cohen_kappa_score(yt, yp)
        print(f"  trial {trial}: ours={mine:.6f}  sklearn={theirs:.6f}  "
              f"diff={abs(mine - theirs):.2e}")
        assert abs(mine - theirs) < 1e-9
    yt = rng.integers(0, 3, 300); yp = rng.integers(0, 3, 300)
    acc = (yt == yp).mean()
    simplified = (acc - 1 / 3) / (1 - 1 / 3)
    print(f"\n  on random predictions: true Cohen's kappa={cohen_kappa(yt, yp):.4f}  "
          f"vs simplified (acc-p0)/(1-p0)={simplified:.4f}")
    print("[PASS] matches sklearn to <1e-9, and is demonstrably a DIFFERENT quantity "
          "from the simplified chance-corrected formula -- so kappa is comparable "
          "with TSFNet's, but must not be compared against papers using the "
          "simplified version without noting it.")
except ImportError:
    print("  [SKIP] sklearn not available")

# ---------------------------------------------------------------------------
banner("FIX 3/5: CLI flags, seed control and output isolation present in the source")
checks = {
    "--seed flag":                     '"--seed"' in src_text,
    "--val-mode flag":                 '"--val-mode"' in src_text,
    "torch.manual_seed":               "torch.manual_seed(SEED)" in src_text,
    "cudnn.deterministic":             "cudnn.deterministic = True" in src_text,
    "per-config output folder":        'f"seed{SEED}_val{VAL_MODE}"' in src_text,
    "TSFNet-format CSV columns":       "'subject_id':" in src_text and "'mean_kappa':" in src_text,
    "accuracy converted to fraction":  "/ 100.0" in src_text,
    "val_loader passed to train_fold": "train_fold(model, cfg, train_loader, val_loader" in src_text,
    "test evaluated separately":       "evaluate(model, test_loader" in src_text,
    "kappa recorded per fold":         "'kappa':" in src_text,
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    assert v, k
print("[PASS] all wiring present.")

# ---------------------------------------------------------------------------
banner("FIX 1g: the ORIGINAL leak pattern is gone from the source")
assert "train_fold(model, cfg, train_loader, test_loader" not in src_text, \
    "the original leak (test_loader passed as validation) is STILL present"
print("  [PASS] 'train_fold(..., train_loader, test_loader, ...)' no longer appears")
print("[PASS] the test set is no longer used for model selection anywhere.")

print("\nALL MSTFDN FIX TESTS PASSED against the real source file.")

# ---------------------------------------------------------------------------
banner("FIX 7: best checkpoint is saved as a dict and loaded via ['model_state']")
assert "ckpt['model_state']" in src_text and "weights_only=False" in src_text, \
    "checkpoint must be loaded with ckpt['model_state'] and weights_only=False"
assert "model.load_state_dict(torch.load(ckpt_path" not in src_text, \
    "the old broken load (whole dict passed to load_state_dict) is still present"
try:
    import tempfile
    import torch
    m = torch.nn.Linear(3, 2)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ck.pt")
        torch.save({'epoch': 5, 'model_state': m.state_dict(),
                    'val_metrics': {'acc': np.float64(0.5), 'preds': [0, 1]}}, p)
        m2 = torch.nn.Linear(3, 2)
        ck = torch.load(p, map_location='cpu', weights_only=False)
        m2.load_state_dict(ck['model_state'])
        assert torch.equal(m.weight, m2.weight)
    print("[PASS] checkpoint round-trip works.")
except ImportError:
    print("  [SKIP] torch not available for round-trip test")

print("\nALL CHECKPOINT TESTS PASSED.")
