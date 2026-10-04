"""
chance_level_permutation_test.py -- Priority A-3: empirical chance level
=========================================================================
Trial-level label-permutation test for classification accuracy/kappa
significance (Combrisson & Jerbi, 2015, J. Neurosci. Methods, "Exceeding
chance level by chance"). Zero GPU cost: works directly on already-saved
per-window predictions, no retraining needed.

WHY TRIAL-LEVEL, NOT WINDOW-LEVEL:
Windows belonging to the same n-back trial are not independent (they come
from overlapping/adjacent segments of the same underlying brain response).
Permuting labels at the WINDOW level would break this dependency structure
and make the null distribution too narrow -- systematically underestimating
the true chance level and inflating apparent significance. Permuting at the
TRIAL level (every window of a trial moves together) preserves the
project's actual dependency structure, exactly mirroring the trial-level
train/val split already used elsewhere in this project's protocol.

USAGE (once you have saved per-window predictions for a subject/session):

    from chance_level_permutation_test import trial_level_permutation_test

    result = trial_level_permutation_test(
        true_labels=true_labels,      # (N,) int array, window-level
        pred_labels=pred_labels,      # (N,) int array, window-level model predictions
        trial_ids=trial_ids,          # (N,) int array, which trial each window belongs to
        n_permutations=10000,
        seed=42,
    )
    print(result["observed_accuracy"], result["p_value"], result["chance_95th_percentile"])

To pool across all 26 subjects (recommended for the final reported number),
concatenate true_labels/pred_labels/trial_ids across subjects first, but
make trial_ids GLOBALLY UNIQUE per subject (e.g. f"{subject_id}_{trial_id}")
before pooling, so trial-level permutation never accidentally mixes trials
from different subjects into the same permuted block within one subject.
"""

import numpy as np


def trial_level_permutation_test(true_labels, pred_labels, trial_ids,
                                  n_permutations=10000, seed=42,
                                  metric="accuracy"):
    """
    Parameters
    ----------
    true_labels : (N,) array of int -- window-level ground-truth class.
    pred_labels : (N,) array of int -- window-level model prediction
        (kept FIXED across all permutations -- only the true-label
        assignment is shuffled, which is what "permutation test for
        classification accuracy" means: are the model's actual outputs
        associated with the true labels more than chance would predict).
    trial_ids : (N,) array -- trial identifier per window. All windows
        sharing a trial_id are assumed to share the same true label (the
        n-back block design guarantees this); this is asserted below.
    n_permutations : number of label-shuffle iterations for the null
        distribution (10000 is standard in Combrisson & Jerbi 2015 for a
        stable p-value estimate at alpha=0.05; use less, e.g. 1000, for a
        quick look).
    metric : "accuracy" or "kappa".

    Returns
    -------
    dict with:
        observed_accuracy / observed_kappa : the real, unpermuted score.
        null_distribution : (n_permutations,) array of the metric under
            random trial-level label reassignment.
        p_value : Phipson & Smyth (2010) bias-corrected permutation
            p-value: (1 + #{null >= observed}) / (n_permutations + 1).
            The "+1" correction avoids ever reporting p=0, which is not
            a meaningful claim from a finite number of permutations.
        chance_95th_percentile / chance_99th_percentile : empirical
            chance-level thresholds -- the model's real accuracy should
            clear these to be considered "above chance" at those
            confidence levels; this is the direct empirical analogue of
            the naive "1/3 for 3 classes" assumption.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    trial_ids = np.asarray(trial_ids)
    assert true_labels.shape == pred_labels.shape == trial_ids.shape

    # ---- Build the trial -> label mapping, and verify the "all windows in
    # a trial share one label" assumption the whole method depends on. ----
    unique_trials, first_idx = np.unique(trial_ids, return_index=True)
    trial_label = {}
    for t in unique_trials:
        labels_in_trial = true_labels[trial_ids == t]
        assert np.all(labels_in_trial == labels_in_trial[0]), (
            f"Trial {t} has more than one distinct true label "
            f"({np.unique(labels_in_trial)}) -- the trial-level permutation "
            f"assumption is violated for this data."
        )
        trial_label[t] = labels_in_trial[0]

    trial_label_array = np.array([trial_label[t] for t in unique_trials])

    def _score(true_arr, pred_arr):
        if metric == "accuracy":
            return float(np.mean(true_arr == pred_arr))
        elif metric == "kappa":
            return _cohen_kappa(true_arr, pred_arr)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    observed = _score(true_labels, pred_labels)

    # Precompute, for each window, which entry of unique_trials it belongs
    # to, so every permutation only needs a fast array lookup, not a
    # per-window python loop.
    trial_to_pos = {t: i for i, t in enumerate(unique_trials)}
    window_trial_pos = np.array([trial_to_pos[t] for t in trial_ids])

    rng = np.random.default_rng(seed)
    null_dist = np.empty(n_permutations)
    for p in range(n_permutations):
        shuffled_trial_labels = rng.permutation(trial_label_array)
        shuffled_true = shuffled_trial_labels[window_trial_pos]  # expand back to window level
        null_dist[p] = _score(shuffled_true, pred_labels)

    p_value = (1 + np.sum(null_dist >= observed)) / (n_permutations + 1)

    result = {
        f"observed_{metric}": observed,
        "null_distribution": null_dist,
        "p_value": p_value,
        "chance_mean": float(null_dist.mean()),
        "chance_95th_percentile": float(np.percentile(null_dist, 95)),
        "chance_99th_percentile": float(np.percentile(null_dist, 99)),
        "n_trials": len(unique_trials),
        "n_windows": len(true_labels),
    }
    return result


def _cohen_kappa(y_true, y_pred):
    """Minimal, dependency-free Cohen's kappa (avoids requiring sklearn
    here); matches sklearn.metrics.cohen_kappa_score for the unweighted
    case, verified in the debug script below."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(y_true)
    cm = np.zeros((len(classes), len(classes)))
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    po = np.trace(cm) / n
    row_marg = cm.sum(axis=1) / n
    col_marg = cm.sum(axis=0) / n
    pe = np.sum(row_marg * col_marg)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


# ============================================================================
# Numeric debug / verification script -- run this file directly
# (`python chance_level_permutation_test.py`) before trusting the function
# on real data, per this project's standing methodology rule: every new
# analysis gets a numeric test on synthetic data first.
# ============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---- Check 1: Cohen's kappa matches sklearn on a known case ----
    try:
        from sklearn.metrics import cohen_kappa_score
        y_t = rng.integers(0, 3, 500)
        y_p = rng.integers(0, 3, 500)
        k_mine = _cohen_kappa(y_t, y_p)
        k_sklearn = cohen_kappa_score(y_t, y_p)
        assert abs(k_mine - k_sklearn) < 1e-9, (k_mine, k_sklearn)
        print(f"[PASS] check 1: _cohen_kappa matches sklearn "
              f"({k_mine:.6f} vs {k_sklearn:.6f}).")
    except ImportError:
        print("[SKIP] check 1: sklearn not available to cross-check kappa.")

    # ---- Check 2: a genuinely random/uninformative model should NOT be
    # judged significantly above chance (p should be large, e.g. > 0.05
    # most of the time; we assert a loose bound to avoid a flaky test). ----
    n_trials, windows_per_trial, n_classes = 90, 3, 3
    trial_ids = np.repeat(np.arange(n_trials), windows_per_trial)
    trial_true = rng.integers(0, n_classes, n_trials)
    true_labels = np.repeat(trial_true, windows_per_trial)
    random_preds = rng.integers(0, n_classes, len(true_labels))  # no relation to true_labels at all

    res_random = trial_level_permutation_test(
        true_labels, random_preds, trial_ids, n_permutations=2000, seed=1)
    assert res_random["p_value"] > 0.05, (
        f"A genuinely random predictor was flagged as significant "
        f"(p={res_random['p_value']:.4f}) -- something is wrong."
    )
    print(f"[PASS] check 2: a random/uninformative predictor is correctly "
          f"NOT flagged as above chance (observed_accuracy="
          f"{res_random['observed_accuracy']:.3f}, p={res_random['p_value']:.4f}, "
          f"empirical chance ~{res_random['chance_mean']:.3f}).")

    # ---- Check 3: a near-perfect model should be flagged as extremely
    # significant (p should hit the floor set by n_permutations). ----
    good_preds = true_labels.copy()
    flip_idx = rng.choice(len(good_preds), size=int(0.05 * len(good_preds)), replace=False)
    good_preds[flip_idx] = (good_preds[flip_idx] + 1) % n_classes  # 5% noise only

    res_good = trial_level_permutation_test(
        true_labels, good_preds, trial_ids, n_permutations=2000, seed=2)
    assert res_good["p_value"] < 0.001, (
        f"A near-perfect predictor (95% accurate) was NOT flagged as highly "
        f"significant (p={res_good['p_value']:.5f}) -- something is wrong."
    )
    print(f"[PASS] check 3: a near-perfect predictor (95% accuracy) is correctly "
          f"flagged as far above chance (p={res_good['p_value']:.5f}, "
          f"observed_accuracy={res_good['observed_accuracy']:.3f}).")

    # ---- Check 4: the "all windows in a trial share one label" assertion
    # actually fires when violated (protects against silently wrong input). ----
    try:
        bad_true = true_labels.copy()
        bad_true[0] = (bad_true[0] + 1) % n_classes  # break trial 0's label consistency
        trial_level_permutation_test(bad_true, good_preds, trial_ids, n_permutations=10)
        raise AssertionError("expected an AssertionError for inconsistent trial labels")
    except AssertionError as e:
        assert "distinct true label" in str(e)
        print("[PASS] check 4: inconsistent per-trial labels are correctly rejected "
              "with a clear error, instead of silently producing a wrong result.")

    # ---- Check 5: p-value never reports exactly 0, even for an extreme case
    # (Phipson & Smyth correction) ----
    assert res_good["p_value"] >= 1 / (2000 + 1)
    print(f"[PASS] check 5: p-value is bias-corrected and never exactly 0 "
          f"(floor = 1/(n_permutations+1) = {1/2001:.6f}).")

    print("\nALL CHECKS PASSED.")
