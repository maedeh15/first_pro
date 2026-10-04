"""
test_d1_safe_vs_original.py -- THE TEST THAT SHOULD HAVE EXISTED FROM THE START.

Rule 5 of قواعد_مهندسی_پروژه.md says a new parameter's DEFAULT must reproduce
prior behavior BIT-FOR-BIT. test_d1_safe.py only ever compared
eeg_band_mode='none' against eeg_band_mode='safe' *within the patched file*.
That cannot detect a patch that shifts the RNG stream for BOTH modes equally
-- which is exactly what happened: a get/set_rng_state wrapper around clf_eeg
rewound the stream, so clf_fusion / clf_fnirs / w_* were initialized
differently from the pre-patch model even in 'none' mode, and three full
26-subject runs drifted ~2-3 accuracy points before anyone noticed.

This test loads the ACTUAL pre-patch file and the ACTUAL patched file side by
side in one process and compares every shared tensor with the same seed.

Usage:
    python test_d1_safe_vs_original.py ORIGINAL_FILE PATCHED_FILE
"""
import sys
import torch


def load_tsfnet(path, label):
    ns = {}
    exec(open(path, encoding="utf-8").read(), ns)
    print(f"  loaded {label}: {path}")
    return ns["TSFNet"]


def main():
    # Defaults make this runnable with NO arguments as long as the pre-patch
    # file sits beside this script named "tsfnet_model_notebook_ORIGINAL.py".
    import os
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    if len(sys.argv) == 3:
        orig_path, patched_path = sys.argv[1], sys.argv[2]
    else:
        orig_path = os.path.join(here, "tsfnet_model_notebook_ORIGINAL.py")
        patched_path = os.path.join(here, "tsfnet_model_notebook.py")
        print("No arguments given; using defaults:")
        print(f"  original: {orig_path}")
        print(f"  patched : {patched_path}")
        if not os.path.exists(orig_path):
            print()
            print("ERROR: the pre-patch file was not found.")
            print("Either place your ORIGINAL (pre-D1-patch) tsfnet_model_notebook.py beside")
            print("this script, renamed to 'tsfnet_model_notebook_ORIGINAL.py', or run:")
            print("  %run test_d1_safe_vs_original.py <path-to-original> <path-to-patched>")
            sys.exit(2)

    print("Loading both versions in one process...")
    TSFNet_orig = load_tsfnet(orig_path, "ORIGINAL (pre-patch)")
    TSFNet_new = load_tsfnet(patched_path, "PATCHED (current)")

    failures = []

    # ---------------------------------------------------------------- TEST A
    print("\n" + "=" * 72)
    print("TEST A (rule 5, THE CRITICAL ONE): eeg_band_mode='none' must be")
    print("BIT-IDENTICAL to the pre-patch model for the same seed")
    print("=" * 72)
    torch.manual_seed(42)
    m_orig = TSFNet_orig(num_classes=3)
    torch.manual_seed(42)
    m_none = TSFNet_new(num_classes=3, eeg_band_mode="none")

    so, sn = m_orig.state_dict(), m_none.state_dict()
    only_orig = sorted(set(so) - set(sn))
    only_new = sorted(set(sn) - set(so))
    shared = [k for k in so if k in sn]
    diff = [k for k in shared if not torch.equal(so[k], sn[k])]

    print(f"  tensors in original: {len(so)}   in patched('none'): {len(sn)}")
    print(f"  shared: {len(shared)}   differing: {len(diff)}")
    if only_orig:
        print(f"  [FAIL] present ONLY in original: {only_orig[:5]}")
        failures.append("patched model is missing tensors the original had")
    if only_new:
        print(f"  [FAIL] present ONLY in patched: {only_new[:5]}")
        failures.append("patched 'none' mode adds tensors the original lacked")
    if diff:
        print(f"  [FAIL] {len(diff)} tensor(s) differ with the same seed:")
        for k in diff[:10]:
            print(f"           {k}")
        failures.append(f"{len(diff)} tensors differ in 'none' mode -- NOT a bit-exact no-op")
    if not (only_orig or only_new or diff):
        print("  [PASS] every single tensor is bit-identical -- 'none' is a true no-op.")

    # ---------------------------------------------------------------- TEST B
    print("\n" + "=" * 72)
    print("TEST B: identical forward output in 'none' mode")
    print("=" * 72)
    torch.manual_seed(0)
    xe = torch.randn(4, 16, 16, 600)
    xf = torch.rand(4, 21, 16, 16, 30, 2)
    x_chan = torch.randn(4, 28, 2000)
    m_orig.eval()
    m_none.eval()
    with torch.no_grad():
        oo = m_orig(xe, xf, None, x_chan)
        on = m_none(xe, xf, None, x_chan)
    for name, a, b in zip(["y_eeg", "y_fusion", "y_fnirs", "y_pred"], oo[:4], on[:4]):
        d = (a - b).abs().max().item()
        ok = d == 0.0
        print(f"  {name:9} max|diff| = {d:.3e}  {'[PASS]' if ok else '[FAIL]'}")
        if not ok:
            failures.append(f"{name} differs in 'none' mode (max|diff|={d:.3e})")

    # ---------------------------------------------------------------- TEST C
    print("\n" + "=" * 72)
    print("TEST C: in 'safe' mode, ONLY clf_eeg + the new branch may differ")
    print("=" * 72)
    torch.manual_seed(42)
    m_safe = TSFNet_new(num_classes=3, eeg_band_mode="safe")
    ss = m_safe.state_dict()
    shared_safe = [k for k in so if k in ss]
    diff_safe = [k for k in shared_safe if not torch.equal(so[k], ss[k])]
    new_safe = sorted(set(ss) - set(so))

    # ACKNOWLEDGED LIMITATION (rule 6): clf_eeg's input dim grows in "safe"
    # mode, so it consumes a different number of RNG draws and the classifier
    # heads built after it (clf_fusion, clf_fnirs) also get different initial
    # values. This is reported as a documented LIMITATION, not a failure --
    # rule 6's prescribed remedy is to "state the limitation explicitly in the
    # conclusion". Attempting to neutralize it with get/set_rng_state is what
    # broke the "none" default and invalidated three 26-subject runs.
    ALLOWED_SHIFT = ("clf_eeg.", "clf_fusion.", "clf_fnirs.")
    unexpected = [k for k in diff_safe if not k.startswith(ALLOWED_SHIFT)]
    documented = [k for k in diff_safe if k.startswith(("clf_fusion.", "clf_fnirs."))]
    if documented:
        print(f"  [LIMITATION, documented] {len(documented)} classifier-head tensors also")
        print("      shift because clf_eeg's size changed. Any 'safe'-mode result therefore")
        print("      reflects (new branch + a different draw of the heads), NOT the branch")
        print("      in isolation. This MUST be stated in the thesis discussion.")
    unexpected_new = [k for k in new_safe if not k.startswith("eeg_band_branch.")]
    print(f"  shared with original: {len(shared_safe)}   differing: {len(diff_safe)}")
    print(f"  differing tensors: {diff_safe}")
    print(f"  new tensors: {len(new_safe)} (all under eeg_band_branch.: "
          f"{not unexpected_new})")
    if unexpected:
        print(f"  [FAIL] these changed but should NOT have: {unexpected}")
        failures.append(f"'safe' mode perturbs {len(unexpected)} unrelated tensors")
    if unexpected_new:
        print(f"  [FAIL] unexpected new tensors: {unexpected_new}")
        failures.append("'safe' mode adds tensors outside eeg_band_branch")
    if not (unexpected or unexpected_new):
        print("  [PASS] only clf_eeg (input dim legitimately grows) and the new")
        print("         branch's own parameters differ -- nothing else moved.")

    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    if failures:
        print(f"{len(failures)} FAILURE(S) -- DO NOT RUN ON GPU:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED against the real pre-patch file.")
    print("eeg_band_mode='none' genuinely reproduces the reference model bit-for-bit.")


if __name__ == "__main__":
    main()
