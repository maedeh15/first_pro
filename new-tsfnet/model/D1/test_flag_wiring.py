"""
test_flag_wiring.py -- catches the class of bug that made three full GPU runs
(seeds 42/999/2024, ~24 GPU-hours) silently meaningless: a CLI flag that is
defined, parsed, stored in a module-level variable, and PRINTED IN THE
ABLATION BANNER, but never actually passed down into run_subject_cross_session
-- so the model was always built with the default.

The existing test_d1_safe.py could not catch this: it constructs TSFNet
DIRECTLY with eeg_band_mode='safe', bypassing the entire CLI -> variable ->
run_subject_cross_session -> TSFNet chain where the break actually was.

THIS IS NOW A MANDATORY CHECK (proposed rule 10 for قواعد_مهندسی_پروژه.md):
  "A flag printed in the [ABLATION] banner is NOT proof it reached the model.
   Verify the full wiring chain statically before trusting any run."

Run: python test_flag_wiring.py
"""
import re
import sys

RUN_PATH = "run_all_subjects.py"
MODEL_PATH = "tsfnet_model_notebook.py"


def _balanced_call(text, start_marker):
    """Extract a full foo(...) call from start_marker to its matching paren."""
    s = text.index(start_marker)
    p = text.index("(", s)
    depth = 0
    for i, ch in enumerate(text[p:], start=p):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[s:i + 1]
    raise ValueError(f"unbalanced parens after {start_marker!r}")


def main():
    run = open(RUN_PATH, encoding="utf-8").read()
    model = open(MODEL_PATH, encoding="utf-8").read()

    call = _balanced_call(run, "run_subject_cross_session(")
    sig = _balanced_call(model, "def run_subject_cross_session(")
    tsf_call = _balanced_call(model, "TSFNet(num_classes=3, fusion_combination")
    tof_call = _balanced_call(model, "train_one_fold(")

    failures = []

    # ---- CHECK 1: every CLI flag with a module-level variable reaches the call
    cli_args = sorted(set(re.findall(r'add_argument\("--([a-z0-9\-]+)"', run)))
    EXEMPT = {"seed"}  # seed is applied globally before the subject loop, by design
    print("CHECK 1: every CLI flag reaches run_subject_cross_session(...)")
    for a in cli_args:
        if a in EXEMPT:
            continue
        var = a.replace("-", "_").upper()
        py = a.replace("-", "_")
        if f"{var} =" not in run:
            continue  # flag consumed some other way; not a wiring claim
        if var not in call and f"{py}=" not in call:
            failures.append(f"CLI --{a} (var {var}) never reaches run_subject_cross_session")
            print(f"   [FAIL] --{a}")
    if not failures:
        print("   [PASS] all non-exempt CLI flags are forwarded")

    # ---- CHECK 2: every eeg_band_* param is wired all the way to TSFNet
    print("\nCHECK 2: the D-1-safe chain specifically, end to end")
    chain = {
        "eeg_band_mode":        (call, sig, tsf_call),
        "eeg_band_d_model":     (call, sig, tsf_call),
        "eeg_band_embed_dim":   (call, sig, tsf_call),
        "eeg_band_weight_decay": (call, sig, tof_call),  # goes to the optimizer, not the model
    }
    for name, (c, s_, final) in chain.items():
        in_call = f"{name}=" in c
        in_sig = f"{name}=" in s_
        in_final = name in final
        ok = in_call and in_sig and in_final
        status = "PASS" if ok else "FAIL"
        print(f"   [{status}] {name:23} call={in_call} signature={in_sig} destination={in_final}")
        if not ok:
            failures.append(f"{name} is broken in the chain "
                            f"(call={in_call}, sig={in_sig}, dest={in_final})")

    # ---- CHECK 3: banner claims match reality
    print("\nCHECK 3: no flag is printed in the [ABLATION] banner without being wired")
    banner_vars = set(re.findall(r'\{([A-Z_0-9]+)[!:}]', run))
    for var in sorted(banner_vars):
        if not var.startswith("EEG_BAND"):
            continue
        if var not in call:
            failures.append(f"{var} appears in the banner but is not passed to the function")
            print(f"   [FAIL] {var} printed in banner but never forwarded")
    if not any("banner" in f for f in failures):
        print("   [PASS] every EEG_BAND_* banner variable is genuinely forwarded")

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} WIRING FAILURE(S):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL WIRING CHECKS PASSED -- flags in the banner genuinely reach the model.")


if __name__ == "__main__":
    main()
