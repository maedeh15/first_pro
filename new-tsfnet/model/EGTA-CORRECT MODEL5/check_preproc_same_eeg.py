"""
check_preproc_same_eeg.py -- after a fNIRS-only preprocessing change, prove that EVERYTHING EXCEPT the fNIRS windows is bit-identical to the reference folder.
For each subject compares {sid}_eeg_windows.npy (in chunks, memory-mapped), _eeg_channel_windows.npy, _labels.npy, _trial_ids.npy, _session_ids.npy (must be identical),
and reports whether _fnirs_windows.npy has the same shape (must) and DIFFERENT values (must, otherwise the filter did nothing).
Exit code 0 only if every subject passes. A single False means the run would change two variables: stop.
  python check_preproc_same_eeg.py --base "E:\\...\\TSFNET_PAPER_EXACT_nback" --new "E:\\...\\TSFNET_PAPER_EXACT_nback_fnirsLP01" --subjects 1-26
"""
import argparse, os, sys, tempfile
import numpy as np

SAME = ("eeg_windows", "eeg_channel_windows", "labels", "trial_ids", "session_ids")


def same_array(p, q, chunk=64):
    if not (os.path.isfile(p) and os.path.isfile(q)): return None
    a, b = np.load(p, mmap_mode="r"), np.load(q, mmap_mode="r")
    if a.shape != b.shape or a.dtype != b.dtype: return False
    return all(np.array_equal(a[i:i + chunk], b[i:i + chunk]) for i in range(0, len(a), chunk))


def check_subject(base, new, sid):
    res = {k: same_array(os.path.join(base, sid, f"{sid}_{k}.npy"), os.path.join(new, sid, f"{sid}_{k}.npy")) for k in SAME}
    pf, qf = os.path.join(base, sid, f"{sid}_fnirs_windows.npy"), os.path.join(new, sid, f"{sid}_fnirs_windows.npy")
    fa, fb = np.load(pf, mmap_mode="r"), np.load(qf, mmap_mode="r")
    res["fnirs_same_shape"] = fa.shape == fb.shape
    res["fnirs_values_differ"] = (not same_array(pf, qf)) if res["fnirs_same_shape"] else None
    return res


def verdict(res):
    ok = all(res[k] in (True, None) for k in SAME) and res["fnirs_same_shape"] and res["fnirs_values_differ"]
    return bool(ok) and res["labels"] is True and res["eeg_windows"] is True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True); ap.add_argument("--new", required=True); ap.add_argument("--subjects", default="1-26"); a = ap.parse_args(argv)
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2] if "-" in a.subjects else (a.subjects, a.subjects); allok = True
    for i in range(int(lo), int(hi) + 1):
        sid = f"VP{i:03d}"
        try:
            r = check_subject(a.base, a.new, sid)
        except OSError as e:
            print(f"[{sid}] cannot read: {e}"); allok = False; continue
        ok = verdict(r); allok &= ok
        print(f"[{sid}] {'PASS' if ok else 'FAIL'} | " + " ".join(f"{k}={'same' if r[k] else 'DIFF' if r[k] is False else 'missing'}" for k in SAME) +
              f" | fnirs shape {'same' if r['fnirs_same_shape'] else 'DIFF'}, values {'differ (expected)' if r['fnirs_values_differ'] else 'IDENTICAL (filter did nothing)'}", flush=True)
    print("\nALL SUBJECTS PASS: only the fNIRS windows changed." if allok else "\nFAILED: do not train on this folder (more than the fNIRS filter changed, or the filter had no effect).")
    return allok


def self_test():
    d = tempfile.mkdtemp(); rng = np.random.RandomState(0)
    for tag in ("base", "new_ok", "new_bad"):
        os.makedirs(os.path.join(d, tag, "VP001"))
    eeg = rng.randn(130, 4, 4, 20).astype(np.float32); ch = rng.randn(130, 28, 50).astype(np.float32); lab = rng.randint(0, 3, 130); fn = rng.randn(130, 3, 4, 4, 30, 2).astype(np.float32)
    for tag in ("base", "new_ok", "new_bad"):
        p = lambda n: os.path.join(d, tag, "VP001", f"VP001_{n}.npy")
        np.save(p("eeg_windows"), eeg + (1e-3 if tag == "new_bad" else 0)); np.save(p("eeg_channel_windows"), ch); np.save(p("labels"), lab)
        np.save(p("trial_ids"), np.arange(130)); np.save(p("session_ids"), np.ones(130, int)); np.save(p("fnirs_windows"), fn if tag == "base" else fn * 0.5)
    assert main(["--base", os.path.join(d, "base"), "--new", os.path.join(d, "new_ok"), "--subjects", "1"]) is True
    assert main(["--base", os.path.join(d, "base"), "--new", os.path.join(d, "new_bad"), "--subjects", "1"]) is False
    print("SELFTEST PASSED")


if __name__ == "__main__":
    self_test() if "--self-test" in sys.argv else sys.exit(0 if main() else 1)
