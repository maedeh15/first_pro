"""
input_scale_probe.py -- C-2 post-hoc audit #2 (~15-30 minutes, GPU optional).

WHAT IT ANSWERS: is the weak C-2 result caused by an INPUT-CONTRACT mismatch
rather than by the foundation model being unhelpful?

THE HYPOTHESIS: LaBraM's published preprocessing scales EEG so that "the unit
is 0.1 mV, to guarantee the value mainly between -1 to 1" (Jiang et al. 2024).
This project's `_eeg_channel_windows.npy` is per-channel z-scored instead, so
each channel has unit variance and values mostly within +-3. Two things differ:

  (a) OVERALL SCALE -- fixable here, and that is what this script tests.
  (b) BETWEEN-CHANNEL AMPLITUDE STRUCTURE -- destroyed by per-channel
      z-scoring and NOT recoverable from the saved arrays. Testing (b)
      requires re-running preprocessing to also save a microvolt-scale copy;
      see the companion roadmap document. This script deliberately does not
      pretend to test (b).

So: a NEGATIVE result here rules out the cheap half of the hypothesis but
leaves (b) open. A POSITIVE result (some scale clearly beats 1.0) would mean
the C-2 runs were handed out-of-contract inputs and must be redone.

WHAT IT DOES: for each scale, runs ONLY the frozen backbone over a few
subjects' windows, then fits a plain logistic regression under this project's
own leave-one-session-out protocol. No TSFNet training at all -- this is the
"cheap route before spending GPU" rule the project follows.

USAGE (run from the folder holding labram_eeg_branch_v3.py):
    python input_scale_probe.py --data-root "E:/.../preprocessed" \
        --subjects VP001 VP002 VP003 VP004
"""

# ---------------------------------------------------------------------------
# Environment hygiene, applied BEFORE braindecode is imported anywhere.
#
# 1. braindecode's `eegpt` module builds an MNE montage at IMPORT time, which
#    emits a FutureWarning about 'standard_1020' being renamed. It comes from
#    inside the library, is unrelated to anything this script does, and cannot
#    be fixed here -- so it is silenced rather than left to clutter the output.
#    (Note: this script's own model path does NOT use 3D coordinates at all;
#    LaBraM maps electrodes to its learned position embedding by channel NAME.)
#
# 2. If a local Hugging Face cache is present, run fully offline. Without this,
#    every head construction pings the Hub and prints an "unauthenticated
#    requests" warning, and risks rate-limiting partway through a long probe.
# ---------------------------------------------------------------------------
import os as _os
import warnings as _warnings

_warnings.filterwarnings("ignore", category=FutureWarning, module=r"braindecode\..*")
_warnings.filterwarnings("ignore", message=r".*standard_10\d\d.*")

if "HF_HOME" in _os.environ and _os.path.isdir(_os.environ["HF_HOME"]):
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")



import argparse
import os
import time

import numpy as np
import torch


DEFAULT_SCALES = [0.1, 0.25, 0.5, 1.0, 2.0]


def load_subject(data_root, sid):
    """Loads one subject's channel-space windows, labels and session ids."""
    sub_dir = os.path.join(data_root, sid)
    def _p(name):
        return os.path.join(sub_dir, f"{sid}_{name}.npy")
    chan = np.load(_p("eeg_channel_windows"))       # (N, 28, 2000)
    y = np.load(_p("labels"))                       # (N,)
    sess = np.load(_p("session_ids"))               # (N,)
    if chan.shape[0] != y.shape[0] or chan.shape[0] != sess.shape[0]:
        raise ValueError(f"{sid}: length mismatch chan={chan.shape[0]} "
                         f"labels={y.shape[0]} sessions={sess.shape[0]}")
    return chan.astype(np.float32), y.astype(np.int64), sess.astype(np.int64)


def loso_score(emb, y, sess):
    """Leave-one-session-out logistic regression; returns (mean_acc, mean_kappa).

    A plain linear model on frozen features is used deliberately: it is the
    same probe family as C-2's own trainable head, so the comparison across
    scales isolates the REPRESENTATION, not the classifier.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    accs, kappas = [], []
    for test_sess in sorted(np.unique(sess)):
        tr = sess != test_sess
        te = sess == test_sess
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        # Standardizing the FEATURES (not the raw EEG) is fit on train only.
        sc = StandardScaler().fit(emb[tr])
        # NOTE: `multi_class` was removed in recent scikit-learn; the default
        # is already multinomial for >2 classes, so it is simply not passed.
        clf = LogisticRegression(max_iter=2000)
        clf.fit(sc.transform(emb[tr]), y[tr])
        pred = clf.predict(sc.transform(emb[te]))
        accs.append(accuracy_score(y[te], pred))
        kappas.append(cohen_kappa_score(y[te], pred))
    if not accs:
        return float("nan"), float("nan")
    return float(np.mean(accs)), float(np.mean(kappas))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="folder containing one sub-folder per subject")
    ap.add_argument("--subjects", nargs="+",
                    default=["VP001", "VP002", "VP003", "VP004"])
    ap.add_argument("--scales", nargs="+", type=float, default=DEFAULT_SCALES)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--readout", default="mean", choices=["mean", "cls", "cls_mean"])
    args = ap.parse_args()

    from labram_eeg_branch_v3 import LaBraMFrozenEEGHeadV3, precompute_embeddings

    print(f"device={args.device} | readout={args.readout!r} | scales={args.scales}")
    print(f"subjects={args.subjects}\n")

    # Load every subject ONCE; the raw arrays are reused across all scales.
    data = {}
    for sid in args.subjects:
        try:
            data[sid] = load_subject(args.data_root, sid)
            print(f"  loaded {sid}: {data[sid][0].shape[0]} windows")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [skip] {sid}: {exc}")
    if not data:
        print("\nERROR: no subjects loaded. Check --data-root and --subjects.")
        return

    # The backbone is identical for every scale (scaling happens on the INPUT,
    # inside embed()), so build one head per scale but reuse nothing else.
    results = {}
    for scale in args.scales:
        t0 = time.time()
        head = LaBraMFrozenEEGHeadV3(
            pretrained=True, num_classes=3, input_scale=scale,
            readout=args.readout, verbose=False)
        per_subject = []
        for sid, (chan, y, sess) in data.items():
            emb = precompute_embeddings(head, chan,
                                        batch_size=args.batch_size,
                                        device=args.device).numpy()
            acc, kappa = loso_score(emb, y, sess)
            per_subject.append((sid, acc, kappa))
        del head
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        accs = [a for _, a, _ in per_subject if a == a]
        kaps = [k for _, _, k in per_subject if k == k]
        results[scale] = (float(np.mean(accs)), float(np.mean(kaps)), per_subject)
        print(f"\nscale={scale:<5} mean_acc={np.mean(accs) * 100:6.2f}%  "
              f"mean_kappa={np.mean(kaps):.4f}   ({time.time() - t0:.0f}s)")
        for sid, a, k in per_subject:
            print(f"    {sid}: acc={a * 100:6.2f}%  kappa={k:.4f}")

    # ---- verdict ----
    print("\n" + "=" * 68)
    print("SUMMARY (frozen-embedding quality vs input scale)")
    print("=" * 68)
    print(f"{'scale':>8} {'acc %':>9} {'kappa':>9}   note")
    baseline = results.get(1.0, (float('nan'),))[0]
    best_scale, best_acc = None, -1.0
    for scale in args.scales:
        acc, kappa, _ = results[scale]
        note = "<- current pipeline (z-scored, scale=1.0)" if scale == 1.0 else ""
        print(f"{scale:>8} {acc * 100:>9.2f} {kappa:>9.4f}   {note}")
        if acc > best_acc:
            best_scale, best_acc = scale, acc

    print()
    if baseline != baseline:
        print("  scale=1.0 was not run, so no baseline to compare against.")
    else:
        gain = (best_acc - baseline) * 100
        print(f"  best scale = {best_scale} ({best_acc * 100:.2f}%), "
              f"vs scale=1.0 ({baseline * 100:.2f}%) -> {gain:+.2f} points")
        if best_scale == 1.0 or gain < 1.0:
            print("\n  => NEGATIVE. Overall input scale is not the bottleneck. The cheap")
            print("     half of the input-contract hypothesis is ruled out; the remaining")
            print("     half (per-channel z-scoring destroying BETWEEN-channel amplitude")
            print("     structure) still requires a preprocessing re-run to test.")
        elif gain < 3.0:
            print("\n  => WEAK POSITIVE. Scale matters a little. Probably not enough on its")
            print("     own to change C-2's verdict, but worth reporting as a limitation.")
        else:
            print("\n  => POSITIVE. The C-2 runs were fed out-of-contract inputs. Re-run C-2")
            print(f"     with input_scale={best_scale} before treating the rejection as final.")

    print("\n  NOTE: these numbers are NOT comparable to the project's headline")
    print("  figures. This is a linear probe on the EEG branch alone, over a few")
    print("  subjects -- it measures EMBEDDING QUALITY, not end-to-end performance.")


if __name__ == "__main__":
    main()
