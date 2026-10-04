"""
channel_sensitivity_probe.py -- C-2 post-hoc audit #3 (~20 minutes, GPU optional).

WHAT IT ANSWERS: how much does the C-2 result depend on the two channels that
had to be APPROXIMATED when mapping this project's montage onto LaBraM's?

THE ISSUE: 26 of this project's 28 electrodes match a canonical LaBraM channel
name exactly. The remaining two -- AFF5h and AFF6h, which are 10-5-system
intermediate positions -- do not exist in LaBraM's 128-name vocabulary, so
they were mapped to their nearest canonical 10-20 neighbours, AF7 and AF8.

That approximation is not free: AFF5h/AFF6h sit over PREFRONTAL cortex, which
is precisely the region most implicated in working-memory load. If the frozen
representation is sensitive to how those two channels are labelled, that is a
real limitation of C-2 and belongs in the thesis; if it is insensitive, the
approximation can be dismissed in one sentence with evidence behind it.

THE FOUR CONFIGURATIONS COMPARED:
  af7_af8   AFF5h->AF7, AFF6h->AF8   (what every C-2 run actually used)
  f5_f6     AFF5h->F5,  AFF6h->F6    (the other plausible nearest neighbour)
  fp1_fp2   AFF5h->FP1, AFF6h->FP2   (a deliberately WORSE, more anterior choice,
                                      included as a negative control: if even
                                      this makes no difference, the model is
                                      simply not using channel identity much)
  dropped   the two channels removed entirely (26 channels)

METHOD: frozen backbone only, then a plain leave-one-session-out logistic
regression -- identical harness to input_scale_probe.py, so the two audits are
directly comparable. No TSFNet training.

USAGE (run from the folder holding labram_eeg_branch_v3.py):
    python channel_sensitivity_probe.py --data-root "E:/.../preprocessed" \
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


# index positions of AFF5h and AFF6h inside PROJECT_EEG_CHANNELS
AFF_INDICES = (2, 3)

CONFIGS = {
    "af7_af8":  {"AFF5h": "AF7", "AFF6h": "AF8"},   # what C-2 actually used
    "f5_f6":    {"AFF5h": "F5",  "AFF6h": "F6"},
    "fp1_fp2":  {"AFF5h": "FP1", "AFF6h": "FP2"},   # negative control
    "dropped":  None,                                # remove both channels
}


def load_subject(data_root, sid):
    sub_dir = os.path.join(data_root, sid)
    def _p(name):
        return os.path.join(sub_dir, f"{sid}_{name}.npy")
    chan = np.load(_p("eeg_channel_windows")).astype(np.float32)
    y = np.load(_p("labels")).astype(np.int64)
    sess = np.load(_p("session_ids")).astype(np.int64)
    if not (chan.shape[0] == y.shape[0] == sess.shape[0]):
        raise ValueError(f"{sid}: length mismatch")
    return chan, y, sess


def loso_score(emb, y, sess):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    accs, kappas = [], []
    for test_sess in sorted(np.unique(sess)):
        tr, te = sess != test_sess, sess == test_sess
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(emb[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(emb[tr]), y[tr])
        pred = clf.predict(sc.transform(emb[te]))
        accs.append(accuracy_score(y[te], pred))
        kappas.append(cohen_kappa_score(y[te], pred))
    if not accs:
        return float("nan"), float("nan")
    return float(np.mean(accs)), float(np.mean(kappas))


def build_head(name, subs, readout, verbose=False):
    """Builds a frozen head for one channel configuration.

    Returns (head, keep_indices) where keep_indices selects the channels of the
    raw (N, 28, T) array that this configuration actually feeds to the model.
    """
    import labram_eeg_branch_v3 as lb

    base = list(lb.PROJECT_EEG_CHANNELS)
    if subs is None:
        keep = [i for i in range(len(base)) if i not in AFF_INDICES]
        ch_names = [base[i] for i in keep]
    else:
        keep = list(range(len(base)))
        ch_names = list(base)

    # Temporarily swap in this configuration's substitution table, since
    # resolve_labram_channel_names() reads it from module scope.
    original = lb.NONCANONICAL_SUBSTITUTIONS
    try:
        lb.NONCANONICAL_SUBSTITUTIONS = subs if subs is not None else {}
        head = lb.LaBraMFrozenEEGHeadV3(
            pretrained=True, num_classes=3, ch_names=ch_names,
            readout=readout, verbose=verbose)
    finally:
        lb.NONCANONICAL_SUBSTITUTIONS = original
    return head, keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--subjects", nargs="+",
                    default=["VP001", "VP002", "VP003", "VP004"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                    choices=list(CONFIGS.keys()))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--readout", default="mean", choices=["mean", "cls", "cls_mean"])
    args = ap.parse_args()

    import labram_eeg_branch_v3 as lb

    print(f"device={args.device} | readout={args.readout!r}")
    print(f"configs={args.configs}\nsubjects={args.subjects}\n")

    data = {}
    for sid in args.subjects:
        try:
            data[sid] = load_subject(args.data_root, sid)
            print(f"  loaded {sid}: {data[sid][0].shape[0]} windows")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [skip] {sid}: {exc}")
    if not data:
        print("\nERROR: no subjects loaded. Check --data-root / --subjects.")
        return

    results = {}
    for name in args.configs:
        subs = CONFIGS[name]
        t0 = time.time()
        # The singleton cache is keyed by readout/scale only, NOT by channel
        # names, so it must be cleared between configurations or every config
        # after the first would silently reuse the first one's head.
        lb._SHARED_FROZEN_HEADS.clear()
        head, keep = build_head(name, subs, args.readout)
        per_subject = []
        for sid, (chan, y, sess) in data.items():
            emb = lb.precompute_embeddings(head, chan[:, keep, :],
                                           batch_size=args.batch_size,
                                           device=args.device).numpy()
            per_subject.append((sid, *loso_score(emb, y, sess)))
        del head
        lb._SHARED_FROZEN_HEADS.clear()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        accs = [a for _, a, _ in per_subject if a == a]
        kaps = [k for _, _, k in per_subject if k == k]
        results[name] = (float(np.mean(accs)), float(np.mean(kaps)))
        desc = "2 channels removed (26 total)" if subs is None else str(subs)
        print(f"\nconfig={name:<9} mean_acc={np.mean(accs) * 100:6.2f}%  "
              f"mean_kappa={np.mean(kaps):.4f}   ({time.time() - t0:.0f}s)  {desc}")
        for sid, a, k in per_subject:
            print(f"    {sid}: acc={a * 100:6.2f}%  kappa={k:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY (frozen-embedding quality vs channel-mapping choice)")
    print("=" * 70)
    print(f"{'config':>10} {'acc %':>9} {'kappa':>9}   note")
    for name in args.configs:
        acc, kappa = results[name]
        note = {"af7_af8": "<- what C-2 actually used",
                "fp1_fp2": "   (deliberately worse: negative control)",
                "dropped": "   (26 channels)"}.get(name, "")
        print(f"{name:>10} {acc * 100:>9.2f} {kappa:>9.4f}   {note}")

    if "af7_af8" in results:
        ref = results["af7_af8"][0]
        spread = max(a for a, _ in results.values()) - min(a for a, _ in results.values())
        print(f"\n  spread across configurations: {spread * 100:.2f} points")
        if spread < 0.02:
            print("\n  => INSENSITIVE. How those two prefrontal channels are labelled barely")
            print("     moves the frozen representation. The AFF5h/AFF6h approximation can")
            print("     be dismissed in the thesis with this measurement as evidence.")
            print("     (Caveat: if even the fp1_fp2 negative control changes nothing, the")
            print("     model may be making little use of channel identity at all, which is")
            print("     itself worth one sentence.)")
        else:
            best = max(results.items(), key=lambda kv: kv[1][0])
            print(f"\n  => SENSITIVE: the choice of channel mapping moves the frozen")
            print(f"     representation by {spread * 100:.2f} points, so it is NOT a free choice")
            print("     and must be reported as a limitation of C-2.")
            if best[0] == "af7_af8":
                print(f"     Fortunately the mapping C-2 actually used ({ref * 100:.2f}%) is the best")
                print("     of those tested, so the rejection is not an artifact of a poor")
                print("     channel choice -- state exactly that in the thesis.")
            else:
                print(f"     WARNING: {best[0]!r} ({best[1][0] * 100:.2f}%) beats the mapping C-2 used")
                print(f"     ({ref * 100:.2f}%) by {(best[1][0] - ref) * 100:.2f} points. Re-run C-2 with that")
                print("     mapping before treating the rejection as final.")

    print("\n  NOTE: linear probe on the EEG branch over a few subjects -- these")
    print("  numbers measure EMBEDDING QUALITY and are not comparable to the")
    print("  project's end-to-end figures.")


if __name__ == "__main__":
    main()
