"""
check_block_order.py -- from the PREPROCESSED data (not from any result file): the n-back order of the 9 blocks of every (subject, session), and whether each session
has ONE order shared by all subjects, and whether the sessions differ from each other.
Reads {sid}_labels.npy (0 = 0-back, 1 = 2-back, 2 = 3-back; checked against the dataset description: marker 7/8/9 and 112/128/144), {sid}_session_ids.npy, {sid}_trial_ids.npy.
Per (subject, session): one label per block (all windows of a block must carry the same label), blocks ordered by trial id. A block that lost ALL its windows to artifact
rejection shortens that sequence; such cases are listed separately instead of being silently compared.
  python check_block_order.py --data-root "E:\\...\\TSFNET_PAPER_EXACT_nback" --subjects 1-26
"""
import argparse, os, sys, tempfile
from collections import Counter
import numpy as np

NB = {0: "0", 1: "2", 2: "3"}


def sequences(data_root, subjects):
    seqs, problems = {}, []
    for i in subjects:
        sid = f"VP{i:03d}"; d = os.path.join(data_root, sid)
        try:
            y = np.load(os.path.join(d, f"{sid}_labels.npy")); s = np.load(os.path.join(d, f"{sid}_session_ids.npy")); t = np.load(os.path.join(d, f"{sid}_trial_ids.npy"))
        except OSError as e:
            problems.append(f"{sid}: cannot read ({e})"); continue
        for ses in sorted(np.unique(s)):
            m = s == ses; seq = []
            for tid in sorted(np.unique(t[m])):
                lab = np.unique(y[m & (t == tid)])
                if len(lab) != 1: problems.append(f"{sid} session {ses} trial {tid}: windows carry labels {lab.tolist()}"); continue
                seq.append(int(lab[0]))
            seqs[(sid, int(ses))] = "".join(NB[v] for v in seq)
    return seqs, problems


def report(seqs, problems):
    sessions = sorted({s for _, s in seqs}); subs = sorted({a for a, _ in seqs})
    print(f"{len(subs)} subjects, {len(seqs)} (subject, session) sequences; labels read as 0 = 0-back, 1 = 2-back, 2 = 3-back\n")
    shared = True; orders = {}
    for ses in sessions:
        c = Counter(v for (sid, s), v in seqs.items() if s == ses); orders[ses] = c
        top, n = c.most_common(1)[0]
        print(f"session {ses}: {len(c)} distinct order(s); most common {','.join(top)} in {n}/{sum(c.values())} subjects")
        for o, k in c.most_common()[1:]:
            print(f"    other: {','.join(o)} ({len(o)} blocks) in {k} subject(s): " + ", ".join(sid for (sid, s), v in seqs.items() if s == ses and v == o)); shared = False
    tops = [orders[s].most_common(1)[0][0] for s in sessions]
    print(f"\nthe sessions differ from each other: {'YES' if len(set(tops)) == len(tops) else 'NO (at least two sessions share the same order)'}")
    print(f"each session has ONE order shared by all subjects: {'YES' if shared and all(len(c) == 1 for c in orders.values()) else 'NO (see the lines above)'}")
    full = [v for v in seqs.values() if len(v) == 9]
    print(f"sequences with all 9 blocks: {len(full)}/{len(seqs)}" + ("" if len(full) == len(seqs) else "  (the others lost a whole block to rejection)"))
    for p in problems: print("PROBLEM:", p)
    return shared and not problems and len(set(tops)) == len(tops)


def self_test():
    d = tempfile.mkdtemp(); orders = {1: "210102021", 2: "102021210", 3: "021210102"}; rng = np.random.RandomState(0)
    for i in (1, 2, 3):
        sid = f"VP{i:03d}"; os.makedirs(os.path.join(d, sid)); y, s, t = [], [], []
        for ses in (1, 2, 3):
            for k, ch in enumerate(orders[ses]):
                n = rng.randint(8, 35); y += [int(ch)] * n; s += [ses] * n; t += [(ses - 1) * 9 + k] * n
        for nm, a in (("labels", y), ("session_ids", s), ("trial_ids", t)): np.save(os.path.join(d, sid, f"{sid}_{nm}.npy"), np.array(a))
    seqs, pr = sequences(d, [1, 2, 3]); assert not pr and seqs[("VP001", 1)] == "320203032" and seqs[("VP002", 2)] == "203032320"
    assert report(seqs, pr) is True
    seqs[("VP003", 2)] = "203032323"; assert report(seqs, pr) is False
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root"); ap.add_argument("--subjects", default="1-26"); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    if not a.data_root: sys.exit("need --data-root")
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2] if "-" in a.subjects else (a.subjects, a.subjects)
    report(*sequences(a.data_root, range(int(lo), int(hi) + 1)))


if __name__ == "__main__":
    main()
