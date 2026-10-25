"""
pretrain_cross_subject.py -- priority-XSUBJ: cross-subject pretraining for
the per-subject TSFNet, without leaking any target subject.

WHY
===
Every subject's model is currently trained from random weights on ONE
subject's two training sessions: 18 independent n-back blocks (35 heavily
overlapping windows each) for a ~415K-parameter network that reaches
train_acc=1.0 by epoch 4. The other 25 subjects' 675 blocks are never used
for that subject. Pretraining on other subjects and then fine-tuning on the
target subject's own two training sessions changes the DATA regime rather
than the model, which is what the project's own history says is needed
(capacity-adding interventions fail; the model has nothing to feed them).

PROTOCOL (no leakage, test protocol unchanged)
==============================================
Subjects are split into --n-groups groups, deterministically (sorted IDs,
subject i -> group i % n_groups; independent of the seed). For group g:
  * pretraining POOL   = every subject NOT in group g (all 3 sessions),
  * pretraining VAL    = --n-val-subjects subjects drawn from that pool
                         (seeded), used only for early stopping,
  * fine-tune TARGETS  = the subjects IN group g.
One model is pretrained per group. run_all_subjects.py --pretrained-dir then
starts all three folds of each target subject from its group's model and runs
the usual leave-one-session-out fine-tuning/evaluation unchanged (same
train/val split, same AdaBN on the held-out session, same metrics).
A target subject never contributes a single window to the model it is
initialized from; the manifest records the pools and run_all_subjects.py
re-checks this before training.

--n-groups trades pretraining data against compute:
    2  -> pools of 13 subjects, 2 pretraining runs
    4  -> pools of 19-20 subjects, 4 runs   (default)
    26 -> pools of 25 subjects, 26 runs (leave-one-subject-out; slowest)

MEMORY
======
One subject's preprocessed windows are ~3 GB (EEG 2.05 MB + fNIRS 1.29 MB per
window, ~888 windows). A 19-subject pool is ~56 GB, more than the RAM, so the
pool is MEMORY-MAPPED (np.load(mmap_mode="r")) and read from disk on demand.
--window-stride subsamples windows WITHIN each block: consecutive windows
overlap by 90% (1 s step, 10 s window), so stride 3 (every 3rd window, 3 s
apart) keeps nearly all information while cutting I/O and epoch time ~3x.

TRAINING LOOP
=============
The pretraining loop IS the project's train_one_fold(): same loss, optimizer,
parameter groups, early stopping on validation accuracy, and checkpointing.
adabn=False (there is no single target session to adapt to here, and the
fine-tuning run applies AdaBN on each held-out session anyway), and stage 2
is disabled so the saved file is exactly the best stage-1 checkpoint.

Defaults reproduce run60's training flags (normalize-ypred, no-decay-scalars,
stage2 none) and architecture flags (egta, sigmoid EFGF gate, init -2.0).
Architecture flags MUST match the fine-tuning run; a mismatch is caught there
by load_state_dict(strict=True).

USAGE
=====
    python pretrain_cross_subject.py --seed 42
    # then:
    python run_all_subjects.py --seed 42 <run60 flags> --pretrained-dir <out dir printed above>

The script is resumable: a group whose checkpoint is already recorded in the
manifest is skipped. Use --groups 0,2 to (re)run specific groups only.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--data-root", type=str,
               default=r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback")
p.add_argument("--out-dir", type=str, default=None,
               help="Default: <data-root>/pretrain_<id>_seed<SEED>")
p.add_argument("--n-groups", type=int, default=4)
p.add_argument("--n-val-subjects", type=int, default=2)
p.add_argument("--window-stride", type=int, default=3,
               help="keep every k-th window within each block (1 = all windows)")
p.add_argument("--groups", type=str, default=None, help="comma list of group indices to run")
p.add_argument("--session-protocol", type=str, default="pooled", choices=["pooled", "loso"],
               help="pooled (default, prior behavior): one model per group trained on ALL 3 sessions "
                    "of the pool subjects. loso: one model per (group, held-out session s) trained "
                    "only on the pool subjects' sessions != s, so a target subject's test session s "
                    "is never seen in ANY subject's data -- the leave-one-session-out protocol "
                    "applied to pretraining too. 3x the pretraining cost.")
p.add_argument("--subjects", type=str, default="1-26")
p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--num-workers", type=int, default=0,
               help="DataLoader workers; 0 is safest on Windows")
# training (defaults = run60)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--max-epochs", type=int, default=30,
               help="one pretraining epoch is ~10x a per-subject epoch even at stride 3")
p.add_argument("--patience", type=int, default=6)
p.add_argument("--lambda-eeg", type=float, default=0.2)
p.add_argument("--lambda-fnirs", type=float, default=0.2)
p.add_argument("--no-normalize-ypred", dest="normalize_ypred", action="store_false", default=True)
p.add_argument("--decay-scalars", dest="no_decay_scalars", action="store_false", default=True)
# architecture (defaults = run60; must match the fine-tuning run)
p.add_argument("--fnirs-temporal-align", type=str, default="egta", choices=["none", "egta"])
p.add_argument("--efgf-gate-mode", type=str, default="sigmoid", choices=["clamp", "sigmoid"])
p.add_argument("--efgf-gate-init", type=float, default=-2.0)
p.add_argument("--fusion-combination", type=str, default="paper_mean",
               choices=["paper_mean", "wsum", "gated"])
p.add_argument("--fnirs-num-segments", type=int, default=21)
p.add_argument("--cafe-d-model", type=int, default=64)
p.add_argument("--dropout", type=float, default=0.5)
p.add_argument("--cafe-dropout", type=float, default=0.3)
p.add_argument("--conv-dropout", type=float, default=None)
p.add_argument("--conv-dropout-type", type=str, default="channel", choices=["channel", "element"])
p.add_argument("--eeg-temporal-mode", type=str, default="gap", choices=["gap", "gap_logvar", "bilstm"])
p.add_argument("--fusion-fnirs-inject", type=str, default="none", choices=["none", "stage1", "both"])
p.add_argument("--fnirs-num-chromophores", type=int, default=2, choices=[2, 3])
ARGS = p.parse_args()

if ARGS.window_stride < 1 or ARGS.n_groups < 2 or ARGS.n_val_subjects < 1:
    sys.exit("ERROR: need --window-stride >= 1, --n-groups >= 2, --n-val-subjects >= 1")

# ---- project definitions (same mechanism as run_all_subjects.py) ----------
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.getcwd()
if "TSFNet" not in globals():
    exec(open(os.path.join(_this_dir, "tsfnet_model_notebook.py"), encoding="utf-8").read(),
         globals())

ARCH = dict(
    num_classes=3, dropout=ARGS.dropout, cafe_dropout=ARGS.cafe_dropout,
    cafe_d_model=ARGS.cafe_d_model, fusion_combination=ARGS.fusion_combination,
    eeg_temporal_mode=ARGS.eeg_temporal_mode, fnirs_num_segments=ARGS.fnirs_num_segments,
    fnirs_num_chromophores=ARGS.fnirs_num_chromophores,
    fnirs_temporal_align=ARGS.fnirs_temporal_align, efgf_gate_mode=ARGS.efgf_gate_mode,
    efgf_gate_init=ARGS.efgf_gate_init, conv_dropout=ARGS.conv_dropout,
    conv_dropout_type=ARGS.conv_dropout_type, fusion_fnirs_inject=ARGS.fusion_fnirs_inject,
)
TRAIN = dict(lr=ARGS.lr, weight_decay=ARGS.weight_decay, max_epochs=ARGS.max_epochs,
             patience=ARGS.patience, lambda_eeg=ARGS.lambda_eeg, lambda_fnirs=ARGS.lambda_fnirs,
             normalize_ypred=ARGS.normalize_ypred, no_decay_scalars=ARGS.no_decay_scalars,
             window_stride=ARGS.window_stride, batch_size=ARGS.batch_size,
             n_groups=ARGS.n_groups, n_val_subjects=ARGS.n_val_subjects)
if ARGS.session_protocol != "pooled":
    # Only added when non-default, so the pooled run keeps its existing id (32dedf7a etc.)
    # while a loso run can never share an id -- or an output folder -- with a pooled one.
    TRAIN["session_protocol"] = ARGS.session_protocol
# The id identifies everything that determines the pretrained weights except
# the seed, so the three seeds of one configuration share an id.
RUN_ID = hashlib.md5(json.dumps({"arch": ARCH, "train": TRAIN}, sort_keys=True)
                     .encode()).hexdigest()[:8]


def parse_subjects(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [f"VP{i:03d}" for i in out]


def assign_groups(subjects, n_groups):
    """Deterministic and seed-independent: sorted IDs, subject i -> group i % n."""
    groups = {g: [] for g in range(n_groups)}
    for i, sid in enumerate(sorted(subjects)):
        groups[i % n_groups].append(sid)
    return groups


class MmapWindowDataset(Dataset):
    """Windows of several subjects read lazily from memory-mapped .npy files.

    Returns the same (eeg, fnirs, label) 3-tuple as NBackDataset for a
    subject folder without optional files, so train_one_fold's _unpack_batch
    handles it unchanged.
    """

    def __init__(self, data_root, subjects, stride, exclude_session=None):
        self.eeg, self.fnirs, self.labels, self.index = [], [], [], []
        self.n_per_subject = {}
        self.sessions_seen = set()
        for k, sid in enumerate(subjects):
            d = os.path.join(data_root, sid)
            eeg = np.load(os.path.join(d, f"{sid}_eeg_windows.npy"), mmap_mode="r")
            fn = np.load(os.path.join(d, f"{sid}_fnirs_windows.npy"), mmap_mode="r")
            lab = np.load(os.path.join(d, f"{sid}_labels.npy"))
            tri = np.load(os.path.join(d, f"{sid}_trial_ids.npy"))
            if not (len(eeg) == len(fn) == len(lab) == len(tri)):
                raise ValueError(f"{sid}: window arrays have different lengths")
            keep = []
            if exclude_session is None:
                # legacy path, untouched: the pooled run must reproduce bit-for-bit
                for t in np.unique(tri):
                    keep.extend(np.flatnonzero(tri == t)[::stride].tolist())
            else:
                sess = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
                if len(sess) != len(lab):
                    raise ValueError(f"{sid}: session_ids length differs from labels")
                allowed = sess != exclude_session
                if not allowed.any():
                    raise ValueError(f"{sid}: excluding session {exclude_session} leaves no windows")
                # stride is applied per (session, trial): a trial id that were reused across
                # sessions must not be subsampled across the session boundary
                for ss in np.unique(sess[allowed]):
                    for t in np.unique(tri[sess == ss]):
                        keep.extend(np.flatnonzero((tri == t) & (sess == ss))[::stride].tolist())
                self.sessions_seen.update(int(x) for x in np.unique(sess[allowed]))
                # hard post-condition: not one window of the excluded session got through
                assert not np.isin(sess[keep], [exclude_session]).any(), "session leak in pretraining pool"
            keep.sort()
            self.eeg.append(eeg)
            self.fnirs.append(fn)
            self.labels.append(lab)
            self.index.extend((k, w) for w in keep)
            self.n_per_subject[sid] = len(keep)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        k, w = self.index[i]
        return (torch.from_numpy(np.array(self.eeg[k][w], dtype=np.float32)),
                torch.from_numpy(np.array(self.fnirs[k][w], dtype=np.float32)),
                torch.tensor(int(self.labels[k][w]), dtype=torch.long))


def _discover_sessions(data_root, subjects):
    """Sorted session ids, required to be IDENTICAL for every subject: a subject missing a session
    would make "the other sessions" mean different things for different pool members."""
    found = {}
    for sid in subjects:
        sess = np.load(os.path.join(data_root, sid, f"{sid}_session_ids.npy"))
        found[sid] = sorted(int(x) for x in np.unique(sess))
    ref = found[subjects[0]]
    odd = {sid: v for sid, v in found.items() if v != ref}
    if odd or len(ref) < 2:
        sys.exit(f"ERROR: loso needs the same >= 2 sessions for every subject; reference {ref}, "
                 f"differing: {odd}")
    return ref


def main():
    subjects = [s for s in parse_subjects(ARGS.subjects)
                if os.path.isdir(os.path.join(ARGS.data_root, s))]
    if len(subjects) < ARGS.n_groups + ARGS.n_val_subjects + 1:
        sys.exit(f"ERROR: only {len(subjects)} subjects with data under {ARGS.data_root}")
    out_dir = ARGS.out_dir or os.path.join(ARGS.data_root, f"pretrain_{RUN_ID}_seed{ARGS.seed}")
    os.makedirs(out_dir, exist_ok=True)
    man_path = os.path.join(out_dir, "pretrain_manifest.json")

    manifest = {"id": RUN_ID, "seed": ARGS.seed, "data_root": ARGS.data_root,
                "arch": ARCH, "train": TRAIN, "groups": {}}
    if os.path.isfile(man_path):
        with open(man_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("id") != RUN_ID or int(old.get("seed", -1)) != ARGS.seed:
            sys.exit(f"ERROR: {man_path} belongs to a different configuration "
                     f"(id {old.get('id')}, seed {old.get('seed')}). Use another --out-dir.")
        manifest["groups"] = old.get("groups", {})

    groups = assign_groups(subjects, ARGS.n_groups)
    todo = (sorted(groups) if ARGS.groups is None
            else [int(g) for g in ARGS.groups.split(",")])
    device = torch.device(ARGS.device)

    print("=" * 78)
    print(f"CROSS-SUBJECT PRETRAINING | id={RUN_ID} seed={ARGS.seed} groups={ARGS.n_groups} "
          f"stride={ARGS.window_stride} device={device}")
    print(f"  out: {out_dir}")
    for g in sorted(groups):
        print(f"  group {g}: targets {groups[g]}")
    print("=" * 78)

    loso = ARGS.session_protocol == "loso"
    sessions = None
    if loso:
        sessions = _discover_sessions(ARGS.data_root, subjects)
        manifest["session_protocol"] = "loso"
        manifest["sessions"] = sessions
        print(f"  session protocol: loso over sessions {sessions} "
              f"({len(sessions)} models per group)")

    for g in todo:
        key = str(g)
        targets = groups[g]
        pool = [s for s in subjects if s not in targets]
        rng = np.random.default_rng(ARGS.seed * 1000 + g)
        pool_val = sorted(rng.choice(pool, size=ARGS.n_val_subjects, replace=False).tolist())
        pool_train = [s for s in pool if s not in pool_val]
        assert not (set(targets) & set(pool)), "leakage guard"

        # one fold = one model: pooled -> a single fold using every session; loso -> one fold per
        # held-out session, trained on the OTHER sessions of the pool subjects only
        folds = [(None, f"pretrain_g{g}.pt")] if not loso else \
                [(sn, f"pretrain_g{g}_s{sn}.pt") for sn in sessions]
        for held, ckpt_name in folds:
            ckpt = os.path.join(out_dir, ckpt_name)
            tag = f"group {g}" + (f" | held-out session {held}" if held is not None else "")
            entry = manifest["groups"].get(key)
            if loso:
                done = bool(entry) and str(held) in entry.get("folds", {}) and os.path.isfile(ckpt)
            else:
                done = bool(entry) and os.path.isfile(ckpt)
            if done:
                va = (entry["folds"][str(held)] if loso else entry)["val_acc"]
                print(f"\n[{tag}] already done (best val acc {va:.4f}) -- skipping")
                continue

            t0 = time.time()
            train_ds = MmapWindowDataset(ARGS.data_root, pool_train, ARGS.window_stride, held)
            val_ds = MmapWindowDataset(ARGS.data_root, pool_val, ARGS.window_stride, held)
            print(f"\n[{tag}] pool_train={len(pool_train)} subjects ({len(train_ds)} windows) | "
                  f"pool_val={pool_val} ({len(val_ds)} windows)"
                  + (f" | train sessions {sorted(train_ds.sessions_seen)}" if loso else ""))
            train_loader = DataLoader(train_ds, batch_size=ARGS.batch_size, shuffle=True,
                                      num_workers=ARGS.num_workers)
            val_loader = DataLoader(val_ds, batch_size=ARGS.batch_size, shuffle=False,
                                    num_workers=ARGS.num_workers)

            torch.manual_seed(ARGS.seed)
            np.random.seed(ARGS.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(ARGS.seed)
            model = TSFNet(**ARCH).to(device)

            val_acc, val_kappa, model, report = train_one_fold(
                model, train_loader, val_loader, val_loader, device,
                lambda_eeg=ARGS.lambda_eeg, lambda_fnirs=ARGS.lambda_fnirs,
                lr=ARGS.lr, weight_decay=ARGS.weight_decay,
                stage1_max_epochs=ARGS.max_epochs, stage1_patience=ARGS.patience,
                stage2_max_epochs=0, stage2_stop_mode="none",
                ckpt_path=ckpt, adabn=False,
                normalize_ypred=ARGS.normalize_ypred, no_decay_scalars=ARGS.no_decay_scalars)

            rec = {"checkpoint": ckpt_name, "val_acc": float(val_acc), "val_kappa": float(val_kappa),
                   "n_train_windows": len(train_ds), "n_val_windows": len(val_ds),
                   "minutes": round((time.time() - t0) / 60, 1)}
            if loso:
                rec["train_sessions"] = sorted(train_ds.sessions_seen)
                # the guard run_all_subjects re-checks: the held-out session must be absent
                assert held not in rec["train_sessions"], "held-out session in pretraining data"
                manifest["groups"].setdefault(key, {"targets": targets, "pool_train": pool_train,
                                                    "pool_val": pool_val, "folds": {}})
                manifest["groups"][key]["folds"][str(held)] = rec
            else:
                manifest["groups"][key] = {"targets": targets, "pool_train": pool_train,
                                           "pool_val": pool_val, **rec}
            with open(man_path, "w", encoding="utf-8") as f:       # after every model: resumable
                json.dump(manifest, f, indent=2)
            print(f"[{tag}] done in {rec['minutes']} min | val acc on unseen pool subjects "
                  f"{pool_val}: {val_acc:.4f} (kappa {val_kappa:.4f})")
            del model, train_loader, val_loader, train_ds, val_ds
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def _group_complete(g):
        e = manifest["groups"].get(str(g))
        if not e:
            return False
        return (not loso) or set(e.get("folds", {})) == {str(x) for x in sessions}
    missing = [g for g in groups if not _group_complete(g)]
    print()
    if missing:
        print(f"NOT COMPLETE: groups {missing} still to run.")
    else:
        print("ALL GROUPS DONE. Fine-tune with:")
        print(f'  --seed {ARGS.seed} --pretrained-dir "{out_dir}"')


if __name__ == "__main__":
    main()
