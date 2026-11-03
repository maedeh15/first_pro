"""
apply_evidence_head_patch.py -- adds the opt-in  --fnirs-head {mlp,linear}  (default mlp = bit-exact prior behaviour).

  python apply_evidence_head_patch.py "E:\\kntu\\Thesis\\codes"          # folder containing the two files

Edits tsfnet_model_notebook.py and run_all_subjects.py by EXACT-STRING replacement. Every edit asserts the anchor occurs the
expected number of times; nothing is written unless ALL anchors match. Backups: *.bak_evidence. Re-running is a no-op.
Rule 6 (RNG): the linear head is built from the pre-construction RNG state and the post-baseline state is restored, so every
module built after clf_fnirs initialises exactly as in the 'mlp' configuration.
Rule 5 (tag): the flag enters the results-folder tag (_fhead-linear), so runs never overwrite the baseline folder.
"""
import os, py_compile, shutil, sys

HEAD_CLASS = '''

class LinearEvidenceHead(nn.Module):
    """fnirs_head='linear': logit = W . x + b on the SAME enhanced_fnirs vector, then softmax (like BranchClassifier).
    A single Linear makes the fNIRS logit a sum over EGTA segments (see fnirs_evidence.py). `net[-1]` stays the Linear so the
    existing diagnostics hooks (clf_fnirs.net[-1]) still return logits."""

    def __init__(self, in_dim, num_classes, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.net = nn.Sequential(nn.Linear(in_dim, num_classes))

    def forward(self, x):
        return torch.softmax(self.net(self.drop(x)), dim=-1)

'''

NEW_HEAD = '''        if fnirs_head not in ("mlp", "linear"):
            raise ValueError(f"fnirs_head must be 'mlp' or 'linear', got {fnirs_head!r}")
        if fnirs_head == "linear" and fusion_combination in ("stanet2", "stanet3"):
            raise ValueError("fnirs_head='linear' is not supported with fusion_combination 'stanet2'/'stanet3' (they read clf_fnirs.net[0])")
        if fnirs_head == "mlp":
            self.clf_fnirs = BranchClassifier(feat_dim, num_classes, dropout=dropout)
        else:
            _rng_before = torch.get_rng_state()
            BranchClassifier(feat_dim, num_classes, dropout=dropout)       # advance the RNG exactly as the baseline would (rule 6)
            _rng_after = torch.get_rng_state()
            torch.set_rng_state(_rng_before)
            self.clf_fnirs = LinearEvidenceHead(feat_dim, num_classes, dropout=fnirs_head_dropout)
            torch.set_rng_state(_rng_after)
        self.fnirs_head = fnirs_head
'''

ARGS = '''_arg_parser.add_argument("--fnirs-head", type=str, default="mlp", choices=["mlp", "linear"],
                          help="'mlp' (DEFAULT, bit-exact prior). 'linear': the fNIRS decision head is ONE Linear layer on enhanced_fnirs, so its logit is an "
                               "exact sum of per-segment EGTA evidence (fnirs_evidence.py). Requires --fnirs-temporal-align egta.")
_arg_parser.add_argument("--fnirs-head-dropout", type=float, default=0.3,
                          help="Only used with --fnirs-head linear: dropout on the input of the linear head.")
'''

EDITS = {
    "tsfnet_model_notebook.py": [
        ("        logits = self.net(x)\n        return torch.softmax(logits, dim=-1)\n", 1,
         lambda s: s + HEAD_CLASS),
        ('                 bandtok_d_model=32, bandtok_heads=4, bandtok_norm="none"):', 1,
         lambda s: s[:-2] + ',\n                 fnirs_head="mlp", fnirs_head_dropout=0.3):'),
        ("        self.clf_fnirs = BranchClassifier(feat_dim, num_classes, dropout=dropout)\n", 1, lambda s: NEW_HEAD),
        ('                               fnirs_temporal_align="none", egta_k_init="zero",\n                               eeg_band_mode="none", eeg_band_d_model=32,', 1,
         lambda s: s.replace('egta_k_init="zero",', 'egta_k_init="zero",\n                               fnirs_head="mlp", fnirs_head_dropout=0.3,')),
        ("                        fnirs_temporal_align=fnirs_temporal_align, egta_k_init=egta_k_init,\n                        eeg_temporal_mode=eeg_temporal_mode,", 1,
         lambda s: s.replace("egta_k_init=egta_k_init,", "egta_k_init=egta_k_init,\n                        fnirs_head=fnirs_head, fnirs_head_dropout=fnirs_head_dropout,")),
    ],
    "run_all_subjects.py": [
        ('_arg_parser.add_argument("--egta-k-init", type=str, default="zero",', 1, lambda s: ARGS + s),
        ("EGTA_K_INIT = SEED_ARGS.egta_k_init\n", 1,
         lambda s: s + 'FNIRS_HEAD = SEED_ARGS.fnirs_head\nFNIRS_HEAD_DROPOUT = SEED_ARGS.fnirs_head_dropout\n'
                       'if FNIRS_HEAD != "mlp" and SEED_ARGS.fnirs_temporal_align != "egta":\n'
                       '    raise ValueError("--fnirs-head linear requires --fnirs-temporal-align egta")\n'),
        ('_EGTA_SUBTAG += "_kinit-noisy" if EGTA_K_INIT == "noisy" else ""   # MUST be in the tag (rule 5)\n', 1,
         lambda s: s + '_EGTA_SUBTAG += (f"_fhead-{FNIRS_HEAD}" + (f"-d{FNIRS_HEAD_DROPOUT}" if FNIRS_HEAD_DROPOUT != 0.3 else "")) if FNIRS_HEAD != "mlp" else ""   # rule 5\n'),
        ('                egta_k_init=EGTA_K_INIT,                      # "zero" (default, accepted) vs "noisy" (tested fix, -1.34pp)\n', 1,
         lambda s: s + "                fnirs_head=FNIRS_HEAD, fnirs_head_dropout=FNIRS_HEAD_DROPOUT,\n"),
    ],
}


def main(folder):
    new = {}
    for fn, edits in EDITS.items():
        path = os.path.join(folder, fn)
        txt = open(path, encoding="utf8").read()
        if "fnirs_head" in txt:
            print(f"{fn}: already patched, skipped"); continue
        for anchor, expected, fix in edits:
            n = txt.count(anchor)
            if n != expected:
                sys.exit(f"ABORT {fn}: anchor found {n}x (expected {expected}):\n{anchor[:120]!r}\nNothing was written.")
            txt = txt.replace(anchor, fix(anchor))
        new[fn] = txt
    for fn, txt in new.items():
        path = os.path.join(folder, fn)
        shutil.copy2(path, path + ".bak_evidence")
        open(path, "w", encoding="utf8").write(txt)
        py_compile.compile(path, doraise=True)
        print(f"{fn}: patched + compiles OK (backup: {fn}.bak_evidence)")
    if not new:
        print("nothing to do")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
