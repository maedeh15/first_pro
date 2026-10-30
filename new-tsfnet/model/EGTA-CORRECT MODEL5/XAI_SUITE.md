# XAI suite — what exists, what changed, what was added

Status of verification: the existing scripts were **syntax-checked and their checkpoint-loading path tested against real member-branch
models** (below); they were NOT re-run on data here. The two new scripts have synthetic self-tests (planted effects recovered, noise gives
no false claim); they have not been run on the real data either.

## What the system is, for explanation purposes
`P = 0.5 * P_trunk + 0.5 * P_member` (~0.67 window accuracy, `--riem-head-fit post`, hash `cf64a71127`). Two components, two explanations:

| component | explained by | exact or post-hoc |
|---|---|---|
| deep trunk (EEG, fNIRS, fusion, EGTA/EFGF attention) | the existing suite below | post-hoc (attention, gradients, SHAP) |
| linear member (logistic regression on multiband tangent-space EEG) | `xai_member_coefficients.py` (NEW) | **exact** (linear) |
| the split between the two | `xai_component_attribution.py` (NEW) | exact (fixed average of two probability vectors) |

## Existing scripts (explain the TRUNK)
| script | question | key arguments |
|---|---|---|
| `run_xai_40b7d8ea53.py` | SHAP, gating weights, EFGF/CAFE attention, Grad-CAM, t-SNE for one subject (template: edit the three paths at the top) | `RESULTS_DIR`, `DATA_ROOT`, `OUTPUT_DIR` |
| `xai_biomarker_validation.py` | is the attention a valid biomarker (vs HbO/theta/P300)? | `--results-dir --data-root` |
| `xai_load_level_anova.py` | do attention/gate signals differ across 0/2/3-back (Friedman + Kendall W)? | `--results-dir --data-root` |
| `xai_multimethod_consensus.py` | do attention, saliency and integrated gradients agree on fNIRS segments? | `--results-dir --data-root --ig-steps` |
| `attention_faithfulness_check.py` | does ablating high-attention inputs hurt more than low-attention ones? | `--results-dir --data-root` |
| `xai_stability_check.py` | is EGTA attention reproducible across seeds (Spearman, ICC)? | `--data-root --variant-hash --seeds` |

### What changed in them
Every one builds a plain `TSFNet` and loaded the checkpoint with `strict=True`. A checkpoint from a `--riem-branch member` run carries extra
`riem_head.*` tensors, so the load failed with a key error. They now load through `xai_trunk.trunk_state`, which drops only those tensors
and prints one line saying the explanation is of the trunk. Tested: for `sgd`, `lr_all`, `post` and `post_oof` models the plain trunk loads
strict from the member model's state_dict, and (for the closed-form modes) every trunk tensor is unchanged.

Not changed, on purpose: they explain the deep network, which is the right object for attention/SHAP/Grad-CAM. Their numbers (for example the
faithfulness of EGTA attention) concern the trunk and must not be quoted as properties of the 0.67 system.

Known caveat: for the `post` and assembled models the trunk was trained and selected exactly like a plain run, so the trunk explanations are
comparable with the baseline's. For `84c18210ab` (gradient head, joint training) the trunk was trained against the head; say so.

## New scripts
### `xai_member_coefficients.py` — explains the linear member exactly
- importance of bands (broad/theta/alpha/beta) for the **load contrast** (3-back minus 0-back), split into diagonal (band power) and off-diagonal (coupling);
- channel importance per band (whitened coordinates, approximate electrodes);
- stability: fold-to-fold cosine of the contrast vector against a block-permutation null, and agreement between subjects;
- **causal deletion test**: accuracy lost by deleting the top-5 ranked channels vs 5 random vs the bottom 5, and by deleting each band;
- two pre-stated literature checks (frontal theta up, parietal alpha down with load) when electrode names are supplied, Holm-corrected.
Two sources for the head: re-fit exactly as the model does (default), or read the buffers from the checkpoints (`--results-dir`).
```
%run xai_member_coefficients.py --data-root "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback" --results-dir "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\results_cf64a71127_seed42" --channel-names "<28 electrode names, data order>.txt" --out-dir "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\xai_member"
```
Self-test: `python xai_member_coefficients.py --self-test`.

### `xai_component_attribution.py` — who decides, and where each helps
correctness table and oracle ceiling, rescue/harm per class, who wins disagreements, probability share per component, per-class recall, block-level
version, and whether the same subjects are hard for both. From saved posteriors (CPU, seconds).
```
%run xai_component_attribution.py --integrated "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\ensemble_eval_cf64a71127_seeds-42-999-2024\posteriors"
```
(or `--deep-dir <plain baseline posteriors> --member-dir <member_ea_posteriors>`). Self-test: `--self-test`.

## What was considered and NOT added
- Haufe activation patterns for the member (would turn weights into scalp patterns; needs per-fold covariance bookkeeping through the whitening): listed as future work.
- SHAP/Grad-CAM on the blended system: the member's input is a precomputed covariance feature, not the raw EEG, so gradients do not reach the signal through it.
