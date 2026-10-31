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

## Update after an external review (what was checked, what was changed)
Reviewer points were checked against the code before acting; the ones that held are fixed below. One literature claim was verified at the abstract level
([Ravindran & Contreras-Vidal 2023, Sci Rep](https://www.nature.com/articles/s41598-023-43871-8): sanity checks by weight and label randomisation, saliency
not class- or model-specific, DeepLift most reliable); the other cited reviews were not checked and should be verified before they are cited.

| point | held? | change |
|---|---|---|
| agreement between methods is not validity; add randomisation sanity checks | yes | NEW `xai_sanity_check.py` (cascading parameter randomisation, with a model-independent control that must stay at rho = 1) |
| IG / saliency explain the TRUE label via a saturating probability | yes (`true_class_prob`) | `xai_multimethod_consensus.py --target {true,pred} --objective {prob,logprob}`; defaults unchanged; for `pred` the class is fixed once from the real input |
| unseeded bootstrap, one-sided test, hard-coded 0.5 | yes | `--seed`, `--rho-threshold` (printed, with the fraction of subjects above it); text now says the one-sided test is passed by almost any positive rho |
| `np.allclose` constancy test discards tiny gradients | found while testing | scale-free test in the consensus, sanity and faithfulness scripts |
| `xai_component_attribution.py` did not check the blend identity | yes | `--integrated` now verifies (1-w) P_deep + w P_riem == P_final and refuses a wrong `--w` |
| member weights can rank a suppressor feature as important; scalp maps need patterns | yes | Haufe activation patterns added to `xai_member_coefficients.py` (a synthetic suppressor shows weights 0.87x vs pattern 0.11x) |
| ground-truth validation | yes | `xai_member_coefficients.py --benchmark` plants a known load effect in the REAL features and scores weights vs patterns (AUC). The same for the trunk (planting a signal in training AND test fNIRS and retraining) needs GPU runs and is NOT implemented |
| DeepLift | not added | needs captum, not verified here; the cited study found it most reliable, so it is a reasonable next method |
| randomised-LABEL variant of the sanity check | not added | needs one retraining per configuration |

## Second update: ground-truth and randomisation tools (all written and unit-tested on CPU; none run on real data here)
| tool | what it does | needs |
|---|---|---|
| `xai_sanity_check.py --deeplift` | adds DeepLift (captum 0.9.0, pip-installable here) with a completeness gate: mean\|delta\| / mean\|f(x)-f(baseline)\| must be < 0.1, because TSFNet's functional sigmoid/softmax fall back to the gradient rule inside captum | a trained checkpoint set |
| `run_all_subjects.py --plant-fnirs "seg=K,strength=S"` | plants a known load-dependent response in fNIRS segment K of every window (patch, HRF, HbO +1 / HbR -0.5, -1/0/+1 by class); sets `TSFNET_PLANT`, default off and bit-identical | GPU, ~6 min/subject |
| `xai_plant_recovery.py` | rank percentile / hit@1 / hit@3 of segment K per method, plus the causal check (zeroing K vs other segments) | the planted run |
| `run_all_subjects.py --shuffle-train-labels SEED` | permutes the labels of whole blocks of the training sessions (test session keeps true labels); default off | GPU, ~6 min/subject |
| `xai_label_randomisation.py` | rho between the explanations of a true-label and a shuffled-label network on the same held-out windows; refuses to interpret if the shuffled network is above chance | both runs |
| `test_xai_groundtruth.py` | CPU tests of the planting and the block shuffle | none |
Run order and pre-registered rules: `NEXT_RUN_xai_groundtruth.md`.

### MSTFDN (not XAI)
`mstfdn/MSTFDN_train_LOSO_3class.py` already implements the paper's cross-session training on its OWN preprocessing (`MSTFDN_PREPROCESSED_NBACK_V3`: 5 frequency bands x 28
channels, fNIRS in channel space with HbO/HbR/HbT). It could not be fed with the TSFNet preprocessing as is, because TSFNet stores fNIRS interpolated on a 16x16 grid. What was
missing was a seed: it now seeds everything with `SEED = 42` (set `SEED = None` to restore the old unseeded behaviour) and logs that the result is single-seed.
