# Late fusion of TSFNet with a per-session-aligned multiband EEG covariance member

Protocol: 26 subjects (Shin et al. 2018), leave-one-session-out, 3 folds per subject, means over
subjects. Fusion `P = (1-w) * P_deep + w * P_eeg`, with **w = 0.5 pre-registered** in the docstring
of `late_fusion_riemann.py` before the first run; every other w is a sensitivity curve, not a test.
Deep posteriors are those of the accepted baseline (`--seed 42 --adabn --fnirs-temporal-align egta
--efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars
--stage2-stop-mode none`, variant hash `40b7d8ea53`); TSFNet is NOT retrained. Tests are paired
Wilcoxon over the 26 subjects.

WHAT IS AND IS NOT PRE-REGISTERED. Pre-registered: w = 0.5; the window-level `fused - deep`
contrast; the attribution reading of `fusion_controls.py` (its reference member stays pinned to
`riem_disp|multi`). NOT pre-registered, and therefore descriptive: the choice of the EA member over
the Riemannian one (made after seeing both), every block-level and 2-vs-3 contrast, the per-class
recall decomposition, and the w curve.

## 1. Main table (w = 0.5)

| system                                       | window | block  | 2-vs-3 |
|----------------------------------------------|-------:|-------:|-------:|
| TSFNet, single seed                          | 0.6042 | 0.6225 | 0.5922 |
| TSFNet, 3-seed ensemble                      | 0.6167 | 0.6382 | 0.5996 |
| EEG member alone, EA multiband               | 0.6449 | 0.7108 | 0.5818 |
| EEG member alone, Riemannian multiband       | 0.6497 | 0.7179 | 0.5861 |
| **Fused, EA member (reported system)**       | **0.6760** | **0.7179** | 0.6135 |
| Fused, Riemannian member (second variant)    | 0.6765 | 0.7208 | 0.6131 |
| Fused, log-variance member (control)         | 0.6222 | 0.6624 |      - |

Headline numbers, window level:

| comparison                                              | gain   | wins  | p      | Holm (m=2) |
|---------------------------------------------------------|-------:|-------|-------:|-----------:|
| **single seed + EA member (0.6711) vs single seed**     | **+0.0669** | 22/26 | <0.0001 | **0.0001** |
| ensemble + EA member (0.6760) vs ensemble               | +0.0593 | 21/26 | 0.0002 | **0.0004** |
| ensemble + EA member vs the original single-seed TSFNet | +0.0718 | -     | -      | -          |

The first row is the fairest comparison: one trained network on each side. The +7.18 figure mixes
the ensemble gain with the member gain and must carry the label "ensemble + member" wherever it
appears.

## 2. The two members are equivalent; EA is reported for simplicity

Fused accuracy is 0.6760 (EA) against 0.6765 (Riemannian): a difference of -0.0005 with p = 0.71,
i.e. no difference. The EA member is reported because it is simpler (arithmetic-mean whitening, no
Karcher iteration, incrementally updatable), **a choice made after seeing both results**, not
because it scored higher. Both are tabulated. The apparent advantage of EA in the `fused - member`
column (+0.0311 vs +0.0267) is an artefact of its weaker member alone (0.6449 vs 0.6497), not a
better fusion, and the two p-values (0.0036, 0.0120) are not a test of the difference between
members.

What the pre-registered attribution test does establish (`fusion_controls.py`): covariance features
and the multiband decomposition are what matter, not Riemannian geometry.
  - fused(riem_disp|multi) - fused(logvar)        = +0.0543, Holm p < 0.0001
  - fused(riem_disp|multi) - fused(riem_disp|broad) = +0.0410, Holm p = 0.0007
  - fused(EA|multi) - fused(riem_disp|multi)      = -0.0005, p = 0.71

## 3. Complementarity holds at window level, NOT at block level

`block_member_contrast.py`, deep member = ensemble, w = 0.5:

| level  | deep   | EA member alone | fused  | fused - deep         | fused - member        |
|--------|-------:|----------------:|-------:|---------------------:|----------------------:|
| window | 0.6167 | 0.6449          | 0.6760 | +0.0593 (21/26, 0.0002) | +0.0311 (19/26, 0.0036) |
| block  | 0.6382 | 0.7108          | 0.7179 | +0.0798 (22/26, 0.0001) | +0.0071 (12/26, 1.0000) |
| 2-vs-3 | 0.5996 | 0.5818          | 0.6135 | +0.0139 (16/26, 0.2799) | +0.0317 (17/26, 0.0176) |

(Riemannian member: block `fused - member` = +0.0028, 11/26, p = 0.93; 2-vs-3 = +0.0270, p = 0.046.)

This must be stated as it is. **At block level the simple linear EEG member alone (0.7108) is
statistically indistinguishable from the fused system (0.7179), and beats the whole deep model
(0.6382) by 7.3 points.** The deep model's contribution is visible at window level, and specifically
on the 2-vs-3 pair, where the member alone (0.5818) is below the deep model (0.5996) and fusion
recovers +0.0317 over the member. The honest reading: the two members are complementary because
each is strong where the other is weak -- the EEG member carries 0-back, the deep model carries the
2-vs-3 discrimination -- and at block level, where only 9 blocks per fold are available, there is no
power to resolve a 0.7-point difference.

## 4. Where the gain comes from (EA member, window level)

| class  | deep  | EA member alone | fused | fused - deep per subject | wins  | Holm (m=9) |
|--------|------:|----------------:|------:|-------------------------:|-------|-----------:|
| 0-back | 78.8% | 87.0%           | 89.4% | **+10.61**               | 24/26 | **0.0011** |
| 2-back | 51.9% | 53.3%           | 55.0% | +3.11                    | 18/26 | 0.8147     |
| 3-back | 54.6% | 53.3%           | 58.7% | +4.08                    | 18/26 | 0.1779     |

Error structure, pooled over all folds: total window errors 8827 -> 7454 (-1373). Errors involving
class 0-back 3313 -> 1913 (-1400). Confusions between 2-back and 3-back 5514 -> 5541 (**+27**).

So the net error reduction is accounted for entirely by errors involving 0-back, while the 2-vs-3
confusions are unchanged in the pooled matrix; the share of all errors that are 2<->3 rises from
62.5% to 74.3%. The 2-back and 3-back recalls do rise nominally by ~3-4 points, but neither survives
Holm correction, so they are descriptive only.

The log-variance control gains nothing on 0-back (+1.19, Holm p = 1.00), which rules out "any extra
EEG member would do this" and attributes the effect to the multiband covariance representation.

## 5. Limitations to disclose

- **Transductive.** The EEG member's per-session centring uses the held-out session's UNLABELED
  windows. No label of the held-out session is used. This is the same transductive assumption
  already present in the baseline through priority-E3 (per-session normalisation) and AdaBN, and it
  must be reported together with them. The results must not be compared against a strictly
  inductive baseline without stating this.
- **Post-hoc choices.** The EA-vs-Riemannian choice was made after seeing results (they are
  equivalent, so this affects simplicity, not the claim). Only w = 0.5 and the attribution reading
  were fixed in advance.
- **The system is "TSFNet + an EEG member", not an improved TSFNet.** TSFNet's own weights are
  unchanged; this is offline posterior fusion. The trained-integration variant
  (`--riem-branch member`) is a separate, as yet incomplete experiment.
- **The 2-vs-3 bottleneck is not solved.** 0.5996 -> 0.6135, and the pooled 2<->3 confusion count
  does not fall.
- **Block-level metrics rest on 9 blocks per fold**, hence the near-total lack of power for the
  `fused - member` contrast there.
- Seven member variants were evaluated on the same folds; Holm correction within that family is
  reported in `fusion_controls.py`.
