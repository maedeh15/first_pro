# Late fusion of TSFNet with a per-session-aligned EEG covariance member — final numbers

Protocol: 26 subjects, leave-one-session-out (3 folds/subject), means over subjects.
Fusion: `P = (1 - w) * P_deep + w * P_eeg`, with **w = 0.5 pre-registered** (the rest of the w grid
is a sensitivity curve, not a test). TSFNet is NOT retrained; the deep posteriors are those of the
accepted baseline configuration (`--seed 42 --adabn --fnirs-temporal-align egta --efgf-gate-mode
sigmoid --efgf-gate-init -2.0 --normalize-ypred --no-decay-scalars --stage2-stop-mode none`,
variant hash `40b7d8ea53`). Tests are paired Wilcoxon over the 26 subjects, Holm-corrected within
each family of the two primary comparisons (vs the deep model alone; vs the EEG member alone).

## Main table (deep member = 3-seed ensemble; w = 0.5)

| system                                         | window acc | block acc | 2-vs-3 |
|------------------------------------------------|-----------:|----------:|-------:|
| TSFNet, single seed                            |     0.6042 |    0.6225 | 0.5922 |
| TSFNet, 3-seed ensemble                        |     0.6167 |    0.6382 | 0.5996 |
| EEG member alone, EA multiband                 |     0.6449 |    0.7108 | 0.5818 |
| EEG member alone, Riemannian multiband         |     0.6497 |    0.7179 | 0.5861 |
| **Fused, EA member (final)**                   | **0.6760** | **0.7179**| 0.6135 |
| Fused, Riemannian member (second variant)      |     0.6765 |    0.7208 | 0.6131 |

## Primary comparisons, window level, w = 0.5

| variant   | deep member | contrast          |   diff | wins  |      p | Holm (m=2) |
|-----------|-------------|-------------------|-------:|-------|-------:|-----------:|
| EA        | ensemble    | vs deep alone     | +0.0593| 21/26 | 0.0002 | **0.0004** |
| EA        | ensemble    | vs EEG member alone| +0.0311| 19/26 | 0.0036 | **0.0036** |
| EA        | single seed | vs deep alone     | +0.0669| 22/26 | 0.0000 | **0.0001** |
| EA        | single seed | vs EEG member alone| +0.0262| 19/26 | 0.0102 | **0.0102** |
| Riemannian| ensemble    | vs deep alone     | +0.0597| 20/26 | 0.0003 | **0.0006** |
| Riemannian| ensemble    | vs EEG member alone| +0.0267| 17/26 | 0.0120 | **0.0120** |
| Riemannian| single seed | vs deep alone     | +0.0652| 22/26 | 0.0001 | **0.0002** |
| Riemannian| single seed | vs EEG member alone| +0.0197| 15/26 | 0.0842 |     0.0842 |

Block level, EA, ensemble: +0.0798, p = 0.0001. Riemannian, ensemble: +0.0826, p = 0.0001.

## Sensitivity curve in w (window acc, ensemble deep member)

| w    | EA     | Riemannian |
|------|-------:|-----------:|
| 0.00 | 0.6167 |     0.6167 |
| 0.25 | 0.6465 |     0.6469 |
| 0.50 | 0.6760 |     0.6765 |
| 0.75 | 0.6591 |     0.6625 |
| 1.00 | 0.6449 |     0.6497 |

The gain is not a knife-edge effect of w: every w in (0, 1) beats w = 0, and the curve peaks at the
pre-registered w = 0.5 at window level. At block level the curve is flat from 0.5 to 0.75.

## Statements that the data support

1. Fusing TSFNet with a linear classifier on per-session-aligned multiband EEG covariances adds
   ~+6 points window / ~+8 points block, significant after Holm, consistent in 21/26 subjects, and
   present against both a single deep seed and the 3-seed ensemble.
2. The gain is not just "an extra member": the fused system also beats the EEG member alone
   (+3.1 points, Holm p = 0.0036), i.e. the two members are complementary.
3. Riemannian geometry is NOT what produces the effect. EA (arithmetic-mean whitening) and the
   Frechet-mean + dispersion-normalisation variant are within 0.05 points of each other at window
   level and 0.3 at block level. What matters, per the pre-registered attribution test in
   `fusion_controls.py`, is (a) covariance rather than log-variance features and (b) the multiband
   decomposition. The simpler EA member is therefore reported as the final system.
4. The improvement is concentrated in 0-back (+9.77 points recall, 24/26 subjects, Holm p < 0.0001).
   2-back (+4.04, Holm 0.335) and 3-back (+4.05, Holm 0.130) are NOT significant. The 2-vs-3
   bottleneck is essentially untouched (0.5996 -> 0.6135).

## Statements that must be disclosed as limitations

- The EEG member's per-session centring is **transductive**: it uses the held-out session's
  unlabelled windows, exactly as priority-E3 (per-session normalisation) and AdaBN already do in
  this baseline. No label of the held-out session is used anywhere. This must be stated together
  with E3/AdaBN and must not be compared against a strictly inductive baseline without saying so.
- The choice between the EA and Riemannian member was made **after** seeing both results. They are
  statistically indistinguishable here, so the choice is justified on simplicity and
  reproducibility, not on accuracy; both are reported. Only w = 0.5 was pre-registered.
- The fused system is "TSFNet + a Riemannian/EA EEG member", not an improved TSFNet. TSFNet's own
  weights are unchanged. The trained-integration variant (`--riem-branch member`) is a separate
  experiment.
