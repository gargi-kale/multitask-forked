# Negative-transfer study: multi-task vs single-task

Fine-tuned small BERT, sweep-winner settings, seeds [1, 2, 3, 4, 42] (n per task: [5]). Multi-task runs per seed: s1: `mtl_lam0_s1`, s2: `mtl_lam0_s2`, s3: `mtl_lam0_s3`, s4: `full_s4`, s42: `mtl_lam0_s42`.

Δ = multi-task − single-task test weighted F1 ×100, paired by seed; 95% CI from a t distribution; p from a paired t-test, Holm-adjusted over 10 tasks.

| Task | Paper single | Paper multi | Paper Δ | Our single | Our multi | Δ [95% CI] | p (Holm) | Verdict |
|---|---|---|---|---|---|---|---|---|
| Propaganda | 63.07 | 61.74 | -1.33 | 66.8 ± 0.3 | 65.2 ± 1.2 | -1.7 [-3.1, -0.3] | 0.242 | suggestive loss (CI excludes 0, not after Holm) |
| Disagree/Agree | 71.15 | 71.38 | +0.23 | 62.4 ± 3.2 | 52.0 ± 10.1 | -10.4 [-24.4, +3.6] | 0.559 | no clear difference |
| Emotion/Fact | 68.11 | 63.93 | -4.18 | 71.5 ± 0.8 | 68.5 ± 0.8 | -3.0 [-4.3, -1.6] | 0.031 | **negative transfer** |
| Attacking/Respectful | 67.46 | 68.07 | +0.61 | 71.8 ± 1.1 | 71.0 ± 0.3 | -0.8 [-2.0, +0.4] | 0.559 | no clear difference |
| Nasty/Nice | 66.90 | 73.69 | +6.79 | 75.0 ± 1.3 | 74.4 ± 1.3 | -0.6 [-3.1, +1.9] | 1.000 | no clear difference |
| Personal/Audience | 63.25 | 65.69 | +2.44 | 66.5 ± 3.2 | 62.0 ± 3.6 | -4.5 [-6.1, -2.9] | 0.015 | **negative transfer** |
| Defeater/Undercutter | 45.97 | 55.65 | +9.68 | 59.2 ± 0.8 | 55.3 ± 3.3 | -3.9 [-8.8, +1.0] | 0.559 | no clear difference |
| Negotiate/Attack | 64.76 | 64.81 | +0.05 | 69.9 ± 0.4 | 69.1 ± 0.6 | -0.8 [-1.8, +0.2] | 0.559 | no clear difference |
| Questioning/Asserting | 59.61 | 63.23 | +3.62 | 67.5 ± 0.5 | 60.1 ± 5.3 | -7.5 [-14.5, -0.5] | 0.290 | suggestive loss (CI excludes 0, not after Holm) |
| Argument Quality | 80.93 | 79.17 | -1.76 | 78.1 ± 2.7 | 78.5 ± 4.2 | +0.5 [-6.8, +7.7] | 1.000 | no clear difference |
| **Mean over 10** | 65.12 | 66.74 | +1.61 | 68.9 ± 0.5 | 65.6 ± 1.2 | -3.3 [-4.9, -1.6] | p=0.005 (unadjusted) | |

**Δm** (mean relative change over tasks): -4.95% [-7.41, -2.48], p = 0.005, n = 5 seeds. Paper: +3.20%.

## AUC (paper reports none)

| Task | Our single AUC | Our multi AUC | Δ AUC [95% CI] |
|---|---|---|---|
| Propaganda | 69.4 ± 0.5 | 66.9 ± 0.9 | -2.5 [-4.0, -0.9] |
| Disagree/Agree | 67.5 ± 0.6 | 61.1 ± 1.8 | -6.4 [-8.6, -4.2] |
| Emotion/Fact | 80.7 ± 0.6 | 77.4 ± 0.5 | -3.3 [-4.5, -2.1] |
| Attacking/Respectful | 75.2 ± 0.9 | 73.7 ± 0.2 | -1.4 [-2.5, -0.4] |
| Nasty/Nice | 77.5 ± 0.6 | 76.6 ± 0.2 | -0.9 [-1.8, +0.0] |
| Personal/Audience | 66.5 ± 0.5 | 61.3 ± 1.4 | -5.2 [-7.2, -3.1] |
| Defeater/Undercutter | 59.4 ± 0.4 | 58.2 ± 0.9 | -1.2 [-2.1, -0.2] |
| Negotiate/Attack | 76.1 ± 0.8 | 76.1 ± 0.4 | -0.0 [-1.0, +0.9] |
| Questioning/Asserting | 66.8 ± 0.9 | 60.2 ± 0.6 | -6.6 [-8.0, -5.3] |
| Argument Quality | 81.2 ± 1.0 | 81.7 ± 1.5 | +0.5 [-2.0, +2.9] |

## Gradient metrics (shared parameters: encoder + shared layers)

From the probed full runs (seeds [4, 42]), averaged over every probe after step 0 (about twice per epoch) and over runs. Per-task losses are class-weighted, without the task-type weights. Row/column labels: Prop = Propaganda, AQ = Argument Quality, D/A = Disagree/Agree, A/R = Attacking/Respectful, E/F = Emotion/Fact, N/N = Nasty/Nice, P/A = Personal/Audience, D/U = Defeater/Undercutter, N/At = Negotiate/Attack, Q/A = Questioning/Asserting.

### Gradient cosine similarity

| row → col | Prop | AQ | D/A | A/R | E/F | N/N | P/A | D/U | N/At | Q/A |
|---|---|---|---|---|---|---|---|---|---|---|
| **Prop** | — | +0.15 | +0.02 | +0.15 | +0.15 | +0.12 | +0.00 | +0.05 | +0.05 | -0.08 |
| **AQ** | +0.15 | — | +0.12 | +0.05 | +0.13 | +0.06 | +0.17 | +0.18 | +0.07 | -0.00 |
| **D/A** | +0.02 | +0.12 | — | +0.39 | +0.18 | +0.39 | -0.06 | +0.06 | +0.11 | +0.06 |
| **A/R** | +0.15 | +0.05 | +0.39 | — | +0.45 | +0.85 | -0.10 | +0.26 | +0.41 | +0.08 |
| **E/F** | +0.15 | +0.13 | +0.18 | +0.45 | — | +0.37 | +0.00 | +0.22 | +0.47 | +0.08 |
| **N/N** | +0.12 | +0.06 | +0.39 | +0.85 | +0.37 | — | -0.18 | +0.20 | +0.32 | +0.01 |
| **P/A** | +0.00 | +0.17 | -0.06 | -0.10 | +0.00 | -0.18 | — | +0.03 | +0.07 | +0.11 |
| **D/U** | +0.05 | +0.18 | +0.06 | +0.26 | +0.22 | +0.20 | +0.03 | — | +0.25 | +0.02 |
| **N/At** | +0.05 | +0.07 | +0.11 | +0.41 | +0.47 | +0.32 | +0.07 | +0.25 | — | +0.11 |
| **Q/A** | -0.08 | -0.00 | +0.06 | +0.08 | +0.08 | +0.01 | +0.11 | +0.02 | +0.11 | — |

### Conflict rate (share of probe batches where the two gradients have cosine < 0)

| row → col | Prop | AQ | D/A | A/R | E/F | N/N | P/A | D/U | N/At | Q/A |
|---|---|---|---|---|---|---|---|---|---|---|
| **Prop** | — | 0.38 | 0.54 | 0.47 | 0.47 | 0.53 | 0.65 | 0.50 | 0.61 | 0.57 |
| **AQ** | 0.38 | — | 0.48 | 0.45 | 0.38 | 0.38 | 0.46 | 0.39 | 0.42 | 0.36 |
| **D/A** | 0.54 | 0.48 | — | 0.38 | 0.33 | 0.33 | 0.49 | 0.53 | 0.48 | 0.40 |
| **A/R** | 0.47 | 0.45 | 0.38 | — | 0.24 | 0.04 | 0.58 | 0.32 | 0.35 | 0.64 |
| **E/F** | 0.47 | 0.38 | 0.33 | 0.24 | — | 0.28 | 0.49 | 0.34 | 0.38 | 0.46 |
| **N/N** | 0.53 | 0.38 | 0.33 | 0.04 | 0.28 | — | 0.54 | 0.43 | 0.33 | 0.51 |
| **P/A** | 0.65 | 0.46 | 0.49 | 0.58 | 0.49 | 0.54 | — | 0.44 | 0.39 | 0.36 |
| **D/U** | 0.50 | 0.39 | 0.53 | 0.32 | 0.34 | 0.43 | 0.44 | — | 0.51 | 0.54 |
| **N/At** | 0.61 | 0.42 | 0.48 | 0.35 | 0.38 | 0.33 | 0.39 | 0.51 | — | 0.44 |
| **Q/A** | 0.57 | 0.36 | 0.40 | 0.64 | 0.46 | 0.51 | 0.36 | 0.54 | 0.44 | — |

### Task affinity, Adam-sized lookahead step

Row i, column j: 1 − L_j(after a step on task i) / L_j(before), ×100. Positive = a step for task i also lowers task j's loss.

| row → col | Prop | AQ | D/A | A/R | E/F | N/N | P/A | D/U | N/At | Q/A |
|---|---|---|---|---|---|---|---|---|---|---|
| **Prop** | -606.56 | -297.46 | -40.14 | -128.62 | -101.08 | -165.15 | -13.38 | -2.91 | -41.23 | -5.31 |
| **AQ** | -163.90 | -1401.45 | -20.23 | -48.33 | -56.03 | -66.50 | -12.79 | -1.71 | -40.19 | -6.74 |
| **D/A** | -191.17 | -481.50 | -79.08 | -161.65 | -125.51 | -179.18 | -16.16 | -3.14 | -72.88 | -13.76 |
| **A/R** | -269.90 | -299.76 | -151.71 | -268.74 | -250.97 | -302.85 | -16.88 | -16.52 | -95.86 | -6.48 |
| **E/F** | -235.87 | -460.44 | -112.15 | -244.29 | -278.87 | -292.27 | -20.67 | -12.56 | -107.25 | -4.06 |
| **N/N** | -278.16 | -318.15 | -143.12 | -260.99 | -249.24 | -285.81 | -15.10 | -15.26 | -96.47 | -7.17 |
| **P/A** | -339.83 | -732.88 | -67.28 | -187.96 | -207.36 | -251.33 | -23.63 | -12.11 | -82.31 | -7.28 |
| **D/U** | -274.30 | -375.49 | -146.02 | -307.39 | -358.48 | -351.82 | -42.08 | -30.37 | -137.09 | -4.58 |
| **N/At** | -215.91 | -612.01 | -98.22 | -204.64 | -213.60 | -229.01 | -14.34 | -11.52 | -89.60 | -5.57 |
| **Q/A** | -320.26 | -561.94 | -74.86 | -167.32 | -156.04 | -211.45 | -17.08 | -4.60 | -69.49 | -5.87 |

**Warning:** non-positive diagonal for ['propaganda_max_pool', 'WA_bi', 'disagree_agree_bi', 'attacking_respectful_bi', 'emotion_fact_bi', 'nasty_nice_bi', 'personal_audience_bi', 'defeater_undercutter_bi', 'negotiate_attack_bi', 'questioning_asserting_bi']: the step overshoots, so this matrix measures curvature as much as transfer.

### Task affinity, a tenth of that step (near first order)

Row i, column j: 1 − L_j(after a step on task i) / L_j(before), ×100. Positive = a step for task i also lowers task j's loss.

| row → col | Prop | AQ | D/A | A/R | E/F | N/N | P/A | D/U | N/At | Q/A |
|---|---|---|---|---|---|---|---|---|---|---|
| **Prop** | -60.77 | -1.58 | -6.14 | -17.51 | -10.78 | -20.41 | -1.84 | -0.61 | -4.26 | -0.95 |
| **AQ** | -8.15 | -138.22 | -1.66 | -3.52 | -4.08 | -3.98 | -3.25 | -0.22 | -5.37 | -1.31 |
| **D/A** | -16.29 | -18.10 | -15.14 | -29.86 | -11.59 | -27.88 | -5.23 | -0.27 | -9.80 | -2.69 |
| **A/R** | -30.88 | -10.33 | -38.97 | -78.59 | -59.26 | -81.83 | -4.32 | +0.21 | -27.54 | -2.78 |
| **E/F** | -23.17 | -11.57 | -15.85 | -54.26 | -60.41 | -67.66 | -3.15 | -0.08 | -26.19 | -1.49 |
| **N/N** | -30.36 | -15.14 | -45.89 | -89.77 | -69.67 | -88.40 | -3.73 | -0.13 | -32.05 | -3.01 |
| **P/A** | -24.09 | -30.26 | -9.98 | -34.98 | -37.08 | -50.04 | +0.53 | -1.46 | -15.32 | -1.07 |
| **D/U** | -14.92 | -4.63 | -21.75 | -55.48 | -78.07 | -67.93 | -7.20 | -1.17 | -29.41 | -1.84 |
| **N/At** | -14.42 | -26.19 | -15.18 | -39.98 | -41.30 | -44.27 | -2.19 | +0.47 | -17.16 | -2.15 |
| **Q/A** | -23.58 | -28.63 | -8.97 | -23.45 | -18.63 | -30.13 | -3.33 | -1.04 | -9.19 | +0.96 |

**Warning:** non-positive diagonal for ['propaganda_max_pool', 'WA_bi', 'disagree_agree_bi', 'attacking_respectful_bi', 'emotion_fact_bi', 'nasty_nice_bi', 'defeater_undercutter_bi', 'negotiate_attack_bi']: the step overshoots, so this matrix measures curvature as much as transfer.

### Mean off-diagonal cosine by epoch

| seed | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | epoch 6 | epoch 7 | epoch 8 | epoch 9 | epoch 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | +0.131 | +0.174 | +0.109 |  |  |  |  |  |  |  |
| 42 | +0.102 | +0.157 | +0.146 | +0.126 |  |  |  |  |  |  |

### Most negative pairs (candidates for leave-one-task-out confirmation)

- step on **Nasty/Nice** → Attacking/Respectful: affinity -89.774, cosine +0.849
- step on **Attacking/Respectful** → Nasty/Nice: affinity -81.832, cosine +0.849
- step on **Defeater/Undercutter** → Emotion/Fact: affinity -78.071, cosine +0.219
- step on **Nasty/Nice** → Emotion/Fact: affinity -69.667, cosine +0.375
- step on **Defeater/Undercutter** → Nasty/Nice: affinity -67.929, cosine +0.195

Seed-42 check: mean test F1 66.08 (λ=0 run) vs 65.11 (plain probed run); GPU runs aren't bit-reproducible, and earlier same-seed reruns drifted about 1 point.

Note: in the training CSVs the Propaganda rows also carry rule-derived IAC labels (e.g. Name_Calling → nasty = 1), and they're in the test set too, so the IAC tasks' single-task baselines also learn that rule.
