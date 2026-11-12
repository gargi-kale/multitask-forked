# Is the extra correlation harmful? (IAC rows)

## Plain (lr0.0001, seeds 1,2,3,42)

### 1. Leak test

8 of 56 ordered pairs have a coefficient ≠ 0 at q < 0.05: 4 negative (head carries another head's signal), 4 positive (head misses information the other head has). Coefficient = change in the log-odds of the label per 1 SD of the added head's logit, holding the label's own head fixed.

| Label | Added head | coefficient [95% CI] | q | seeds sig. same sign |
|---|---|---|---|---|
| Attacking/Respectful | Defeater/Undercutter | +0.21 [+0.11, +0.30] | 0.001 | 4/4 |
| Attacking/Respectful | Personal/Audience | +0.20 [+0.10, +0.29] | 0.003 | 4/4 |
| Nasty/Nice | Defeater/Undercutter | +0.20 [+0.10, +0.31] | 0.003 | 4/4 |
| Nasty/Nice | Personal/Audience | +0.18 [+0.07, +0.29] | 0.019 | 4/4 |
| Negotiate/Attack | Defeater/Undercutter | -0.19 [-0.33, -0.06] | 0.042 | 4/4 |
| Disagree/Agree | Nasty/Nice | -0.19 [-0.33, -0.06] | 0.042 | 3/4 |
| Attacking/Respectful | Emotion/Fact | -0.23 [-0.39, -0.07] | 0.042 | 2/4 |
| Disagree/Agree | Attacking/Respectful | -0.20 [-0.34, -0.06] | 0.042 | 3/4 |

### 2. Cost test: recombine the eight scores (fit on val, scored on test)

Weighted F1 and AUC on test IAC rows only (the paper's Table 3 F1 is over all labelled test rows, so it is shown for reference, not as a like-for-like comparison). Δ = recombined − original, 95% interval.

| Output | Paper (Table 3 F1) | F1 original | F1 recombined | ΔF1 [95% CI] | AUC original | AUC recombined | ΔAUC [95% CI] |
|---|---|---|---|---|---|---|---|
| Disagree/Agree | 71.38 | 65.2 | 63.0 | -2.2 [-6.0, +2.1] | 0.608 | 0.619 | +0.010 [-0.003, +0.024] |
| Emotion/Fact | 63.93 | 65.6 | 66.3 | +0.7 [-0.1, +1.6] | 0.709 | 0.716 | +0.006 [+0.001, +0.014] |
| Attacking/Respectful | 68.07 | 67.8 | 68.6 | +0.8 [-2.7, +3.8] | 0.674 | 0.683 | +0.009 [+0.001, +0.018] |
| Nasty/Nice | 73.69 | 70.3 | 68.7 | -1.6 [-4.0, +0.9] | 0.688 | 0.696 | +0.008 [+0.001, +0.016] |
| Personal/Audience | 65.69 | 65.2 | 62.6 | -2.6 [-10.4, +6.1] | 0.616 | 0.619 | +0.003 [-0.018, +0.026] |
| Defeater/Undercutter | 55.65 | 53.4 | 55.3 | +1.8 [-2.5, +6.8] | 0.531 | 0.554 | +0.023 [-0.010, +0.051] |
| Negotiate/Attack | 64.81 | 66.0 | 66.8 | +0.8 [-1.7, +4.0] | 0.712 | 0.715 | +0.002 [-0.008, +0.013] |
| Questioning/Asserting | 63.23 | 60.0 | 57.6 | -2.4 [-4.5, -0.6] | 0.639 | 0.644 | +0.005 [-0.006, +0.019] |
| **Mean of 8** | 65.81 | 64.2 | 63.6 | -0.6 [-2.4, +1.1] | 0.647 | 0.656 | +0.008 [+0.003, +0.014] |

Mean |phi| over the 28 pairs: labels 0.167, original predictions 0.345, recombined predictions 0.354.

Recombination weights (rows: label predicted; columns: heads' standardised logits; mean over seeds):

| Label | Disagree/Agree | Emotion/Fact | Attacking/Respectful | Nasty/Nice | Personal/Audience | Defeater/Undercutter | Negotiate/Attack | Questioning/Asserting |
|---|---|---|---|---|---|---|---|---|
| Disagree/Agree | +0.27 | +0.03 | -0.24 | +0.37 | +0.32 | +0.11 | +0.26 | -0.16 |
| Emotion/Fact | -0.05 | +1.45 | -0.41 | +0.12 | +0.44 | +0.12 | +0.30 | -0.25 |
| Attacking/Respectful | +0.10 | -0.29 | +0.68 | -0.23 | +0.15 | +0.17 | -0.57 | -0.22 |
| Nasty/Nice | -0.01 | -0.06 | +0.59 | +0.01 | +0.17 | +0.13 | -0.30 | -0.13 |
| Personal/Audience | -0.14 | +0.59 | -0.58 | +0.67 | +0.62 | -0.03 | +0.59 | -0.05 |
| Defeater/Undercutter | -0.19 | -0.19 | +0.78 | +0.03 | +0.17 | -0.07 | +0.62 | -0.07 |
| Negotiate/Attack | -0.26 | -0.29 | +0.17 | +0.16 | -0.22 | -0.08 | +0.49 | -0.08 |
| Questioning/Asserting | +0.22 | -0.17 | -0.84 | +0.43 | +0.22 | -0.22 | -0.44 | +0.26 |

## FAMO + consistency (best) (famo_consf1_m5, seeds 1,2,3,42)

### 1. Leak test

24 of 56 ordered pairs have a coefficient ≠ 0 at q < 0.05: 9 negative (head carries another head's signal), 15 positive (head misses information the other head has). Coefficient = change in the log-odds of the label per 1 SD of the added head's logit, holding the label's own head fixed.

| Label | Added head | coefficient [95% CI] | q | seeds sig. same sign |
|---|---|---|---|---|
| Nasty/Nice | Defeater/Undercutter | +0.41 [+0.28, +0.53] | <0.001 | 4/4 |
| Attacking/Respectful | Defeater/Undercutter | +0.35 [+0.24, +0.46] | <0.001 | 4/4 |
| Emotion/Fact | Defeater/Undercutter | +0.30 [+0.18, +0.41] | <0.001 | 4/4 |
| Nasty/Nice | Personal/Audience | +0.32 [+0.19, +0.44] | <0.001 | 4/4 |
| Nasty/Nice | Attacking/Respectful | +1.92 [+1.17, +2.67] | <0.001 | 4/4 |
| Attacking/Respectful | Personal/Audience | +0.27 [+0.16, +0.38] | <0.001 | 4/4 |
| Attacking/Respectful | Emotion/Fact | -0.32 [-0.45, -0.18] | <0.001 | 4/4 |
| Negotiate/Attack | Personal/Audience | -0.38 [-0.56, -0.19] | <0.001 | 4/4 |
| Attacking/Respectful | Nasty/Nice | -1.37 [-2.04, -0.69] | <0.001 | 4/4 |
| Negotiate/Attack | Defeater/Undercutter | -0.27 [-0.40, -0.13] | <0.001 | 4/4 |
| Disagree/Agree | Nasty/Nice | -0.22 [-0.35, -0.09] | 0.003 | 4/4 |
| Disagree/Agree | Defeater/Undercutter | +0.18 [+0.07, +0.28] | 0.003 | 4/4 |
| Nasty/Nice | Disagree/Agree | +0.24 [+0.10, +0.37] | 0.003 | 2/4 |
| Negotiate/Attack | Questioning/Asserting | -0.41 [-0.65, -0.17] | 0.004 | 4/4 |
| Attacking/Respectful | Disagree/Agree | +0.23 [+0.09, +0.37] | 0.004 | 2/4 |
| Disagree/Agree | Attacking/Respectful | -0.23 [-0.37, -0.09] | 0.005 | 4/4 |
| Nasty/Nice | Questioning/Asserting | +0.18 [+0.05, +0.30] | 0.015 | 3/4 |
| Negotiate/Attack | Emotion/Fact | +0.32 [+0.10, +0.55] | 0.015 | 3/4 |
| Nasty/Nice | Emotion/Fact | -0.22 [-0.39, -0.05] | 0.028 | 2/4 |
| Disagree/Agree | Negotiate/Attack | +0.13 [+0.03, +0.23] | 0.028 | 4/4 |
| Disagree/Agree | Emotion/Fact | -0.13 [-0.23, -0.03] | 0.028 | 4/4 |
| Defeater/Undercutter | Attacking/Respectful | +0.18 [+0.04, +0.32] | 0.029 | 4/4 |
| Defeater/Undercutter | Nasty/Nice | +0.18 [+0.04, +0.32] | 0.029 | 4/4 |
| Attacking/Respectful | Questioning/Asserting | +0.14 [+0.03, +0.25] | 0.037 | 3/4 |

### 2. Cost test: recombine the eight scores (fit on val, scored on test)

Weighted F1 and AUC on test IAC rows only (the paper's Table 3 F1 is over all labelled test rows, so it is shown for reference, not as a like-for-like comparison). Δ = recombined − original, 95% interval.

| Output | Paper (Table 3 F1) | F1 original | F1 recombined | ΔF1 [95% CI] | AUC original | AUC recombined | ΔAUC [95% CI] |
|---|---|---|---|---|---|---|---|
| Disagree/Agree | 71.38 | 64.1 | 62.2 | -2.0 [-6.7, +2.4] | 0.645 | 0.654 | +0.009 [-0.001, +0.019] |
| Emotion/Fact | 63.93 | 66.6 | 67.6 | +1.1 [+0.3, +1.9] | 0.719 | 0.730 | +0.011 [+0.006, +0.016] |
| Attacking/Respectful | 68.07 | 70.2 | 72.5 | +2.2 [+0.4, +4.4] | 0.717 | 0.727 | +0.010 [+0.002, +0.018] |
| Nasty/Nice | 73.69 | 74.0 | 75.9 | +1.9 [-1.7, +5.1] | 0.720 | 0.731 | +0.011 [+0.003, +0.019] |
| Personal/Audience | 65.69 | 63.3 | 60.9 | -2.4 [-17.6, +16.3] | 0.653 | 0.655 | +0.002 [-0.012, +0.016] |
| Defeater/Undercutter | 55.65 | 57.0 | 58.2 | +1.2 [-2.3, +4.8] | 0.564 | 0.582 | +0.019 [-0.009, +0.044] |
| Negotiate/Attack | 64.81 | 69.1 | 69.6 | +0.6 [-1.1, +2.0] | 0.734 | 0.740 | +0.006 [-0.006, +0.017] |
| Questioning/Asserting | 63.23 | 63.1 | 63.3 | +0.2 [-2.1, +2.3] | 0.661 | 0.672 | +0.012 [-0.003, +0.028] |
| **Mean of 8** | 65.81 | 65.9 | 66.3 | +0.3 [-1.6, +2.9] | 0.677 | 0.687 | +0.010 [+0.005, +0.015] |

Mean |phi| over the 28 pairs: labels 0.167, original predictions 0.387, recombined predictions 0.362.

Recombination weights (rows: label predicted; columns: heads' standardised logits; mean over seeds):

| Label | Disagree/Agree | Emotion/Fact | Attacking/Respectful | Nasty/Nice | Personal/Audience | Defeater/Undercutter | Negotiate/Attack | Questioning/Asserting |
|---|---|---|---|---|---|---|---|---|
| Disagree/Agree | +0.44 | +0.08 | +0.06 | +0.11 | +0.50 | +0.09 | +0.39 | -0.36 |
| Emotion/Fact | +0.01 | +1.40 | +0.05 | -0.25 | +0.23 | +0.37 | +0.17 | -0.13 |
| Attacking/Respectful | +0.20 | -0.25 | +1.07 | -0.61 | +0.25 | +0.32 | -0.77 | -0.61 |
| Nasty/Nice | +0.22 | +0.12 | +0.69 | -0.35 | +0.53 | +0.14 | -0.36 | -0.54 |
| Personal/Audience | +0.35 | +0.35 | -0.40 | -0.39 | +0.54 | +0.19 | -0.53 | -0.29 |
| Defeater/Undercutter | -0.04 | -0.20 | +0.95 | -0.43 | -0.18 | +0.18 | +0.12 | +0.01 |
| Negotiate/Attack | +0.01 | -0.02 | -0.18 | +0.28 | -0.20 | -0.30 | +0.99 | +0.09 |
| Questioning/Asserting | +0.13 | -0.20 | -1.24 | +0.59 | +0.45 | -0.08 | -0.91 | +0.01 |
