# Shares and correlations: predictions vs labels (test set)

Bootstrap: 5000 draws over test rows (8848) and seeds. Hard predictions use each run's saved threshold (Youden's J on val).

## Setups

| Setup | run name | seeds | test F1 (mean of 10, x100) | Paper (Table 3 F1) |
|---|---|---|---|---|
| Plain | lr0.0001 | 1,2,3,42 | 68.1 ± 1.0 | 66.74 |
| FAMO fixed | famo_fixed | 1,2,3,42 | 69.5 ± 0.7 | 66.74 |
| FAMO + consistency | famo_consf1_m5 | 1,2,3,42 | 69.9 ± 0.7 | 66.74 |
| Noisy-OR pool | famo_consf1_pnor | 1,2,3 | 68.9 ± 0.4 | 66.74 |

## 1. Shares of 'yes'

Columns: true share; predicted share at the threshold and its difference from the true share with a 95% interval and Holm-adjusted p (family = all cells of that setup); mean score and its difference; McNemar = how many seeds have significantly more false positives than false negatives (+) or the reverse (−) at p < 0.05.

### Plain (4 seeds)

| Output | Source | n | Paper (Table 3 F1) | our F1 | true share | pred share | pred − true [95% CI] | Holm p | mean score | score − true [95% CI] | McNemar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Propaganda | Propaganda | 3350 | 61.74 | 64.8 | 38.1% | 31.4% | -6.6 [-9.8, -3.0] | 0.002 | 44.5% | +6.4 [+3.6, +8.9] | −−− |
| Disagree/Agree | IAC | 3051 | 71.38 | 65.1 | 76.7% | 58.0% | -18.8 [-31.7, -9.4] | 0.014 | 51.2% | -25.6 [-29.1, -22.2] | −−−− |
| Disagree/Agree | Propaganda (rule labels) | 280 | 71.38 | 65.1 | 100.0% | 86.0% | -14.0 [-19.1, -9.1] | <0.001 | 73.3% | -26.7 [-30.0, -23.4] | −−−− |
| Emotion/Fact | IAC | 3038 | 63.93 | 69.8 | 49.3% | 39.2% | -10.2 [-16.8, -4.8] | 0.014 | 44.1% | -5.3 [-7.6, -2.5] | −−−− |
| Emotion/Fact | Propaganda (rule labels) | 906 | 63.93 | 69.8 | 100.0% | 86.0% | -14.0 [-17.6, -10.0] | <0.001 | 80.4% | -19.6 [-22.8, -16.6] | −−−− |
| Attacking/Respectful | IAC | 3050 | 68.07 | 70.8 | 27.1% | 28.5% | +1.4 [-1.6, +4.3] | 1.000 | 40.0% | +12.8 [+10.3, +15.2] | ++ |
| Attacking/Respectful | Propaganda (rule labels) | 372 | 68.07 | 70.8 | 100.0% | 79.3% | -20.7 [-25.8, -15.8] | <0.001 | 76.9% | -23.1 [-26.1, -20.3] | −−−− |
| Nasty/Nice | IAC | 3056 | 73.69 | 74.2 | 19.1% | 26.9% | +7.9 [+2.6, +13.8] | 0.050 | 36.5% | +17.4 [+14.1, +20.6] | +++ |
| Nasty/Nice | Propaganda (rule labels) | 352 | 73.69 | 74.2 | 100.0% | 80.3% | -19.7 [-24.0, -15.7] | <0.001 | 76.4% | -23.6 [-27.3, -20.3] | −−−− |
| Personal/Audience | IAC | 1194 | 65.69 | 66.6 | 77.8% | 67.5% | -10.3 [-18.3, -3.5] | 0.051 | 54.1% | -23.7 [-26.4, -21.1] | −−−− |
| Personal/Audience | Propaganda (rule labels) | 52 | 65.69 | 66.6 | 36.5% | 48.1% | +11.5 [-11.4, +33.8] | 1.000 | 45.7% | +9.1 [-5.5, +23.8] | + |
| Defeater/Undercutter | IAC | 1208 | 55.65 | 57.7 | 63.7% | 67.8% | +4.1 [-9.2, +16.8] | 1.000 | 50.6% | -13.1 [-15.9, -10.4] | ++− |
| Defeater/Undercutter | Propaganda (rule labels) | 68 | 55.65 | 57.7 | 0.0% | 11.4% | +11.4 [+5.0, +19.5] | 0.022 | 35.6% | +35.6 [+32.6, +38.7] | ++ |
| Negotiate/Attack | IAC | 1203 | 64.81 | 69.4 | 60.3% | 61.3% | +0.9 [-6.3, +7.4] | 1.000 | 53.6% | -6.7 [-9.8, -3.6] | ++− |
| Negotiate/Attack | Propaganda (rule labels) | 268 | 64.81 | 69.4 | 7.5% | 15.1% | +7.6 [+2.0, +13.3] | 0.051 | 24.4% | +16.9 [+13.0, +20.6] | +++ |
| Questioning/Asserting | IAC | 1196 | 63.23 | 59.9 | 34.7% | 40.1% | +5.4 [+1.6, +9.5] | 0.050 | 49.3% | +14.6 [+11.9, +17.4] | +++ |
| Questioning/Asserting | Propaganda (rule labels) | 463 | 63.23 | 59.9 | 18.4% | 54.5% | +36.2 [+26.5, +44.8] | <0.001 | 54.3% | +35.9 [+32.3, +39.5] | ++++ |
| Argument Quality | IBM | 2406 | 79.17 | 82.9 | 93.6% | 75.6% | -18.0 [-21.4, -14.9] | <0.001 | 73.7% | -20.0 [-26.3, -13.4] | −−−− |

### FAMO fixed (4 seeds)

| Output | Source | n | Paper (Table 3 F1) | our F1 | true share | pred share | pred − true [95% CI] | Holm p | mean score | score − true [95% CI] | McNemar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Propaganda | Propaganda | 3350 | 61.74 | 66.0 | 38.1% | 34.8% | -3.3 [-6.8, +0.4] | 0.499 | 48.2% | +10.1 [+8.1, +12.1] | −− |
| Disagree/Agree | IAC | 3051 | 71.38 | 62.4 | 76.7% | 48.7% | -28.0 [-36.9, -20.3] | <0.001 | 51.6% | -25.1 [-28.9, -21.0] | −−−− |
| Disagree/Agree | Propaganda (rule labels) | 280 | 71.38 | 62.4 | 100.0% | 81.5% | -18.5 [-23.4, -13.8] | <0.001 | 75.3% | -24.7 [-29.8, -19.5] | −−−− |
| Emotion/Fact | IAC | 3038 | 63.93 | 71.4 | 49.3% | 37.6% | -11.8 [-13.9, -9.7] | <0.001 | 42.6% | -6.7 [-9.7, -4.1] | −−−− |
| Emotion/Fact | Propaganda (rule labels) | 906 | 63.93 | 71.4 | 100.0% | 90.2% | -9.8 [-12.2, -7.7] | <0.001 | 84.4% | -15.6 [-17.6, -13.8] | −−−− |
| Attacking/Respectful | IAC | 3050 | 68.07 | 73.4 | 27.1% | 29.6% | +2.5 [-0.8, +5.3] | 0.611 | 39.0% | +11.9 [+8.9, +15.0] | +++− |
| Attacking/Respectful | Propaganda (rule labels) | 372 | 68.07 | 73.4 | 100.0% | 84.5% | -15.5 [-19.6, -11.7] | <0.001 | 79.7% | -20.3 [-24.1, -16.6] | −−−− |
| Nasty/Nice | IAC | 3056 | 73.69 | 76.7 | 19.1% | 25.0% | +5.9 [+1.7, +9.3] | 0.026 | 36.3% | +17.2 [+14.0, +20.1] | +++ |
| Nasty/Nice | Propaganda (rule labels) | 352 | 73.69 | 76.7 | 100.0% | 84.7% | -15.3 [-20.1, -11.3] | <0.001 | 81.2% | -18.8 [-22.1, -15.5] | −−−− |
| Personal/Audience | IAC | 1194 | 65.69 | 68.6 | 77.8% | 71.0% | -6.8 [-17.1, +3.6] | 0.779 | 57.8% | -20.0 [-23.2, -16.5] | +−− |
| Personal/Audience | Propaganda (rule labels) | 52 | 65.69 | 68.6 | 36.5% | 40.4% | +3.8 [-16.7, +24.6] | 1.000 | 38.7% | +2.1 [-13.0, +17.2] | 0 |
| Defeater/Undercutter | IAC | 1208 | 55.65 | 60.1 | 63.7% | 66.1% | +2.5 [-9.0, +11.8] | 1.000 | 51.8% | -11.9 [-15.7, -8.5] | ++− |
| Defeater/Undercutter | Propaganda (rule labels) | 68 | 55.65 | 60.1 | 0.0% | 9.9% | +9.9 [+3.9, +17.8] | 0.045 | 30.4% | +30.4 [+25.2, +34.9] | +++ |
| Negotiate/Attack | IAC | 1203 | 64.81 | 71.2 | 60.3% | 57.7% | -2.7 [-8.4, +2.9] | 1.000 | 54.7% | -5.7 [-9.1, -2.4] | −− |
| Negotiate/Attack | Propaganda (rule labels) | 268 | 64.81 | 71.2 | 7.5% | 12.5% | +5.0 [+0.1, +10.1] | 0.372 | 19.3% | +11.9 [+7.2, +16.4] | ++ |
| Questioning/Asserting | IAC | 1196 | 63.23 | 64.3 | 34.7% | 39.3% | +4.6 [-0.8, +9.1] | 0.499 | 49.5% | +14.8 [+11.4, +18.1] | +++ |
| Questioning/Asserting | Propaganda (rule labels) | 463 | 63.23 | 64.3 | 18.4% | 37.6% | +19.2 [+8.1, +30.7] | 0.012 | 49.1% | +30.7 [+26.4, +35.0] | ++++ |
| Argument Quality | IBM | 2406 | 79.17 | 81.2 | 93.6% | 71.8% | -21.8 [-27.5, -14.2] | <0.001 | 53.9% | -39.7 [-45.2, -32.1] | −−−− |

### FAMO + consistency (4 seeds)

| Output | Source | n | Paper (Table 3 F1) | our F1 | true share | pred share | pred − true [95% CI] | Holm p | mean score | score − true [95% CI] | McNemar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Propaganda | Propaganda | 3350 | 61.74 | 66.1 | 38.1% | 35.9% | -2.2 [-5.7, +1.8] | 1.000 | 47.8% | +9.7 [+7.8, +11.6] | +−− |
| Disagree/Agree | IAC | 3051 | 71.38 | 64.6 | 76.7% | 52.5% | -24.2 [-26.9, -21.6] | <0.001 | 51.7% | -25.0 [-28.1, -21.9] | −−−− |
| Disagree/Agree | Propaganda (rule labels) | 280 | 71.38 | 64.6 | 100.0% | 81.8% | -18.2 [-23.1, -13.8] | <0.001 | 75.8% | -24.2 [-28.4, -20.5] | −−−− |
| Emotion/Fact | IAC | 3038 | 63.93 | 71.2 | 49.3% | 37.9% | -11.4 [-15.3, -7.7] | <0.001 | 43.5% | -5.8 [-8.9, -2.8] | −−−− |
| Emotion/Fact | Propaganda (rule labels) | 906 | 63.93 | 71.2 | 100.0% | 90.0% | -10.0 [-12.9, -7.6] | <0.001 | 85.3% | -14.7 [-16.5, -13.1] | −−−− |
| Attacking/Respectful | IAC | 3050 | 68.07 | 74.0 | 27.1% | 26.6% | -0.6 [-3.6, +2.7] | 1.000 | 38.5% | +11.4 [+8.6, +14.9] | +−− |
| Attacking/Respectful | Propaganda (rule labels) | 372 | 68.07 | 74.0 | 100.0% | 82.9% | -17.1 [-21.0, -13.5] | <0.001 | 80.2% | -19.8 [-23.0, -16.9] | −−−− |
| Nasty/Nice | IAC | 3056 | 73.69 | 77.1 | 19.1% | 24.3% | +5.3 [+1.4, +9.3] | 0.094 | 35.4% | +16.3 [+13.6, +19.6] | +++ |
| Nasty/Nice | Propaganda (rule labels) | 352 | 73.69 | 77.1 | 100.0% | 84.2% | -15.8 [-19.8, -12.0] | <0.001 | 81.8% | -18.2 [-21.0, -15.6] | −−−− |
| Personal/Audience | IAC | 1194 | 65.69 | 69.4 | 77.8% | 71.4% | -6.4 [-12.2, +1.0] | 0.541 | 59.1% | -18.7 [-22.3, -14.7] | +−−− |
| Personal/Audience | Propaganda (rule labels) | 52 | 65.69 | 69.4 | 36.5% | 36.5% | +0.0 [-18.5, +18.5] | 1.000 | 36.1% | -0.4 [-15.6, +14.9] | 0 |
| Defeater/Undercutter | IAC | 1208 | 55.65 | 59.7 | 63.7% | 63.4% | -0.2 [-7.5, +6.3] | 1.000 | 51.7% | -12.0 [-15.5, -8.6] | +− |
| Defeater/Undercutter | Propaganda (rule labels) | 68 | 55.65 | 59.7 | 0.0% | 7.4% | +7.4 [+2.4, +13.4] | 0.094 | 23.4% | +23.4 [+17.1, +30.0] | + |
| Negotiate/Attack | IAC | 1203 | 64.81 | 71.6 | 60.3% | 57.5% | -2.8 [-5.9, +0.3] | 0.541 | 54.9% | -5.5 [-9.9, -1.7] | − |
| Negotiate/Attack | Propaganda (rule labels) | 268 | 64.81 | 71.6 | 7.5% | 11.8% | +4.4 [-0.5, +9.4] | 0.541 | 18.2% | +10.7 [+6.3, +15.0] | + |
| Questioning/Asserting | IAC | 1196 | 63.23 | 62.9 | 34.7% | 42.6% | +7.9 [-3.1, +19.7] | 0.879 | 48.9% | +14.2 [+10.2, +18.6] | ++− |
| Questioning/Asserting | Propaganda (rule labels) | 463 | 63.23 | 62.9 | 18.4% | 39.4% | +21.1 [+9.6, +33.0] | 0.005 | 45.9% | +27.5 [+22.7, +32.5] | ++++ |
| Argument Quality | IBM | 2406 | 79.17 | 82.8 | 93.6% | 74.3% | -19.3 [-25.1, -12.3] | <0.001 | 56.6% | -37.1 [-44.5, -31.0] | −−−− |

### Noisy-OR pool (3 seeds)

| Output | Source | n | Paper (Table 3 F1) | our F1 | true share | pred share | pred − true [95% CI] | Holm p | mean score | score − true [95% CI] | McNemar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Propaganda | Propaganda | 3350 | 61.74 | 65.1 | 38.1% | 37.4% | -0.7 [-3.1, +1.8] | 1.000 | 44.7% | +6.6 [+1.2, +11.7] | − |
| Disagree/Agree | IAC | 3051 | 71.38 | 61.2 | 76.7% | 45.0% | -31.7 [-34.2, -29.1] | <0.001 | 53.1% | -23.7 [-27.2, -19.9] | −−− |
| Disagree/Agree | Propaganda (rule labels) | 280 | 71.38 | 61.2 | 100.0% | 78.3% | -21.7 [-27.2, -16.1] | <0.001 | 74.5% | -25.5 [-30.9, -21.0] | −−− |
| Emotion/Fact | IAC | 3038 | 63.93 | 70.9 | 49.3% | 36.7% | -12.7 [-17.0, -8.8] | <0.001 | 41.0% | -8.4 [-11.4, -4.9] | −−− |
| Emotion/Fact | Propaganda (rule labels) | 906 | 63.93 | 70.9 | 100.0% | 88.5% | -11.5 [-15.6, -7.7] | <0.001 | 81.8% | -18.2 [-21.0, -15.1] | −−− |
| Attacking/Respectful | IAC | 3050 | 68.07 | 73.2 | 27.1% | 29.0% | +1.9 [-3.7, +9.1] | 1.000 | 39.6% | +12.5 [+9.3, +16.2] | +− |
| Attacking/Respectful | Propaganda (rule labels) | 372 | 68.07 | 73.2 | 100.0% | 82.4% | -17.6 [-21.9, -13.4] | <0.001 | 79.0% | -21.0 [-24.4, -17.3] | −−− |
| Nasty/Nice | IAC | 3056 | 73.69 | 76.1 | 19.1% | 26.0% | +6.9 [+1.0, +12.1] | 0.116 | 37.4% | +18.3 [+14.1, +23.1] | ++ |
| Nasty/Nice | Propaganda (rule labels) | 352 | 73.69 | 76.1 | 100.0% | 82.5% | -17.5 [-22.2, -13.3] | <0.001 | 80.4% | -19.6 [-23.4, -15.7] | −−− |
| Personal/Audience | IAC | 1194 | 65.69 | 68.3 | 77.8% | 65.6% | -12.2 [-19.1, -5.2] | 0.008 | 58.6% | -19.2 [-24.8, -14.5] | −−− |
| Personal/Audience | Propaganda (rule labels) | 52 | 65.69 | 68.3 | 36.5% | 34.0% | -2.6 [-19.4, +14.5] | 1.000 | 39.3% | +2.7 [-11.9, +17.2] | 0 |
| Defeater/Undercutter | IAC | 1208 | 55.65 | 59.2 | 63.7% | 59.8% | -3.8 [-14.6, +4.3] | 1.000 | 52.7% | -10.9 [-13.7, -8.2] | − |
| Defeater/Undercutter | Propaganda (rule labels) | 68 | 55.65 | 59.2 | 0.0% | 13.2% | +13.2 [+5.9, +21.3] | 0.009 | 28.3% | +28.3 [+22.1, +34.4] | +++ |
| Negotiate/Attack | IAC | 1203 | 64.81 | 71.5 | 60.3% | 56.3% | -4.1 [-7.1, -0.9] | 0.095 | 54.2% | -6.1 [-10.9, -2.0] | −− |
| Negotiate/Attack | Propaganda (rule labels) | 268 | 64.81 | 71.5 | 7.5% | 11.9% | +4.5 [-0.4, +9.6] | 0.467 | 18.6% | +11.1 [+6.4, +15.8] | + |
| Questioning/Asserting | IAC | 1196 | 63.23 | 64.8 | 34.7% | 37.2% | +2.5 [-1.3, +6.1] | 0.951 | 48.7% | +14.0 [+10.3, +17.8] | + |
| Questioning/Asserting | Propaganda (rule labels) | 463 | 63.23 | 64.8 | 18.4% | 36.7% | +18.4 [+6.3, +29.2] | 0.019 | 48.1% | +29.7 [+24.4, +35.1] | +++ |
| Argument Quality | IBM | 2406 | 79.17 | 78.5 | 93.6% | 67.2% | -26.4 [-31.1, -19.8] | <0.001 | 60.0% | -33.7 [-36.3, -31.2] | −−− |

## 2. Correlations between outputs

phi = correlation of two yes/no variables (−1 to 1). 'true' uses the labels, 'pred' the hard predictions, 'indep.' what pred would be if the two heads' mistakes were independent, 'score r' the Pearson correlation of the raw scores. Δ = pred − true with a 95% interval; q = Benjamini–Hochberg adjusted p over this setup's pairs. Pairs need ≥ 50 rows with both labels and both labels between 2% and 98% yes.

### Plain (4 seeds)

| Pair | Source | n | true phi | pred phi | Δ [95% CI] | q | indep. phi | pred − indep. q | score r |
|---|---|---|---|---|---|---|---|---|---|
| Personal/Audience × Questioning/Asserting | IAC | 1138 | +0.10 | +0.54 | +0.44 [+0.35, +0.54] | <0.001 | +0.00 | <0.001 | +0.86 |
| Negotiate/Attack × Questioning/Asserting | IAC | 1139 | -0.23 | -0.64 | -0.41 [-0.50, -0.32] | <0.001 | -0.02 | <0.001 | -0.84 |
| Attacking/Respectful × Negotiate/Attack | IAC | 1181 | -0.31 | -0.67 | -0.36 [-0.43, -0.29] | <0.001 | -0.03 | <0.001 | -0.85 |
| Nasty/Nice × Negotiate/Attack | IAC | 1188 | -0.30 | -0.66 | -0.36 [-0.42, -0.29] | <0.001 | -0.03 | <0.001 | -0.85 |
| Emotion/Fact × Negotiate/Attack | IAC | 1175 | -0.28 | -0.63 | -0.36 [-0.47, -0.25] | <0.001 | -0.03 | <0.001 | -0.82 |
| Attacking/Respectful × Questioning/Asserting | IAC | 1175 | +0.10 | +0.40 | +0.31 [+0.22, +0.39] | <0.001 | +0.01 | <0.001 | +0.58 |
| Emotion/Fact × Nasty/Nice | IAC | 3003 | +0.27 | +0.57 | +0.30 [+0.21, +0.39] | <0.001 | +0.02 | <0.001 | +0.80 |
| Emotion/Fact × Attacking/Respectful | IAC | 3000 | +0.27 | +0.57 | +0.30 [+0.17, +0.43] | <0.001 | +0.02 | <0.001 | +0.78 |
| Nasty/Nice × Questioning/Asserting | IAC | 1179 | +0.09 | +0.37 | +0.29 [+0.21, +0.36] | <0.001 | +0.00 | <0.001 | +0.55 |
| Emotion/Fact × Questioning/Asserting | IAC | 1166 | +0.11 | +0.38 | +0.27 [+0.11, +0.46] | 0.004 | +0.01 | <0.001 | +0.52 |
| Disagree/Agree × Nasty/Nice | IAC | 3017 | +0.21 | +0.43 | +0.22 [+0.15, +0.32] | <0.001 | +0.01 | <0.001 | +0.70 |
| Personal/Audience × Negotiate/Attack | IAC | 1140 | -0.19 | -0.38 | -0.19 [-0.26, -0.12] | <0.001 | -0.01 | <0.001 | -0.67 |
| Disagree/Agree × Attacking/Respectful | IAC | 3010 | +0.27 | +0.45 | +0.18 [+0.10, +0.28] | <0.001 | +0.01 | <0.001 | +0.73 |
| Emotion/Fact × Personal/Audience | IAC | 1164 | +0.06 | +0.22 | +0.16 [+0.07, +0.27] | 0.002 | +0.00 | <0.001 | +0.31 |
| Disagree/Agree × Negotiate/Attack | IAC | 1189 | -0.05 | -0.20 | -0.15 [-0.30, -0.02] | 0.049 | -0.00 | 0.003 | -0.43 |
| Attacking/Respectful × Nasty/Nice | IAC | 3018 | +0.74 | +0.89 | +0.15 [+0.08, +0.20] | <0.001 | +0.05 | <0.001 | +0.99 |
| Defeater/Undercutter × Negotiate/Attack | IAC | 1152 | -0.12 | +0.03 | +0.15 [+0.00, +0.29] | 0.073 | -0.00 | 0.727 | +0.17 |
| Attacking/Respectful × Personal/Audience | IAC | 1172 | +0.15 | +0.27 | +0.12 [+0.05, +0.19] | 0.003 | +0.01 | <0.001 | +0.47 |
| Disagree/Agree × Emotion/Fact | IAC | 2999 | +0.04 | +0.15 | +0.11 [+0.00, +0.24] | 0.106 | +0.00 | 0.022 | +0.31 |
| Emotion/Fact × Defeater/Undercutter | IAC | 1179 | +0.03 | -0.08 | -0.11 [-0.22, +0.00] | 0.073 | +0.00 | 0.135 | -0.28 |
| Nasty/Nice × Defeater/Undercutter | IAC | 1193 | +0.13 | +0.02 | -0.11 [-0.30, +0.08] | 0.311 | +0.00 | 0.897 | -0.06 |
| Attacking/Respectful × Defeater/Undercutter | IAC | 1189 | +0.14 | +0.04 | -0.11 [-0.27, +0.07] | 0.294 | +0.00 | 0.727 | -0.02 |
| Nasty/Nice × Personal/Audience | IAC | 1179 | +0.14 | +0.23 | +0.09 [+0.03, +0.16] | 0.010 | +0.01 | <0.001 | +0.42 |
| Disagree/Agree × Questioning/Asserting | IAC | 1181 | +0.05 | +0.14 | +0.09 [-0.09, +0.27] | 0.368 | +0.00 | 0.151 | +0.29 |
| Disagree/Agree × Personal/Audience | IAC | 1178 | +0.13 | +0.21 | +0.08 [-0.09, +0.22] | 0.368 | +0.00 | 0.008 | +0.35 |
| Defeater/Undercutter × Questioning/Asserting | IAC | 1147 | -0.03 | -0.00 | +0.03 [-0.06, +0.12] | 0.601 | -0.00 | 0.928 | -0.07 |
| Personal/Audience × Defeater/Undercutter | IAC | 1140 | +0.08 | +0.06 | -0.01 [-0.12, +0.09] | 0.819 | +0.00 | 0.194 | +0.25 |
| Disagree/Agree × Defeater/Undercutter | IAC | 1195 | +0.09 | +0.10 | +0.00 [-0.14, +0.14] | 0.950 | +0.00 | 0.165 | +0.09 |

### FAMO fixed (4 seeds)

| Pair | Source | n | true phi | pred phi | Δ [95% CI] | q | indep. phi | pred − indep. q | score r |
|---|---|---|---|---|---|---|---|---|---|
| Negotiate/Attack × Questioning/Asserting | IAC | 1139 | -0.23 | -0.71 | -0.48 [-0.57, -0.39] | <0.001 | -0.02 | <0.001 | -0.86 |
| Attacking/Respectful × Questioning/Asserting | IAC | 1175 | +0.10 | +0.53 | +0.43 [+0.31, +0.54] | <0.001 | +0.01 | <0.001 | +0.69 |
| Personal/Audience × Questioning/Asserting | IAC | 1138 | +0.10 | +0.51 | +0.41 [+0.26, +0.56] | <0.001 | +0.00 | <0.001 | +0.91 |
| Nasty/Nice × Questioning/Asserting | IAC | 1179 | +0.09 | +0.49 | +0.40 [+0.27, +0.53] | <0.001 | +0.01 | <0.001 | +0.67 |
| Nasty/Nice × Negotiate/Attack | IAC | 1188 | -0.30 | -0.68 | -0.38 [-0.45, -0.30] | <0.001 | -0.04 | <0.001 | -0.86 |
| Attacking/Respectful × Negotiate/Attack | IAC | 1181 | -0.31 | -0.66 | -0.35 [-0.42, -0.28] | <0.001 | -0.04 | <0.001 | -0.84 |
| Disagree/Agree × Defeater/Undercutter | IAC | 1195 | +0.09 | +0.43 | +0.34 [+0.24, +0.43] | <0.001 | +0.00 | <0.001 | +0.54 |
| Emotion/Fact × Negotiate/Attack | IAC | 1175 | -0.28 | -0.60 | -0.32 [-0.40, -0.24] | <0.001 | -0.03 | <0.001 | -0.78 |
| Emotion/Fact × Nasty/Nice | IAC | 3003 | +0.27 | +0.54 | +0.27 [+0.22, +0.31] | <0.001 | +0.03 | <0.001 | +0.78 |
| Disagree/Agree × Questioning/Asserting | IAC | 1181 | +0.05 | +0.29 | +0.24 [+0.16, +0.31] | <0.001 | +0.00 | <0.001 | +0.46 |
| Disagree/Agree × Attacking/Respectful | IAC | 3010 | +0.27 | +0.50 | +0.24 [+0.16, +0.32] | <0.001 | +0.02 | <0.001 | +0.74 |
| Disagree/Agree × Nasty/Nice | IAC | 3017 | +0.21 | +0.44 | +0.23 [+0.17, +0.31] | <0.001 | +0.01 | <0.001 | +0.68 |
| Emotion/Fact × Questioning/Asserting | IAC | 1166 | +0.11 | +0.34 | +0.23 [+0.12, +0.34] | <0.001 | +0.01 | <0.001 | +0.47 |
| Personal/Audience × Negotiate/Attack | IAC | 1140 | -0.19 | -0.42 | -0.23 [-0.32, -0.13] | <0.001 | -0.01 | <0.001 | -0.73 |
| Attacking/Respectful × Personal/Audience | IAC | 1172 | +0.15 | +0.37 | +0.22 [+0.10, +0.33] | <0.001 | +0.01 | <0.001 | +0.65 |
| Emotion/Fact × Attacking/Respectful | IAC | 3000 | +0.27 | +0.49 | +0.22 [+0.16, +0.27] | <0.001 | +0.03 | <0.001 | +0.73 |
| Disagree/Agree × Negotiate/Attack | IAC | 1189 | -0.05 | -0.27 | -0.21 [-0.29, -0.14] | <0.001 | -0.00 | <0.001 | -0.42 |
| Defeater/Undercutter × Questioning/Asserting | IAC | 1147 | -0.03 | +0.16 | +0.19 [+0.06, +0.31] | 0.004 | -0.00 | 0.007 | +0.17 |
| Disagree/Agree × Personal/Audience | IAC | 1178 | +0.13 | +0.31 | +0.19 [+0.08, +0.28] | <0.001 | +0.00 | <0.001 | +0.59 |
| Nasty/Nice × Personal/Audience | IAC | 1179 | +0.14 | +0.31 | +0.18 [+0.06, +0.28] | 0.003 | +0.01 | <0.001 | +0.61 |
| Attacking/Respectful × Nasty/Nice | IAC | 3018 | +0.74 | +0.86 | +0.12 [+0.09, +0.16] | <0.001 | +0.07 | <0.001 | +0.99 |
| Emotion/Fact × Defeater/Undercutter | IAC | 1179 | +0.03 | -0.09 | -0.12 [-0.23, -0.02] | 0.030 | +0.00 | 0.052 | -0.25 |
| Emotion/Fact × Personal/Audience | IAC | 1164 | +0.06 | +0.18 | +0.12 [+0.02, +0.21] | 0.019 | +0.00 | <0.001 | +0.29 |
| Defeater/Undercutter × Negotiate/Attack | IAC | 1152 | -0.12 | -0.04 | +0.07 [-0.02, +0.16] | 0.151 | -0.00 | 0.297 | +0.03 |
| Attacking/Respectful × Defeater/Undercutter | IAC | 1189 | +0.14 | +0.21 | +0.07 [-0.02, +0.16] | 0.174 | +0.01 | <0.001 | +0.23 |
| Personal/Audience × Defeater/Undercutter | IAC | 1140 | +0.08 | +0.13 | +0.05 [-0.11, +0.27] | 0.600 | +0.00 | 0.177 | +0.41 |
| Disagree/Agree × Emotion/Fact | IAC | 2999 | +0.04 | +0.07 | +0.03 [-0.05, +0.13] | 0.584 | +0.00 | 0.150 | +0.18 |
| Nasty/Nice × Defeater/Undercutter | IAC | 1193 | +0.13 | +0.15 | +0.02 [-0.07, +0.11] | 0.673 | +0.00 | <0.001 | +0.15 |

### FAMO + consistency (4 seeds)

| Pair | Source | n | true phi | pred phi | Δ [95% CI] | q | indep. phi | pred − indep. q | score r |
|---|---|---|---|---|---|---|---|---|---|
| Negotiate/Attack × Questioning/Asserting | IAC | 1139 | -0.23 | -0.68 | -0.45 [-0.53, -0.38] | <0.001 | -0.02 | <0.001 | -0.84 |
| Personal/Audience × Questioning/Asserting | IAC | 1138 | +0.10 | +0.51 | +0.41 [+0.34, +0.49] | <0.001 | +0.00 | <0.001 | +0.91 |
| Attacking/Respectful × Questioning/Asserting | IAC | 1175 | +0.10 | +0.47 | +0.37 [+0.27, +0.46] | <0.001 | +0.01 | <0.001 | +0.65 |
| Nasty/Nice × Negotiate/Attack | IAC | 1188 | -0.30 | -0.65 | -0.35 [-0.43, -0.27] | <0.001 | -0.04 | <0.001 | -0.86 |
| Nasty/Nice × Questioning/Asserting | IAC | 1179 | +0.09 | +0.43 | +0.34 [+0.24, +0.45] | <0.001 | +0.01 | <0.001 | +0.62 |
| Attacking/Respectful × Negotiate/Attack | IAC | 1181 | -0.31 | -0.65 | -0.34 [-0.42, -0.26] | <0.001 | -0.04 | <0.001 | -0.85 |
| Disagree/Agree × Defeater/Undercutter | IAC | 1195 | +0.09 | +0.42 | +0.33 [+0.24, +0.42] | <0.001 | +0.00 | <0.001 | +0.35 |
| Emotion/Fact × Negotiate/Attack | IAC | 1175 | -0.28 | -0.59 | -0.32 [-0.42, -0.22] | <0.001 | -0.03 | <0.001 | -0.77 |
| Emotion/Fact × Nasty/Nice | IAC | 3003 | +0.27 | +0.54 | +0.27 [+0.20, +0.33] | <0.001 | +0.03 | <0.001 | +0.77 |
| Emotion/Fact × Questioning/Asserting | IAC | 1166 | +0.11 | +0.33 | +0.22 [+0.09, +0.35] | <0.001 | +0.01 | <0.001 | +0.43 |
| Emotion/Fact × Attacking/Respectful | IAC | 3000 | +0.27 | +0.49 | +0.22 [+0.14, +0.30] | <0.001 | +0.03 | <0.001 | +0.73 |
| Personal/Audience × Negotiate/Attack | IAC | 1140 | -0.19 | -0.41 | -0.22 [-0.31, -0.13] | <0.001 | -0.01 | <0.001 | -0.68 |
| Disagree/Agree × Questioning/Asserting | IAC | 1181 | +0.05 | +0.27 | +0.22 [+0.11, +0.33] | <0.001 | +0.00 | <0.001 | +0.45 |
| Disagree/Agree × Attacking/Respectful | IAC | 3010 | +0.27 | +0.48 | +0.21 [+0.17, +0.25] | <0.001 | +0.02 | <0.001 | +0.77 |
| Disagree/Agree × Nasty/Nice | IAC | 3017 | +0.21 | +0.42 | +0.21 [+0.17, +0.25] | <0.001 | +0.01 | <0.001 | +0.72 |
| Disagree/Agree × Negotiate/Attack | IAC | 1189 | -0.05 | -0.26 | -0.21 [-0.29, -0.12] | <0.001 | -0.00 | <0.001 | -0.45 |
| Disagree/Agree × Personal/Audience | IAC | 1178 | +0.13 | +0.31 | +0.18 [+0.04, +0.33] | 0.019 | +0.00 | <0.001 | +0.52 |
| Attacking/Respectful × Personal/Audience | IAC | 1172 | +0.15 | +0.31 | +0.17 [+0.07, +0.26] | <0.001 | +0.01 | <0.001 | +0.57 |
| Attacking/Respectful × Nasty/Nice | IAC | 3018 | +0.74 | +0.89 | +0.15 [+0.12, +0.18] | <0.001 | +0.07 | <0.001 | +0.99 |
| Defeater/Undercutter × Questioning/Asserting | IAC | 1147 | -0.03 | +0.12 | +0.15 [+0.03, +0.27] | 0.020 | -0.00 | 0.032 | +0.18 |
| Nasty/Nice × Personal/Audience | IAC | 1179 | +0.14 | +0.26 | +0.13 [+0.04, +0.21] | 0.007 | +0.01 | <0.001 | +0.52 |
| Personal/Audience × Defeater/Undercutter | IAC | 1140 | +0.08 | +0.20 | +0.12 [-0.06, +0.28] | 0.190 | +0.00 | 0.021 | +0.43 |
| Emotion/Fact × Defeater/Undercutter | IAC | 1179 | +0.03 | -0.08 | -0.11 [-0.29, +0.05] | 0.211 | +0.00 | 0.312 | -0.28 |
| Emotion/Fact × Personal/Audience | IAC | 1164 | +0.06 | +0.14 | +0.08 [-0.01, +0.17] | 0.083 | +0.00 | <0.001 | +0.23 |
| Disagree/Agree × Emotion/Fact | IAC | 2999 | +0.04 | +0.09 | +0.05 [-0.00, +0.11] | 0.078 | +0.00 | <0.001 | +0.24 |
| Defeater/Undercutter × Negotiate/Attack | IAC | 1152 | -0.12 | -0.08 | +0.04 [-0.12, +0.21] | 0.683 | -0.00 | 0.346 | +0.07 |
| Nasty/Nice × Defeater/Undercutter | IAC | 1193 | +0.13 | +0.10 | -0.03 [-0.16, +0.10] | 0.683 | +0.00 | 0.118 | +0.04 |
| Attacking/Respectful × Defeater/Undercutter | IAC | 1189 | +0.14 | +0.17 | +0.02 [-0.10, +0.15] | 0.701 | +0.00 | 0.006 | +0.10 |

### Noisy-OR pool (3 seeds)

| Pair | Source | n | true phi | pred phi | Δ [95% CI] | q | indep. phi | pred − indep. q | score r |
|---|---|---|---|---|---|---|---|---|---|
| Personal/Audience × Questioning/Asserting | IAC | 1138 | +0.10 | +0.55 | +0.45 [+0.36, +0.56] | <0.001 | +0.01 | <0.001 | +0.92 |
| Negotiate/Attack × Questioning/Asserting | IAC | 1139 | -0.23 | -0.69 | -0.45 [-0.55, -0.34] | <0.001 | -0.02 | <0.001 | -0.83 |
| Attacking/Respectful × Questioning/Asserting | IAC | 1175 | +0.10 | +0.50 | +0.40 [+0.30, +0.50] | <0.001 | +0.01 | <0.001 | +0.67 |
| Nasty/Nice × Questioning/Asserting | IAC | 1179 | +0.09 | +0.45 | +0.36 [+0.23, +0.52] | <0.001 | +0.01 | <0.001 | +0.64 |
| Disagree/Agree × Defeater/Undercutter | IAC | 1195 | +0.09 | +0.43 | +0.34 [+0.22, +0.47] | <0.001 | +0.00 | <0.001 | +0.42 |
| Nasty/Nice × Negotiate/Attack | IAC | 1188 | -0.30 | -0.64 | -0.33 [-0.42, -0.25] | <0.001 | -0.04 | <0.001 | -0.85 |
| Attacking/Respectful × Negotiate/Attack | IAC | 1181 | -0.31 | -0.64 | -0.33 [-0.40, -0.25] | <0.001 | -0.04 | <0.001 | -0.84 |
| Emotion/Fact × Negotiate/Attack | IAC | 1175 | -0.28 | -0.59 | -0.31 [-0.39, -0.24] | <0.001 | -0.03 | <0.001 | -0.80 |
| Personal/Audience × Negotiate/Attack | IAC | 1140 | -0.19 | -0.48 | -0.29 [-0.37, -0.21] | <0.001 | -0.01 | <0.001 | -0.71 |
| Attacking/Respectful × Personal/Audience | IAC | 1172 | +0.15 | +0.42 | +0.28 [+0.18, +0.37] | <0.001 | +0.01 | <0.001 | +0.65 |
| Emotion/Fact × Nasty/Nice | IAC | 3003 | +0.27 | +0.54 | +0.27 [+0.19, +0.36] | <0.001 | +0.03 | <0.001 | +0.78 |
| Disagree/Agree × Nasty/Nice | IAC | 3017 | +0.21 | +0.47 | +0.26 [+0.20, +0.33] | <0.001 | +0.01 | <0.001 | +0.67 |
| Disagree/Agree × Personal/Audience | IAC | 1178 | +0.13 | +0.38 | +0.26 [+0.13, +0.39] | <0.001 | +0.00 | <0.001 | +0.55 |
| Disagree/Agree × Attacking/Respectful | IAC | 3010 | +0.27 | +0.52 | +0.25 [+0.17, +0.34] | <0.001 | +0.02 | <0.001 | +0.72 |
| Nasty/Nice × Personal/Audience | IAC | 1179 | +0.14 | +0.38 | +0.25 [+0.12, +0.36] | <0.001 | +0.01 | <0.001 | +0.60 |
| Emotion/Fact × Attacking/Respectful | IAC | 3000 | +0.27 | +0.52 | +0.24 [+0.17, +0.33] | <0.001 | +0.03 | <0.001 | +0.76 |
| Emotion/Fact × Questioning/Asserting | IAC | 1166 | +0.11 | +0.34 | +0.24 [+0.10, +0.36] | <0.001 | +0.01 | <0.001 | +0.49 |
| Disagree/Agree × Questioning/Asserting | IAC | 1181 | +0.05 | +0.25 | +0.20 [+0.08, +0.31] | <0.001 | +0.00 | <0.001 | +0.41 |
| Disagree/Agree × Negotiate/Attack | IAC | 1189 | -0.05 | -0.23 | -0.18 [-0.31, -0.06] | 0.009 | -0.00 | <0.001 | -0.38 |
| Emotion/Fact × Personal/Audience | IAC | 1164 | +0.06 | +0.21 | +0.16 [+0.06, +0.26] | 0.003 | +0.00 | <0.001 | +0.33 |
| Attacking/Respectful × Defeater/Undercutter | IAC | 1189 | +0.14 | +0.29 | +0.15 [+0.06, +0.24] | 0.002 | +0.01 | <0.001 | +0.18 |
| Defeater/Undercutter × Questioning/Asserting | IAC | 1147 | -0.03 | +0.10 | +0.13 [+0.03, +0.23] | 0.009 | -0.00 | 0.011 | +0.13 |
| Attacking/Respectful × Nasty/Nice | IAC | 3018 | +0.74 | +0.86 | +0.12 [+0.07, +0.17] | <0.001 | +0.07 | <0.001 | +0.99 |
| Nasty/Nice × Defeater/Undercutter | IAC | 1193 | +0.13 | +0.23 | +0.10 [-0.03, +0.22] | 0.119 | +0.00 | <0.001 | +0.13 |
| Emotion/Fact × Defeater/Undercutter | IAC | 1179 | +0.03 | -0.05 | -0.08 [-0.17, +0.02] | 0.109 | +0.00 | 0.205 | -0.21 |
| Defeater/Undercutter × Negotiate/Attack | IAC | 1152 | -0.12 | -0.05 | +0.06 [-0.03, +0.16] | 0.210 | -0.00 | 0.212 | +0.07 |
| Disagree/Agree × Emotion/Fact | IAC | 2999 | +0.04 | +0.10 | +0.06 [-0.02, +0.13] | 0.153 | +0.00 | 0.011 | +0.20 |
| Personal/Audience × Defeater/Undercutter | IAC | 1140 | +0.08 | +0.09 | +0.02 [-0.16, +0.18] | 0.844 | +0.00 | 0.273 | +0.38 |

## 3. Summary scores (one number per setup)

| Score | Plain | FAMO fixed | FAMO + consistency | Noisy-OR pool |
|---|---|---|---|---|
| mean |pred − true share|, hard | 13.0 [10.9, 15.5] | 11.5 [10.7, 12.9] | 10.7 [9.9, 12.2] | 12.0 [10.9, 13.7] |
| mean |mean score − true share| | 18.7 [17.5, 20.0] | 17.7 [16.3, 19.6] | 16.5 [15.0, 18.5] | 17.2 [16.1, 18.9] |
| mean |phi|, labels | 0.167 [0.152, 0.183] | 0.167 [0.152, 0.183] | 0.167 [0.152, 0.183] | 0.167 [0.152, 0.183] |
| mean |phi|, predictions | 0.352 [0.321, 0.384] | 0.385 [0.361, 0.410] | 0.377 [0.345, 0.413] | 0.396 [0.357, 0.431] |
| mean |score r| | 0.516 [0.489, 0.549] | 0.564 [0.538, 0.587] | 0.548 [0.523, 0.574] | 0.548 [0.531, 0.568] |
| mean (|pred phi| − |true phi|) | 0.185 [0.149, 0.219] | 0.218 [0.190, 0.246] | 0.210 [0.175, 0.248] | 0.228 [0.188, 0.266] |
| RMS (pred phi − true phi) | 0.247 [0.203, 0.294] | 0.269 [0.237, 0.306] | 0.255 [0.224, 0.290] | 0.270 [0.237, 0.305] |
| mean (pred phi − indep. phi) | 0.098 [0.080, 0.116] | 0.134 [0.122, 0.146] | 0.124 [0.105, 0.146] | 0.151 [0.122, 0.174] |

Share scores are in percentage points; others in correlation units. Paper (Table 3 F1) has no counterpart for these scores; see the Setups table.

### Setup vs setup (row-paired bootstrap; Holm over all rows of this table)

| Score | A | B | B − A [95% CI] | Holm p |
|---|---|---|---|---|
| mean |pred − true share|, hard | Plain | FAMO fixed | -1.5 [-3.7, +0.7] | 1.000 |
| mean |pred − true share|, hard | Plain | FAMO + consistency | -2.3 [-4.5, +0.0] | 1.000 |
| mean |pred − true share|, hard | Plain | Noisy-OR pool | -1.1 [-3.4, +1.5] | 1.000 |
| mean |pred − true share|, hard | FAMO fixed | FAMO + consistency | -0.8 [-1.8, +0.3] | 1.000 |
| mean |pred − true share|, hard | FAMO fixed | Noisy-OR pool | +0.4 [-0.7, +1.8] | 1.000 |
| mean |pred − true share|, hard | FAMO + consistency | Noisy-OR pool | +1.2 [+0.1, +2.6] | 1.000 |
| mean |mean score − true share| | Plain | FAMO fixed | -1.0 [-2.5, +0.8] | 1.000 |
| mean |mean score − true share| | Plain | FAMO + consistency | -2.2 [-3.9, -0.2] | 0.523 |
| mean |mean score − true share| | Plain | Noisy-OR pool | -1.5 [-2.7, +0.1] | 0.851 |
| mean |mean score − true share| | FAMO fixed | FAMO + consistency | -1.2 [-3.3, +0.8] | 1.000 |
| mean |mean score − true share| | FAMO fixed | Noisy-OR pool | -0.5 [-2.2, +1.3] | 1.000 |
| mean |mean score − true share| | FAMO + consistency | Noisy-OR pool | +0.7 [-1.1, +2.6] | 1.000 |
| RMS (pred phi − true phi) | Plain | FAMO fixed | +0.022 [-0.030, +0.076] | 1.000 |
| RMS (pred phi − true phi) | Plain | FAMO + consistency | +0.008 [-0.043, +0.060] | 1.000 |
| RMS (pred phi − true phi) | Plain | Noisy-OR pool | +0.024 [-0.028, +0.074] | 1.000 |
| RMS (pred phi − true phi) | FAMO fixed | FAMO + consistency | -0.014 [-0.055, +0.027] | 1.000 |
| RMS (pred phi − true phi) | FAMO fixed | Noisy-OR pool | +0.001 [-0.041, +0.041] | 1.000 |
| RMS (pred phi − true phi) | FAMO + consistency | Noisy-OR pool | +0.015 [-0.026, +0.055] | 1.000 |
| mean (|pred phi| − |true phi|) | Plain | FAMO fixed | +0.033 [-0.004, +0.070] | 1.000 |
| mean (|pred phi| − |true phi|) | Plain | FAMO + consistency | +0.025 [-0.017, +0.070] | 1.000 |
| mean (|pred phi| − |true phi|) | Plain | Noisy-OR pool | +0.044 [-0.004, +0.089] | 1.000 |
| mean (|pred phi| − |true phi|) | FAMO fixed | FAMO + consistency | -0.008 [-0.044, +0.029] | 1.000 |
| mean (|pred phi| − |true phi|) | FAMO fixed | Noisy-OR pool | +0.010 [-0.032, +0.049] | 1.000 |
| mean (|pred phi| − |true phi|) | FAMO + consistency | Noisy-OR pool | +0.019 [-0.031, +0.063] | 1.000 |
