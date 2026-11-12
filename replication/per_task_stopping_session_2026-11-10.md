# Per-task early stopping and self-distillation (10 Nov 2026)

This note records one working session on the main argument data: the multi-task model with the 10 outputs that the paper reports in Table 3. It covers:

1. Whether Caruana (1997) rotates which task is the "main task", and what he says about early stopping.
2. One run that stops training on a single main task (Disagree/Agree) instead of the mean of all 10.
3. A replay of saved runs where each task is scored at its own best epoch (4 seeds).
4. Self-distillation (approach 2): a task stops learning from its labels at its best epoch and copies its own best-epoch scores after that. Three runs: plain loss, FAMO with validation F1, and FAMO with validation loss.

**Short version:** scoring each task at its own best epoch gives +0.9 test F1 over one shared stopping point (4 seeds, significant) and ties single-task. But it needs 3–4 saved models. Stopping on Disagree/Agree alone made things worse. None of the three self-distillation runs kept that gain in one model. In both FAMO runs, the model finally kept came from an epoch before any task had switched, so it contained no distillation at all.

---

## Timeline

All times are US Central (CST). They come from Modal's app and result-file times and from the session log. A "~" means the time is estimated from the events around it.

| Date and time | What happened |
|---|---|
| 10 Nov, ~10:20 | Caruana (1997), "Multitask Learning", *Machine Learning* 28:41–75, downloaded and read. |
| 10 Nov, 10:37 | Main-task run launched: `famo_consf1_m5`, stop on Disagree/Agree, seed 42. |
| 10 Nov, 11:05 | Per-task replay worked out from the saved `famo_consf1_m5` runs (seeds 42, 1, 2, 3). |
| 10 Nov, 12:47 | Self-distillation run on the plain loss launched (`sd_p2_s42`). A first launch at 12:06 was replaced. |
| 10 Nov, 13:42 | Main-task run finished and saved. |
| 10 Nov, 14:02 | Plain self-distillation run finished and saved. |
| 10 Nov, 16:03 | FAMO self-distillation run launched (validation F1, patience 2): `famo_consf1_m5_sd_p2_s42`. |
| 10 Nov, 17:19 | FAMO self-distillation run launched (validation loss, patience 3): `famo_consf1_m5_sd_p3_loss_s42`. |
| 10 Nov, 17:35 | FAMO validation-F1 run finished and saved. |
| 10 Nov, 18:50 | FAMO validation-loss run finished and saved. |
| 10 Nov, ~19:00 | Reports pulled and this note written. |

---

## Terms used here

- **Output / task:** one of the 10 things the model predicts, e.g. Disagree/Agree. Propaganda is the max-pool over the technique outputs.
- **F1:** weighted F1 × 100. The decision threshold is tuned on the validation split. "Mean of 10" is the average over the 10 outputs.
- **Val / test:** every table shows the validation score first, then test. Validation picks the epoch and the thresholds. Test is never used to make any choice; it is only reported.
- **Paper (Table 3 F1):** the paper's multi-task F1. The paper does not say whether that is validation or test; we compare it with our test.
- **One stopping point ("mean stop"):** our usual rule. After every epoch, the mean validation F1 of the 10 outputs is computed, and the best epoch is kept for all outputs together.
- **Per-task epochs (replay):** take one saved run and score each output at the epoch where *that output's own* validation F1 was highest. This needs one saved model per distinct epoch used.
- **Single-task:** 10 separate models, one per output, each with its own stopping point.
- **`famo_consf1_m5`:** our best multi-task setup. FAMO balances the losses of the outputs, plus the masked-word term and the filtered consistency term on unlabelled cells. "FAMO" in this note always means this setup.
- **Plain:** the same model with the losses simply added up (lr 1e-4), no FAMO.
- **10-epoch cap:** the learning rate falls linearly to 0 over 10 epochs, so training cannot usefully go further. The cap is not the stopping rule.

---

## 1. Caruana (1997)

- No experiment rotates which task is the main task.
- His measure is RMSE (an error size), never F1.
- In Table 1 (1D-ALVINN, 9 outputs from one multi-task network), 6 outputs are significantly better than single-task and none significantly worse.
- Section 7.1 advises early stopping **per task**: save a snapshot of the network whenever a task reaches its best validation score, and use that snapshot for that task. Section 7.2 suggests a separate learning rate for each task, but gives no numbers.

This is what led to the per-task replay (Section 3) and to self-distillation (Section 4).

---

## 2. Stopping on one main task (Disagree/Agree), seed 42

**Setup:** `famo_consf1_m5`, but the epoch is chosen on Disagree/Agree's validation F1 alone. Training stops once its validation F1 and AUC have both failed to improve for 3 epochs. Launch: `modal_replicate.py::main_task_stop --task disagree_agree_bi --seeds 42`. Run: `paper_replication_retune_famo/famo_consf1_m5_main_disagree_agree_bi_s42`.

It ran for 8 epochs and kept epoch 5.

| Setup | Epoch | D/A val F1 | D/A test F1 | D/A test AUC | Mean of 10 test F1 |
|---|---|---|---|---|---|
| Paper (Table 3 F1) | – | – | 71.38 | – | 66.74 |
| Single-task | 5 | 69.2 | 69.1 | 64.2 | – |
| Multi-task, stop on mean (saved run) | 2 | 66.0 | 64.7 | 67.1 | 69.9 |
| Same saved run scored at its D/A-best epoch | 5 | 70.3 | 69.6 | 66.6 | 69.1 |
| **Multi-task, stop on D/A (new run)** | 5 | **65.4** | **63.5** | 65.6 | **67.9** |

Mean validation / test F1 of all 10, per epoch, in the new run (D/A in brackets):

| Epoch | Mean val / test | D/A val / test |
|---|---|---|
| 1 | 69.5 / 68.5 | 64.0 / 63.1 |
| 2 | 70.6 / 69.8 | 65.2 / 63.4 |
| 3 | 69.5 / 68.5 | 59.9 / 57.7 |
| 4 | 68.1 / 66.8 | 59.9 / 57.5 |
| 5 (kept) | 69.4 / 67.9 | 65.4 / 63.5 |
| 6 | 69.6 / 68.1 | 63.4 / 61.1 |
| 7 | 69.8 / 68.3 | 63.3 / 61.1 |
| 8 | 68.9 / 67.3 | 58.9 / 56.4 |

Per output at the kept epoch:

| Task | Paper (Table 3 F1) | Stop on D/A (val / test) | Stop on mean, saved run (val / test) |
|---|---|---|---|
| Propaganda | 61.74 | 64.1 / 63.6 | 64.8 / 66.7 |
| Disagree/Agree | 71.38 | 65.4 / 63.5 | 66.0 / 64.7 |
| Emotion/Fact | 63.93 | 74.0 / 69.1 | 73.2 / 69.7 |
| Attacking/Respectful | 68.07 | 72.0 / 69.9 | 75.3 / 74.4 |
| Nasty/Nice | 73.69 | 76.4 / 74.7 | 79.2 / 78.4 |
| Personal/Audience | 65.69 | 72.3 / 70.3 | 68.3 / 67.5 |
| Defeater/Undercutter | 55.65 | 58.7 / 57.3 | 61.9 / 60.4 |
| Negotiate/Attack | 64.81 | 71.3 / 71.3 | 70.6 / 71.7 |
| Questioning/Asserting | 63.23 | 61.9 / 61.1 | 67.3 / 65.8 |
| Argument Quality | 79.17 | 78.1 / 78.5 | 79.4 / 79.3 |
| **Mean of 10** | **66.74** | **69.4 / 67.9** | **70.6 / 69.9** |

**What this shows:** stopping on Disagree/Agree made Disagree/Agree *worse* (63.5 vs 64.7 test) and the mean of 10 worse (67.9 vs 69.9). Retraining with the same seed did not reproduce the saved run's curve: there, D/A reached 70.3 validation at epoch 5; here it reached 65.4. GPU training is not deterministic, so the saved run's 69.6 at epoch 5 was partly a lucky epoch.

---

## 3. Each task at its own best epoch (replay of saved runs, 4 seeds)

**Setup:** no new training. In each saved `famo_consf1_m5` run (seeds 42, 1, 2, 3), each output is scored at the epoch where its own validation F1 was highest. The epoch is picked on validation only. 95% confidence intervals use the t-distribution with 3 degrees of freedom.

| Task | Paper (Table 3 F1) | One stopping point (val / test) | Each task's own best epoch (val / test) | Single-task (val / test) |
|---|---|---|---|---|
| Propaganda | 61.74 | 63.9 / 66.1 | 64.5 / 66.3 | 65.8 / 67.6 |
| Disagree/Agree | 71.38 | 65.7 / 64.6 | 68.2 / 67.1 | 69.7 / 69.1 |
| Emotion/Fact | 63.93 | 74.4 / 71.2 | 74.6 / 71.2 | 75.6 / 72.6 |
| Attacking/Respectful | 68.07 | 75.3 / 74.0 | 75.4 / 74.1 | 74.1 / 72.4 |
| Nasty/Nice | 73.69 | 77.9 / 77.1 | 78.2 / 77.4 | 77.5 / 76.1 |
| Personal/Audience | 65.69 | 71.3 / 69.4 | 72.8 / 70.2 | 71.2 / 69.2 |
| Defeater/Undercutter | 55.65 | 61.4 / 59.7 | 62.4 / 60.4 | 62.5 / 60.0 |
| Negotiate/Attack | 64.81 | 71.3 / 71.6 | 71.7 / 71.6 | 69.8 / 69.5 |
| Questioning/Asserting | 63.23 | 65.0 / 62.9 | 67.1 / 65.7 | 68.4 / 67.6 |
| Argument Quality | 79.17 | 82.4 / 82.8 | 83.9 / 84.5 | 83.8 / 83.7 |
| **Mean of 10** | **66.74** | **70.9 / 69.9** | **71.9 / 70.9** | **71.8 / 70.8** |

- Own best epoch vs one stopping point: validation +1.0 [+0.7, +1.3], test **+0.9 [+0.5, +1.3]**. Every seed went up.
- Own best epoch vs single-task: validation +0.0 [−0.5, +0.6], test +0.1 [−0.8, +0.9]. A tie: the gap to single-task closes.
- Per seed, mean test F1 (one stop / own best / single-task): s42 69.9 / 70.8 / 70.7; s1 69.6 / 70.2 / 70.8; s2 69.4 / 70.6 / 70.6; s3 70.9 / 71.7 / 71.0.
- **Cost:** each seed used 3–4 different epochs (s42: 1, 2, 4, 5; s1: 1–4; s2: 1, 2, 3, 5; s3: 1, 2, 4), so you need 3–4 saved models to serve all 10 outputs.
- **Caveat:** these runs stopped on the mean after 4–5 epochs. Tasks that peak later were cut off, so this may understate the gain.

---

## 4. Self-distillation: one model with per-task stopping

**The idea (approach 2):** train as usual and watch each output's validation score. When an output has not improved for *patience* epochs, it "switches":
- the weights from its best epoch (kept in CPU memory, from this same run) score every training row once; and
- from then on, that output's loss on its true labels is replaced by a loss that pulls its predictions towards those stored scores (binary KL divergence, on the same labelled rows with the same class weights).

The aim is to freeze each task at its peak while the shared layers keep training for the others. Training stops when all 10 have switched (or at the 10-epoch cap). The final model is the epoch with the best mean validation F1, the same rule as usual. Every run trains from scratch; no earlier saved model is used as a teacher.

In the FAMO version, a switched output leaves FAMO and its distillation loss is added outside FAMO with weight 1, as the masked-word and consistency terms are.

**Reading the per-epoch tables:** "Switched by end of epoch" counts outputs that switched when that epoch was scored. Their distillation loss is used from the *next* epoch onwards.

### 4a. Plain loss, switch on validation F1, patience 2 (seed 42)

Run `paper_replication_self_distill/sd_p2_s42`, launched 12:47, finished 14:02. It ran 7 epochs and kept epoch 5. The comparison is the saved plain run `paper_replication_retune/lr0.0001_s42` (same settings, 10 epochs, kept epoch 8).

Switches (output: best epoch → switched after epoch): Defeater/Undercutter 1 → 3; Nasty/Nice, Personal/Audience, Questioning/Asserting, Argument Quality 2 → 4; Attacking/Respectful, Negotiate/Attack 3 → 5; Propaganda, Disagree/Agree 4 → 6; Emotion/Fact 5 → 7.

| Epoch | Switched by end of epoch | Self-distill mean val / test | Saved plain run mean val / test |
|---|---|---|---|
| 1 | 0 | 65.4 / 64.4 | 67.7 / 66.4 |
| 2 | 0 | 67.6 / 66.8 | 67.8 / 66.7 |
| 3 | 1 | 66.4 / 65.1 | 67.3 / 66.3 |
| 4 | 5 | 68.5 / 67.0 | 68.3 / 66.8 |
| 5 (kept) | 7 | 68.7 / 67.2 | 68.6 / 67.0 |
| 6 | 9 | 67.0 / 65.7 | 68.6 / 67.2 |
| 7 | 10 | 67.9 / 66.9 | 67.8 / 66.5 |
| 8 | – | – | 69.3 / 68.0 (kept) |
| 9 | – | – | 68.7 / 67.2 |
| 10 | – | – | 69.0 / 67.6 |

| Task | Paper (Table 3 F1) | Plain, one stop (val / test) | Plain, own best epoch (val / test) | Self-distilled (val / test) | Single-task (val / test) |
|---|---|---|---|---|---|
| Propaganda | 61.74 | 63.8 / 63.8 | 66.0 / 68.0 | 64.0 / 63.8 | 66.2 / 68.0 |
| Disagree/Agree | 71.38 | 67.8 / 66.7 | 71.0 / 69.5 | 65.0 / 63.5 | 69.2 / 69.1 |
| Emotion/Fact | 63.93 | 74.5 / 71.0 | 74.5 / 71.0 | 75.4 / 71.5 | 75.3 / 71.7 |
| Attacking/Respectful | 68.07 | 72.3 / 70.4 | 73.1 / 71.5 | 72.4 / 71.1 | 75.0 / 73.3 |
| Nasty/Nice | 73.69 | 73.3 / 71.7 | 74.7 / 73.5 | 76.8 / 75.5 | 78.4 / 77.1 |
| Personal/Audience | 65.69 | 69.4 / 68.2 | 70.8 / 68.8 | 57.2 / 54.4 | 72.2 / 69.9 |
| Defeater/Undercutter | 55.65 | 57.7 / 55.4 | 60.9 / 59.2 | 59.1 / 56.7 | 62.4 / 59.4 |
| Negotiate/Attack | 64.81 | 67.4 / 69.0 | 68.7 / 68.4 | 68.6 / 70.2 | 70.4 / 69.8 |
| Questioning/Asserting | 63.23 | 63.4 / 60.8 | 67.3 / 63.3 | 64.4 / 61.3 | 66.6 / 66.2 |
| Argument Quality | 79.17 | 83.2 / 82.8 | 86.0 / 85.3 | 83.7 / 83.6 | 82.4 / 82.0 |
| **Mean of 10** | **66.74** | **69.3 / 68.0** | **71.3 / 69.8** | **68.7 / 67.2** | **71.8 / 70.7** |

**What this shows:** self-distillation scored 0.8 test below the plain run's single stopping point. Personal/Audience collapsed (validation 62.6 at its epoch-2 teacher, 52.4 by epoch 7; test 54.4 vs 68.2). In plain runs, Personal/Audience learns late (validation ~70 at epochs 8–10), so switching it at epoch 4 froze it before its real peak. Nasty/Nice gained (+3.8 test).

### 4b. FAMO, switch on validation F1, patience 2 (seed 42)

Run `paper_replication_self_distill/famo_consf1_m5_sd_p2_s42`, launched 16:03, finished 17:35. It ran 5 epochs and **kept epoch 1, before any switch**. The comparison is the saved `paper_replication_retune_famo/famo_consf1_m5_s42` (5 epochs, kept epoch 2).

Switches: Propaganda, Personal/Audience, Defeater/Undercutter, Questioning/Asserting 1 → 3; Attacking/Respectful, Nasty/Nice, Argument Quality 2 → 4; Disagree/Agree, Emotion/Fact, Negotiate/Attack 3 → 5.

| Epoch | Switched by end of epoch | Self-distill mean val / test | Saved `famo_consf1_m5_s42` mean val / test |
|---|---|---|---|
| 1 (kept) | 0 | 70.4 / 69.3 | 69.7 / 68.8 |
| 2 | 0 | 69.5 / 68.8 | 70.6 / 69.9 (kept) |
| 3 | 4 | 69.9 / 69.0 | 68.1 / 67.4 |
| 4 | 7 | 69.5 / 68.1 | 68.4 / 67.6 |
| 5 | 10 | 68.5 / 66.7 | 70.3 / 69.1 |

| Task | Paper (Table 3 F1) | FAMO, one stop (val / test) | FAMO, own best epoch (val / test) | Self-distilled (val / test) | Single-task (val / test) |
|---|---|---|---|---|---|
| Propaganda | 61.74 | 64.8 / 66.7 | 64.8 / 66.7 | 64.8 / 65.5 | 66.2 / 68.0 |
| Disagree/Agree | 71.38 | 66.0 / 64.7 | 70.3 / 69.6 | 65.0 / 63.8 | 69.2 / 69.1 |
| Emotion/Fact | 63.93 | 73.2 / 69.7 | 73.6 / 69.9 | 72.6 / 68.5 | 75.3 / 71.7 |
| Attacking/Respectful | 68.07 | 75.3 / 74.4 | 75.3 / 74.4 | 74.3 / 73.0 | 75.0 / 73.3 |
| Nasty/Nice | 73.69 | 79.2 / 78.4 | 79.2 / 78.4 | 76.1 / 75.8 | 78.4 / 77.1 |
| Personal/Audience | 65.69 | 68.3 / 67.5 | 74.4 / 70.5 | 70.8 / 68.8 | 72.2 / 69.9 |
| Defeater/Undercutter | 55.65 | 61.9 / 60.4 | 62.7 / 61.3 | 62.6 / 61.8 | 62.4 / 59.4 |
| Negotiate/Attack | 64.81 | 70.6 / 71.7 | 71.0 / 71.3 | 69.7 / 69.9 | 70.4 / 69.8 |
| Questioning/Asserting | 63.23 | 67.3 / 65.8 | 67.3 / 65.8 | 67.5 / 65.4 | 66.6 / 66.2 |
| Argument Quality | 79.17 | 79.4 / 79.3 | 79.9 / 80.4 | 80.2 / 80.5 | 82.4 / 82.0 |
| **Mean of 10** | **66.74** | **70.6 / 69.9** | **71.9 / 70.8** | **70.4 / 69.3** | **71.8 / 70.7** |

**What this shows:** the kept model is just `famo_consf1_m5` at epoch 1; the distillation never reached it. After the switches, the mean kept falling (68.5 / 66.7 by epoch 5). Distillation did not hold the scores in place.

### 4c. FAMO, switch on the task's own validation loss, patience 3 (seed 42)

**Change from 4b:** an output's best epoch is now where its own class-weighted validation loss is lowest (lower is better), not where its validation F1 is highest. It must fail to improve for 3 epochs instead of 2. Run `paper_replication_self_distill/famo_consf1_m5_sd_p3_loss_s42`, launched 17:19, finished 18:50. It ran 5 epochs and **kept epoch 2, again before any switch**.

Switches: Propaganda, Disagree/Agree, Nasty/Nice, Personal/Audience, Defeater/Undercutter, Negotiate/Attack, Questioning/Asserting 1 → 4; Emotion/Fact, Attacking/Respectful, Argument Quality 2 → 5. Validation loss bottomed out at epoch 1 or 2 for every output. This was expected: total validation loss rises from epoch 2 in every run because the model becomes overconfident.

| Epoch | Switched by end of epoch | Self-distill mean val / test | Mean val AUC / test AUC | Saved `famo_consf1_m5_s42` mean val / test |
|---|---|---|---|---|
| 1 | 0 | 70.8 / 69.6 | 72.7 / 72.2 | 69.7 / 68.8 |
| 2 (kept) | 0 | 71.8 / 70.9 | 73.2 / 72.7 | 70.6 / 69.9 (kept) |
| 3 | 0 | 69.4 / 68.6 | 72.3 / 71.8 | 68.1 / 67.4 |
| 4 | 7 | 69.0 / 67.7 | 71.7 / 70.7 | 68.4 / 67.6 |
| 5 | 10 | 69.1 / 67.6 | 71.8 / 70.8 | 70.3 / 69.1 |

| Task | Paper (Table 3 F1) | FAMO, one stop (val / test) | FAMO, own best epoch (val / test) | Self-distill run, kept epoch 2 (val / test) | Single-task (val / test) |
|---|---|---|---|---|---|
| Propaganda | 61.74 | 64.8 / 66.7 | 64.8 / 66.7 | 64.0 / 66.2 | 66.2 / 68.0 |
| Disagree/Agree | 71.38 | 66.0 / 64.7 | 70.3 / 69.6 | 70.5 / 69.7 | 69.2 / 69.1 |
| Emotion/Fact | 63.93 | 73.2 / 69.7 | 73.6 / 69.9 | 75.5 / 71.9 | 75.3 / 71.7 |
| Attacking/Respectful | 68.07 | 75.3 / 74.4 | 75.3 / 74.4 | 75.3 / 73.4 | 75.0 / 73.3 |
| Nasty/Nice | 73.69 | 79.2 / 78.4 | 79.2 / 78.4 | 78.4 / 77.1 | 78.4 / 77.1 |
| Personal/Audience | 65.69 | 68.3 / 67.5 | 74.4 / 70.5 | 74.5 / 72.5 | 72.2 / 69.9 |
| Defeater/Undercutter | 55.65 | 61.9 / 60.4 | 62.7 / 61.3 | 62.1 / 60.6 | 62.4 / 59.4 |
| Negotiate/Attack | 64.81 | 70.6 / 71.7 | 71.0 / 71.3 | 71.7 / 72.2 | 70.4 / 69.8 |
| Questioning/Asserting | 63.23 | 67.3 / 65.8 | 67.3 / 65.8 | 62.9 / 62.4 | 66.6 / 66.2 |
| Argument Quality | 79.17 | 79.4 / 79.3 | 79.9 / 80.4 | 83.1 / 82.7 | 82.4 / 82.0 |
| **Mean of 10** | **66.74** | **70.6 / 69.9** | **71.9 / 70.8** | **71.8 / 70.9** | **71.8 / 70.7** |

**What this shows:** the kept 71.8 / 70.9 looks like a gain, but it is not from distillation. The kept model is `famo_consf1_m5` retrained from scratch and stopped at epoch 2, exactly like the saved run (70.6 / 69.9 at epoch 2). Same settings, same seed, same epoch: the ~1-point gap is run-to-run randomness from non-deterministic GPU training. Scores had already dropped at epoch 3, before anything switched, and stayed flat after the switches.

---

## Conclusions

1. **Scoring each task at its own best epoch works** (+0.9 test F1 over one stopping point, 4 seeds, significant; ties single-task), but it needs 3–4 saved models.
2. **Stopping on one main task does not help.** Stopping on Disagree/Agree lowered both Disagree/Agree and the mean of 10 (seed 42).
3. **Self-distillation did not turn the per-task gain into one model:**
   - Plain loss: −0.8 test vs one stopping point; Personal/Audience collapsed because it was switched before its late peak.
   - FAMO, validation F1, patience 2: kept epoch 1, before any switch.
   - FAMO, validation loss, patience 3: kept epoch 2, before any switch.
   - In every run, mean scores stayed flat or fell after tasks switched. Copying a task's own earlier scores on the training rows did not stop the shared layers drifting, so validation scores still went down.
4. **Why:** FAMO runs peak on the mean at epoch 1–2. A task can only switch `patience` epochs after its own peak, so by then the best mean epoch has already passed. Per-task validation F1 also jumps by up to ±5 points between neighbouring epochs, so a short patience switches tasks on noise.
5. **Run-to-run noise is about 1 point.** The same seed and settings gave 69.9 and 70.9 test at the same epoch. Single-seed differences under ~1 point should not be read as effects.

**All self-distillation and main-task results are from one seed (42).** None has been repeated on more seeds.

---

## Code and how to reproduce

- Training script: `replication/test_no_prop_overlap.py`.
  - `--main-task <output>`: choose the epoch and stop on one output (`EpochScorer(main_output=...)`).
  - `--self-distill`, `--self-distill-patience N` (default 2), `--self-distill-signal {f1,loss}` (default f1). Classes `SelfDistillModel`, `SelfDistillFamoModel`, `SelfDistillCallback`. The scorer now also records each output's own validation loss.
- Tests: `replication/test_self_distill.py` (9 tests) and 2 new tests in `replication/test_epoch_scorer.py`. Run with `modal run modal_replicate.py::masked_lm_tests`.
- Launch and report (from `multitask-forked/`):
  - `modal run --detach modal_replicate.py::main_task_stop --task disagree_agree_bi --seeds 42`, then `main_task_report --task disagree_agree_bi --seeds 42`
  - `modal run --detach modal_replicate.py::self_distill_runs --seeds 42`, then `self_distill_report --seeds 42`
  - `modal run --detach modal_replicate.py::self_distill_runs --seeds 42 --famo`, then `self_distill_report --seeds 42 --famo`
  - `modal run --detach modal_replicate.py::self_distill_runs --seeds 42 --famo --patience 3 --signal loss`, then `self_distill_report --seeds 42 --famo --patience 3 --signal loss`
- Results on the Modal volume `multitask-outputs`: `paper_replication_self_distill/` and `paper_replication_retune_famo/famo_consf1_m5_main_disagree_agree_bi_s42.json`.
- The per-task replay (Section 3) was a one-off script and is not in the repo. It reads `epoch_scores` and `best_epoch` from the saved `famo_consf1_m5_s<seed>.json` and single-task `only_<output>_s<seed>.json` files.
