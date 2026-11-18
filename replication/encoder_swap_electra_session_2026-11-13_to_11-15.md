# Encoder swap: ELECTRA-small in place of the small BERT (13–15 Nov 2026)

This note records the encoder-swap work on the main argument data: the multi-task model with the 10 outputs that the paper reports in Table 3. It covers:

1. ELECTRA-small (12 layers) against the small BERT, both fine-tuned, both at 256 tokens.
2. ELECTRA-small cut down to its first 2 or 4 layers, to see whether the gain survives a smaller encoder.
3. The frozen stage 2 (re-train each task's own block on a frozen model) on the 12-layer ELECTRA.

**Short version:** the full 12-layer ELECTRA-small beats the small BERT: with the plain loss, 69.3 ± 0.9 against 67.5 ± 0.6 test F1 over 4 seeds (+1.9 [-0.3, +4.1], not significant) and +3.3 [+0.7, +5.8] test AUC (significant). With FAMO on seed 42 it reaches 71.6 test F1 against 69.9 for BERT. Cut to 4 layers with FAMO it falls to 67.9, below BERT, so the gain comes from the full depth. Frozen stage 2 on the 12-layer model lifts seed 42 from 71.1 to 72.0 test F1, the highest in the project so far. Everything with FAMO on ELECTRA is one seed only.

---

## Timeline

All times are US Central (CST). "Finished" times are the result-file times on the Modal volume; launch times come from Modal's app and container start times and the session log. A "~" means the time is estimated from the events around it.

| Date and time | What happened |
|---|---|
| 13 Nov, ~08:45 | `--encoder electra-small` added to the training script; tests written. |
| 13 Nov, ~09:05 | Tests run on Modal: 20 of 20 pass. |
| 13 Nov, ~09:11 | 18 runs launched: ELECTRA and small BERT × plain and FAMO × seeds 42, 1, 2, 3 (ELECTRA on H100, BERT on L4), plus 2 learning-rate check runs for ELECTRA. |
| 13 Nov, 09:44 | First result: small BERT, plain, seed 42. |
| 13 Nov, ~09:48 | Job cut back for budget: ELECTRA FAMO seeds 1, 2, 3 and both learning-rate check runs stopped. They left no results. |
| 13 Nov, 09:51 | ELECTRA FAMO seed 42 finished (it had stopped early after 4 epochs). |
| 13 Nov, 10:00–10:17 | ELECTRA plain finished: seed 2 at 10:00, seed 3 at 10:04, seed 42 at 10:15, seed 1 at 10:17. |
| 13 Nov, 10:19–10:50 | Small BERT plain finished: seed 1 at 10:19, seed 2 at 10:31, seed 3 at 10:50. |
| 13 Nov, 10:58–11:59 | Small BERT FAMO finished: seed 42 at 10:58, seed 1 at 11:11, seed 2 at 11:36, seed 3 at 11:59. |
| 13 Nov, 13:26 | Option to keep only the first 2, 4 or 6 ELECTRA layers added. |
| 13 Nov, ~13:40 | Tests run on Modal: 21 of 21 pass. |
| 13 Nov, 13:49 | 2-layer and 4-layer ELECTRA launched: plain, seed 42. |
| 13 Nov, 14:04 | 2-layer ELECTRA plain finished. |
| 13 Nov, 14:08 | 4-layer ELECTRA plain finished. |
| 15 Nov, ~15:35 | Report command hung: macOS had moved three project files to iCloud. Files downloaded again; report works. |
| 15 Nov, ~16:31 | 4-layer ELECTRA with FAMO launched, seed 42. |
| 15 Nov, 16:53 | 4-layer ELECTRA FAMO finished. |
| 15 Nov, ~19:55 | Stage 2 launcher extended to other encoders; new ELECTRA stage 2 test passes on Modal. |
| 15 Nov, 20:03 | 12-layer ELECTRA, FAMO, seed 42, with frozen stage 2 launched. |
| 15 Nov, 20:54 | Its stage 1 finished (5 epochs); stage 1 weights saved. |
| 15 Nov, 21:00 | Stage 2 finished and saved. Container shut down at 21:01. |
| 15 Nov, 22:10 | This note written. |

---

## Terms used here

- **Output / task:** one of the 10 things the model predicts, e.g. Disagree/Agree. Propaganda is the max-pool over the 18 technique outputs.
- **F1:** weighted F1 × 100, with the decision threshold tuned on the validation split. "Mean of 10" is the average over the 10 outputs.
- **AUC:** area under the ROC curve × 100, mean over the 10 outputs. It does not depend on a threshold.
- **Val / test:** validation picks the epoch and the thresholds. Test is never used to make any choice; it is only reported.
- **Paper (Table 3 F1):** the paper's multi-task F1 (mean 66.74). The paper does not say whether that is validation or test; we compare it with our test and show our validation too. The paper reports no AUC.
- **Encoder:** the pretrained network that turns a text into a vector. The shared layers and task blocks sit on top of it.
- **Small BERT:** `google/bert_uncased_L-2_H-128_A-2`, 2 layers, 4.4M weights. The encoder in all earlier runs.
- **ELECTRA-small:** `google/electra-small-discriminator`, 12 layers, 13.5M weights. Same vocabulary as BERT, so the input tokens are identical. Pretrained by spotting replaced words instead of guessing hidden words.
- **First 2 / 4 layers:** ELECTRA-small with only its first 2 or 4 layers kept (pretrained weights intact), the rest dropped: 5.6M / 7.2M weights. There is no official ELECTRA checkpoint smaller than ELECTRA-small.
- **Fine-tuned:** the encoder's weights are updated during training. Every run here is fine-tuned (stage 2 then freezes them).
- **256 tokens:** texts are cut at 256 tokens here, not 512, because the 12-layer ELECTRA at 512 tokens × batch 256 does not fit in an 80 GB GPU. About 2% of training rows are longer than 256 tokens (7% of the IAC rows, none of the others). The small BERT was re-run at 256 tokens so the comparison is like for like.
- **Plain:** fixed task weights, losses simply added up.
- **FAMO / `famo_consf1_m5`:** FAMO balances the losses of the outputs, plus the filtered consistency term on a copy of each text with 5% of tokens masked. The masked-word loss has weight 0 (ELECTRA's checkpoint has no masked-word head). "FAMO" in this note always means this setup.
- **Stopping rule:** stop when the mean validation F1 has not improved for 3 epochs, keep the best epoch; at most 10 epochs. "Epoch picked / run" shows the kept epoch and how many epochs ran.
- **Settings:** learning rate 1e-4 (tuned for the small BERT, not re-tuned for ELECTRA), batch 256, weight decay 0.01, warm-up 5%, dropout 0, the "wide" layout.
- **Frozen stage 2:** take the trained model (stage 1), freeze the encoder, the shared layers and the task-family blocks, and re-train only each task's own block on its real labels (no FAMO, no consistency term). Each output keeps its own best epoch; epoch 0 means it kept its stage 1 block.
- **Single-task:** 10 separate small-BERT models at 512 tokens, one per output (earlier study; mean test F1 70.8 over 4 seeds).
- **Paired difference:** the difference computed seed by seed, with a 95% t interval in brackets. An interval that includes 0 is not significant.

---

## 1. Summary of all runs

Mean of the 10 outputs, × 100, mean ± standard deviation over seeds. The two "512 tokens" rows are earlier runs, shown for reference.

| Encoder | Layers | Weights | Tokens | Method | Seeds | Epoch picked / run | Val F1 | Test F1 | Test AUC | Paper (Table 3 F1) |
|---|---|---|---|---|---|---|---|---|---|---|
| Small BERT (earlier runs) | 2 | 4.4M | 512 | plain | 42, 1, 2, 3 | 5.5 / 8.2 | 69.3 ± 0.8 | 68.1 ± 1.0 | 69.1 ± 0.6 | 66.74 |
| Small BERT | 2 | 4.4M | 256 | plain | 42, 1, 2, 3 | 5.8 / 8.2 | 68.6 ± 0.4 | 67.5 ± 0.6 | 69.1 ± 0.7 | 66.74 |
| ELECTRA-small, first 2 layers | 2 | 5.6M | 256 | plain | 42 | 7.0 / 10.0 | 68.8 | 67.5 | 69.3 | 66.74 |
| ELECTRA-small, first 4 layers | 4 | 7.2M | 256 | plain | 42 | 6.0 / 9.0 | 69.1 | 68.2 | 70.8 | 66.74 |
| ELECTRA-small, full | 12 | 13.5M | 256 | plain | 42, 1, 2, 3 | 4.0 / 8.8 | 70.2 ± 0.9 | 69.3 ± 0.9 | 72.4 ± 0.9 | 66.74 |
| Small BERT (earlier runs) | 2 | 4.4M | 512 | FAMO (`famo_consf1_m5`) | 42, 1, 2, 3 | 1.8 / 4.8 | 70.9 ± 0.6 | 69.9 ± 0.7 | 72.6 ± 0.2 | 66.74 |
| Small BERT | 2 | 4.4M | 256 | FAMO (`famo_consf1_m5`) | 42, 1, 2, 3 | 3.2 / 6.2 | 69.8 ± 0.4 | 68.9 ± 0.7 | 71.3 ± 1.1 | 66.74 |
| ELECTRA-small, first 4 layers | 4 | 7.2M | 256 | FAMO (`famo_consf1_m5`) | 42 | 2.0 / 5.0 | 69.2 | 67.9 | 73.4 | 66.74 |
| ELECTRA-small, full | 12 | 13.5M | 256 | FAMO (`famo_consf1_m5`) | 42 | 1.0 / 4.0 | 72.1 | 71.6 | 75.6 | 66.74 |

## 2. Every run, with the time it finished

| Finished (CST) | Run name | Encoder | Method | Seed | Epoch picked / run | Val F1 | Test F1 | Test AUC | Paper (Table 3 F1) |
|---|---|---|---|---|---|---|---|---|---|
| 13 Nov 09:44 | `bert_plain_s42` | Small BERT | plain | 42 | 3 / 6 | 69.2 | 68.2 | 70.2 | 66.74 |
| 13 Nov 09:51 | `electra_famo_consf1_m5_s42` | ELECTRA-small, full | FAMO (`famo_consf1_m5`) | 42 | 1 / 4 | 72.1 | 71.6 | 75.6 | 66.74 |
| 13 Nov 10:00 | `electra_plain_s2` | ELECTRA-small, full | plain | 2 | 4 / 7 | 70.2 | 69.4 | 72.7 | 66.74 |
| 13 Nov 10:04 | `electra_plain_s3` | ELECTRA-small, full | plain | 3 | 3 / 8 | 71.0 | 69.8 | 73.2 | 66.74 |
| 13 Nov 10:15 | `electra_plain_s42` | ELECTRA-small, full | plain | 42 | 2 / 10 | 69.0 | 68.0 | 71.2 | 66.74 |
| 13 Nov 10:17 | `electra_plain_s1` | ELECTRA-small, full | plain | 1 | 7 / 10 | 70.8 | 70.0 | 72.5 | 66.74 |
| 13 Nov 10:19 | `bert_plain_s1` | Small BERT | plain | 1 | 6 / 9 | 68.6 | 67.7 | 69.0 | 66.74 |
| 13 Nov 10:31 | `bert_plain_s2` | Small BERT | plain | 2 | 9 / 10 | 68.4 | 67.2 | 69.1 | 66.74 |
| 13 Nov 10:50 | `bert_plain_s3` | Small BERT | plain | 3 | 5 / 8 | 68.3 | 66.8 | 68.4 | 66.74 |
| 13 Nov 10:58 | `bert_famo_consf1_m5_s42` | Small BERT | FAMO (`famo_consf1_m5`) | 42 | 2 / 5 | 70.2 | 69.9 | 72.6 | 66.74 |
| 13 Nov 11:11 | `bert_famo_consf1_m5_s1` | Small BERT | FAMO (`famo_consf1_m5`) | 1 | 2 / 5 | 69.4 | 68.4 | 71.6 | 66.74 |
| 13 Nov 11:36 | `bert_famo_consf1_m5_s2` | Small BERT | FAMO (`famo_consf1_m5`) | 2 | 3 / 6 | 69.6 | 68.4 | 71.3 | 66.74 |
| 13 Nov 11:59 | `bert_famo_consf1_m5_s3` | Small BERT | FAMO (`famo_consf1_m5`) | 3 | 6 / 9 | 70.2 | 68.9 | 69.8 | 66.74 |
| 13 Nov 14:04 | `electra2_plain_s42` | ELECTRA-small, first 2 layers | plain | 42 | 7 / 10 | 68.8 | 67.5 | 69.3 | 66.74 |
| 13 Nov 14:08 | `electra4_plain_s42` | ELECTRA-small, first 4 layers | plain | 42 | 6 / 9 | 69.1 | 68.2 | 70.8 | 66.74 |
| 15 Nov 16:53 | `electra4_famo_consf1_m5_s42` | ELECTRA-small, first 4 layers | FAMO (`famo_consf1_m5`) | 42 | 2 / 5 | 69.2 | 67.9 | 73.4 | 66.74 |
| 15 Nov 21:00 | `electra_famo_consf1_m5_stage2_s42` (stage 1) | ELECTRA-small, full | FAMO (`famo_consf1_m5`) | 42 | 1 / 5 | 72.2 | 71.1 | – | 66.74 |
| 15 Nov 21:00 | `electra_famo_consf1_m5_stage2_s42` (stage 2) | ELECTRA-small, full | FAMO, then frozen stage 2 | 42 | per output | 73.2 | 72.0 | – | 66.74 |

The stage 2 report does not print AUC, so those two cells are empty.

## 3. Full ELECTRA-small against the small BERT (256 tokens)

Differences on test, mean of 10 outputs, × 100, paired by seed.

| Comparison | Seeds | Δ test F1 | Δ test AUC | Δ val F1 |
|---|---|---|---|---|
| ELECTRA − BERT, plain | 4 | +1.9 [-0.3, +4.1] | +3.3 [+0.7, +5.8] | +1.6 [-0.4, +3.6] |
| ELECTRA − BERT, FAMO | 1 | +1.7 (1 seed) | +3.0 (1 seed) | +2.0 (1 seed) |
| FAMO − plain, BERT | 4 | +1.4 [+0.4, +2.4] | +2.2 [+1.4, +3.0] | +1.2 [+0.5, +2.0] |
| FAMO − plain, ELECTRA | 1 | +3.5 (1 seed) | +4.4 (1 seed) | +3.1 (1 seed) |

- **Plain, 4 seeds:** ELECTRA is ahead on F1 but the interval includes 0. The AUC gain is significant.
- **FAMO, seed 42 only:** the same direction and a similar size, but one seed cannot establish it.
- **256 against 512 tokens for BERT:** cutting to 256 tokens costs the small BERT −0.6 [−2.2, +0.9] test F1 with plain and −1.1 [−2.4, +0.2] with FAMO (4 seeds each); neither is significant.
- **Against single-task BERT (70.8):** ELECTRA plain is below it; the one ELECTRA FAMO run (71.6) is above it.

### Per output, plain, 4 seeds (test F1)

| Output | Paper (Table 3 F1) | Small BERT | Full ELECTRA-small | ELECTRA − BERT |
|---|---|---|---|---|
| Propaganda | 61.74 | 64.0 ± 0.9 | 67.1 ± 0.3 | +3.0 [+1.3, +4.7] |
| Disagree/Agree | 71.38 | 61.9 ± 3.9 | 65.2 ± 1.5 | +3.3 [-2.5, +9.1] |
| Emotion/Fact | 63.93 | 70.2 ± 1.8 | 69.6 ± 0.5 | -0.6 [-3.2, +2.1] |
| Attacking/Respectful | 68.07 | 69.9 ± 0.3 | 72.2 ± 0.4 | +2.3 [+1.3, +3.3] |
| Nasty/Nice | 73.69 | 73.4 ± 0.9 | 76.4 ± 1.2 | +3.0 [+2.0, +4.1] |
| Personal/Audience | 65.69 | 64.5 ± 2.7 | 64.5 ± 1.8 | +0.0 [-4.5, +4.6] |
| Defeater/Undercutter | 55.65 | 57.9 ± 1.5 | 57.5 ± 3.3 | -0.4 [-7.6, +6.8] |
| Negotiate/Attack | 64.81 | 69.7 ± 0.6 | 70.6 ± 1.7 | +0.9 [-2.2, +4.1] |
| Questioning/Asserting | 63.23 | 59.9 ± 3.0 | 66.1 ± 4.4 | +6.2 [-5.0, +17.3] |
| Argument Quality | 79.17 | 83.3 ± 1.5 | 84.2 ± 1.7 | +0.9 [-3.9, +5.6] |
| Mean of 10 | 66.74 | 67.5 ± 0.6 | 69.3 ± 0.9 | +1.9 [-0.3, +4.1] |

The clear gains are Propaganda, Attacking/Respectful and Nasty/Nice. Questioning/Asserting has the largest mean gain but swings a lot between seeds.

## 4. Does a smaller ELECTRA keep the gain? (seed 42 only)

| Encoder | Layers | Weights | Method | Epoch picked / run | Val F1 | Test F1 | Test AUC | Paper (Table 3 F1) |
|---|---|---|---|---|---|---|---|---|
| Small BERT | 2 | 4.4M | plain | 3 / 6 | 69.2 | 68.2 | 70.2 | 66.74 |
| ELECTRA-small, first 2 layers | 2 | 5.6M | plain | 7 / 10 | 68.8 | 67.5 | 69.3 | 66.74 |
| ELECTRA-small, first 4 layers | 4 | 7.2M | plain | 6 / 9 | 69.1 | 68.2 | 70.8 | 66.74 |
| ELECTRA-small, full | 12 | 13.5M | plain | 2 / 10 | 69.0 | 68.0 | 71.2 | 66.74 |
| Small BERT | 2 | 4.4M | FAMO (`famo_consf1_m5`) | 2 / 5 | 70.2 | 69.9 | 72.6 | 66.74 |
| ELECTRA-small, first 4 layers | 4 | 7.2M | FAMO (`famo_consf1_m5`) | 2 / 5 | 69.2 | 67.9 | 73.4 | 66.74 |
| ELECTRA-small, full | 12 | 13.5M | FAMO (`famo_consf1_m5`) | 1 / 4 | 72.1 | 71.6 | 75.6 | 66.74 |

- **Plain:** all four encoders are within 0.7 F1 of each other on this seed, inside the usual seed-to-seed spread (about 0.9). Seed 42 is also a seed where the full ELECTRA shows no F1 gain with plain, so these two runs cannot answer the question. AUC rises with depth among the ELECTRA versions.
- **FAMO:** here the full model does show a gain on seed 42, and the 4-layer version loses it: 3.7 F1 below the full model and 2.0 below the small BERT. It keeps a small part of the AUC gain (+0.8 over BERT).
- **Reading:** ELECTRA's advantage comes with the full 12-layer depth. One caveat: these layers were pretrained as the bottom of a 12-layer model, so cutting them is harsher than a model built small from the start.
- The 2-layer version was not run with FAMO.

### Per output, seed 42 (test F1)

| Output | Paper (Table 3 F1) | Small BERT, plain | ELECTRA-small, first 2 layers, plain | ELECTRA-small, first 4 layers, plain | ELECTRA-small, full, plain | Small BERT, FAMO | ELECTRA-small, first 4 layers, FAMO | ELECTRA-small, full, FAMO |
|---|---|---|---|---|---|---|---|---|
| Propaganda | 61.74 | 65.2 | 62.7 | 65.4 | 67.1 | 66.0 | 68.2 | 69.0 |
| Disagree/Agree | 71.38 | 63.6 | 69.3 | 68.4 | 65.6 | 67.1 | 61.4 | 70.2 |
| Emotion/Fact | 63.93 | 68.9 | 69.4 | 71.6 | 69.7 | 71.8 | 69.6 | 71.1 |
| Attacking/Respectful | 68.07 | 70.2 | 69.9 | 69.6 | 71.7 | 73.1 | 73.4 | 75.6 |
| Nasty/Nice | 73.69 | 72.2 | 75.0 | 75.6 | 75.2 | 78.5 | 77.2 | 79.9 |
| Personal/Audience | 65.69 | 66.0 | 56.7 | 70.1 | 67.1 | 66.0 | 61.6 | 64.7 |
| Defeater/Undercutter | 55.65 | 58.9 | 60.1 | 50.3 | 53.2 | 59.5 | 53.2 | 61.2 |
| Negotiate/Attack | 64.81 | 70.0 | 67.6 | 70.8 | 68.4 | 72.1 | 72.3 | 73.0 |
| Questioning/Asserting | 63.23 | 62.4 | 64.3 | 57.3 | 60.0 | 61.4 | 60.6 | 69.1 |
| Argument Quality | 79.17 | 84.4 | 79.6 | 82.9 | 82.4 | 83.0 | 81.5 | 81.8 |
| Mean of 10 | 66.74 | 68.2 | 67.5 | 68.2 | 68.0 | 69.9 | 67.9 | 71.6 |

### Per output, seed 42 (test AUC; the paper reports none)

| Output | Small BERT, plain | ELECTRA-small, first 2 layers, plain | ELECTRA-small, first 4 layers, plain | ELECTRA-small, full, plain | Small BERT, FAMO | ELECTRA-small, first 4 layers, FAMO | ELECTRA-small, full, FAMO |
|---|---|---|---|---|---|---|---|
| Propaganda | 66.3 | 62.6 | 66.2 | 70.8 | 68.9 | 69.6 | 72.7 |
| Disagree/Agree | 65.3 | 63.9 | 67.2 | 62.8 | 66.4 | 69.2 | 69.7 |
| Emotion/Fact | 78.6 | 76.9 | 78.1 | 76.8 | 80.0 | 80.9 | 81.1 |
| Attacking/Respectful | 74.6 | 73.2 | 74.0 | 75.7 | 76.8 | 77.1 | 80.1 |
| Nasty/Nice | 76.7 | 75.0 | 75.9 | 78.0 | 78.7 | 78.3 | 81.9 |
| Personal/Audience | 62.8 | 66.6 | 67.6 | 64.6 | 65.5 | 66.4 | 68.8 |
| Defeater/Undercutter | 59.8 | 60.7 | 59.0 | 57.4 | 61.9 | 61.5 | 62.0 |
| Negotiate/Attack | 76.8 | 76.3 | 78.4 | 75.9 | 79.3 | 79.3 | 80.5 |
| Questioning/Asserting | 59.9 | 61.4 | 62.2 | 65.9 | 64.6 | 67.8 | 73.2 |
| Argument Quality | 80.8 | 76.6 | 79.9 | 84.1 | 84.2 | 84.4 | 85.9 |
| Mean of 10 | 70.2 | 69.3 | 70.8 | 71.2 | 72.6 | 73.4 | 75.6 |

## 5. Frozen stage 2 on the 12-layer ELECTRA (seed 42)

Stage 1 was trained again in the same job (its weights had not been saved before). It kept epoch 1 of 5. Stage 2 ran 10 epochs at learning rate 1e-4 (the rate that won on the small BERT). The check that the frozen weights are unchanged passed.

| Model | Tokens | Val F1 | Test F1 | Paper (Table 3 F1) |
|---|---|---|---|---|
| Full ELECTRA-small, FAMO, stage 1 | 256 | 72.2 | 71.1 | 66.74 |
| Full ELECTRA-small, FAMO, stage 2 | 256 | 73.2 | 72.0 | 66.74 |
| Small BERT, FAMO, stage 1 (earlier run) | 512 | 70.5 | 70.0 | 66.74 |
| Small BERT, FAMO, stage 2 (earlier run) | 512 | 71.7 | 71.1 | 66.74 |
| Ten single-task small BERT models (earlier runs) | 512 | 71.8 | 70.7 | 66.74 |

- **Stage 2 gain on ELECTRA:** +0.9 test F1 and +1.0 validation F1, close to the small BERT's gain on this seed (+1.1 test). Over 4 seeds the small BERT's stage 2 gain was +1.3 [+0.1, +2.4].
- **Against BERT on this seed:** +0.9 over BERT stage 2 and +1.3 over the ten single-task BERT models. This is not like for like: different encoder size and 256 against 512 tokens, and there is no single-task ELECTRA baseline.
- **Run-to-run variation:** this stage 1 scored 72.2 / 71.1 (val / test); the 13 Nov run with the same seed and settings scored 72.1 / 71.6. Training on the GPU is not exactly repeatable.

### Per output (ELECTRA, seed 42)

| Output | Paper (Table 3 F1) | Stage 1 val F1 | Stage 1 test F1 | Stage 2 val F1 | Stage 2 test F1 | Change in test F1 | Stage 2 epoch kept |
|---|---|---|---|---|---|---|---|
| Propaganda | 61.74 | 67.1 | 69.0 | 67.1 | 69.0 | +0.0 | 0 |
| Disagree/Agree | 71.38 | 61.3 | 59.7 | 64.2 | 63.1 | +3.4 | 4 |
| Emotion/Fact | 63.93 | 75.4 | 72.2 | 75.4 | 72.2 | +0.0 | 0 |
| Attacking/Respectful | 68.07 | 75.6 | 74.4 | 76.2 | 75.1 | +0.6 | 6 |
| Nasty/Nice | 73.69 | 81.0 | 79.4 | 81.0 | 79.4 | +0.0 | 0 |
| Personal/Audience | 65.69 | 71.1 | 68.4 | 73.0 | 70.9 | +2.6 | 1 |
| Defeater/Undercutter | 55.65 | 62.8 | 60.8 | 62.8 | 60.8 | +0.0 | 0 |
| Negotiate/Attack | 64.81 | 72.4 | 71.6 | 73.6 | 72.0 | +0.4 | 9 |
| Questioning/Asserting | 63.23 | 71.6 | 71.6 | 72.6 | 72.2 | +0.6 | 3 |
| Argument Quality | 79.17 | 83.9 | 83.2 | 85.8 | 85.1 | +1.9 | 6 |
| Mean of 10 | 66.74 | 72.2 | 71.1 | 73.2 | 72.0 | +0.9 | |

- Four outputs kept their stage 1 block (epoch 0): no stage 2 epoch beat stage 1 on validation.
- Disagree/Agree is the weak spot: 63.1 after stage 2, below the paper's 71.38. It is also unstable: 70.2 in the 13 Nov run with the same seed, 59.7 in this stage 1.

---

## Caveats

- **One seed for everything with FAMO on ELECTRA** (full model, 4-layer model, stage 2). The seed-to-seed spread is about 0.7–0.9 F1.
- **Size:** the full ELECTRA-small has 13.5M weights against 4.4M, and the 4-layer result suggests depth is what matters. The gain cannot be credited to ELECTRA's pretraining alone.
- **Learning rate:** 1e-4 was tuned for the small BERT. The two ELECTRA learning-rate check runs (3e-5 and 3e-4) were stopped before they produced results.
- **Text length:** all new runs cut texts at 256 tokens.
- **RoBERTa was not run:** the smallest official model (about 125M weights) is roughly 28 times the small BERT, would not fit at batch 256, and uses a different vocabulary.

## Not run

- Full ELECTRA-small with FAMO on seeds 1, 2, 3 (launched 13 Nov, stopped for budget, no results).
- Stage 2 on ELECTRA for seeds 1, 2, 3.
- 2-layer ELECTRA with FAMO; 6-layer ELECTRA (option exists, never run); 2- and 4-layer on other seeds.
- Single-task ELECTRA models.

## Cost (estimates, not from the bill)

H100 at about $3.95 per hour, L4 at about $0.80 per hour. Check the Modal dashboard for the real figures.

| What | Estimate |
|---|---|
| 13 Nov morning: 4 ELECTRA plain runs, 1 ELECTRA FAMO run, 8 small BERT runs | about $20–23 |
| 13 Nov morning: 3 ELECTRA FAMO runs and 2 learning-rate check runs, stopped without results | about $10 |
| 13 Nov: 2-layer and 4-layer ELECTRA, plain, seed 42 (about 15 and 19 minutes) | about $2.30 |
| 15 Nov: 4-layer ELECTRA, FAMO, seed 42 (about 22 minutes) | about $1.50 |
| 15 Nov: full ELECTRA, FAMO, stage 1 + stage 2, seed 42 (58 minutes) | about $3.80 |
| Total | about $38–41 |

## Code and files

- **Training script:** `replication/test_no_prop_overlap.py` — `--encoder electra-small`, `electra-small-L2`, `-L4`, `-L6`. `NoMLMHead` stands in for the masked-word head ELECTRA lacks (only when that loss has weight 0); encoders without a pooler read the [CLS] position.
- **Library:** `EncoderConfig.encoder_kwargs` in `multitask/config.py`, passed to the HuggingFace loader in `multitask/models/hierarchical.py` (used for `num_hidden_layers`).
- **Tests:** `replication/test_electra_encoder.py` (6 tests); run with `modal run modal_replicate.py::encoder_tests`.
- **Launchers and reports** in `modal_replicate.py`:
  - `encoder_swap --encoders electra,bert,electra2,electra4,electra6 --methods plain,famo_consf1_m5 --seeds ...` (skips finished runs)
  - `encoder_swap_report`
  - `stage2_runs --seeds 42 --lrs 1e-4 --encoder electra` and `stage2_report --seeds 42 --encoder electra`
- **Results on the Modal volume `multitask-outputs`:** `paper_replication_encoder/<run name>.json` (+ `_predictions.npz`) and `paper_replication_stage2/electra_famo_consf1_m5_stage2_s42.json` (+ predictions and `_stage1_weights.npz`, so more stage 2 variants on seed 42 need no new stage 1).
- **If `modal run` hangs or fails with `Errno 89`:** macOS has moved project files to iCloud. `find . -type f -flags +dataless` lists them; `brctl download <path>` brings them back.
