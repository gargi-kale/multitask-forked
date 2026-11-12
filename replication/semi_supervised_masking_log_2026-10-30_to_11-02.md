# Semi-supervised masking experiments: full log

**Period:** 2026-10-30 to 2026-11-02 (one Claude Code chat; record written 2026-11-04)
**Times:** US Central (CST), taken from Modal result-file timestamps where available. Analyses on 2026-10-30 have dates only.
**Setup for every run:** small BERT (`google/bert_uncased_L-2_H-128_A-2`), "wide" head layers, learning rate 1e-4, batch 256, dropout 0 in the head layers, stopping on validation F1 with patience 3, seeds 42, 1, 2, 3. Scores are the mean weighted F1 (or AUC) over the paper's 10 Table-3 outputs, ×100.

---

## 1. In short

| Question | Answer |
|---|---|
| Can the word-guessing loss reach the shared and task layers? | Yes, by running each word's vector through those layers (designed, **not run**) |
| Option 1: masked sentence must keep its real label | No gain over `famo_fixed` (+0.1 F1, not significant) |
| Option 2: masked answer must match full-sentence answer | No gain over `famo_fixed` (−0.1 F1, not significant) |
| Does hiding 15% of words destroy label information? | Yes for rudeness labels (Nasty −6.4 F1, Attacking −4.7); Propaganda only −1.9 |
| Option 2 alone, 5% hidden, with per-head confidence filter | **69.9 F1**, the highest so far. +1.0 vs FAMO alone, not significant; ties `famo_fixed` (+0.4) |
| Clearest finding | Option 2 without a filter makes every head's scores bunch together (20–30% less spread); the filter prevents this |
| Does word-guessing itself help (FAMO vs FAMO + word-guessing)? | +0.6 F1, +1.5 AUC, not significant |

**Term definitions used throughout:**
- **Cell:** one row × one label. Most cells are empty, because each corpus has only some labels.
- **Word-guessing (MLM):** hide 15% of words and have the model guess them. This was the earlier semi-supervised experiment.
- **`famo_fixed`:** FAMO (automatic task weighting) + word-guessing at weight 0.03. Previous best: 69.5 F1, 72.7 AUC.
- **FAMO alone (`famo`):** 69.0 F1, 71.2 AUC.
- **95% range `[a, b]`:** where the true effect most likely lies, from paired seed differences. If it includes 0, the effect can't be told apart from luck.
- **AUC:** the chance that a random true-yes row scores higher than a random true-no row. 0.5 = coin flip.

---

## 2. Timeline

| Date / time (CST) | What happened |
|---|---|
| 2026-10-29 14:39–16:48 | (Before this chat) FAMO alone and `famo_fixed` baselines finished, 4 seeds each. Later reused as comparisons. |
| 2026-10-30 | Chat starts. User asks: (1) send the word-guessing loss into the shared and task layers; (2) make the model learn about labels, not just words. |
| 2026-10-30 | Label-coverage analysis of the training data; pseudo-label quality check on validation. Plan proposed: Experiment 1 (deeper word-guessing) and Options 1–3. |
| 2026-11-02 | Plain-language re-explanations: Control / E1a / E1b, Options 1 and 2, how the consistency loss is calculated. User says go. |
| 2026-11-02 ~14:31 | Code built; library test suite run (207 pass, 8 old unrelated failures); replication tests 47/47 pass. |
| 2026-11-02 14:42–14:43 | Smoke run finished: 1 epoch of each option, no memory problems, 174–189 s per epoch. |
| 2026-11-02 14:54 | 16 runs launched at once (both options × weights 0.3 and 1 × 4 seeds). |
| 2026-11-02 15:07–15:35 | All 16 runs finished. Result: no gain. |
| 2026-11-02 | User pastes an outside critique. Assessed point by point. |
| 2026-11-02 | Masking check built (48/48 tests pass) and run. |
| 2026-11-02 16:03 | Masking check finished: IAC rudeness labels suffer most. |
| 2026-11-02 | Explained separating the two hiding rates and the per-head confidence filter. Checked a single 0.8 cut-off on real scores. Compared FAMO with and without word-guessing directly. |
| 2026-11-02 | Confidence filter built (50/50 tests pass). 12 runs launched: Option 2 alone, 5% hidden. |
| 2026-11-02 17:22–17:40 | All 12 runs finished. Filtered Option 2: 69.9 F1. |

---

## 3. Starting point: label coverage (2026-10-30)

Training set: 219,014 rows (IAC 61,069; IBM 96,036; Propaganda 61,909) × 10 outputs = 2,190,140 cells.

- **Only 538,360 cells (24.6%) have a real label.**
- 36,002 IAC rows have the first four discourse labels but not the second four. That's **144,008 empty cells from the same kind of text**, the safest place for pseudo-labels.
- Those rows differ from fully labelled IAC rows: Disagree is 68% "yes" there vs 84%.
- Propaganda rows carry rule-made IAC labels: 100% "yes" for Disagree, Emotion, Attacking and Nasty; 0% for Defeater.
- `WA_bi` uses an explicit −1 (missing) on IAC and Propaganda rows, so simple NaN counts are misleading.

### Pseudo-label quality check (validation, 4-seed `famo_fixed` ensemble)

For each second-set label: of the 10% of rows the model was most sure were "yes" (or "no"), how many really were?

| Label | Paper (Table 3 F1) | AUC | Confident "yes" correct (always-yes rate) | Confident "no" correct (always-no rate) | Verdict |
|---|---|---|---|---|---|
| Personal/Audience | 65.69 | 0.66 | 97% (78%) | 47% (22%) | trust "yes" only |
| Defeater/Undercutter | 55.65 | 0.57 | 78% (65%) | 44% (35%) | too weak |
| Negotiate/Attack | 64.81 | 0.74 | 90% (61%) | 73% (39%) | trust "yes" more |
| Questioning/Asserting | 63.23 | 0.68 | 58% (34%) | 93% (66%) | trust "no" only |

- **No head is reliable on both sides**, so hard pseudo-labels would be one-sided and skew the heads, like zero-fill did.
- Averaging 4 seeds adds only about 1 AUC point over one model.
- Conclusion: prefer consistency (no invented labels) and, if pseudo-labelling, soft labels on same-domain cells only.

---

## 4. Experiment 1: word-guessing in deeper layers (designed, not run)

**Why it can't simply be switched on:**
- The shared layers only see the sentence summary vector, while word-guessing needs one vector per word.
- The final output layer is one number, so it can't name one of 30,522 words.

**Design:** run each word's vector through the same shared (and task) layers, then a new guessing head tied to BERT's word table.

| Arm | Word-guessing happens after | Head | Purpose |
|---|---|---|---|
| Baseline (`famo_fixed`) | BERT | Google's pretrained head | already run |
| Control | BERT | new head | isolates the effect of replacing the head |
| E1a | shared layers | new head | the user's experiment 1 |
| E1b | one randomly chosen task's own layers each step | new head | the user's experiment 2 |

The guessing head in E1a/E1b has to be new: Google's pretrained head only understands BERT's word vectors. **Status:** not built. The chat moved to Options 1–2, which address the user's concern that word-guessing teaches language, not labels.

---

## 5. Label-aware options (as explained in the chat)

- **Option 1 (masked copy + real label):** the masked sentence must still get its real label. It applies only to labelled cells and blocks single-word shortcuts.
- **Option 2 (consistency):** on every cell, including empty ones, the masked copy's answer must match the model's **own** full-sentence answer. That answer is held fixed. Each head is compared only with itself, never with another head. It invents no labels.
- **Option 3 (teacher → student, soft labels):** fill the 144k same-domain empty cells with an ensemble's soft scores. **Not built.**

Both Option 1 and Option 2 reach every layer, including the final output layer, unlike word-guessing.

---

## 6. Code built (2026-11-02)

All in `multitask-forked/`:

| File | Change |
|---|---|
| `multitask/models/hierarchical.py` | New `MultiTaskModel.forward_from_encoding()`: runs the shared, task and output layers on an already-encoded batch. `call()` now uses it, with unchanged behaviour. |
| `replication/test_no_prop_overlap.py` | `mlm_loss_from_hidden()` split out of `mlm_loss()`. New `LabelAwareMaskedModel` / `LabelAwareMaskedTrainer` built on `famo_fixed`; the masked copy reuses the word-guessing BERT pass. New `confident_cells()` and `mask_sweep()`. New flags: `--masked-label-weight`, `--consistency-weight`, `--consistency-rampup-epochs` (default 1), `--consistency-keep`, `--mask-sweep`, `--mask-sweep-repeats`. |
| `replication/test_label_aware_masking.py` | New: 9 tests (weight 0 = `famo_fixed` exactly; Option 1 ignores empty rows; Option 2 target gets no gradient; zero loss when nothing is hidden; ramp; step = weighted sum of all terms; graph-mode fit; filter keeps both ends; filter skips marked rows). |
| `replication/test_epoch_scorer.py` | New test: masking check at rate 0 changes nothing. |
| `modal_replicate.py` | New `replicate` settings (`masked_label_weight`, `consistency_weight`, `mask_sweep`, `mlm_mask_prob`, `consistency_keep`). New entry points: `label_aware`, `label_aware_smoke`, `label_aware_report` (with `--baseline` / `--prefixes`), `mask_check`, `mask_check_report`, `consistency_filter`. New tests added to `masked_lm_tests`. |

**Details of Option 2's loss:** binary KL divergence (0 when the two answers match), averaged over the cells used. The weight rises from 0 to full over the first epoch.

**Tests:** replication suite 50/50 pass at the end. The library suite has 8 failures in `tests/test_optimizers.py`. They come from the Lion optimizer not being supported in `trainer.py` and the earlier `WarmUpWrapper` change in `compat.py`, and existed before this work.

**Commands:**
```
modal run modal_replicate.py::masked_lm_tests
modal run --detach modal_replicate.py::label_aware --stage 1 --seeds 42,1,2,3
modal run modal_replicate.py::label_aware_report
modal run modal_replicate.py::mask_check
modal run modal_replicate.py::mask_check_report
modal run --detach modal_replicate.py::consistency_filter
modal run modal_replicate.py::label_aware_report --baseline famo --prefixes famo_cons1_m5,famo_consf
```

---

## 7. Run 1: Options 1 and 2 on top of `famo_fixed` (finished 2026-11-02 15:35)

15% of words hidden, word-guessing on (weight 0.03). The weight for each option was picked on mean validation F1 over 4 seeds. Stage 2 was skipped by running both weights on all 4 seeds.

| Version | Test F1 | Δ F1 vs `famo_fixed` [95%] | Test AUC | Δ AUC | Paper (Table 3 F1) |
|---|---|---|---|---|---|
| `famo_fixed` | 69.5 | – | 72.7 | – | 66.74 |
| Option 1, weight 0.3 | 69.3 | −0.3 [−1.2, +0.7] | 72.0 | −0.7 | 66.74 |
| **Option 1, weight 1** (picked) | 69.7 | +0.1 [−0.9, +1.2] | 72.2 | −0.5 | 66.74 |
| **Option 2, weight 0.3** (picked) | 69.4 | −0.1 [−1.5, +1.3] | 72.4 | −0.3 | 66.74 |
| Option 2, weight 1 | 69.3 | −0.2 [−1.0, +0.6] | 71.9 | −0.8 | 66.74 |

- No gain; any real effect is probably under 1 F1.
- AUC slightly lower in all four versions (not significant).
- The best epoch moved later (1.5 → 2.5–3.8).
- No lazy heads: score spread went up.

Smoke run, for reference (1 epoch, seed 42): val F1 66.5 (Option 1) and 68.0 (Option 2) vs `famo_fixed`'s own epoch 1 of 70.3. Word-guessing accuracy matched `famo_fixed` (28.3–28.6%), which confirms the shared BERT pass works.

---

## 8. The outside critique and assessment

| Critique point | Assessment |
|---|---|
| No confidence gate, so the model learns to agree with its own mistakes | Right about the problem. A single global cut-off is a bad fix here, because heads are lopsided (see section 10). |
| Random masking can delete the word a label rests on | Right. About 13.5% of words actually change (15% picked, 90% of those altered). |
| One setting controls both hiding rates | Right. Separating them needs a second masked copy and a third BERT pass: roughly 30–50% slower per epoch (an earlier "10–15%" estimate was too low). |
| Target comes from a noisy pass | Right, and it applies: `--dropout 0` only covers the head layers, while **BERT's internal dropout (0.1) is on in training**. |
| 1-epoch ramp too short | Disagreed. Runs peak around epochs 1–6, so 1 epoch is already 12–20% of training. |
| Multi-task is −3.3 F1 below single-task | That number came from the old settings and hadn't been rerun at the time. |
| Suggested a cheap masking check | Good idea. Built and run (section 9). There was no saved model to load, so it runs inside a retraining run. |

---

## 9. Masking check (finished 2026-11-02 16:03)

`famo_fixed`, seed 42, retrained (best epoch 1). Validation set, BERT's masking rule, dropout off, 3 random maskings per rate. Compared with the model's own full-sentence answers.

**F1 change vs full sentences (×100):**

| Label | Paper (Table 3 F1) | Full-sentence val F1 | 5% | 10% | 15% | 30% |
|---|---|---|---|---|---|---|
| Nasty/Nice | 73.69 | 77.0 | −1.9 | −4.3 | −6.4 | −16.3 |
| Attacking/Respectful | 68.07 | 75.2 | −1.3 | −2.9 | −4.7 | −10.9 |
| Questioning/Asserting | 63.23 | 65.4 | −1.0 | −2.1 | −3.3 | −6.6 |
| Negotiate/Attack | 64.81 | 70.5 | −1.0 | −2.2 | −2.6 | −8.9 |
| Propaganda | 61.74 | 65.1 | −0.9 | −0.6 | −1.9 | −4.9 |
| Argument Quality | 79.17 | 84.1 | −0.3 | −0.8 | −1.9 | −5.2 |
| Emotion/Fact | 63.93 | 75.1 | −0.2 | −1.1 | −1.4 | −7.1 |
| Defeater/Undercutter | 55.65 | 60.7 | −0.6 | −0.9 | −0.9 | −3.4 |
| Personal/Audience | 65.69 | 70.2 | +0.5 | +0.5 | +1.3 | +1.3 |
| Disagree/Agree | 71.38 | 62.4 | +0.2 | +1.0 | +1.4 | +4.2 |

**Share of yes/no answers that flip at 15% hidden:**

| Label | Paper (Table 3 F1) | Labelled rows | Unlabelled rows |
|---|---|---|---|
| Propaganda | 61.74 | 13.8% | 23.3% |
| Nasty/Nice | 73.69 | 12.5% | 19.9% |
| Attacking/Respectful | 68.07 | 12.6% | 18.4% |
| Emotion/Fact | 63.93 | 11.6% | 18.9% |
| Disagree/Agree | 71.38 | 11.2% | 11.1% |
| Defeater/Undercutter | 55.65 | 10.9% | 8.9% |
| Negotiate/Attack | 64.81 | 10.4% | 12.1% |
| Questioning/Asserting | 63.23 | 8.8% | 9.9% |
| Argument Quality | 79.17 | 8.8% | 6.4% |
| Personal/Audience | 65.69 | 8.0% | 8.6% |

- **Opposite of the critique's guess:** the IAC rudeness labels lose most, not Propaganda.
- About 1 answer in 8 flips at 15%. The model is much less stable than Option 2's tiny loss (0.006) suggested, because many scores sit right at the cut-off.
- Disagree/Agree and Personal/Audience go **up** when words are hidden. Unexplained.
- 5% hiding is gentle; 30% is destructive.
- Not measured: whether models trained with Options 1/2 flip less.

---

## 10. Why one global confidence cut-off fails (checked on `famo_fixed` seed 42 validation scores)

| Head | Paper (Table 3 F1) | Its yes/no cut-off | Cells above 0.8 | Cells below 0.2 | Kept, yes : no |
|---|---|---|---|---|---|
| Propaganda | 61.74 | 0.54 | 4.5% | 0.2% | 97 : 3 |
| Disagree/Agree | 71.38 | 0.48 | 15.5% | 0.0% | 100 : 0 |
| Emotion/Fact | 63.93 | 0.51 | 24.1% | 4.6% | 84 : 16 |
| Attacking/Respectful | 68.07 | 0.46 | 20.5% | 2.9% | 87 : 13 |
| Nasty/Nice | 73.69 | 0.48 | 21.7% | 3.8% | 85 : 15 |
| Personal/Audience | 65.69 | 0.42 | 0.0% | 7.0% | 0 : 100 |
| Defeater/Undercutter | 55.65 | 0.51 | 0.0% | 12.7% | 0 : 100 |
| Negotiate/Attack | 64.81 | 0.59 | 8.3% | 23.8% | 26 : 74 |
| Questioning/Asserting | 63.23 | 0.51 | 0.0% | 0.4% | 0 : 100 |
| Argument Quality | 79.17 | 0.31 | 10.0% | 25.7% | 28 : 72 |

- Every head is lopsided, in both directions, not only toward "no". Questioning/Asserting would be switched off almost entirely.
- **About 90% of the confident-yes cells for Disagree, Emotion, Attacking and Nasty are Propaganda rows** (e.g. 1,103 of 1,242 for Nasty). Those are the rule-made labels, so an unguarded filter would reinforce that shortcut.

---

## 11. Does word-guessing help? (direct comparison, 2026-11-02)

| Setup | Test F1 | Test AUC | Paper (Table 3 F1) |
|---|---|---|---|
| Plain (no FAMO, no word-guessing) | 68.1 | 69.1 | 66.74 |
| FAMO alone | 69.0 | 71.2 | 66.74 |
| FAMO + word-guessing (`famo_fixed`) | 69.5 | 72.7 | 66.74 |
| Effect of word-guessing (paired) | +0.6 [−1.3, +2.4] | +1.5 [−1.6, +4.6] | |

F1 rose on 3 of 4 seeds and AUC on all 4, but neither is significant. Word-guessing isn't needed for Option 2.

---

## 12. Run 2: Option 2 alone with the confidence filter (finished 2026-11-02 17:40)

**How the filter works:** for each head and batch, keep the cells whose full-sentence score is in the **lowest 10% or highest 10%**; drop the unsure middle. Ranking keeps both sides equal in size. Propaganda rows are skipped for the 8 IAC heads. Targets stay soft (no rounding to 0/1).

**Settings:** word-guessing off (`--mlm-lambda 0`), 5% of words hidden (`--mlm-mask-prob 0.05`). Compared with FAMO alone.

| Version | Test F1 | Δ F1 vs FAMO [95%] | Test AUC | Δ AUC | Paper (Table 3 F1) |
|---|---|---|---|---|---|
| FAMO alone | 69.0 | – | 71.2 | – | 66.74 |
| Option 2, no filter, weight 1 (control) | 69.2 | +0.3 [−1.3, +1.8] | 71.9 | +0.7 | 66.74 |
| **Option 2 + filter, weight 1** (picked on val, 70.9 vs 70.3) | **69.9** | +1.0 [−0.9, +2.8] | 72.6 | +1.4 | 66.74 |
| Option 2 + filter, weight 3 | 69.4 | +0.4 [−1.7, +2.5] | 71.9 | +0.7 | 66.74 |

**Extra paired comparisons:**

| Comparison | Δ F1 [95%] | Δ AUC [95%] |
|---|---|---|
| Filtered (weight 1) vs `famo_fixed` | +0.4 [−0.4, +1.2] | −0.1 [−0.5, +0.3] |
| Filtered vs no filter | +0.7 [−1.5, +2.9] (mostly one seed, +2.7) | +0.7 [−0.6, +2.0] |

**Per label (test F1):**

| Label | Paper (Table 3 F1) | FAMO alone | No filter | Filter, weight 1 | Filter, weight 3 |
|---|---|---|---|---|---|
| Propaganda | 61.74 | 64.4 | 64.4 | 66.1 | 64.6 |
| Disagree/Agree | 71.38 | 66.9 | 63.5 | 64.6 | 65.4 |
| Emotion/Fact | 63.93 | 70.7 | 71.4 | 71.2 | 70.8 |
| Attacking/Respectful | 68.07 | 72.7 | 72.2 | 74.0 | 73.3 |
| Nasty/Nice | 73.69 | 76.0 | 75.2 | 77.1 | 77.4 |
| Personal/Audience | 65.69 | 64.9 | 67.5 | 69.4 | 69.3 |
| Defeater/Undercutter | 55.65 | 58.1 | 59.2 | 59.7 | 58.1 |
| Negotiate/Attack | 64.81 | 70.5 | 70.6 | 71.6 | 71.7 |
| Questioning/Asserting | 63.23 | 63.6 | 63.8 | 62.9 | 59.5 |
| Argument Quality | 79.17 | 82.0 | 84.5 | 82.8 | 83.5 |

**Findings:**
- Filtered Option 2 has the highest test F1 so far (69.9) but **ties `famo_fixed`**. It adds about as much as word-guessing does, and neither is proven with 4 seeds.
- **Clearest effect:** without the filter, every head's validation scores lost 20–30% of their spread (heads drifting toward safe middle answers). The filter kept the spread at FAMO's level.
- Filtered runs were the most consistent across seeds (F1 ±0.7, AUC ±0.2).
- Propaganda +1.7 [+0.5, +2.9] vs FAMO was the only interval above 0, but it's one of about 30 comparisons and only +0.1 vs `famo_fixed`.
- Weight 3 was worse than weight 1. Questioning/Asserting fell to 59.5, probably because the pull toward stability starts competing with getting labels right.
- Two runs showed local "connection error" messages at the end, but all 12 results were saved.

---

## 13. Open items at the end of this chat

- **Experiment 1** (word-guessing through shared and task layers): designed, not built.
- **Option 3** (teacher → student, soft labels on the 144k same-domain cells): not built. Would need teacher runs that save training-row predictions.
- **Settling the ~+1 F1:** 4 more seeds each of FAMO alone and filtered Option 2 (8 runs) would roughly halve the uncertainty.
- **Not measured:** whether models trained with Options 1/2 flip less when words are hidden.
- **Known noise source:** BERT's internal dropout (0.1) perturbs Option 2's full-sentence target during training.
- **No run saves model weights.** Checks that need a trained model must retrain.

Later runs on 2026-11-03 and 2026-11-04 (e.g. `famo_consq2_m5`, `famo_consq3f2_m5`, `famo_consf1_pnor` / `plse` / `ptech`, `famo_consf1_norule`) came from other sessions and aren't covered here.

---

## 14. Where results live

Modal volume `multitask-outputs`:
- `paper_replication_retune_famo/famo_fixed_{mlab,cons}{0.3,1}_s{42,1,2,3}.json` (+ `_predictions.npz`): run 1
- `paper_replication_retune_famo/famo_cons1_m5_*`, `famo_consf1_m5_*`, `famo_consf3_m5_*`: run 2
- `paper_replication_retune_famo/famo_s*`, `famo_fixed_s*`: baselines (2026-10-29)
- `paper_replication_mask_check/famo_fixed_s42.json`: masking check (`mask_sweep` key)
- `paper_replication_label_aware_smoke/`: 1-epoch smoke runs
