# Hate speech experiments: results (4–5 Nov 2026)

This note records the hate speech experiments: hidden-label (masking) runs, the FAMO and consistency methods, IAC + hate speech in one model, and a single-task baseline.

**Short version**

- **Hiding labels costs little until most are gone.** Hiding 25–75% of the hate speech training labels lowers test F1 by only 0.1–0.6. Hiding 90% costs 1.9 and hiding 95% costs 2.7.
- **Neither FAMO nor `famo_consf1_m5` beats plain training at any hiding level.** That holds on test and on the hidden training labels; every difference is 0.3 points or less. `famo_consf1_m5` is the 69.9 setup from the main data.
- **One model for IAC + hate speech does not beat separate models.** Over all 18 heads it is 0.5–0.8 F1 lower, which is not significant. The one consistent gain is Nasty/Nice, +1.1 to +2.1 F1 under all three methods.
- **Single-task models beat the multi-task hate speech model by about 1 F1** (79.1 against 78.0–78.1, one seed), with equal AUC. The gap comes from violence, genocide and hatespeech.
- **With 60% of hate speech labels hidden, adding IAC still doesn't help the hate speech heads.** The joint model is 1.0–1.4 F1 lower than hate speech alone (one seed), probably because it stops at an earlier epoch.

The paper has no hate speech results, so hate speech tables have no paper column. IAC tables show the paper's Table 3 multi-task F1.

---

## Terms

- **Item:** one of the 10 hate speech survey questions, each a yes/no output: sentiment, respect, insult, humiliate, status, dehumanize, violence, genocide, attack_defend, hatespeech.
- **Cut-offs:** an item is "yes" when its average annotator score is above 2 on the 0–4 scale. For hatespeech, which uses a 0–2 scale, "yes" means a score of at least 1.
- **Hidden labels (label masking):** a random share of each item's training labels is removed before training. The same seed always hides the same cells, so runs can be compared cell by cell. Validation and test labels are never hidden.
- **Word covering:** inside `famo_consf1_m5`, a copy of each text has 5% of its words replaced by [MASK]. The model's answers on the copy should match its answers on the full text. Nothing is guessed: the masked-word loss has weight 0. This is separate from label hiding.
- **plain:** fixed task weights.
- **FAMO:** task weights that change during training, according to how fast each task's loss is falling.
- **`famo_consf1_m5`:** FAMO plus the consistency loss on the word-covered copy. The loss uses only the 10% most confident scores at each end, per head. It is the best setup on IAC/IBM/Propaganda (69.9).
- **Test F1:** weighted F1 ×100 on the 3,949 test comments. The model never saw them.
- **Hidden-label F1:** weighted F1 on the training comments, counting only the labels hidden from that model. The model saw the text and the comment's other labels, but not the label being scored. This measures "performance on the unsupervised part".
- **Δ:** a difference paired by seed, with a 95% t interval in brackets. If the interval excludes 0, the difference counts as real.

## Setup

- **Data:** Measuring Hate Speech (`hatespeech_dlab.csv`), cleaned to match CMP style (ASCII, apostrophes removed, @handles and emoji removed).
  - 39,490 comments: train 31,592, val 3,949, test 3,949.
  - Split with seed 42, stratified by platform and by hatespeech yes/no.
- **Model and training:** the same as the main re-tuned runs.
  - small BERT (`google/bert_uncased_L-2_H-128_A-2`), inputs up to 512 word pieces;
  - "wide" head layers;
  - learning rate 1e-4, weight decay 0.01, 5% warmup, dropout 0, batch 256;
  - up to 10 epochs, keeping the epoch with the best validation F1 and stopping after 3 epochs without improvement;
  - yes/no thresholds tuned on validation.
- **Seeds:** 42, 1 and 2 unless stated.
- **Code:** `multitask-forked/hatespeech/`: `prepare_hatespeech.py`, `run_hatespeech.py` and `modal_hatespeech.py`. It imports the main model, loss and scorer without changing them.
  - Two small trainer subclasses fix FAMO's task count. The main code assumes a pooled Propaganda output; hate speech has none.
- **IAC data:** IAC forum posts only, from `no_prop_overlap_*.csv`: 61,069 / 1,546 / 3,092 rows. There are no news rows, so no rule-made labels.

## Timeline (CST)

| When | What |
|---|---|
| Wed 4 Nov, ~17:40 | Hate speech data cleaned and split |
| Thu 5 Nov, 10:17 | Upload; first test run (all labels, seed 42): test F1 78.1 |
| 10:43–11:50 | Plain runs at 0 / 25 / 50 / 75% hidden (12 runs; one relaunch after connection drops) |
| 12:21 | IAC alone and IAC + hate speech, plain (6 runs) |
| 12:37–14:30 | Set A: plain at 90 / 95%; FAMO and `famo_consf1_m5` at 0 / 50 / 90 / 95%. Set B: FAMO and `famo_consf1_m5` on IAC alone and on IAC + hate speech (48 runs in total, all finished) |
| ~14:40 | Single-task baseline, 10 runs, seed 42 |
| 15:08 | IAC + hate speech with 90% of hate speech labels hidden: started, then stopped on request |
| ~15:10–16:30 | 60% hidden: joint and alone × plain and `famo_consf1_m5`, seed 42 (4 runs, done) |

---

## 1. Hate speech alone: test F1 (mean of the 10 items)

| Labels hidden | plain | FAMO | `famo_consf1_m5` |
|---|---|---|---|
| none | 78.1 ± 0.2 | 78.2 ± 0.2 | 78.0 ± 0.2 |
| 25% | 78.0 ± 0.4 | — | — |
| 50% | 77.7 ± 0.2 | 77.8 ± 0.2 | 77.7 ± 0.5 |
| 75% | 77.5 ± 0.6 | — | — |
| 90% | 76.2 ± 0.5 | 76.3 ± 0.3 | 76.2 ± 0.0 |
| 95% | 75.4 ± 0.9 | 75.3 ± 0.2 | 75.1 ± 0.9 |

How much hiding costs the plain model, against all labels:

| Labels hidden | Δ F1 | Δ AUC |
|---|---|---|
| 25% | −0.1 [−0.7, +0.5] | −0.1 |
| 50% | −0.4 [−0.5, −0.2] | −0.5 |
| 75% | −0.6 [−2.7, +1.4] | −1.1 [−1.9, −0.3] |
| 90% | −1.9 [−2.6, −1.2] | −2.6 [−3.5, −1.6] |
| 95% | −2.7 [−5.5, +0.1] | −4.1 [−6.5, −1.7] |

FAMO and `famo_consf1_m5` against plain at the same hiding level:
- F1 is within −0.3 to +0.1 everywhere, and every interval includes 0.
- Test AUC: plain falls from 84.1 to 80.0 between none and 95% hidden, and the methods stay within 0.3 of plain throughout.

## 2. Hate speech alone: F1 on the hidden training labels

| Labels hidden | plain | FAMO | `famo_consf1_m5` |
|---|---|---|---|
| 25% | 79.9 ± 0.5 | — | — |
| 50% | 79.4 ± 0.1 | 79.5 ± 0.1 | 79.2 ± 0.4 |
| 75% | 79.2 ± 0.4 | — | — |
| 90% | 77.0 ± 0.7 | 77.4 ± 0.2 | 77.3 ± 0.3 |
| 95% | 76.1 ± 0.9 | 75.9 ± 0.3 | 75.7 ± 0.8 |

For comparison, here is the all-labels plain model, which saw these labels, scored on the same cells:

| Labels hidden | Masked model | All-labels model | Difference |
|---|---|---|---|
| 25% | 79.9 | 80.8 | −0.9 |
| 50% | 79.4 | 80.9 | −1.4 |
| 75% | 79.2 | 80.9 | −1.7 |

The 90% and 95% comparisons aren't available. The all-labels plain runs were made before those levels were added.

## 3. Hate speech alone: test F1 per item

| Item | plain, none hidden | plain 50% | plain 75% | plain 90% | plain 95% | FAMO 90% | `famo_consf1_m5` 90% |
|---|---|---|---|---|---|---|---|
| sentiment | 80.9 ± 0.2 | 80.5 ± 1.1 | 80.6 ± 0.8 | 80.3 ± 0.3 | 79.0 ± 1.0 | 80.0 ± 1.4 | 78.6 ± 1.2 |
| respect | 79.9 ± 0.7 | 80.2 ± 0.9 | 79.2 ± 0.6 | 78.0 ± 0.8 | 78.5 ± 1.0 | 78.7 ± 2.0 | 78.6 ± 0.1 |
| insult | 77.5 ± 0.2 | 77.0 ± 0.5 | 76.7 ± 0.7 | 75.9 ± 0.4 | 75.3 ± 0.9 | 75.9 ± 0.7 | 75.1 ± 0.8 |
| humiliate | 74.7 ± 0.5 | 74.2 ± 0.2 | 74.1 ± 0.4 | 73.0 ± 0.6 | 72.0 ± 0.8 | 73.0 ± 0.4 | 73.1 ± 0.5 |
| status | 72.3 ± 0.6 | 72.1 ± 0.5 | 71.8 ± 0.6 | 70.1 ± 1.1 | 69.9 ± 0.2 | 70.4 ± 0.0 | 70.6 ± 0.7 |
| dehumanize | 71.0 ± 0.2 | 70.1 ± 0.5 | 70.1 ± 0.7 | 70.3 ± 0.0 | 69.9 ± 1.4 | 69.5 ± 0.6 | 70.6 ± 0.5 |
| violence | 82.5 ± 1.5 | 80.8 ± 1.5 | 81.5 ± 3.1 | 76.7 ± 3.5 | 75.2 ± 7.4 | 79.0 ± 1.9 | 78.8 ± 2.2 |
| genocide | 89.7 ± 1.5 | 89.0 ± 1.0 | 88.5 ± 0.7 | 87.1 ± 0.7 | 87.3 ± 1.9 | 87.6 ± 1.3 | 87.1 ± 1.8 |
| attack_defend | 75.9 ± 0.5 | 75.8 ± 0.0 | 75.1 ± 0.6 | 74.1 ± 0.7 | 72.7 ± 0.8 | 73.7 ± 0.3 | 73.5 ± 0.3 |
| hatespeech | 76.6 ± 0.5 | 77.3 ± 1.2 | 77.2 ± 0.4 | 76.8 ± 0.8 | 74.2 ± 1.4 | 75.5 ± 0.5 | 75.5 ± 1.7 |

## 4. Single-task vs multi-task on hate speech (all labels, seed 42)

Each single-task model has one head trained on one item; everything else is the same as the multi-task runs. Δ is multi-task minus single-task on the same seed, so a negative value means single-task did better.

| Item | Single-task | plain multi | Δ | FAMO multi | Δ | `famo_consf1_m5` multi | Δ |
|---|---|---|---|---|---|---|---|
| sentiment | 81.1 | 80.7 | −0.4 | 79.5 | −1.6 | 80.0 | −1.1 |
| respect | 80.6 | 80.1 | −0.5 | 81.1 | +0.5 | 81.3 | +0.7 |
| insult | 77.6 | 77.5 | −0.1 | 76.3 | −1.2 | 76.9 | −0.7 |
| humiliate | 74.8 | 74.9 | +0.1 | 74.9 | +0.1 | 75.1 | +0.3 |
| status | 72.1 | 71.6 | −0.5 | 72.7 | +0.6 | 71.9 | −0.2 |
| dehumanize | 72.1 | 70.9 | −1.2 | 71.5 | −0.6 | 70.3 | −1.8 |
| violence | 86.1 | 84.2 | −1.9 | 81.2 | −4.9 | 80.5 | −5.7 |
| genocide | 90.9 | 88.3 | −2.6 | 88.1 | −2.8 | 89.0 | −1.9 |
| attack_defend | 75.8 | 75.4 | −0.4 | 76.4 | +0.6 | 76.3 | +0.5 |
| hatespeech | 79.6 | 76.9 | −2.7 | 77.9 | −1.6 | 78.6 | −0.9 |
| **Mean** | **79.1** | **78.1** | **−1.0** | **78.0** | **−1.1** | **78.0** | **−1.1** |

- **AUC:** mean test AUC is 84.1 single-task and 84.0 plain multi-task. Ranking quality is the same, so the F1 gap on the rare items (violence 10% "yes", genocide 4%) may come from where the yes/no cut-off lands. That is untested.
- **Seeds:** with one single-task seed there are no intervals. Gaps under about 1 point may be noise.

## 5. IAC + hate speech: one joint model vs separate models (all labels)

- **Joint:** one model with shared layers, an IAC branch (8 heads) and a hate speech branch (10 heads). Each row has labels only from its own dataset; the other heads skip it.
- **Separate:** an IAC-only model and a hate speech-only model, scored together.
- **Scoring:** IAC heads are scored on IAC forum posts only. The paper's numbers are on its full test split, which includes rule-labelled news rows, so they aren't directly comparable.
- **Labels hidden:** none, apart from the cells that are empty because each dataset has only its own labels.
- **Word covering:** only the `famo_consf1_m5` runs used it.

### Averages

| Method | Model | Hate speech heads (10) | IAC heads (8) | All 18 heads | All 18, test AUC |
|---|---|---|---|---|---|
| plain | separate | 78.1 ± 0.2 | 66.0 ± 0.5 | 72.7 ± 0.3 | 76.7 ± 0.2 |
| plain | joint | 77.4 ± 0.4 | 65.8 ± 0.3 | 72.2 ± 0.3 | 76.8 ± 0.1 |
| FAMO | separate | 78.2 ± 0.2 | 66.9 ± 0.7 | 73.2 ± 0.2 | 77.0 ± 0.5 |
| FAMO | joint | 77.8 ± 0.2 | 65.7 ± 1.1 | 72.4 ± 0.6 | 76.8 ± 0.1 |
| `famo_consf1_m5` | separate | 78.0 ± 0.2 | 66.4 ± 1.3 | 72.9 ± 0.5 | 76.8 ± 0.3 |
| `famo_consf1_m5` | joint | 77.4 ± 0.1 | 66.1 ± 1.2 | 72.4 ± 0.5 | 76.9 ± 0.1 |
| *Paper (Table 3 F1)* | — | *n/a* | *65.81* | *n/a* | *n/a* |

Joint minus separate (F1, paired by seed):

| Method | Hate speech heads | IAC heads | All 18 heads |
|---|---|---|---|
| plain | −0.7 [−2.1, +0.6] | −0.1 [−2.0, +1.7] | −0.5 [−2.0, +1.1] |
| FAMO | −0.4 [−1.2, +0.5] | **−1.2 [−2.3, −0.1]** | −0.8 [−1.7, +0.2] |
| `famo_consf1_m5` | **−0.6 [−0.8, −0.5]** | −0.4 [−3.5, +2.8] | −0.5 [−1.8, +0.8] |

### Per head (test F1)

| Head | Paper (Table 3 F1) | plain separate | plain joint | FAMO separate | FAMO joint | `famo_consf1_m5` separate | `famo_consf1_m5` joint |
|---|---|---|---|---|---|---|---|
| sentiment | n/a | 80.9 ± 0.2 | 79.8 ± 1.3 | 81.0 ± 1.3 | 80.6 ± 0.1 | 80.1 ± 0.1 | 80.1 ± 0.7 |
| respect | n/a | 79.9 ± 0.7 | 79.0 ± 1.5 | 81.0 ± 0.3 | 79.8 ± 0.6 | 81.3 ± 0.3 | 79.4 ± 0.4 |
| insult | n/a | 77.5 ± 0.2 | 77.0 ± 0.2 | 77.3 ± 0.9 | 77.5 ± 0.5 | 77.7 ± 0.8 | 77.0 ± 0.8 |
| humiliate | n/a | 74.7 ± 0.5 | 74.2 ± 0.1 | 74.5 ± 0.4 | 74.4 ± 0.5 | 74.7 ± 0.7 | 74.8 ± 0.2 |
| status | n/a | 72.3 ± 0.6 | 71.8 ± 0.4 | 72.7 ± 0.1 | 71.8 ± 0.1 | 72.4 ± 0.5 | 72.3 ± 0.3 |
| dehumanize | n/a | 71.0 ± 0.2 | 69.8 ± 0.6 | 70.3 ± 1.1 | 70.4 ± 0.3 | 70.8 ± 0.6 | 71.0 ± 0.5 |
| violence | n/a | 82.5 ± 1.5 | 80.5 ± 2.4 | 81.7 ± 1.1 | 82.1 ± 0.7 | 80.3 ± 2.1 | 78.6 ± 1.2 |
| genocide | n/a | 89.7 ± 1.5 | 88.6 ± 1.9 | 89.5 ± 1.7 | 89.0 ± 2.1 | 89.8 ± 1.7 | 88.7 ± 1.4 |
| attack_defend | n/a | 75.9 ± 0.5 | 76.0 ± 0.4 | 76.1 ± 0.3 | 75.8 ± 0.1 | 76.1 ± 0.4 | 75.3 ± 0.4 |
| hatespeech | n/a | 76.6 ± 0.5 | 77.1 ± 1.1 | 77.7 ± 0.4 | 77.0 ± 1.2 | 77.1 ± 1.5 | 77.0 ± 1.0 |
| Disagree/Agree | 71.38 | 62.8 ± 7.0 | 62.5 ± 4.9 | 62.2 ± 2.1 | 61.6 ± 3.2 | 66.0 ± 1.4 | 63.3 ± 2.6 |
| Emotion/Fact | 63.93 | 67.7 ± 0.3 | 67.9 ± 0.8 | 67.2 ± 0.1 | 67.9 ± 0.3 | 67.7 ± 0.7 | 67.4 ± 0.3 |
| Attacking/Respectful | 68.07 | 71.9 ± 1.2 | 73.1 ± 1.0 | 73.0 ± 1.6 | 72.1 ± 3.9 | 73.4 ± 0.3 | 72.9 ± 2.0 |
| **Nasty/Nice** | 73.69 | 74.0 ± 0.6 | **75.6 ± 0.2** | 74.7 ± 3.0 | **75.9 ± 0.9** | 75.6 ± 0.9 | **77.6 ± 0.5** |
| Personal/Audience | 65.69 | 67.5 ± 3.7 | 63.1 ± 7.5 | 68.6 ± 3.5 | 65.9 ± 3.3 | 67.7 ± 3.0 | 68.8 ± 1.9 |
| Defeater/Undercutter | 55.65 | 54.4 ± 4.8 | 56.2 ± 0.9 | 57.7 ± 3.4 | 53.2 ± 1.5 | 49.7 ± 7.6 | 50.9 ± 8.3 |
| Negotiate/Attack | 64.81 | 68.3 ± 1.4 | 67.3 ± 1.3 | 69.2 ± 2.1 | 68.5 ± 1.4 | 67.5 ± 4.1 | 70.2 ± 0.4 |
| Questioning/Asserting | 63.23 | 61.1 ± 2.1 | 60.8 ± 3.0 | 62.6 ± 5.1 | 60.3 ± 1.8 | 63.8 ± 2.4 | 57.4 ± 5.9 |

- **Nasty/Nice** improves in every method. The plain gain is +1.6 [+0.4, +2.8].
- The IAC validation set is small (1,546 rows), so several IAC heads swing by ±3–8 between seeds.

### IAC numbers from the main experiments (IAC + IBM + Propaganda model)

For reference:
- **Full test split, with news rows:**

| Model | Seeds | Mean of 8 IAC heads |
|---|---|---|
| plain | 4 | 66.7 |
| FAMO | 4 | 67.9 |
| `famo_fixed` | 4 | 68.5 |
| `famo_consf1_m5` | 4 | 68.8 |
| single-task | 4 | 69.6 |

- **IAC forum posts only** (the same rows as IAC alone above): plain multi-task 65.3 (seeds 42, 1, 2), against 66.0 for IAC alone.

## 6. IAC + hate speech with 60% of hate speech labels hidden (seed 42)

All 4 runs hide the same 60% of hate speech training labels (seed 42); IAC labels are never hidden. They were run Thursday 5 Nov, ~15:10–16:30 CST. A 90% version was started at 15:08 and stopped on request, so it has no results. There is one seed, so there are no intervals. Treat differences under about 1 point as possible noise.

### Averages

| Model | Method | Best epoch | Hate speech test F1 | Hate speech test AUC | Hidden-label F1 | IAC test F1 | IAC test AUC |
|---|---|---|---|---|---|---|---|
| hate speech alone | plain | 10 | **77.8** | 83.5 | **79.9** | — | — |
| IAC + hate speech | plain | 5 | 76.4 | 83.0 | 78.2 | 65.8 | 67.6 |
| hate speech alone | `famo_consf1_m5` | 9 | **77.8** | 83.5 | 79.3 | — | — |
| IAC + hate speech | `famo_consf1_m5` | 3 | 76.8 | 83.3 | 78.0 | 66.2 | 67.8 |
| *Paper (Table 3 F1)* | — | — | *n/a* | *n/a* | *n/a* | *65.81* | *n/a* |

Seed-42 references from earlier runs:

| Run | Hate speech test F1 | Hidden-label F1 | IAC test F1 |
|---|---|---|---|
| hate speech alone, plain, 0% hidden | 78.1 | — | — |
| hate speech alone, plain, 50% hidden | 77.6 | 79.3 | — |
| hate speech alone, plain, 75% hidden | 77.8 | 79.4 | — |
| joint, plain, 0% hidden | 77.4 | — | 65.6 |
| joint, `famo_consf1_m5`, 0% hidden | 77.4 | — | 65.3 |
| IAC alone, plain | — | — | 66.1 |
| IAC alone, `famo_consf1_m5` | — | — | 67.1 |

What the 60% runs show:

- **Training with IAC doesn't help the hate speech heads when hate speech labels are scarce; it costs them.** With plain training, the joint model is 1.4 lower on test and 1.7 lower on hidden labels than hate speech alone. With `famo_consf1_m5` it is 1.0 and 1.3 lower.
- **`famo_consf1_m5` adds nothing.** Alone it ties plain on test (77.8 vs 77.8) and is 0.6 lower on hidden labels. In the joint model it is +0.4 on test and −0.2 on hidden labels against plain.
- **IAC heads are barely affected by hiding hate speech labels.** Joint at 60% scores 65.8 and 66.2, against 65.6 and 65.3 for joint at 0% and 66.1 and 67.1 for IAC alone.
- **A likely reason the joint model trails on hate speech: epoch choice.** The joint model picks its epoch by mean validation F1 over all 18 heads. The IAC heads have a small, noisy validation set (1,546 rows) and peak early, so the joint runs stop at epochs 3–5. The hate speech-only runs go to epochs 9–10. The same pattern holds in every joint run (mean best epoch 3.3–4.3 vs 7–8 for hate speech alone). This is a hypothesis, not tested.

### Per hate speech item (test F1)

| Item | hate speech alone, plain | joint, plain | hate speech alone, `famo_consf1_m5` | joint, `famo_consf1_m5` |
|---|---|---|---|---|
| sentiment | 82.6 | 77.7 | 80.9 | 80.9 |
| respect | 79.6 | 79.4 | 80.1 | 78.9 |
| insult | 76.7 | 75.9 | 77.4 | 76.6 |
| humiliate | 74.4 | 73.1 | 74.3 | 74.1 |
| status | 71.8 | 70.5 | 71.8 | 69.8 |
| dehumanize | 70.7 | 69.9 | 71.2 | 66.2 |
| violence | 81.4 | 76.1 | 80.8 | 81.2 |
| genocide | 89.3 | 89.9 | 87.2 | 87.5 |
| attack_defend | 75.6 | 74.2 | 76.1 | 74.7 |
| hatespeech | 76.0 | 76.7 | 78.1 | 77.6 |

### Per IAC head (test F1, joint models at 60% hidden)

| Head | Paper (Table 3 F1) | joint, plain | joint, `famo_consf1_m5` |
|---|---|---|---|
| Disagree/Agree | 71.38 | 59.6 | 58.5 |
| Emotion/Fact | 63.93 | 67.8 | 68.0 |
| Attacking/Respectful | 68.07 | 72.4 | 71.8 |
| Nasty/Nice | 73.69 | 79.9 | 76.0 |
| Personal/Audience | 65.69 | 68.6 | 70.4 |
| Defeater/Undercutter | 55.65 | 49.4 | 54.6 |
| Negotiate/Attack | 64.81 | 64.4 | 70.6 |
| Questioning/Asserting | 63.23 | 64.1 | 59.8 |

Nasty/Nice is high again: 79.9 with plain, against 74.6 for IAC alone on the same seed. That is consistent with section 5, but on one seed.

## 7. Which source is most like hate speech?

These checks used simple word-count models (TF-IDF + logistic regression), not BERT.

| Source | Word-use difference vs hate speech (lower = closer) | Mean AUC of its best task's model on hate speech items |
|---|---|---|
| IAC | 0.17 | 61 (nasty, attacking) |
| Propaganda | 0.23 | ~53 |
| IBM | 0.28 | 47 (reversed) |

- A word model trained on hate speech itself gets 83.5.
- IAC is the closest source, which matches the Nasty/Nice gain in section 5.

## Open items

- Possible, not started:
  - a joint model that picks its epoch from the hate speech heads only, to test the early-stopping explanation in section 6;
  - 2 more single-task seeds for violence, genocide and hatespeech;
  - a check, from the saved results, of whether the rare-item F1 gap comes from the threshold;
  - FAMO and `famo_consf1_m5` at 75% hidden (the closest match to how sparse the labels are in the main data).
