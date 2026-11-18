# OpenAI embeddings as the encoder: setup and one-seed pilot (15 Nov 2026)

This note records one working session. It covers:

1. How OpenAI's embedding API was wired into the multitask training on the main data.
2. What it cost.
3. A pilot of two training runs (seed 42): plain task weights, and FAMO task weighting.

**Short version:** on seed 42, the multitask layers trained on fixed OpenAI embeddings score about the same test F1 as the fine-tuned small BERT (plain 68.8 vs 68.0, FAMO 70.1 vs 70.2) and about 3 points more test AUC. This is one seed, so the F1 differences are inside the usual seed-to-seed spread (about 1 point) and nothing is established yet. Seeds 1, 2 and 3 have not been run.

---

## Timeline

All times are US Central (CST), 15 Nov 2026. They come from file save times, Modal's file times and log lines. A "~" means the time is estimated from the events around it.

| Time | What happened |
|---|---|
| 19:31 | OpenAI key saved in `multitask-forked/.env` (by the user). |
| 19:37–19:38 | Code written: `replication/openai_embeddings.py` (19:37:15), `modal_common.py` (19:37:33), `modal_embed.py` (19:37:50), `replication/test_no_prop_overlap.py` (19:38:34). |
| ~19:38 | Local test with the real key on 17 texts: resume after a partial run, empty texts, one over-long text, row order. All passed. |
| 19:39 | Modal secret `openai-api` created from `.env`. |
| ~19:39 | Cost estimate on the Modal copy of the data, no API call: 228,340 unique texts, 9,747,721 tokens, about $0.19. |
| 19:40 | Paid test on Modal: 200 texts embedded and stored (5,441 tokens). |
| ~19:41 | First attempt at the full embedding run was blocked by a permission check; nothing was sent. |
| ~19:45 | Launcher and report added to `modal_replicate.py`. |
| ~19:50 | FAMO method added to the launcher at the user's request. Full embedding run started. |
| 19:52–20:15 | Embedding: 228,140 texts at about 155 texts per second. First full shard saved 19:52, last 20:15. |
| ~20:50 | The two training runs launched on Modal. |
| 20:55 | `openai-small_plain_lr0.0001_s42` finished. |
| 21:00 | `openai-small_famo_lr0.0001_s42` finished. |
| ~21:00 | Report run. A small report fix was saved afterwards (21:00:49): the AUC table no longer shows the paper's F1 column. The fixed report has not been re-run. |
| ~21:50 | Fine-tuned BERT runs for all four seeds pulled from Modal to put seed 42 in context (section 4). |
| 22:07 | This note written. |

---

## Terms used in this note

- **Embedding:** a list of numbers that stands for a text. Texts with similar meaning get similar lists. `text-embedding-3-small` returns 1,536 numbers per text.
- **Encoder:** the part of the model that turns a text into such a list. In earlier studies this was the small BERT (`google/bert_uncased_L-2_H-128_A-2`), which trains along with the rest of the model ("fine-tuned").
- **Frozen encoder:** an encoder whose numbers never change during training. OpenAI's model is frozen by nature: we only receive its output.
- **Shared and task layers:** the dense layers on top of the encoder. Here they are the only part that trains.
- **Plain:** the standard loss with fixed task weights (pooled Propaganda loss, the "wide" layer sizes, dropout 0).
- **FAMO:** a method that re-weights the tasks' losses during training so no task falls behind. Here it is used alone, with no word masking.
- **`famo_consf1_m5`:** the best setup so far on BERT: FAMO plus a consistency loss on copies of the text with some words masked. It cannot be used with OpenAI embeddings, because masking needs the individual words and the API returns one list per whole text.
- **F1:** the weighted F1 score (×100), the paper's metric. The yes/no cut-off for each task is tuned on the validation split.
- **AUC:** how well the scores rank examples (×100), before any cut-off is applied. 50 is chance.
- **Val / test:** the validation and test splits. The epoch is picked on val; test is read at that epoch.
- **Mean of 10:** the average over the 10 outputs in the paper's Table 3.
- **Seed:** the number that fixes the random choices in a run. Different seeds give slightly different results.
- **Token:** the unit OpenAI bills by, roughly three-quarters of a word.

---

## 1. What was built

The texts are embedded once and stored; training then reads the stored lists. The library's own OpenAI encoder was not used, because it calls the API on every training batch and would pay for the whole dataset again every epoch.

| Step | Where | What it does |
|---|---|---|
| Embed | `modal_embed.py::embed` | Sends every unique text of the train, val and test files to OpenAI and stores the results on the Modal cache volume at `/cache/openai_embeddings/<model>/`. Re-running it embeds only what is missing. |
| Train | `modal_replicate.py::replicate --encoder openai-small` | Trains the shared and task layers on the stored lists. Needs no key. |
| Launch and report | `modal_replicate.py::openai_embeddings`, `::openai_embeddings_report` | Starts the runs and prints the tables in section 3. |

Details that affect results:

- **Scaling.** OpenAI's lists have length 1, so each number is about 0.026. They are multiplied by √1536 (about 39) before training so the numbers are about the size the dense layers expect at the start. Every text is scaled the same way, so no similarity changes.
- **Empty texts.** Five training rows have no text. The API rejects empty input, so they get a list of zeros.
- **Long texts.** The model accepts at most 8,191 tokens per text. No text in the data is longer, so nothing was cut.
- **Epoch cap.** 30 instead of the usual 10, with the same stopping rule (val F1, patience 3). Both runs peaked at epoch 2, so the cap was never reached.
- **Methods available.** Plain and FAMO only. The script refuses masked-word options with these encoders.

The key is stored as the Modal secret `openai-api` and is attached only to the embedding function. `.env` is ignored by git and is no longer copied into Modal containers.

Not tested: `openai-large` (`text-embedding-3-large`, 3,072 numbers per text) is wired in but has never been run.

---

## 2. Cost

| Item | Amount |
|---|---|
| Rows in the data (train / val / test) | 219,014 / 5,733 / 8,848 |
| Unique non-empty texts | 228,340 |
| Tokens billed by OpenAI, full run | 9,742,280 |
| Tokens billed, tests (17 local texts + 200 on Modal) | about 14,000 |
| OpenAI cost at list price ($0.02 per million tokens) | about $0.19 |
| Estimate for `text-embedding-3-large` ($0.13 per million tokens), not run | about $1.27 |

The dollar figures use the list prices as I know them, not a bill; check the OpenAI usage page for the actual charge. The stored embeddings are reused by every later run at no further OpenAI cost.

Training ran on one L4 GPU per run, a few minutes each (about 44 seconds per epoch for FAMO). The Modal charge was not looked up.

---

## 3. Pilot results, seed 42

Both runs: `text-embedding-3-small`, learning rate 1e-4, batch size 256, epoch picked on val F1 with patience 3. The BERT columns are the same methods on the fine-tuned small BERT (512 tokens, learning rate 1e-4, same seed and stopping rule), from the earlier `retune` and `retune_famo` studies.

The paper does not say whether its Table 3 is on val or test, so both are shown.

### Mean of the 10 Table-3 outputs (×100)

| Setup | Best epoch | Epochs run | Val F1 | Test F1 | Test AUC | Paper (Table 3 F1) |
|---|---|---|---|---|---|---|
| OpenAI embeddings, FAMO | 2 | 5 | 71.2 | 70.1 | 75.4 | 66.74 |
| OpenAI embeddings, plain | 2 | 9 | 70.5 | 68.8 | 72.3 | 66.74 |
| Small BERT fine-tuned, FAMO | 2 | 5 | 71.3 | 70.2 | 72.4 | 66.74 |
| Small BERT fine-tuned, plain | 8 | 10 | 69.3 | 68.0 | 68.7 | 66.74 |

- **Test F1:** OpenAI minus BERT is +0.8 for plain and −0.1 for FAMO.
- **Test AUC:** OpenAI minus BERT is +3.6 for plain and +3.0 for FAMO.
- **FAMO vs plain on OpenAI embeddings:** +1.3 test F1, +3.1 test AUC.
- **Overfitting:** both OpenAI runs peak at epoch 2. In the FAMO run the training loss keeps falling (0.17 at epoch 4, 0.11 at epoch 5) while the val loss rises (1.02, then 1.40). Dropout is 0 in this setup.

### Test F1 per task (×100)

| Task | Paper (Table 3 F1) | OpenAI FAMO | OpenAI plain | BERT FAMO | BERT plain |
|---|---|---|---|---|---|
| Propaganda | 61.74 | 66.4 | 66.0 | 66.0 | 63.8 |
| Disagree/Agree | 71.38 | 59.4 | 57.3 | 64.6 | 66.7 |
| Emotion/Fact | 63.93 | 73.9 | 71.7 | 70.9 | 71.0 |
| Attacking/Respectful | 68.07 | 75.9 | 73.7 | 74.6 | 70.4 |
| Nasty/Nice | 73.69 | 76.5 | 77.2 | 77.8 | 71.7 |
| Personal/Audience | 65.69 | 66.2 | 64.6 | 70.2 | 68.2 |
| Defeater/Undercutter | 55.65 | 55.2 | 58.3 | 60.9 | 55.4 |
| Negotiate/Attack | 64.81 | 73.5 | 73.1 | 71.4 | 69.0 |
| Questioning/Asserting | 63.23 | 65.6 | 60.3 | 64.1 | 60.8 |
| Argument Quality | 79.17 | 88.3 | 86.2 | 81.4 | 82.8 |
| Mean of 10 | 66.74 | 70.1 | 68.8 | 70.2 | 68.0 |

### Val F1 per task (×100)

| Task | Paper (Table 3 F1) | OpenAI FAMO | OpenAI plain | BERT FAMO | BERT plain |
|---|---|---|---|---|---|
| Propaganda | 61.74 | 66.7 | 66.6 | 64.9 | 63.8 |
| Disagree/Agree | 71.38 | 60.8 | 59.6 | 66.3 | 67.8 |
| Emotion/Fact | 63.93 | 77.5 | 75.8 | 74.9 | 74.5 |
| Attacking/Respectful | 68.07 | 77.2 | 76.3 | 76.3 | 72.3 |
| Nasty/Nice | 73.69 | 77.4 | 78.5 | 78.9 | 73.3 |
| Personal/Audience | 65.69 | 67.1 | 66.3 | 72.5 | 69.4 |
| Defeater/Undercutter | 55.65 | 55.7 | 59.6 | 62.0 | 57.7 |
| Negotiate/Attack | 64.81 | 73.8 | 72.5 | 70.3 | 67.4 |
| Questioning/Asserting | 63.23 | 67.8 | 62.4 | 65.5 | 63.4 |
| Argument Quality | 79.17 | 88.3 | 87.2 | 81.1 | 83.2 |
| Mean of 10 | 66.74 | 71.2 | 70.5 | 71.3 | 69.3 |

### Test AUC per task (×100; the paper reports no AUC)

| Task | OpenAI FAMO | OpenAI plain | BERT FAMO | BERT plain |
|---|---|---|---|---|
| Propaganda | 70.2 | 71.5 | 68.7 | 63.7 |
| Disagree/Agree | 72.6 | 68.0 | 66.9 | 64.1 |
| Emotion/Fact | 83.4 | 81.7 | 80.2 | 77.8 |
| Attacking/Respectful | 80.8 | 77.8 | 77.2 | 73.5 |
| Nasty/Nice | 82.5 | 80.2 | 78.9 | 75.6 |
| Personal/Audience | 68.2 | 64.7 | 64.7 | 60.0 |
| Defeater/Undercutter | 61.6 | 58.1 | 61.9 | 58.1 |
| Negotiate/Attack | 79.9 | 78.8 | 78.1 | 76.2 |
| Questioning/Asserting | 69.8 | 60.4 | 64.0 | 59.5 |
| Argument Quality | 84.7 | 82.2 | 83.9 | 78.2 |
| Mean of 10 | 75.4 | 72.3 | 72.4 | 68.7 |

**Disagree/Agree** is the one task where OpenAI embeddings are clearly worse on F1 (5 to 9 points lower on test) while being better on AUC (72.6 vs 66.9 for FAMO). That pattern would fit scores that rank examples well but a yes/no cut-off that lands badly. This was not checked.

---

## 4. Seed 42 in context

Seed 42 alone can mislead. These are the fine-tuned small BERT runs for all four seeds (test F1, mean of 10, ×100, same stopping rule), pulled from Modal on 15 Nov:

| Fine-tuned small BERT | Seed 42 | Seed 1 | Seed 2 | Seed 3 | Mean ± SD | Mean val F1 | Mean test AUC | Paper (Table 3 F1) |
|---|---|---|---|---|---|---|---|---|
| plain | 68.0 | 69.5 | 67.0 | 67.9 | 68.1 ± 1.0 | 69.3 | 69.1 | 66.74 |
| FAMO only | 70.2 | 68.2 | 69.0 | 68.6 | 69.0 ± 0.9 | 70.1 | 71.2 | 66.74 |
| `famo_consf1_m5` (best so far) | 69.9 | 69.6 | 69.4 | 70.9 | 69.9 ± 0.7 | 70.9 | 72.6 | 66.74 |

- The **69.9** quoted in earlier notes is the four-seed mean of `famo_consf1_m5`.
- The **70.2** in section 3 is FAMO-only on seed 42, which is that method's highest seed: 1.2 points above its own four-seed mean (69.0).
- So the OpenAI FAMO result (70.1, seed 42) should be compared with 70.2, not with 69.9, and it may come down when other seeds are added.

---

## 5. What is and is not established

- **Established:** the pipeline works end to end. The embeddings are stored and reusable. On seed 42, frozen OpenAI embeddings match the fine-tuned small BERT on mean F1 and beat it on mean AUC for both methods.
- **Not established:** whether any of this holds across seeds. No significance test is possible with one seed.
- **Not tried:** seeds 1, 2, 3; dropout above 0 or other settings to slow the overfitting; `text-embedding-3-large`; any check of the Disagree/Agree cut-off.

---

## 6. Commands

Run from `multitask-forked/`.

```bash
# Tokens and cost for whatever is not stored yet, no API call
modal run modal_embed.py::embed --estimate-only

# Embed and store (needs the Modal secret openai-api)
modal run --detach modal_embed.py::embed

# The two pilot runs, and the three remaining seeds
modal run --detach modal_replicate.py::openai_embeddings --seeds 42
modal run --detach modal_replicate.py::openai_embeddings --seeds 1,2,3

# Tables
modal run modal_replicate.py::openai_embeddings_report
```

Result files on Modal (`multitask-outputs` volume): `paper_replication_openai/openai-small_<method>_lr0.0001_s42.json`, each with a `_predictions.npz` beside it.

The code changes from this session are not committed to git.
