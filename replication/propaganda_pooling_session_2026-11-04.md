# Propaganda pooling study and related questions (3–4 Nov 2026)

This note records one working session. It covers:

1. How the loss treats the three data sources, and whether IAC and IBM could use the same max-pooling as Propaganda.
2. Three replacements for the max-pool, built and tested on Modal (12 training runs).
3. Whether each data source should get its own kind of loss.
4. Whether the paper's Table 3 F1 scores are on the validation or the test split.

**Short version:** none of the three pooling options beat the current best setup (`famo_consf1_m5`, mean test F1 69.9), so the max-pool stays. The code for the options stays in the repo behind flags that are off by default. The paper never says which split Table 3 uses.

---

## Timeline

All times are US Central (CST). They come from file save times, Modal's result-file times and log lines. A "~" means the time is estimated from the events around it, not recorded directly.

| Date and time | What happened |
|---|---|
| 2 Nov, 17:23–17:32 | Baseline runs `famo_consf1_m5` (seeds 2, 3, 1, 42) saved on Modal. These come from an earlier session and are the comparison point here. |
| 3 Nov, evening, before 22:35 | Discussion: how the loss works, why IAC can't be max-pooled, whether Propaganda F1 is already good (section 1). The user asked to try all three pooling options. |
| 3 Nov, 22:35 | Library changes saved: `config.py`, `trainers/trainer.py` (22:35:29) and `models/hierarchical.py` (22:35:52). The script changes were made at the same time. |
| 3 Nov, 22:36 | New tests written in `test_pooled_output_loss.py` (22:36:43). |
| 3 Nov, ~22:38–22:44 | Tests run on Modal: first the full suite, then the pooled-loss file alone (its log shows 22:43:48). 19 of 19 pooled-loss tests pass. |
| 3 Nov, 22:47 | The 12 runs launched on Modal, detached so they keep going if the laptop disconnects. |
| 4 Nov, 00:21–00:37 | Noisy-OR runs finish (seeds 1, 2, 3, 42). |
| 4 Nov, 00:47–01:32 | Log-sum-exp runs finish (seeds 42, 3, 2, 1). |
| 4 Nov, 01:17 | Early report on noisy-OR alone: −1.0 F1 against the best. |
| 4 Nov, ~01:20–02:00 | Discussion: should each data source get its own loss? (section 3). |
| 4 Nov, 01:34–02:04 | Max + per-technique runs finish (seeds 1, 42, 2, 3). |
| 4 Nov, ~02:05 | Final report and technique-head check. Decision: keep the max-pool. The study's memory note was updated at 02:05. |
| 4 Nov, 10:25–10:27 | Paper check: both PDFs downloaded (10:25:52 arXiv, 10:26:00 ACL) and read. The split for Table 3 is unstated (section 4). |
| 4 Nov, 16:26 | This note written; the timestamps were added afterwards. |

`modal_replicate.py` and `test_no_prop_overlap.py` show a save time of 4 Nov 13:35. That comes from later, unrelated work on the same day, not from this study.

---

## Terms used in this note

- **Head:** the small output layer that gives one task's score, a probability between 0 and 1.
- **Technique heads:** the 18 Propaganda heads, one per propaganda technique (Name_Calling, Slogans, Bandwagon, …).
- **Max-pool:** the Propaganda score is the highest of the 18 technique scores. The Propaganda label is 1 if any technique is present in the sentence.
- **Gradient:** the learning signal that flows back from the loss and changes the model's weights. A part of the model that gets no gradient doesn't learn.
- **F1:** the weighted F1 score (×100), the paper's metric. The yes/no cut-off for each task is tuned on the validation split.
- **AUC:** ROC AUC (×100). It measures whether the model ranks positive examples above negative ones, without any cut-off. 50 means no signal.
- **Validation / test:** the 10% and 10% held-out splits. Validation picks the setup and the epoch; test is the final score.
- **FAMO:** the loss-balancing method in the current best setup. It reweights the 10 task losses during training according to how fast each is improving.
- **`famo_consf1_m5`:** the current best setup. FAMO, plus a consistency loss: the model's answers on a copy of each sentence with 5% of words hidden should match its answers on the full sentence. That loss uses only the 10% most confident cells on each side, per head. Mean test F1 69.9.
- **Paired difference (Δ):** the difference from the baseline computed seed by seed (same seeds 42, 1, 2, 3), then averaged, with a 95% t interval. An interval that excludes 0 counts as a real difference.

---

## 1. How the loss is built now, and why IAC can't simply be max-pooled

*3 Nov 2026, evening.*

### What the max-pool does for Propaganda

The trainer takes the highest of the 18 technique scores and compares it with the highest of the 18 technique labels, giving one yes/no question: "does this sentence use any propaganda technique?" That question makes sense, and it's what the paper reports. The code is `PooledOutputLossTrainer` in `replication/test_no_prop_overlap.py`.

### Does the loss reach only the head with the max?

Yes, for each sentence. The gradient of a max passes back only through the highest-scoring head, so the other 17 heads get nothing from that sentence. The shared layers and BERT still train through the winning head.

In practice it's the same head nearly every time. The 2 Nov score-distribution study found that one technique head, chosen at random by the seed, was the top score on 96–100% of test rows. The other 17 stayed near their starting values. So the model acts like a single "propaganda yes/no" head.

### Is Propaganda F1 already good?

Yes. FAMO-fixed scores 66.0 and the plain setup 64.8, against the paper's 61.74. The max-pool doesn't hurt the reported number. Its cost is that the 18 technique heads mean little, so any analysis by technique is reading mostly untrained heads.

### Why max-pooling IAC or IBM doesn't work

- **IBM (argument quality)** has one head, so a max over it changes nothing.
- **IAC** has 8 heads that answer different questions: agree vs disagree, emotional vs factual, and so on. Which end of each scale is coded "1" is arbitrary. The max of the 8 labels would mean "on the 1-end of at least one of 8 unrelated scales", which is nothing anyone wants to predict.
- **Max-pooling IAC would also break the results.** Table 3 reports each IAC dimension separately, and only one of the 8 heads would keep learning, the same winner-take-all problem.

### Each source already counts about once in the loss

The trainer groups the heads into three task types: propaganda, argument quality, and the IAC dimensions. Each IAC head's loss is divided by 8, the size of its group, so IAC as a whole carries about one group's weight. FAMO then rebalances during training.

---

## 2. Three replacements for the max-pool

*Built and tested 3 Nov, 22:35–22:44 CST. Runs from 3 Nov 22:47 to 4 Nov 02:04. Decided 4 Nov at ~02:05.*

### The options

| Option | Run name | Formula | What it does |
|---|---|---|---|
| Noisy-OR | `famo_consf1_pnor` | 1 − ∏(1 − pᵢ) | "Probability at least one technique is present." Every head gets gradient. |
| Log-sum-exp, t = 1 | `famo_consf1_plse` | sigmoid(log Σ exp(zᵢ)), zᵢ = logit(pᵢ) | A smooth max. Equals 1 − P(no technique) under a softmax over the 18 techniques plus a "none" option. Each head gets a share of the gradient in proportion to how strongly it fires. |
| Max + per-technique loss | `famo_consf1_ptech` | max-pool term + 0.1 × mean of the 18 per-technique losses | Keeps the paper's pooled output and also trains each head on its own technique label. |

All three were added on top of `famo_consf1_m5`, with the same seeds and settings.

### Design decisions

- **The pooled label is always the max of the technique labels** ("any technique present"), whatever the pooling. Before this change the library pooled labels with the same rule as predictions, which would have given meaningless labels under log-sum-exp.
- **The pooled output keeps its name, `propaganda_max_pool`,** under every option, so existing reports and saved predictions still work.
- **Rare techniques are handled by averaging each technique's loss over its class weights, not its row count.** This is how PyTorch's weighted cross-entropy averages. Some techniques are extremely rare in the 61,909 Propaganda training rows:

  | Technique | Positives | Class weight for "yes" |
  |---|---|---|
  | Loaded_Language | 8,155 | 3.8 |
  | Name_Calling | 4,218 | 7.3 |
  | Red_Herring | 104 | 297.6 |
  | Straw_Men | 48 | 644.9 |
  | Obfuscation | 50 | 619.1 |
  | Bandwagon | 35 | 884.4 |

  With row-count averaging, one Bandwagon positive in a batch would swamp the loss.
- **The per-technique loss is folded into the Propaganda task's loss,** so FAMO still balances 10 tasks.
- **Probabilities are clipped to [1e-7, 1 − 1e-7] before taking logs,** as Keras's binary cross-entropy does.

### Code changes

| File | Change |
|---|---|
| `multitask/models/hierarchical.py` | New `pool_probabilities(stacked, strategy, axis)`: max, mean, min, noisy_or, lse. The model's pooled output uses it. |
| `multitask/config.py` | `noisy_or` and `lse` accepted as pool strategies. |
| `multitask/trainers/trainer.py` | `_derive_pooled_labels` uses the max of labels for `noisy_or` and `lse`. |
| `replication/test_no_prop_overlap.py` | `PooledOutputLossTrainer` gains `pool_strategy`, `technique_loss_weight` and `technique_class_weights`. New flags `--propaganda-pool {max,noisy_or,lse}` and `--technique-loss-weight λ`, both requiring `--pooled-propaganda-loss`. Using them with `--semi-supervised` is refused, because that code pools with its own max. Defaults leave behaviour unchanged. |
| `modal_replicate.py` | `replicate()` gains `propaganda_pool` and `technique_loss_weight`. New entrypoint `propaganda_pooling` and table `PROPAGANDA_POOLING_CONFIGS`. |
| `replication/test_pooled_output_loss.py` | 12 new tests (details below). |

### Tests

Run on Modal CPU with `modal run modal_app.py::smoke_tests --extra-args "replication/test_pooled_output_loss.py"`. The local Python setup couldn't import Keras.

- **All 19 tests in `test_pooled_output_loss.py` pass,** 12 of them new. The new tests check:
  - pooled values against hand calculations;
  - outputs stay finite and between 0 and 1 when scores are exactly 0 or 1;
  - the loss formula for noisy-OR, log-sum-exp and the per-technique term;
  - weight 0 leaves the loss unchanged;
  - every technique head gets gradient under the new options, and rows without Propaganda labels get none;
  - the per-technique term lands in the Propaganda task's loss;
  - bad settings are rejected;
  - pooled labels stay "any technique present".
- **The full suite run:** 251 passed and 8 failed. All 8 failures are in `tests/test_optimizers.py` (the Lion optimizer and learning-rate schedule wrappers), which this change doesn't touch.

### Runs

```
modal run --detach modal_replicate.py::propaganda_pooling
modal run modal_replicate.py::label_aware_report --baseline famo_consf1_m5 --prefixes famo_consf1_pnor,famo_consf1_plse,famo_consf1_ptech
```

- 3 options × 4 seeds (42, 1, 2, 3) = 12 runs, saved in `paper_replication_retune_famo/`. Launched 3 Nov at 22:47 CST; the last one finished 4 Nov at 02:04.

  | Run | Result saved (CST) |
  |---|---|
  | `famo_consf1_pnor` s1 / s2 / s3 / s42 | 4 Nov 00:21 / 00:23 / 00:25 / 00:37 |
  | `famo_consf1_plse` s42 / s3 / s2 / s1 | 4 Nov 00:47 / 01:12 / 01:23 / 01:32 |
  | `famo_consf1_ptech` s1 / s42 / s2 / s3 | 4 Nov 01:34 / 01:47 / 01:48 / 02:04 |
  | `famo_consf1_m5` (baseline) s2 / s3 / s1 / s42 | 2 Nov 17:23 / 17:25 / 17:30 / 17:32 |
- The epoch is picked by the "F1 p3" rule on validation (best validation F1, patience 3).
- All 12 runs finished. The four "FAIL ConnectionError" lines in the launch log came from the laptop losing its network connection while it waited. The runs themselves completed on Modal, and every option has all 4 seeds.

### Results: mean of the 10 Table-3 outputs

Mean ± standard deviation over 4 seeds. Δ is paired by seed with a 95% interval.

| Setup | Validation F1 | Test F1 | Test AUC | ΔF1 vs best | ΔAUC vs best | Paper (Table 3 F1) |
|---|---|---|---|---|---|---|
| **Max-pool, current best (`famo_consf1_m5`)** | 70.9 ± 0.6 | **69.9 ± 0.7** | 72.6 ± 0.2 | — | — | 66.74 |
| Max + per-technique loss (`ptech`) | 70.5 ± 0.5 | 69.6 ± 0.6 | 72.6 ± 0.1 | −0.3 [−0.9, +0.3] | −0.0 [−0.2, +0.2] | 66.74 |
| Log-sum-exp (`plse`) | 70.4 ± 1.2 | 69.4 ± 1.2 | 71.4 ± 0.7 | −0.5 [−1.6, +0.5] | **−1.2 [−2.2, −0.2]** | 66.74 |
| Noisy-OR (`pnor`) | 69.9 ± 0.5 | 68.9 ± 0.3 | 72.1 ± 0.3 | **−1.0 [−2.0, −0.0]** | −0.5 [−1.2, +0.1] | 66.74 |

Propaganda alone, paired against the current best:

| Option | Propaganda ΔF1 | Propaganda ΔAUC |
|---|---|---|
| Max + per-technique loss | +0.2 [−1.9, +2.2] | −0.1 [−0.6, +0.3] |
| Log-sum-exp | −1.0 [−3.3, +1.2] | −2.0 [−4.1, +0.1] |
| Noisy-OR | −0.8 [−2.8, +1.2] | −0.5 [−2.8, +1.8] |

### Results: test F1 per task

| Task | Paper (Table 3 F1) | Max-pool (best) | Log-sum-exp | Noisy-OR | Max + per-technique |
|---|---|---|---|---|---|
| Propaganda | 61.74 | 66.1 ± 1.4 | 65.1 ± 1.2 | 65.3 ± 0.4 | 66.3 ± 0.4 |
| Disagree/Agree | 71.38 | 64.6 ± 1.0 | 64.7 ± 3.8 | 61.9 ± 1.6 | 65.1 ± 2.7 |
| Emotion/Fact | 63.93 | 71.2 ± 1.1 | 70.4 ± 0.7 | 70.8 ± 1.3 | 70.3 ± 0.6 |
| Attacking/Respectful | 68.07 | 74.0 ± 0.4 | 72.4 ± 1.0 | 73.2 ± 1.0 | 73.4 ± 0.8 |
| Nasty/Nice | 73.69 | 77.1 ± 1.3 | 75.9 ± 2.0 | 76.2 ± 1.2 | 76.8 ± 1.1 |
| Personal/Audience | 65.69 | 69.4 ± 2.5 | 69.6 ± 1.7 | 68.0 ± 1.6 | 67.7 ± 0.6 |
| Defeater/Undercutter | 55.65 | 59.7 ± 2.1 | 58.2 ± 1.4 | 59.0 ± 0.7 | 60.2 ± 2.1 |
| Negotiate/Attack | 64.81 | 71.6 ± 0.3 | 71.0 ± 1.0 | 71.5 ± 0.4 | 71.4 ± 0.5 |
| Questioning/Asserting | 63.23 | 62.9 ± 3.9 | 64.7 ± 2.4 | 64.2 ± 1.8 | 63.9 ± 1.8 |
| Argument Quality | 79.17 | 82.8 ± 4.1 | 81.8 ± 4.3 | 79.2 ± 3.3 | 81.3 ± 1.8 |

### How spread out the scores are

This is the standard deviation of each head's validation scores, averaged over seeds. A big drop would mean a head had collapsed toward one value; none did. Log-sum-exp spreads every head's scores 10–35% wider. The other two barely change anything.

| Task | Max-pool | Log-sum-exp | Noisy-OR | Max + per-technique |
|---|---|---|---|---|
| Propaganda | 0.253 | 0.309 | 0.272 | 0.248 |
| Disagree/Agree | 0.240 | 0.283 | 0.249 | 0.243 |
| Personal/Audience | 0.241 | 0.305 | 0.244 | 0.239 |
| Questioning/Asserting | 0.181 | 0.248 | 0.199 | 0.186 |
| Argument Quality | 0.314 | 0.353 | 0.319 | 0.315 |

### Did the technique heads start to mean something?

Computed on the test Propaganda rows, averaged over 4 seeds.

| Setup | Rows where the most common top head is the top head | Mean per-technique AUC | Heads with AUC ≥ 0.6 |
|---|---|---|---|
| Max-pool | 93.2% | 0.578 | 7.0 of 18 |
| Noisy-OR | 76.4% | 0.599 | 8.2 of 18 |
| Log-sum-exp | 56.3% | 0.588 | 7.2 of 18 |
| Max + per-technique | 51.6% | 0.588 | 8.5 of 18 |

Winner-take-all eased under all three options, but each head got only slightly better at spotting its own technique. Even under the max-pool, the heads that get no direct training rank sentences somewhat, because they sit on top of shared layers that do get trained.

### Decision

The rule was to pick the strongest option by validation F1, so the test set plays no part in the choice, and keep it only if it beats 69.9 on test.

- **Max + per-technique loss** was strongest on validation (70.5). On test it ties the current best: −0.3 F1, with an interval that includes 0.
- **Log-sum-exp** is worse on AUC.
- **Noisy-OR** is worse on F1.

**The max-pool stays.** It's the default, so the best setup needs no change.

---

## 3. A different loss for each data source?

*4 Nov, ~01:20–02:00 CST, while the runs were still training.*

The idea of giving each source a loss that fits its labels, then combining them, is sound and is already partly in place:

- Propaganda has its own pooled loss.
- Every head has class weights from its own label balance.
- FAMO combines the losses. It balances them by how fast each one is falling relative to its own size, not by raw size, so mixing different kinds of loss is safe.

Per-source changes already tried, with no gain:

- **Disagree/Agree imbalance:** focal loss and logit adjustment did no better than plain class weights.
- **Propaganda zero-fill** (treating IAC and IBM rows as "no technique"): Propaganda F1 dropped 4–6 points.

**Untried, and the suggested next step:** losses that use how the IAC and IBM labels were made.

- **IAC.** Each label is a −5 to 5 rating cut at 0, so a post rated +0.2 and one rated +4.8 both become "1".
  - **Soft labels:** train toward a target such as 0.52 near the midpoint.
  - **Distance weighting:** keep 0/1 labels but give posts far from 0 more weight.
  - The Propaganda rows carry rule-made IAC labels with no rating behind them, so they would stay hard labels or be left out. Either way the loss would separate real human ratings from the rule labels.
- **IBM.** The 0–1 quality score is cut at 0.5, and 94% of rows come out "good". Soft labels or a ranking loss would fit; the source dataset (IBM-Rank-30k) was built for ranking.

Two caveats:

- Weighted F1 is scored on yes/no answers at a cut-off, so the gain from soft or ranking losses may show up mostly in AUC.
- This is an extension, not part of the replication: the paper trained on the 0/1 labels.

**Status:** proposed, not started. First check that the original ratings exist in the CSVs for every IAC and IBM row.

---

## 4. Is the paper's Table 3 on validation or test?

*4 Nov, 10:25–10:27 CST.*

**The paper doesn't say.** I read both the arXiv preprint (2307.01401) and the published ACL ArgMining 2024 version (pages 46–58). Quotes from the published version:

- **Split:** "We use 80% of the data for training and set aside 10% each for validation and test sets."
- **Hyperparameter choice:** "All training hyperparameters are tuned through a standard grid search over 72 sets of hyperparameters and selected based on validation F1 score."
- **Early stopping:** "stop training after two epochs without a decrease in loss." No split is named.
- **Thresholds:** "threshold tuning, maximizing true positive rate while minimizing false positive rate." No split is named.
- **Table 3 caption:** "Weighted F1 scores comparing baselines, single-task, and multi-task models, as well as multi-task model with ablated layers." No split is named.

The word "test" appears only in the split sentence and in "We test this conjecture."

The two versions differ in some details, but the multi-task Table 3 numbers are the same. The ACL version adds single-task and ablation columns and reports 32,536 trainable parameters, against 17,024 in the arXiv version.

**What this means for our comparisons:**

- **We assume test.** That is the usual practice, and the paper used validation to pick its settings.
- **If Table 3 is actually validation,** the paper's numbers come from the best of 72 settings, picked on that same score, so they would be flattering. The fair comparison would then be our validation F1.
- **Our conclusion holds either way.** The current best scores 70.9 on validation and 69.9 on test, against the paper's mean of 66.74.
- **From now on,** results tables show our validation F1 beside test F1, with a note that the paper's split is unstated.

---

## Open items

1. **Soft labels or distance weighting for IAC and IBM** (section 3). Not started; check the CSV ratings first.
2. **Fix or remove the 8 failing tests in `tests/test_optimizers.py`.** They're unrelated to this work and predate it.
3. **The local Python setup can't import Keras,** so tests currently run only on Modal.
