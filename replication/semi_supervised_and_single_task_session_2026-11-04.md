# Semi-supervised "foreign cell" loss, single-task rerun and the paper's compute argument (3–4 Nov 2026)

This note records one working session with Claude Code. It covers:

1. Whether a head can learn from rows that have no label for its task, and what the literature calls this.
2. A semi-supervised loss (a **rate penalty** and a **confidence penalty**), built, tested and run on Modal (7 training runs).
3. The ablation design that would show whether semi-supervised learning helps (discussed, not run).
4. The single-task baselines rerun at the current tuned settings (40 training runs).
5. Whether the paper's case for multi-task learning is about compute.

**Short version:**
- The semi-supervised loss does what it is told: it moves the average scores to the target rates. But it improves no task. The best version scored 67.3 mean test F1 against 68.2 for the plain baseline, a gap within seed noise. It was not adopted.
- At the tuned settings, single-task models average **70.8** test F1. Every multi-task setup is still below that. The best, `famo_consf1_m5` at 69.9, is −0.8 F1 behind [95% interval −1.7, −0.0].
- The paper's main claim is accuracy. Lower compute is a secondary point. Our data supports multi-task learning as a trade-off (one model, 10× cheaper to run), not as an accuracy gain.

All times are **US Central (CST, UTC−6)**, taken from the session transcript. UTC is given in brackets.

---

## Terms used in this note

- **Head:** the small output layer that gives one task's score, a probability between 0 and 1.
- **Source:** which dataset a row comes from.
  - **Propaganda:** news sentences, 61,909 training rows.
  - **IAC:** forum posts, 61,069 training rows.
  - **IBM:** arguments, 96,036 training rows.
  - Each source has labels only for its own tasks.
- **Propaganda output (pooled):** the Propaganda score is the highest of the 18 technique-head scores (the "max-pool"). The loss is computed on that pooled score.
- **Rule-labelled rows:** some Propaganda sentences carry IAC labels that a fixed rule made from their techniques, not from people. For example, Name_Calling sets attacking, nasty, emotion and disagree to 1. These labels are always 1.
- **Foreign cell:** a head on a row from a source that has no label for that head. Examples: the Propaganda head on an IAC forum post, or the Emotion head on an IBM argument.
- **Rate penalty:** for each chosen (head, source) pair, the average score over that pair's foreign cells in a batch should match a target share. An example target is "about 30% of IAC posts contain propaganda". The penalty is the binary KL divergence between the target share and the batch average. It is 0 when they match. It never labels any single row.
- **Confidence penalty (entropy):** pushes each foreign-cell score toward 0 or 1, away from 0.5. Binary entropy is highest at 0.5 and 0 at 0 or 1.
- **F1:** weighted F1 (×100), the paper's metric. Each task's yes/no cut-off is tuned on validation.
- **AUC:** ROC AUC (×100). It measures ranking quality without a cut-off. 50 means no signal.
- **F1 p3:** early stopping that keeps the epoch with the best validation F1, and stops after 3 epochs without improvement.
- **Plain baseline:** the pooled-loss multi-task model with dropout 0, the wide architecture, batch 256, learning rate 1e-4 and F1 p3. It averages 68.1 over 4 seeds and 68.2 over seeds 42, 1, 2.
- **FAMO:** a loss-balancing method that reweights the task losses by how fast each is improving.
- **`famo_fixed`, `famo_consf1_m5`:** the stronger FAMO variants. `famo_consf1_m5` (69.9) is the best multi-task setup so far. It adds a consistency loss on a copy of each sentence with 5% of the words hidden.
- **Paired Δ:** the difference from the baseline worked out seed by seed, then averaged, with a 95% t interval. If the interval excludes 0, the difference counts as real.
- **Δm:** the mean relative change against single-task, in %, averaged over the 10 outputs.
- **Modal:** the cloud GPU service where every run happens.

---

## Timeline

| When (CST) | UTC | What happened |
|---|---|---|
| Tue 3 Nov, 21:15 | 03:15 | You asked: is there a general loss that lets head A learn from rows without an A label? |
| 22:24 | 04:24 | You pasted a 10-step plan and asked me to build it |
| 22:34 | 04:34 | Tests run on Modal (12 new tests, all pass) |
| 22:38 | 04:38 | 1-seed sanity run `ssl_re1_s42` launched |
| 22:49 | 04:49 | You asked for a detailed explanation of what was running |
| 22:52 | 04:52 | Sanity run finished (7 epochs, kept epoch 3). You chose to stop the full grid |
| 23:52 | 05:52 | You asked: is it built on FAMO? What about the rate penalty alone? Could the rate penalty be an adaptive-weight task? |
| Wed 4 Nov, 00:12 | 06:12 | You said: "set up and run step 1" (rate penalty only, without the news pairs) |
| 00:13 | 06:13 | 6 runs launched (`ssl_nonews_r0.1`, `ssl_nonews_r1` × seeds 42, 1, 2) |
| 01:02 | 07:02 | All 6 finished. Report run (report bug fixed: 1-D arrays) |
| 01:11 | 07:11 | You asked whether there is an ablation study that shows if semi-supervised learning helps |
| 01:12 | 07:12 | You pasted the three-control design. I explained it and offered to build it. Not taken up |
| 11:45 | 17:45 | You asked me to rerun the single-task baselines with the new settings |
| 11:46 | 17:46 | 40 single-task runs launched (10 outputs × seeds 42, 1, 2, 3) |
| 12:04 | 18:04 | All 40 finished. Report run |
| 12:12 | 18:12 | You asked whether the paper's argument is about compute. I checked the paper and counted the rows processed |
| 16:30 | 22:30 | You asked for this note |

---

## 1. Can a head learn from rows that lack its label? (21:15 CST, 3 Nov)

**Your question:** when head A scores a row with no A label, is there a general loss that still teaches A something, with no word guessing and no masking? Have people done this?

**Answer:** yes. Several research lines do this without making up a label for any single row:

- **Multi-label learning with missing labels:** the standard choice is to drop missing cells from the loss, which is what we already do. The other methods below add a signal on top of that.
- **Positive-unlabelled (PU) learning**, for example nnPU: this uses a known prior rate of positives among the unlabelled rows.
- **Expectation / posterior regularisation** (Mann & McCallum; Ganchev et al.): the model's *average* prediction on unlabelled data must match a known rate. This became the **rate penalty** (called "option 2" in your plan).
- **Entropy minimisation** (Grandvalet & Bengio): unlabelled predictions are pushed to be confident. This became the **confidence penalty** ("option 4").
- **Semantic loss / logic constraints:** rules between outputs are enforced ("option 6", not built).

I recommended the batch-level rate constraint. It is the only option that uses a real fact about the data (a rough share of positives) without labelling any row.

---

## 2. The semi-supervised loss: what was built (22:24–22:34 CST, 3 Nov)

### Code

- **`replication/test_no_prop_overlap.py`:**
  - `parse_ssl_rates`: reads `"head@source=rate,…"`.
  - `semi_supervised_terms`: works out the source of each row from its labels. Technique labels mean Propaganda, a WA_bi label means IBM, neither means IAC. It then finds the foreign cells and computes the rate KL and the entropy.
  - `SemiSupervisedModel`: adds the terms to the training loss. The reported and validation loss stays supervised only, so it can be compared with the baseline.
  - `SemiSupervisedTrainer` (extends `PooledOutputLossTrainer`): the confidence penalty stays off for the first 2 epochs.
  - Flags: `--semi-supervised`, `--ssl-rates`, `--ssl-rate-weight` (0.1), `--ssl-entropy-weight` (0), `--ssl-entropy-start-epoch` (2). The flags are refused with FAMO, MLM, teachers or zero-fill, so they can't be combined by accident.
- **`replication/test_semi_supervised.py`:** 12 new tests, all passing (73/73 in the suite). They check that:
  - the extra loss is 0 when there are no foreign cells;
  - the source counts are right;
  - the KL and entropy match hand calculations;
  - only foreign cells get gradient;
  - entropy waits for its start step;
  - with every label present, a step equals a normal multi-task step;
  - the rate term pulls the mean toward the target.
- **`modal_replicate.py`:** entry points `semi_supervised`, `semi_supervised_check --name`, `semi_supervised_report`. Output folder `paper_replication_ssl`.

### Target rates

These are my rough reading of 40 training rows per source, **not** proper annotation:

| Head | on IAC rows | on IBM rows | on Propaganda rows (not rule-labelled) |
|---|---|---|---|
| Propaganda | 0.30 | 0.10 | — |
| Emotion/Fact | — | 0.15 | 0.12 |
| Nasty/Nice | — | 0.03 | 0.05 |
| Attacking/Respectful | — | 0.05 | 0.07 |

Disagree/Agree on news was skipped, as your plan said. The rule-labelled Propaganda rows were kept as in the baseline. The report adds an "IAC rows only" F1 so their effect can be seen.

---

## 3. Sanity run `ssl_re1_s42` (22:38–22:52 CST, 3 Nov)

Settings: rate weight 1, confidence weight 1 from epoch 3, all 8 pairs, seed 42, on the plain model.

**Per epoch:**

| Epoch | Rate loss (train) | Entropy (train) | Confidence weight | Val F1 | Test F1 |
|---|---|---|---|---|---|
| 1 | 0.0550 | 0.398 | 0 | 62.6 | 61.6 |
| 2 | 0.0028 | 0.290 | 0 | 65.1 | 64.3 |
| 3 (kept) | 0.0123 | 0.046 | 1 | 65.5 | 65.1 |
| 4 | 0.0098 | 0.028 | 1 | 64.5 | 63.4 |
| 7 | 0.0089 | 0.020 | 1 | 64.1 | 63.6 |

**What happened:**
- The loss mechanics worked. The rate loss fell and the training averages reached every target.
- Test F1 was **65.1 against 68.0** for the baseline on the same seed, and it was lower at every epoch, including epochs 1–2 before the confidence penalty was on.
- The worst-hit tasks were Emotion (71.0 → 57.3), Disagree (66.7 → 55.6) and Attacking (70.4 → 64.2).

**Why:** the rate targets on news rows (5–12% "yes") clash with the rule labels (always "yes" on Name_Calling sentences and similar). The model can't tell the two groups of news sentences apart. So it pushed the IAC heads toward 0 on **all** news: on rule-labelled rows, the share scored "yes" fell from 83% to 20–49%, and the median score was 0.00. This is the same "source detector" failure that zero-fill caused earlier. The confidence penalty also made foreign-cell scores hard 0/1 within one epoch (entropy 0.29 → 0.05).

**Your decision (22:52):** stop the full 4-setting grid.

---

## 4. Your follow-up questions (23:52 CST, 3 Nov)

- **"Is this built on the best FAMO model?"** No. It was built on the plain pooled model, so the semi-supervised effect could be measured on its own.
- **"What if we keep only the rate penalty?"** That became step 1, below. The news pairs were also dropped to remove the clash with the rule labels.
- **"What if the rate penalty were a task with adaptive weights?"** Possible: it could be an 11th task under FAMO, or gated by gradient cosine. I planned it last for two reasons. First, the rate loss drops to about 0 very quickly, and FAMO's update uses the log of loss ratios, which becomes unstable near 0. Second, it is only worth doing if the fixed-weight version helps first.

---

## 5. Step 1: rate penalty only, no news pairs (00:13–01:02 CST, 4 Nov)

Settings: 5 pairs (Propaganda@IAC, Propaganda@IBM, and Emotion/Nasty/Attacking@IBM), no confidence penalty, rate weight 0.1 or 1, seeds 42, 1, 2.

### Mean over the 10 outputs (test, F1 p3)

| Setup | n | Val F1 | Test F1 | Test AUC | Paper (Table 3 F1) | ΔF1 vs plain | ΔAUC vs plain |
|---|---|---|---|---|---|---|---|
| plain (baseline) | 3 | 69.2 ± 0.9 | 68.2 ± 1.3 | 69.3 ± 0.6 | 66.74 | | |
| ssl_re1 (sanity run) | 1 | 65.5 | 65.1 | 63.6 | 66.74 | −2.8 (n=1) | −5.1 (n=1) |
| ssl_nonews_r0.1 | 3 | 68.7 ± 0.5 | 67.3 ± 0.4 | 68.7 ± 0.3 | 66.74 | −0.8 [−3.7, +2.0] | −0.6 [−2.1, +0.9] |
| ssl_nonews_r1 | 3 | 68.2 ± 1.1 | 66.9 ± 1.3 | 68.1 ± 0.5 | 66.74 | −1.3 [−6.8, +4.2] | −1.2 [−2.9, +0.5] |

### Per-task test F1

| Task | Paper (Table 3 F1) | plain | ssl_re1 | ssl_nonews_r0.1 | ssl_nonews_r1 |
|---|---|---|---|---|---|
| Propaganda | 61.74 | 65.4 ± 1.6 | 63.9 | 64.1 ± 1.0 | 63.5 ± 1.1 |
| Disagree/Agree | 71.38 | 63.6 ± 6.6 | 55.6 | 63.0 ± 6.4 | 58.0 ± 6.3 |
| Emotion/Fact | 63.93 | 69.7 ± 2.0 | 57.3 | 66.9 ± 1.3 | 66.3 ± 1.3 |
| Attacking/Respectful | 68.07 | 71.0 ± 0.7 | 64.2 | 69.8 ± 0.5 | 70.1 ± 1.0 |
| Nasty/Nice | 73.69 | 74.6 ± 2.5 | 69.0 | 73.6 ± 1.1 | 72.9 ± 0.5 |
| Personal/Audience | 65.69 | 65.6 ± 2.7 | 69.3 | 65.7 ± 2.1 | 64.0 ± 5.0 |
| Defeater/Undercutter | 55.65 | 58.1 ± 2.5 | 56.8 | 57.4 ± 1.9 | 59.6 ± 1.3 |
| Negotiate/Attack | 64.81 | 69.6 ± 0.8 | 68.7 | 68.5 ± 0.6 | 68.5 ± 0.2 |
| Questioning/Asserting | 63.23 | 60.4 ± 2.0 | 62.3 | 60.9 ± 2.2 | 62.8 ± 1.8 |
| Argument Quality | 79.17 | 83.7 ± 1.1 | 84.3 | 83.3 ± 3.2 | 82.9 ± 3.4 |

### IAC heads scored on IAC test rows only (rule-labelled news rows removed)

| Task | Paper (Table 3 F1) | plain | ssl_nonews_r0.1 | ssl_nonews_r1 |
|---|---|---|---|---|
| Disagree/Agree | 71.38 | 60.9 ± 7.3 | 61.0 ± 7.2 | 55.8 ± 6.8 |
| Emotion/Fact | 63.93 | 63.5 ± 2.7 | 62.8 ± 2.4 | 62.2 ± 2.5 |
| Attacking/Respectful | 68.07 | 70.2 ± 0.9 | 70.6 ± 1.0 | 71.3 ± 0.8 |
| Nasty/Nice | 73.69 | 74.7 ± 2.5 | 75.3 ± 1.1 | 74.8 ± 0.9 |
| Personal/Audience | 65.69 | 67.0 ± 2.7 | 66.6 ± 2.1 | 65.0 ± 5.3 |
| Defeater/Undercutter | 55.65 | 56.1 ± 2.0 | 55.7 ± 1.8 | 57.7 ± 1.3 |
| Negotiate/Attack | 64.81 | 66.9 ± 1.1 | 67.2 ± 0.7 | 67.2 ± 0.6 |
| Questioning/Asserting | 63.23 | 62.7 ± 1.1 | 61.6 ± 1.9 | 63.0 ± 0.9 |

### Did the averages move to the targets? (test foreign cells, mean over seeds)

| Pair | Target | plain | ssl_re1 | ssl_nonews_r0.1 | ssl_nonews_r1 |
|---|---|---|---|---|---|
| Propaganda on IAC | 0.30 | 0.541 | 0.337 | 0.296 | 0.312 |
| Propaganda on IBM | 0.10 | 0.497 | 0.133 | 0.176 | 0.164 |
| Emotion on IBM | 0.15 | 0.483 | 0.216 | 0.209 | 0.208 |
| Nasty on IBM | 0.03 | 0.420 | 0.085 | 0.070 | 0.066 |
| Attacking on IBM | 0.05 | 0.453 | 0.101 | 0.092 | 0.090 |
| Emotion on news (not targeted in nonews) | 0.12 | 0.769 | 0.252 | 0.682 | 0.661 |

### What it means

- The clash with the rule labels is gone: IAC heads on news rows are back to about 0.6–0.7.
- The targets carry over to test, so the penalty does what it is told.
- No task improves. All differences are small and within seed noise.
- Propaganda F1 is slightly lower (−1.3 / −1.9, not significant). With fewer high Propaganda scores on IAC and IBM rows, the tuned cut-off lets through fewer techniques. For example, Loaded_Language recall fell from 52% to 48% and 46%, and Appeal_to_Authority from 29% to 22%.

**Verdict:** not adopted. The FAMO follow-up depended on a gain here, so it was not run.

---

## 6. Is there an ablation that shows whether semi-supervised learning helps? (01:11–01:12 CST, 4 Nov)

I pointed to Oliver et al. (2018), *Realistic Evaluation of Deep Semi-Supervised Learning Algorithms*. The standard checks are:
- **Label-budget curves:** train with 10%, 25% and 100% of the labels, with and without the semi-supervised loss. Semi-supervised learning usually helps most when labels are scarce.
- **Distribution mismatch:** check whether the unlabelled data comes from the same kind of text as the labelled data.

You then pasted the design you meant, three controls for the consistency loss:

1. **Augmentation only:** the masked copy of each sentence is scored against the *real* labels (the earlier `mlab` runs). This tests whether any gain comes from seeing noisier inputs.
2. **Labelled cells only:** consistency applied only where a label exists.
3. **Unlabelled cells only:** consistency applied only to the empty cells.

If the gain truly comes from the empty label cells, then (3) should match the full method, and (1) and (2) should not.

I offered to build a `--consistency-cells all|labelled|unlabelled` switch and run 3 controls × 4 seeds (12 runs) under FAMO, with a 5% mask and no MLM. The comparison would be against `famo` (69.0) and `famo_consf1_m5` (69.9). **This was not started.**

---

## 7. Single-task baselines at the tuned settings (11:45–12:04 CST, 4 Nov)

**Why:** the old single-task numbers (mean 68.9) used learning rate 3e-4 and stopping on validation loss. Every current multi-task run uses learning rate 1e-4 and stopping on validation F1. The comparison was therefore unfair.

**Setup:** `modal_replicate.py::single_task_retune`. Each run uses one output (`tasks=<output>`), keeps only rows that have that label, and otherwise uses the multi-task settings. That makes 10 outputs × seeds 42, 1, 2, 3 = 40 runs, in folder `paper_replication_single_retune`. The report is `single_task_retune_report`.

### Test F1 (mean ± sd over 4 seeds)

| Task | Paper single | Paper multi (Table 3 F1) | **Single (ours)** | plain MTL | famo | famo_fixed | famo_consf1_m5 |
|---|---|---|---|---|---|---|---|
| Propaganda | 63.07 | 61.74 | **67.6 ± 0.3** | 64.8 ± 1.7 | 64.4 ± 1.4 | 66.0 ± 0.4 | 66.1 ± 1.4 |
| Disagree/Agree | 71.15 | 71.38 | **69.1 ± 1.9** | 65.1 ± 6.1 | 66.9 ± 2.7 | 62.4 ± 4.7 | 64.6 ± 1.0 |
| Emotion/Fact | 68.11 | 63.93 | **72.6 ± 0.8** | 69.8 ± 1.6 | 70.7 ± 1.2 | 71.4 ± 0.6 | 71.2 ± 1.1 |
| Attacking/Respectful | 67.46 | 68.07 | **72.4 ± 0.9** | 70.8 ± 0.8 | 72.7 ± 2.0 | 73.4 ± 0.4 | 74.0 ± 0.4 |
| Nasty/Nice | 66.90 | 73.69 | **76.1 ± 1.0** | 74.2 ± 2.2 | 76.0 ± 1.9 | 76.7 ± 0.9 | 77.1 ± 1.3 |
| Personal/Audience | 63.25 | 65.69 | **69.2 ± 0.5** | 66.6 ± 3.0 | 64.9 ± 8.3 | 68.6 ± 2.7 | 69.4 ± 2.5 |
| Defeater/Undercutter | 45.97 | 55.65 | **60.0 ± 1.3** | 57.7 ± 2.2 | 58.1 ± 3.0 | 60.1 ± 2.3 | 59.7 ± 2.1 |
| Negotiate/Attack | 64.76 | 64.81 | **69.4 ± 0.9** | 69.4 ± 0.8 | 70.5 ± 1.6 | 71.2 ± 0.3 | 71.6 ± 0.3 |
| Questioning/Asserting | 59.61 | 63.23 | **67.6 ± 1.1** | 59.9 ± 1.9 | 63.6 ± 1.8 | 64.3 ± 1.5 | 62.9 ± 3.9 |
| Argument Quality | 80.93 | 79.17 | **83.7 ± 1.6** | 82.9 ± 1.9 | 82.0 ± 4.0 | 81.2 ± 4.4 | 82.8 ± 4.1 |
| **Mean** | 65.12 | 66.74 | **70.8 ± 0.2** | 68.1 ± 1.0 | 69.0 ± 0.9 | 69.5 ± 0.7 | 69.9 ± 0.7 |

### Multi-task minus single-task (paired by seed, 95% t interval; negative means multi-task is worse)

| Task | plain MTL | famo | famo_fixed | famo_consf1_m5 |
|---|---|---|---|---|
| Propaganda | −2.8 [−5.7, +0.1] | −3.2 [−5.2, −1.1] | −1.5 [−2.3, −0.7] | −1.5 [−3.8, +0.8] |
| Disagree/Agree | −4.0 [−13.5, +5.4] | −2.2 [−8.8, +4.4] | −6.6 [−16.7, +3.4] | −4.5 [−7.1, −1.9] |
| Emotion/Fact | −2.8 [−6.5, +1.0] | −1.8 [−4.4, +0.8] | −1.2 [−3.2, +0.9] | −1.4 [−3.2, +0.4] |
| Attacking/Respectful | −1.6 [−3.7, +0.4] | +0.2 [−2.3, +2.8] | +1.0 [−1.0, +3.1] | **+1.6 [+0.6, +2.5]** |
| Nasty/Nice | −1.9 [−5.9, +2.1] | −0.1 [−3.1, +2.8] | +0.6 [−2.4, +3.7] | +1.0 [−1.4, +3.4] |
| Personal/Audience | −2.6 [−6.9, +1.7] | −4.3 [−17.5, +8.9] | −0.6 [−4.5, +3.4] | +0.2 [−3.9, +4.4] |
| Defeater/Undercutter | −2.3 [−6.1, +1.6] | −1.9 [−7.0, +3.3] | +0.1 [−4.7, +5.0] | −0.2 [−4.9, +4.5] |
| Negotiate/Attack | −0.0 [−2.3, +2.3] | +1.1 [−1.2, +3.4] | **+1.8 [+0.4, +3.2]** | **+2.2 [+0.8, +3.5]** |
| Questioning/Asserting | −7.8 [−11.7, −3.8] | −4.0 [−7.2, −0.8] | −3.4 [−4.9, −1.9] | −4.7 [−12.5, +3.1] |
| Argument Quality | −0.8 [−4.7, +3.0] | −1.7 [−6.8, +3.4] | −2.6 [−10.9, +5.8] | −0.9 [−8.0, +6.1] |
| **Mean** | −2.7 [−4.2, −1.1] | −1.8 [−3.4, −0.2] | −1.2 [−2.3, −0.2] | −0.8 [−1.7, −0.0] |
| **Δm** | −3.8% [−6.1, −1.6] | −2.6% [−4.8, −0.3] | −1.7% [−3.3, −0.2] | −1.2% [−2.3, −0.1] |

For comparison, the paper reports mean single 65.12, multi 66.74, Δ +1.62 and Δm +3.2%.

### What it means

- With fair settings, single-task models are 1.9 points better than before (70.8 against 68.9).
- Every multi-task setup is still below single-task. Better methods shrink the gap: −2.7 for plain MTL, −0.8 for `famo_consf1_m5`.
- Multi-task learning wins on **Attacking** (+1.6) and **Negotiate** (+2.2). It loses on **Disagree** (−4.5), **Questioning** (−3.4 to −7.8) and **Propaganda** (−1.5 to −3.2).
- These are 40 comparisons with no multiple-comparison correction, so treat the single-task wins and losses as leads, not firm results.
- Single-task runs pick early epochs (1.2–4.0 on average).

---

## 8. Is the paper's argument about compute? (12:12 CST, 4 Nov)

**What the paper says** (arXiv 2307.01401):
- The title and introduction argue for **accuracy**: training on related tasks together should help each task.
- Efficiency is a secondary point. Section 5.3 trains for 1 epoch on 5–40% subsamples and reports at least 31% less training time at "comparable performance".

**What our runs show:** I counted the training rows processed (rows per epoch × epochs actually run) from the saved run files.

| Setup | Rows processed, whole run | Rows up to the kept epoch | Extra work per step |
|---|---|---|---|
| 10 single-task models (total) | 2.92M | 1.28M | — |
| plain MTL | 1.81M | 1.20M | — |
| famo | 1.42M | 0.93M | about ⅓ more |
| famo_consf1_m5 | 1.04M | 0.38M | about 2× |

- The training-compute saving is modest, and FAMO and the consistency loss use up most of it.
- The clear advantages of multi-task learning are:
  - **10× fewer encoder passes at prediction time**, because one model serves all 10 outputs;
  - one model to store and deploy;
  - one hyperparameter search instead of ten.

**Suggested framing for the thesis:** multi-task learning as a **trade-off**. It costs 0.8–2.7 F1 and gives one model that is 10× cheaper to run. It should not be presented as "multi-task learning improves performance".

---

## Files changed or created in this session

- `multitask-forked/replication/test_no_prop_overlap.py`: semi-supervised loss, model, trainer and flags.
- `multitask-forked/replication/test_semi_supervised.py`: new file, 12 tests.
- `multitask-forked/modal_replicate.py`:
  - semi-supervised launch, check and report entry points (`semi_supervised`, `semi_supervised_check`, `semi_supervised_report`);
  - single-task rerun and report (`single_task_retune`, `single_task_retune_report`);
  - the tests were added to `masked_lm_tests`;
  - report fix: `.reshape(-1)` for 1-D prediction arrays.
- Modal output folders: `paper_replication_ssl` (7 runs) and `paper_replication_single_retune` (40 runs).
- Claude memory notes: `ssl-rate-entropy-study-2026-11.md` and `single-task-retune-2026-11.md`.

## Open items (not started)

- The three-control consistency ablation (12 runs), from section 6.
- Label-budget curves with and without the semi-supervised loss (about 9 runs).
- A leave-one-task-out / task-grouping follow-up. It would test whether removing Disagree, Questioning or Propaganda from the joint model helps the other tasks, given that those three lose to single-task and Attacking and Negotiate gain.
- Lesson for later: don't put rate targets on heads whose labels in that source are made by a rule.
