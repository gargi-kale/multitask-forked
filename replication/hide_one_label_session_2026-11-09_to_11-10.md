# Hide-one-label study, loss weighting and FAMO (6–10 Nov 2026)

This note records the hide-one-label runs for the paper's baseline table, plus the comparisons and checks that came out of them:

1. Hide-one-label runs with plain training (8 hidden labels, 2 datasets), against full-label references.
2. Mean F1 across all labels, with one label hidden.
3. With vs without branches (IAC + hate speech).
4. With vs without the rule-made IAC labels (main data).
5. How the task weights work, and a bug that hit the run hiding Argument Quality.
6. One run with no task weights and no class weights.
7. The same hide-one-label runs with FAMO + SSL (`famo_consf1_m5`).

**Short version**

- Hiding one IAC or hate speech label costs almost nothing. A linear probe on the trained model still reads the hidden label to within about 0–2 AUC of a model that saw it, and the other labels barely move.
- Argument Quality (IBM's only label) is the exception: it loses 5–8 AUC when hidden. The plain run's −3 AUC drop on the *other* labels was mostly a weighting bug. Under FAMO, which doesn't use those weights, the other labels don't change at all (+0.1 AUC).
- FAMO + SSL beats plain on the main data (69.4 vs 67.9 mean test F1; the paper reports 66.7), and the two tie on IAC + hate speech.
- The consistency SSL does not teach a hidden label's own output. It lands on a random direction (AUC 23–36 for 6 of 8 labels, below chance).
- Branches and removing the rule labels change nothing that matters.
- A loss with no task weights and no class weights was slightly *better* than the paper's weighting (68.4 vs 67.9 F1).
- Every run is a single seed (42), so differences under about 1–2 points may be noise.

---

## Timeline

All times are US Central (CST). Run finish times come from the save times of the result files on the Modal volume. Launch times come from Modal's app records. A "~" means the time is estimated from nearby events, not recorded directly.

| Date and time | What happened |
|---|---|
| 6 Nov, 19:35–20:50 | Hide-one-label code written: the probe (`hidden_label_probe.py`), the `--hide-task` / `--probe-tasks` / `--no-rule-labels` flags, branches in the hate speech runner, and the tests (`test_hide_one_label.py`). Full suite passed on Modal. |
| 6 Nov, 20:19 | Pilot finished: IAC + hate speech, plain, violence hidden (31 min, $0.51). |
| 6 Nov, 20:45 | Pilot finished: main data, plain, Nasty/Nice hidden (58 min, $1.05). |
| 9 Nov, evening | Decided: 1 seed, 3 more randomly chosen labels per dataset (random generator seed 20261109), **plain training only**. The user's reasoning: "hiding labels is itself a form of SSL". An SSL + FAMO joint launch was stopped before it ran. |
| 9 Nov, ~21:45 | 6 plain hide-one runs launched on L4 GPUs. |
| 9 Nov, 22:19–22:26 | IAC + hate speech hide-one runs finished (Attacking/Respectful, dehumanize, attack_defend). |
| 9 Nov, ~22:45 | 2 full-label plain references launched (user approved). |
| 9 Nov, 22:42–23:20 | Main-data hide-one runs finished (Negotiate/Attack, Argument Quality, Attacking/Respectful). |
| 9 Nov, 23:17 | IAC + hate speech reference finished. |
| 9 Nov, 23:48 | Main-data reference finished. |
| 10 Nov, ~08:00 | Hide-one results analysed (sections 1–2). |
| 10 Nov, ~09:00–10:00 | Branches comparison and rule-label comparison (sections 3–4), using existing runs. No new training. |
| 10 Nov, 10:07 | 10 FAMO + SSL runs launched (8 hide-one + 2 references). They waited about 20–50 min for free L4 GPUs. |
| 10 Nov, ~11:00–11:30 | Task weighting explained; Argument Quality weighting bug found (section 5). |
| 10 Nov, 11:29–12:34 | IAC + hate speech FAMO runs finished (4 hide-one at 11:29–12:16, reference at 12:34). |
| 10 Nov, 11:35 | Unweighted run launched (`--unweighted-loss`, section 6). |
| 10 Nov, 12:08 | Main-data FAMO reference finished. |
| 10 Nov, 12:42 | Unweighted run finished. |
| 10 Nov, 13:13–15:13 | Main-data FAMO hide-one runs finished. |
| 10 Nov, ~17:45 | FAMO runs analysed (section 7). |
| 10 Nov, ~19:00 | This note written. |

---

## Terms used

- **Main data**: IAC + Propaganda + IBM, with no overlap between datasets.
- **Rule labels**: 45,111 IAC labels on Propaganda rows that were made by rule from the propaganda technique tags, not by annotators. "No rule labels" means they are removed from train, validation and test.
- **Hide one label**: one label is removed from every training row. Validation and test keep it, so it can still be scored.
- **Plain**: fixed loss weights as in the paper (section 5); no FAMO, no SSL.
- **FAMO + SSL** (`famo_consf1_m5`): FAMO adjusts each label's loss weight during training. The SSL part also asks the model to give the same answer on a text and on a copy with 5% of its words masked, using only the 10% most confident predictions.
- **F1**: the library's weighted F1 on test. **AUC**: how well the scores rank "yes" above "no" (50 = chance, 100 = perfect, below 50 = ranked backwards). F1 depends on the yes/no cut-off chosen on validation; AUC doesn't, so AUC is steadier for single-seed comparisons.
- **Probe**: freeze the trained model, take its internal features for each text, and fit a logistic regression (a simple linear classifier) for the label. It is trained on up to 20,000 training labels of that label, with the cut-off tuned on validation. Note that the probe does see labels for the hidden label. It measures whether the hidden label can be *read* from what the model learned, not whether the model predicts it unsupervised. Features are taken from three places: the BERT output ("encoder"), the shared layers ("shared"), and the label's own group branch ("group"). Tables below use "shared" unless they say otherwise.
- **Reference**: the same setup with every label seen, used as the comparison point for each hidden label.
- **Paper** column: the paper's Table 3 multitask F1. The paper has no numbers for the hate speech labels.

All runs: small BERT, wide architecture, learning rate 1e-4, batch 256, dropout 0, the best epoch picked by validation F1 (stop after 3 epochs without improvement), seed 42, L4 GPU.

---

## 1. Hide-one-label, plain

The hidden label's own output gets no training, so its scores are random: AUC 33–80 depending on luck. The probe is the measure.

**Probe test AUC: label seen (reference) vs hidden**

| Dataset | Hidden label | Paper (Table 3 F1) | Reference output, trained (F1) | Probe, seen | Probe, hidden | Change |
|---|---|---|---|---|---|---|
| IAC + hate speech | attack_defend | — | 76.4 | 82.6 | 82.3 | −0.3 |
| IAC + hate speech | Attacking/Respectful | 68.1 | 72.1 | 70.9 | 70.2 | −0.7 |
| IAC + hate speech | dehumanize | — | 70.1 | 78.3 | 77.7 | −0.5 |
| IAC + hate speech | violence | — | 82.2 | 86.9 | 85.1 | −1.8 |
| Main data | Attacking/Respectful | 68.1 | 71.2 | 68.8 | 67.7 | −1.1 |
| Main data | Nasty/Nice | 73.7 | 74.7 | 69.6 | 69.9 | +0.3 |
| Main data | Negotiate/Attack | 64.8 | 68.2 | 72.9 | 72.0 | −0.8 |
| Main data | Argument Quality (WA) | 79.2 | 84.9 | 80.6 | 75.2 | **−5.4** |

- 7 of 8 hidden labels stay within about 2 AUC of the seen version.
- Violence loses a little more. The other label in its branch is genocide.
- Argument Quality loses the most. It is the only label on the 96,036 IBM rows, so hiding it leaves those rows with nothing to learn from.
- Probe F1 swings by up to ±8 while AUC stays flat (for example Nasty/Nice: encoder-probe F1 −8.1, AUC +0.3), because F1 depends on the cut-off.
- **Effect on the other labels** (mean AUC change): IAC + hate speech −0.2 to −0.6; main data −0.2 to −1.1, except −3.1 when Argument Quality is hidden. Section 5 shows that −3.1 is mostly a bug.

**Missing control.** The probe has not been run on the *untrained* small BERT. Without that, it is not shown that training on the other labels taught the model anything about the hidden one, beyond what BERT's pretraining already gives. It costs a few cents and needs no training run.

---

## 2. Mean F1 across all labels, plain

For the hidden label this counts the probe's F1, since its own output never trained.

**Main data (mean over the 10 outputs)**

| Run | Paper (Table 3 F1) | Mean F1, all 10 | Mean AUC, all 10 | Mean F1, the 9 not hidden | Same 9 in reference |
|---|---|---|---|---|---|
| Reference (all labels seen) | 66.7 | **67.9** | 68.3 | — | — |
| Hide Argument Quality | 66.7 | 64.0 | 64.9 | 63.2 | 66.0 |
| Hide Attacking/Respectful | 66.7 | 66.8 | 67.1 | 66.7 | 67.5 |
| Hide Nasty/Nice | 66.7 | 66.9 | 67.6 | 66.5 | 67.2 |
| Hide Negotiate/Attack | 66.7 | 66.8 | 68.0 | 67.0 | 67.9 |

**IAC + hate speech (mean over the 18 labels)**

| Run | Paper (Table 3 F1) | Mean F1, all 18 | Mean AUC, all 18 | IAC labels (8) | Hate speech labels (10) |
|---|---|---|---|---|---|
| Reference (all labels seen) | — | **72.2** | 76.9 | 64.8 | 78.2 |
| Hide attack_defend | — | 72.8 | 76.5 | 66.1 | 78.2 |
| Hide Attacking/Respectful | — | 72.5 | 76.5 | 66.0 | 77.8 |
| Hide dehumanize | — | 72.6 | 76.3 | 66.3 | 77.6 |
| Hide violence | — | 72.0 | 76.6 | 64.8 | 77.7 |

- Main data: hiding one IAC label costs about 1 F1 overall. Hiding Argument Quality costs about 4, but that run was mis-weighted (section 5).
- IAC + hate speech: no cost overall. IAC F1 looks *higher* in three hide runs. They picked a later best epoch (6 vs 4), and their AUC is slightly lower, so this is not a real gain.

---

## 3. IAC + hate speech: with vs without branches

"Branches" means separate layers for the extreme hate speech labels (violence, genocide), the other hate speech labels, and IAC. The comparison uses existing plain runs without branches (`hatespeech_joint/iac_hatespeech_s{42,1,2}`), with the same settings.

| Mean test F1 / AUC | Paper (Table 3 F1) | No branches, seed 42 | No branches, 3 seeds (F1 range) | With branches, seed 42 |
|---|---|---|---|---|
| All 18 labels | — | 72.2 / 76.7 | 72.2 / 76.8 (72.0–72.6) | **72.2 / 76.9** |
| IAC (8) | — | 65.6 / 67.7 | 65.8 / 67.9 (65.6–66.1) | 64.8 / 67.7 |
| Extreme hate speech (2) | — | 85.4 / 87.7 | 84.6 / 87.9 (82.1–86.1) | 86.2 / 88.3 |
| Other hate speech (8) | — | 75.4 / 83.0 | 75.6 / 83.0 (75.4–75.7) | 76.1 / 83.2 |

- No difference overall.
- Hate speech gains 0.5–0.8 F1, within the seed spread.
- IAC F1 drops 0.8, but its AUC is unchanged. Almost all of the drop is Defeater/Undercutter (F1 −8.8, AUC flat), a cut-off effect.

---

## 4. Main data: with vs without rule labels

The with-rule runs (`paper_replication_retune/lr0.0001_s{42,1,2,3}`, plain, same settings) had rule labels in test too. To compare on equal terms, their saved predictions were rescored on exactly the test rows the no-rule model is graded on. Recomputed scores match the library's to 1e-14.

| Output | Paper (Table 3 F1) | With rule labels, all rows incl. rule labels (s42) | With rule labels, same rows (4-seed mean) | **No rule labels (s42)** |
|---|---|---|---|---|
| Propaganda | 61.7 | 63.8 | 64.8 | 65.5 |
| Disagree/Agree | 71.4 | 66.7 | 62.6 | 61.4 |
| Emotion/Fact | 63.9 | 71.0 | 64.0 | 67.5 |
| Attacking/Respectful | 68.1 | 70.4 | 70.1 | 71.2 |
| Nasty/Nice | 73.7 | 71.7 | 74.3 | 74.7 |
| Personal/Audience | 65.7 | 68.2 | 67.9 | 62.5 |
| Defeater/Undercutter | 55.6 | 55.4 | 55.7 | 59.4 |
| Negotiate/Attack | 64.8 | 69.0 | 66.7 | 68.2 |
| Questioning/Asserting | 63.2 | 60.8 | 62.6 | 63.8 |
| Argument Quality | 79.2 | 82.8 | 82.9 | 84.9 |
| **Mean F1** | **66.7** | 68.0 | 67.2 (seeds 65.9–68.1) | **67.9** |
| **Mean AUC** | — | 68.7 | 66.3 | **68.3** |

- Removing the rule labels costs nothing. The no-rule model scores 67.9 F1 against 67.2 on the same rows (inside the seed range), with AUC 2 points higher.
- The rule labels mostly make scores look better: 68.0 when graded on all rows, 67.2 on real IAC rows. The inflation is largest for Emotion/Fact and Negotiate/Attack.
- Personal/Audience −5.4 F1 without rule labels. This label varies by up to 7.5 F1 between seeds, so one seed can't settle it.

---

## 5. How the task weights work, and the Argument Quality bug

**The weighting in plain runs (the paper's rule)**

1. Labels are grouped, roughly one group per dataset. Group weight = 1 ÷ (rows with at least one label in the group). The group weights are rescaled to average 1. In effect each group contributes about the same total, so large datasets don't drown out small ones.
2. The group weight is split equally among the group's labels.
3. Class weights inside each label make the rarer answer count more: weight = labelled rows ÷ (2 × rows with that answer).
4. Loss = sum of weighted errors over visible labels ÷ number of visible labels.

Values used, read from the run logs:

| Dataset | Group | Rows with a label | Group weight | Weight per label |
|---|---|---|---|---|
| IAC + hate speech | Hate speech (10 labels) | 31,592 | 1.32 | 0.132 |
| IAC + hate speech | IAC (8 labels) | 61,069 | 0.68 | 0.085 |
| Main data | Propaganda (one pooled output) | 61,909 | 1.13 | 1.13 |
| Main data | IBM: Argument Quality (1 label) | 96,036 | 0.73 | 0.73 |
| Main data | IAC (8 labels) | 61,069 | 1.14 | 0.143 |

In the training file, IBM rows carry only the Argument Quality label, and no other dataset carries it.

**When one IAC or hate speech label is hidden, nothing much changes.** A row still counts if any other label in its group is visible, and the hidden label still counts as one of the group's labels.

**The bug: hiding Argument Quality.** It is the only label in its group, so its group has 0 rows. The code treats 0 as 1, which gives the empty group weight 1 ÷ 1 = 1, against about 1/61,000 for the others. Rescaling to an average of 1 then shrinks every *real* group's weight about 23,000-fold:

| Group | Normal weight | With Argument Quality hidden |
|---|---|---|
| Propaganda | 1.13 | 0.00005 |
| IBM (Argument Quality) | 0.73 | 3 (no labels, so it multiplies nothing) |
| IAC | 1.14 | 0.00005 |

The empty group's big weight has no effect, since it has no labels. The harm is that the real labels are underweighted. AdamW normally adjusts for the overall size of the loss, but at this size its fixed minimum step size (epsilon 1e-7) probably takes over and slows learning. Signs: that run picked a later best epoch (6 vs 3), and the other labels dropped 3.1 AUC. Section 7 confirms it: under FAMO, which doesn't use these weights, hiding Argument Quality doesn't change the other labels at all.

Only the plain run that hid Argument Quality is affected. Proposed fix: leave groups with no visible labels out of the weight calculation in `Trainer._compute_task_weights`, then rerun that one run (about $1.20). **Not yet done.**

---

## 6. No task weights and no class weights (main data, no rule labels)

New flag `--unweighted-loss` in `test_no_prop_overlap.py`, launched with `modal_replicate.py::norule_unweighted`. Every output gets task weight 1 and both answers get class weight 1. The loss is the plain mean error over all visible labels, so a dataset counts in proportion to how many labels it has. The run log confirms every weight printed as 1. Cost $0.81; best epoch 2 (validation F1 by epoch: 67.9, 68.8, 67.7, 64.3, 65.7).

**Measures in the table:**
- **Macro F1** gives the rare answer the same weight as the common one, so it shows whether the rare cases suffer.
- **Rare-answer recall** is the share of rare-answer test cases the model catches.

| Output | Paper (Table 3 F1) | Weighted (paper's rule): F1 / AUC / macro F1 | **No weights: F1 / AUC / macro F1** | Rare answer's share of test | Rare-answer recall, weighted → no weights |
|---|---|---|---|---|---|
| Propaganda | 61.7 | 65.5 / 67.1 / 62.3 | 66.0 / 68.5 / 63.3 | 38% | 41 → 47 |
| Disagree/Agree | 71.4 | 61.4 / 63.6 / 54.1 | 64.7 / 66.8 / 57.7 | 23% | 61 → 66 |
| Emotion/Fact | 63.9 | 67.5 / 73.0 / 67.5 | 68.5 / 73.7 / 68.5 | 49% | 68 → 74 |
| Attacking/Respectful | 68.1 | 71.2 / 69.8 / 64.2 | 73.5 / 71.9 / 66.1 | 27% | 52 → 48 |
| Nasty/Nice | 73.7 | 74.7 / 71.1 / 62.8 | 75.9 / 72.6 / 64.2 | 19% | 55 → 56 |
| Personal/Audience | 65.7 | 62.5 / 63.2 / 54.2 | 64.2 / 65.4 / 56.8 | 22% | 60 → 68 |
| Defeater/Undercutter | 55.6 | 59.4 / 56.8 / 54.8 | 55.5 / 56.0 / 53.2 | 36% | 33 → 51 |
| Negotiate/Attack | 64.8 | 68.2 / 73.2 / 67.1 | 68.3 / 73.9 / 67.6 | 40% | 65 → 72 |
| Questioning/Asserting | 63.2 | 63.8 / 63.9 / 60.8 | 63.6 / 65.9 / 60.9 | 35% | 55 → 58 |
| Argument Quality | 79.2 | 84.9 / 81.1 / 58.8 | 83.7 / 83.6 / 58.2 | 6% | 63 → 69 |
| **Mean** | **66.7** | **67.9 / 68.3 / 60.7** | **68.4 / 69.8 / 61.7** | | |

- No weights is slightly ahead: +0.5 F1, +1.5 AUC, +1.0 macro F1. On validation it is 68.8 vs 68.5.
- Dropping class weights did not hurt the rare answers. Rare-answer recall rose for 9 of 10 outputs. The likely reason is that each label's yes/no cut-off is already tuned on validation, and that tuning corrects for imbalance by itself.
- Worse: Defeater/Undercutter (F1 −3.9, AUC −0.8) and Argument Quality (F1 −1.2, though AUC +2.5).
- Probe mean AUC 69.3 vs 67.6.
- With one seed, the safe claim is that the paper's weighting doesn't help on this data. Claiming that dropping it helps would need 2 more seeds (about $1.60).

---

## 7. Hide-one-label with FAMO + SSL (`famo_consf1_m5`)

The same 8 hidden labels, plus 2 full-label FAMO references. In these runs the hidden label's own output *does* get a training signal from the consistency SSL. It gets no signal from labels.

**Full models (all labels seen)**

| Dataset | Paper (Table 3 F1) | Plain F1 / AUC | FAMO + SSL F1 / AUC |
|---|---|---|---|
| Main data | 66.7 | 67.9 / 68.3 | **69.4 / 70.5** |
| IAC + hate speech | — | 72.2 / 76.9 | 72.9 / 76.9 |

On the main data FAMO + SSL is ahead on 7 of 10 outputs. On IAC + hate speech the overall result is the same: IAC +2.0 F1 but the same AUC, hate speech −0.4 F1.

**Probe test AUC for the hidden label (shared layers)**

| Dataset | Hidden label | Paper (Table 3 F1) | Plain: seen → hidden | FAMO + SSL: seen → hidden |
|---|---|---|---|---|
| IAC + hate speech | violence | — | 86.9 → 85.1 | 86.9 → 85.0 |
| IAC + hate speech | Attacking/Respectful | 68.1 | 70.9 → 70.2 | 71.9 → 70.3 |
| IAC + hate speech | dehumanize | — | 78.3 → 77.7 | 78.1 → 77.9 |
| IAC + hate speech | attack_defend | — | 82.6 → 82.3 | 82.6 → 82.5 |
| Main data | Nasty/Nice | 73.7 | 69.6 → 69.9 | 72.6 → **71.3** |
| Main data | Negotiate/Attack | 64.8 | 72.9 → 72.0 | 75.3 → **74.0** |
| Main data | Attacking/Respectful | 68.1 | 68.8 → 67.7 | 72.1 → **71.9** |
| Main data | Argument Quality | 79.2 | 80.6 → 75.2 | 81.9 → 74.1 |

- IAC + hate speech: FAMO + SSL and plain are identical.
- Main data: FAMO + SSL's features carry the hidden IAC label better (+1.4 to +4.2 AUC over plain hidden). Hiding costs about the same under both (about 1 point); FAMO + SSL just starts higher.
- Argument Quality is the hardest to recover under both (−5 to −8 AUC).

**Effect of hiding Argument Quality on the other 9 outputs**

| Method | Change in mean AUC | Change in mean F1 |
|---|---|---|
| Plain (uses the buggy group weights) | −3.1 | −2.8 |
| FAMO + SSL (doesn't use them) | **+0.1** | **−0.1** |

So the plain drop was mostly the weighting bug (section 5). Losing IBM's training signal does not hurt the IAC and Propaganda labels. For every other hidden label under FAMO + SSL, the other labels' mean AUC moved by −1.3 to +0.1.

**Mean over all labels (probe score for the hidden label)**

| Dataset | Paper (Table 3 F1) | Plain hide-one runs | FAMO + SSL hide-one runs |
|---|---|---|---|
| Main data | 66.7 | 64.0–66.9 F1 | **68.1–69.4 F1** |
| IAC + hate speech | — | 72.0–72.8 F1 | 72.0–73.3 F1 |

**The hidden label's own output (test AUC)**

| Hidden label | Plain | FAMO + SSL |
|---|---|---|
| violence (joint) | 79.3 | 31.9 |
| Attacking/Respectful (joint) | 34.0 | 32.1 |
| dehumanize (joint) | 32.7 | 23.0 |
| attack_defend (joint) | 79.9 | 81.0 |
| Nasty/Nice (main) | 34.0 | 35.6 |
| Negotiate/Attack (main) | 54.6 | 27.7 |
| Argument Quality (main) | 67.5 | 74.4 |
| Attacking/Respectful (main) | 62.3 | 33.7 |

The consistency SSL pushes the hidden output to agree with itself on masked text, but nothing tells it which way is "yes". Under FAMO + SSL, 6 of 8 ended up ranked backwards (AUC 23–36) and 2 landed the right way round by luck. An AUC of 23 means the output does track the label, just with the sign flipped. Without at least a few labels to fix the direction, the output is not usable, so the probe remains the right measure for hidden labels.

**Caveat.** On the main data, FAMO + SSL picked epoch 1 as its best in the reference and in 3 of 4 hide-one runs, so those results come from a model trained for only one epoch.

---

## Cost

| What | Runs | Cost |
|---|---|---|
| Pilot, 6 Nov (2 hide-one + 1 stopped plain FAMO run) | 3 | $2.65 |
| Plain hide-one + 2 references, 9 Nov | 8 | $7.07 |
| FAMO + SSL hide-one + 2 references, 10 Nov | 10 | $13.02 |
| Unweighted run, 10 Nov | 1 | $0.81 |
| **Total** | | **$23.55** |

Billing from `modal billing report`. Other apps on the account (not started in this session) were charged about $5.50 on 10 Nov. The largest was `ap-N6HIKhNQLK0mX0V5lBN0N2`, a `famo_consf1_m5` main-data run with `--main-task disagree_agree_bi`.

---

## Where things are

- Results (Modal volume `multitask-outputs`): `hatespeech_hide_one/{plain,famo_consf1_m5}_br_{full,hide_<label>}_s42.json` and `paper_replication_hide_one/{plain,famo_consf1_m5}_{full,hide_<label>}_s42.json`, `plain_unweighted_full_s42.json`. Each has a `_predictions.npz` next to it.
- Probe code: `replication/hidden_label_probe.py`. Flags `--hide-task`, `--probe-tasks`, `--no-rule-labels`, `--unweighted-loss` in `replication/test_no_prop_overlap.py`; `--branches`, `--extreme-items`, `--hide-task`, `--probe-tasks` in `hatespeech/run_hatespeech.py`.
- Modal entrypoints: `modal_replicate.py::hide_one`, `norule_baselines`, `norule_unweighted`; `hatespeech/modal_hatespeech.py::hide_one_runs`, `branched_runs`.
- Modal apps: plain joint hides `ap-lli4AEitVQiqehGwj5xgWH`, plain main hides `ap-sDO1WeW1pRcqoFZdajW6XH`, plain joint reference `ap-IGAcPmSxCEtxYAzhy5ypus`, plain main reference `ap-thGiwIz1LzyROY76PtM2ow`, FAMO joint hides `ap-0PHYu8O5MoWRTJUSNl7jJY`, FAMO main hides `ap-h1nAB2W2j6MUh21sQwpesF`, FAMO joint reference `ap-2X1nws0dwVRiEaLzN8NFqe`, FAMO main reference `ap-5j9IZ8BwgsX4SHiQGZ4C2Y`, unweighted `ap-9Cst54syvFXesILyBKlohs`.

## Open items

1. Fix the empty-group weighting bug and rerun plain hide-Argument-Quality (about $1.20).
2. Probe on the untrained small BERT, to show the other labels taught the model something (a few cents).
3. Two more seeds for the unweighted run (about $1.60), and for the branched and no-rule plain references (about $3), to turn "within the seed spread" into a test.
4. Label-budget probe (100 / 1,000 / 20,000 labels of the hidden label), proposed but not decided.
