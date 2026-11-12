# Model perturbation study: dropout views, EMA teacher, Mean Teacher, and wider confidence filters

Record of one working session, 2 to 3 November 2026. All times are US Central (CST, UTC−6). Exact times come from file edit times, Modal result files and log lines. Times marked "about" are inferred from the order of events.

Code changes described here are as of 3 November. `test_no_prop_overlap.py` and `modal_replicate.py` were edited again in later sessions.

---

## Contents

1. [Timeline](#1-timeline)
2. [Background: where things stood](#2-background-where-things-stood)
3. [What "model perturbation" can mean](#3-what-model-perturbation-can-mean)
4. [The design: two switches, three arms](#4-the-design-two-switches-three-arms)
5. [Questions answered along the way](#5-questions-answered-along-the-way)
6. [Check: is the confidence filter using too few cells?](#6-check-is-the-confidence-filter-using-too-few-cells)
7. [Code changes](#7-code-changes)
8. [Tests](#8-tests)
9. [Smoke test](#9-smoke-test)
10. [Full runs and problems during launch](#10-full-runs-and-problems-during-launch)
11. [Results](#11-results)
12. [Conclusions](#12-conclusions)
13. [Thesis plan (deadline 25 November)](#13-thesis-plan-deadline-25-november)
14. [Terms used](#14-terms-used)
15. [Commands to reproduce](#15-commands-to-reproduce)

---

## 1. Timeline

| Date and time (CST) | What happened |
|---|---|
| 2 Nov, evening | Chat started: "this chat is about model perturbation". Searched the project, which had no existing perturbation code, and listed four meanings of perturbation (section 3). |
| 2 Nov, evening | User pasted a note on dropout views vs the EMA (Mean) teacher and asked how to run it. Read the training code and found that DistilBERT's own dropout (0.1) is always on in training (section 4). Proposed the 2 × 2 design. |
| 2 Nov, late evening | User asked how q = 0.1 was chosen, whether the baseline could be famo_consf1_m5, and for more detail on dropout views (section 5). |
| 3 Nov, about 00:20 | User chose **option (a)**: three arms, compared with both FAMO and famo_consf1_m5. User suspected an unsure model meant only a few cells reached the loss, and asked for a check plus a wider q sweep that isn't "too flexible". |
| 3 Nov, 00:32–00:33 | Downloaded the saved results and validation predictions of famo_consf1_m5, famo_cons1_m5 and famo (4 seeds each) from Modal. |
| 3 Nov, about 00:35 | Filter check done (section 6): the share of cells is fixed at 15.1%, but weak heads' "confident" cells are close to 0.5. |
| 3 Nov, 00:35–00:47 | Code written: confidence floor, `ConsistencyMixin`, `TeacherConsistencyModel` / `TeacherConsistencyTrainer`, teacher scoring, new options and launchers. New test file at 00:39; floor test at 00:47. |
| 3 Nov, about 00:40–00:50 | First full test run on Modal: 60 of 61 passed. An older filter test failed because of tied scores (section 8). |
| 3 Nov, about 00:46 | Debug run confirmed the tie explanation. Seeded that test. |
| 3 Nov, about 01:00–01:10 | Second full test run: 60 of 61 passed. The failure was a numerical tolerance in the unrelated gradient-probe test. |
| 3 Nov, 02:33 | First smoke launch. The local Modal client lost its connection, which stopped the app and killed all 3 runs. |
| 3 Nov, 02:50–03:22 | Smoke rerun with `--detach`: all 3 arms finished (results saved 02:50, 03:12, 03:22). |
| 3 Nov, 03:40 | Full launch: 12 teacher runs and 8 filter-sweep runs. The first launcher was stopped too late, so a duplicate teacher app ran as well (section 10). |
| 3 Nov, 05:20 | First full result saved (famo_consq2_m5_s42). |
| 3 Nov, about 07:30 | Duplicate teacher app stopped (`ap-9M5uNPG5FAY1x66xSqL0M2`). |
| 3 Nov, 09:50 | Last result saved (famo_tc_ema_s3). All 20 done. |
| 3 Nov, about 10:00–10:30 | Both comparison reports, plus the two head-to-head Mean Teacher comparisons. |
| 3 Nov, after the results | User: "idk man i dont know what else to do anymore". Summarised what the project has established and suggested stopping new methods. |
| 3 Nov | User: the thesis deadline is 25 November. Proposed a timeline (section 13). |
| 4 Nov | This record written. |

---

## 2. Background: where things stood

Results before this session (FAMO learning rate 1e-4, stop on validation F1 with patience 3, 4 seeds, mean test F1 over the 10 Table-3 outputs):

| Setup | Test F1 | Test AUC | Paper (Table 3 F1) |
|---|---|---|---|
| famo (FAMO balancing alone) | 69.0 | 71.2 | 66.74 |
| famo_fixed (FAMO + 0.03 × word-guessing loss) | 69.5 | 72.7 | 66.74 |
| famo_cons1_m5 (FAMO + Option 2 consistency, 5% masking, weight 1, no filter) | 69.2 | 71.9 | 66.74 |
| **famo_consf1_m5** (same, with the q = 0.1 confidence filter) | **69.9** | **72.6** | 66.74 |
| famo_consf3_m5 (same filter, weight 3) | 69.4 | 71.9 | 66.74 |

"Option 2" is the consistency term from the label-aware study. A masked copy of each sentence goes through the whole model, and its answers are pulled toward the full-sentence answers.

---

## 3. What "model perturbation" can mean

Four readings, offered at the start:

1. **Weight perturbation:** add small random noise to the trained weights and see how much F1 drops. This shows whether the model sits in a flat or a sharp region.
2. **Input perturbation:** change the text slightly and see whether predictions change. This links to the earlier finding that the model uses source and length shortcuts.
3. **Representation perturbation:** add noise to, or remove parts of, the encoder output before the heads.
4. **Perturbation during training:** add noise on purpose while training, for example adversarial embeddings or sharpness-aware minimisation.

The pasted note picked a form of (4): **consistency under dropout**, and the **EMA (Mean) teacher**.

---

## 4. The design: two switches, three arms

### Key finding from the code

The small layers on top of the encoder use dropout 0 (the WINNER settings). But **DistilBERT's own dropout (0.1) is on whenever the model runs in training mode**. So:

- Every earlier run already trained with dropout inside the encoder.
- No run had a student without dropout. Getting one means running that pass in evaluation mode (`training=False`), which turns dropout off but still lets gradients through.
- In the earlier consistency runs (for example famo_consf1_m5) the **target was itself noisy**, because it was computed with dropout on.

### The design

Two switches, with the teacher's dropout always off:

| | Teacher = current student weights | Teacher = EMA copy |
|---|---|---|
| **Student pass, dropout off** | loss is always 0, so this is a code check, not a run | **famo_tc_ema** (better targets, no noise) |
| **Student pass, dropout on** | **famo_tc_drop** (dropout views) | **famo_tc_mt** (Mean Teacher) |

- **famo_tc_mt − famo_tc_drop** = the effect of **better targets**: same noise, averaged teacher instead of current weights.
- **famo_tc_mt − famo_tc_ema** = the effect of **stability under noise**: same teacher, student noise on vs off.

### Settings held fixed

- FAMO, learning rate 1e-4, stop on validation F1 with patience 3, other WINNER settings, seeds 42, 1, 2, 3.
- No masking and no word-guessing loss.
- Consistency weight 1, rising linearly over the first epoch.
- Confidence filter q = 0.1, with Propaganda rows skipped for the 8 IAC heads.
- The supervised pass is unchanged (encoder dropout on).
- **EMA decay 0.99**, starting at min(0.99, 1 − 1/(step + 1)) as in Mean Teacher. There are about 855 steps per epoch (219,000 texts / batch size 256), so 0.99 trails the student by about 100 steps, roughly 0.1 epoch. The more common 0.999 would trail by about 1,000 steps, more than an epoch, while the best epoch is usually 1.5–4.
- The main score is always the **student**. The EMA teacher is scored separately as a read-out and plays no part in picking the epoch.

---

## 5. Questions answered along the way

### How was q = 0.1 chosen? Were other filter sizes tried?

Only q = 0.1 had been run. The filter study varied the weight (1 vs 3) and the filter (on vs off), never its size. As far as the code and notes show, 0.1 was carried over from the label-aware pre-check, which measured accuracy in the **most confident 10%** of validation cells. It was not tuned.

Two caveats about the earlier filter result:
- Filter vs no filter was +0.7 F1 [−1.5, +2.9], mostly from one seed.
- The clear effect of the filter was that it stopped the heads from flattening. Without it, each head's score spread shrank by 20–30%.

### Can the baseline be famo_consf1_m5?

Yes, and it costs nothing, since its runs are saved. Recommended reporting against **both**:
- **vs famo (69.0):** does this kind of consistency help at all?
- **vs famo_consf1_m5 (69.9):** is it better than the best existing setup?

Caveat: famo_consf1_m5's own gain over famo was not significant (+1.0 [−0.9, +2.8]).

Two meanings of "baseline" were offered:
- **(a)** a comparison point only, with the new arms using dropout noise and no masking; **chosen**;
- **(b)** a starting point, adding an EMA teacher on top of famo_consf1_m5 with masking kept.

### Dropout views in more detail

- **Dropout** switches off a random 10% of DistilBERT's neurons on every training pass. The same sentence run twice gives two slightly different answers, or "views".
- **The idea:** if the answer depends a lot on which neurons were dropped, it rests on fragile features. Penalising disagreement between views pushes toward robust answers.
- **Versions:**
  - Π-model: both views noisy, one held fixed.
  - R-Drop: both noisy, both learn.
  - Clean target (used here): the dropout-off answer is held fixed, and the dropout-on answer is pulled toward it.
- **Labelled cells:** the supervised loss already gives the right answer, so consistency only regularises.
- **Unlabelled cells:** consistency is the only signal. It **can't teach the right answer, only a stable one**.
- **Main risk: flattening.** Pushing every score toward 0.5 makes views agree trivially, which is why the filter matters.
- **Expected problem:** the signal may be tiny, because dropout at 0.1 may disturb the model less than masking does.

---

## 6. Check: is the confidence filter using too few cells?

**User's idea:** the model is unsure about most answers, so only a few rows reach the loss, and that limits the gain.

### What the training logs show

The filter is **rank-based**. Per head and per batch it always keeps the top and bottom q share of scores, however confident they are. "consistency_kept" was **exactly 15.1% of cells in every epoch of every seed**. It's 15% rather than 20% because Propaganda rows are skipped for the 8 IAC heads. So an unsure model **doesn't** mean fewer cells.

### What does vary: how confident the kept cells are

Validation scores of famo_consf1_m5 (4 seeds, best epoch, evaluation mode), using the same eligibility rule as training:

| Head | Share of cells already ≤ 0.2 or ≥ 0.8 | Scores at the q = 0.1 cut-off (low / high) | Share of labelled kept cells that are right (low / high side) |
|---|---|---|---|
| Propaganda (max-pool) | 37% | 0.20 / 0.88 | 81% / 74% |
| Disagree/Agree | 21% | **0.29** / 0.90 | **41%** / 72% |
| Emotion/Fact | 45% | 0.10 / 0.97 | 85% / 80% |
| Attacking/Respectful | 52% | 0.09 / 0.94 | 93% / **33%** |
| Nasty/Nice | 57% | 0.07 / 0.95 | 97% / **23%** |
| Personal/Audience | 26% | 0.14 / 0.76 | – / 89% |
| Defeater/Undercutter | 14% | 0.10 / **0.58** | – / 71% |
| Negotiate/Attack | 58% | 0.09 / 0.93 | 66% / 92% |
| Questioning/Asserting | 14% | 0.18 / **0.66** | 78% / 55% |
| Argument Quality | 55% | 0.07 / 0.91 | **42%** / 99% |

("–" means no labelled cells fell on that side in at least one seed.)

**What this shows:**

- **The user's idea is partly right.** The number of cells is fixed, but on the weak heads (Defeater, Questioning, Disagree) the so-called confident cells are close to a coin flip.
- **Confident heads could take many more cells and still be confident.**
- **Confident isn't the same as correct on the rarer side of imbalanced heads.** For example, only 23% of Nasty's confident "yes" cells are actually nasty. Widening the filter doesn't fix that.

### The two rules chosen for the sweep

Both were checked on the same validation scores first:

| Rule | Cells kept on confident heads | Cells kept on Disagree / Defeater / Questioning |
|---|---|---|
| q 0.1 (current) | 20% | 20% |
| q 0.2 | about 40% | 40%, including near-coin-flip targets |
| q 0.3 + floor 0.2, sides equal | 38–51% | 2% / 0% / 1% |

The **floor** rule keeps a cell only if its score is ≤ 0.2 or ≥ 0.8, and cuts both sides to the same size. If one side has no confident cells, that head contributes nothing in that batch. Without equal sides, the weak heads would become one-sided; for example Defeater would keep only "no" cells, and those were right only 26% of the time.

The analysis scripts (`filter_check.py`, `floor_check.py`) were in the session scratchpad and were not kept.

---

## 7. Code changes

All in `multitask-forked/`, made 3 November, 00:35–00:47.

### `replication/test_no_prop_overlap.py`

- **`confident_cells(t, eligible, share, floor=0.0)`:** new `floor` option. With floor > 0, a kept score must also be ≤ floor or ≥ 1 − floor, and both sides shrink to the smaller count, possibly 0. Without a floor, behaviour is unchanged.
- **`check_consistency_filter(keep, floor)`:** shared validation of the filter settings.
- **`ConsistencyMixin`:** `consistency_loss` and `consistency_ramp` moved out of `LabelAwareMaskedModel` so both models share them unchanged. It now passes `consistency_floor` through.
- **`LabelAwareMaskedModel` / `LabelAwareMaskedTrainer`:** accept `consistency_floor`.
- **`TeacherConsistencyModel(ConsistencyMixin, BalancedMultiTaskModel)`:**
  - The teacher is either "current" (the model itself with dropout off) or "ema" (a separate `MultiTaskModel`, copied from the student, then averaged after each step).
  - The student answer is either the supervised pass (dropout on) or a second pass with dropout off.
  - "current" with student dropout off is refused, because the loss would always be 0.
  - The EMA teacher is held in an `_Untracked` wrapper, so its weights stay out of the student's trainable variables, the optimizer and `get_weights`.
  - Logs `consistency_loss`, `consistency_kept`, `consistency_weight`, `teacher_gap` (mean |teacher − student| over all cells) and `ema_decay`.
- **`TeacherConsistencyTrainer(BalancedLossTrainer)`:** builds the student, sets up FAMO, builds and copies the EMA teacher, then attaches the settings.
- **`EpochScorer._teacher_scores`:** each epoch, loads the EMA teacher's weights into the student, scores validation and test with the teacher's own thresholds, then restores the student. Saved as `rec["teacher"]`. It doesn't affect stopping.
- **New command-line options:**
  - `--consistency-floor`
  - `--consistency-teacher {current,ema}`
  - `--consistency-student-dropout {on,off}`
  - `--ema-decay` (default 0.99)
  - Checks: the teacher mode needs `--loss-balancer famo` and `--consistency-weight`, and can't be combined with `--mlm-lambda` or `--masked-label-weight`.

### `modal_replicate.py`

- `replicate(...)` gained `consistency_floor`, `consistency_teacher`, `consistency_student_dropout` and `ema_decay`.
- **`filter_sweep`:** famo_consq2_m5 (q 0.2) and famo_consq3f2_m5 (q 0.3 + floor 0.2). Option 2 at 5% masking, weight 1, no word-guessing loss.
- **`teacher_consistency`:** famo_tc_drop, famo_tc_ema and famo_tc_mt, with `--smoke` for runs named `smoke_<config>_s<seed>`.
- **`label_aware_report`** now also prints the share of cells the consistency term used, and the EMA teacher's scores at the student's picked epoch.
- `masked_lm_tests` now includes the new test file.

---

## 8. Tests

### New: `replication/test_teacher_consistency.py` (10 tests)

- The teacher starts as an exact copy and none of its weights are in the student's variables.
- After one step, each teacher weight equals decay × old teacher + (1 − decay) × student.
- With decay 0, the teacher equals the student, and it never appears in the optimizer.
- "current" teacher with student dropout off is refused.
- The "current" teacher gives exactly 0 loss against a dropout-off student.
- A dropout-off pass is deterministic; a dropout-on pass isn't.
- With consistency weight 0, all three arms take exactly the same step as plain FAMO.
- Training and validation logs exist and are finite, and the EMA decay values follow the schedule.

### Also added

- A floor test in `replication/test_label_aware_masking.py`.

### Results on Modal

- **First full run:** 60 of 61 passed. The failure was `test_filter_uses_only_confident_cells_and_skips_marked_rows`. Its exact reference comparison passed, but a sanity check (kept share < 30%) failed at 35.9%. Run alone, the same test kept the expected 15.6% with no tied scores. Cause: random state left by earlier tests sometimes gives the tiny test model many identical scores, and all tied cells at the cut-off are kept. **Fix:** a fixed seed in that test. The filter code was fine.
- **Second full run:** 60 of 61 passed. The failure was `test_gradient_probe.py::test_probe_records_valid_metrics`, which checks that a gradient's cosine with itself is 1 within 0.00001 and got 0.999974 (rounding over 4.4M parameters). That test passed on the previous run with the same code, and this session didn't touch it. It was left as is.

---

## 9. Smoke test

One epoch on seed 42. The first attempt at 02:33 was killed by a lost connection; the rerun saved results at 02:50, 03:12 and 03:22.

| Arm | Consistency loss | Cells used | Teacher–student gap | Final EMA decay | Val F1 | Test F1 |
|---|---|---|---|---|---|---|
| famo_tc_drop | 0.0030 | 15.1% | 0.032 | – | 68.1 | 67.4 |
| famo_tc_ema | 0.0036 | 15.1% | 0.028 | 0.986 | 67.7 | 66.6 |
| famo_tc_mt | 0.0051 | 15.1% | 0.038 | 0.986 | 68.1 | 67.1 |

All three arms trained, saved and logged sensible values. The consistency loss was already **tiny**, about the same size as the masking version (famo_consf1_m5 epoch 1: 0.0055). Paper mean for reference: 66.74.

---

## 10. Full runs and problems during launch

**Launch at 03:40 on 3 November:** 3 teacher arms × 4 seeds (12 runs) and 2 filter rules × 4 seeds (8 runs). Results went to `multitask-outputs/paper_replication_retune_famo/`.

**Problems:**

1. **Chained launch.** Both launches were first run in one shell command with `;`. Each launcher waits for its runs, so the filter sweep would have started only after every teacher run finished, and could then have launched duplicate runs. The sweep was started separately and the first launcher was stopped.
2. **Duplicate teacher app.** The stopped launcher had already spawned its runs. Its app `ap-9M5uNPG5FAY1x66xSqL0M2` ran alongside the real teacher app `ap-865YM8QKVV7mtlPfcXoRZQ`, writing the same result files. It was stopped at about 07:30. Any file it wrote is a valid run of the same settings, so the only cost was GPU time.
3. **"FAIL" lines** in both launchers (`Connection lost`, `nodename nor servname provided`, `Deadline exceeded`) came from this machine losing its network connection while waiting. The detached runs kept going and all saved.

**Finish times of the full runs:** 05:20 (first) to 09:50 (last).

| Run | Seed 42 | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| famo_tc_drop | 05:38 | 06:01 | 07:18 | 07:29 |
| famo_tc_ema | 07:13 | 07:24 | 07:21 | 09:50 |
| famo_tc_mt | 07:17 | 08:19 | 08:35 | 08:27 |
| famo_consq2_m5 | 05:20 | 05:50 | 05:50 | 06:43 |
| famo_consq3f2_m5 | 07:58 | 09:29 | 07:32 | 08:06 |

---

## 11. Results

All runs: 4 seeds, epoch picked on validation F1 with patience 3. Values are means over the 10 Table-3 outputs × 100, given as mean ± standard deviation over seeds. Δ is paired by seed with a 95% t-interval.

### 11.1 Against FAMO alone

| Setup | Picked epoch | Val F1 | Test F1 | Test AUC | Paper (Table 3 F1) | ΔF1 vs famo | ΔAUC vs famo |
|---|---|---|---|---|---|---|---|
| famo | 4.2 | 70.1 ± 1.0 | 69.0 ± 0.9 | 71.2 ± 2.0 | 66.74 | | |
| famo_consf1_m5 | 1.8 | 70.9 ± 0.6 | 69.9 ± 0.7 | 72.6 ± 0.2 | 66.74 | +1.0 [−0.9, +2.8] | +1.4 [−1.6, +4.4] |
| famo_consq2_m5 | 2.2 | 70.6 ± 0.2 | 69.8 ± 0.4 | 72.5 ± 0.4 | 66.74 | +0.8 [−0.5, +2.1] | +1.4 [−1.8, +4.5] |
| famo_consq3f2_m5 | 3.2 | 70.6 ± 0.6 | 69.5 ± 0.4 | 71.9 ± 1.0 | 66.74 | +0.5 [−0.9, +2.0] | +0.7 [−3.6, +5.0] |
| famo_tc_drop | 4.2 | 70.1 ± 0.5 | 69.1 ± 0.3 | 70.4 ± 0.7 | 66.74 | +0.1 [−0.9, +1.2] | −0.8 [−3.0, +1.3] |
| famo_tc_ema | 3.2 | 70.4 ± 0.1 | 69.2 ± 0.2 | 71.6 ± 0.9 | 66.74 | +0.2 [−1.2, +1.7] | +0.4 [−2.9, +3.7] |
| famo_tc_mt | 2.0 | 70.0 ± 0.7 | 69.1 ± 0.8 | 72.4 ± 0.5 | 66.74 | +0.1 [−0.2, +0.5] | +1.2 [−1.3, +3.6] |

### 11.2 Against the best run so far (famo_consf1_m5)

| Setup | Test F1 | Paper (Table 3 F1) | ΔF1 vs famo_consf1_m5 | ΔAUC vs famo_consf1_m5 |
|---|---|---|---|---|
| famo_consq2_m5 | 69.8 | 66.74 | −0.2 [−1.2, +0.8] | −0.1 [−0.5, +0.4] |
| famo_consq3f2_m5 | 69.5 | 66.74 | −0.4 [−0.9, +0.1] | −0.7 [−2.4, +1.1] |
| famo_tc_drop | 69.1 | 66.74 | −0.8 [−1.7, +0.0] | **−2.2 [−3.1, −1.3]** |
| famo_tc_ema | 69.2 | 66.74 | −0.7 [−1.4, −0.0] | −1.0 [−2.4, +0.4] |
| famo_tc_mt | 69.1 | 66.74 | −0.8 [−2.6, +1.0] | −0.2 [−0.9, +0.4] |

### 11.3 Which mechanism? (head-to-head)

| Comparison | What it isolates | ΔF1 | ΔAUC | Propaganda ΔF1 | Propaganda ΔAUC |
|---|---|---|---|---|---|
| famo_tc_mt − famo_tc_drop | better targets | +0.0 [−1.0, +1.0] | **+2.0 [+1.4, +2.6]** | +1.1 [−0.9, +3.1] | +2.7 [+0.2, +5.2] |
| famo_tc_mt − famo_tc_ema | noise on the student | −0.1 [−1.5, +1.3] | +0.8 [−1.0, +2.5] | +1.1 [−0.5, +2.8] | +1.8 [−1.0, +4.6] |

### 11.4 Share of training cells the consistency term used

| Setup | Cells used |
|---|---|
| famo_consf1_m5, famo_tc_drop, famo_tc_ema, famo_tc_mt (q 0.1) | 15.1% |
| famo_consq2_m5 (q 0.2) | 30.7% |
| famo_consq3f2_m5 (q 0.3 + floor 0.2) | 25.6% |

### 11.5 The EMA teacher itself

Scored at the student's picked epoch; Δ = teacher − student.

| Setup | Teacher test F1 | Teacher test AUC | ΔF1 | ΔAUC |
|---|---|---|---|---|
| famo_tc_ema | 69.2 ± 0.7 | 71.7 ± 0.9 | −0.0 | +0.1 |
| famo_tc_mt | 68.6 ± 0.5 | 72.5 ± 0.5 | −0.5 | +0.2 |

### 11.6 Propaganda against FAMO alone

| Setup | F1 Δ | AUC Δ |
|---|---|---|
| famo_consf1_m5 | +1.7 [+0.5, +2.9] | +2.2 [−0.0, +4.3] |
| famo_consq2_m5 | +1.9 [+1.1, +2.7] | +1.8 [−0.5, +4.0] |
| famo_consq3f2_m5 | +0.9 [−2.3, +4.2] | +0.6 [−3.8, +5.0] |
| famo_tc_drop | −0.1 [−2.9, +2.7] | −1.1 [−5.1, +2.8] |
| famo_tc_ema | −0.1 [−2.7, +2.5] | −0.2 [−4.5, +4.1] |
| famo_tc_mt | +1.0 [−0.3, +2.3] | +1.6 [+0.0, +3.2] |

These are a few comparisons among many, so single intervals that clear 0 should be read with care.

### 11.7 Test weighted F1 per task (× 100)

| Task | Paper (Table 3 F1) | famo | famo_consf1_m5 | famo_consq2_m5 | famo_consq3f2_m5 | famo_tc_drop | famo_tc_ema | famo_tc_mt |
|---|---|---|---|---|---|---|---|---|
| Propaganda | 61.74 | 64.4 ± 1.4 | 66.1 ± 1.4 | 66.3 ± 1.1 | 65.3 ± 1.4 | 64.3 ± 0.7 | 64.3 ± 0.8 | 65.4 ± 0.7 |
| Disagree/Agree | 71.38 | 66.9 ± 2.7 | 64.6 ± 1.0 | 65.5 ± 2.4 | 66.7 ± 4.3 | 66.6 ± 1.7 | 64.8 ± 3.2 | 61.6 ± 2.6 |
| Emotion/Fact | 63.93 | 70.7 ± 1.2 | 71.2 ± 1.1 | 70.9 ± 0.8 | 69.9 ± 1.5 | 69.7 ± 1.3 | 70.8 ± 1.2 | 71.3 ± 0.9 |
| Attacking/Respectful | 68.07 | 72.7 ± 2.0 | 74.0 ± 0.4 | 73.9 ± 0.7 | 72.8 ± 1.3 | 72.2 ± 0.8 | 72.3 ± 1.4 | 73.2 ± 1.0 |
| Nasty/Nice | 73.69 | 76.0 ± 1.9 | 77.1 ± 1.3 | 77.4 ± 0.3 | 76.3 ± 0.8 | 75.3 ± 0.6 | 76.2 ± 0.3 | 76.6 ± 0.7 |
| Personal/Audience | 65.69 | 64.9 ± 8.3 | 69.4 ± 2.5 | 67.1 ± 2.7 | 69.2 ± 3.9 | 68.3 ± 2.7 | 69.7 ± 1.3 | 68.2 ± 2.8 |
| Defeater/Undercutter | 55.65 | 58.1 ± 3.0 | 59.7 ± 2.1 | 60.7 ± 0.9 | 58.1 ± 4.3 | 59.0 ± 1.9 | 56.3 ± 2.3 | 55.8 ± 3.6 |
| Negotiate/Attack | 64.81 | 70.5 ± 1.6 | 71.6 ± 0.3 | 72.0 ± 0.4 | 71.3 ± 0.2 | 71.0 ± 0.7 | 71.1 ± 0.4 | 71.7 ± 0.7 |
| Questioning/Asserting | 63.23 | 63.6 ± 1.8 | 62.9 ± 3.9 | 63.7 ± 2.5 | 61.9 ± 3.4 | 61.0 ± 1.9 | 63.2 ± 1.7 | 64.8 ± 0.9 |
| Argument Quality | 79.17 | 82.0 ± 4.0 | 82.8 ± 4.1 | 80.3 ± 4.3 | 83.9 ± 2.0 | 83.4 ± 3.4 | 83.6 ± 2.0 | 82.5 ± 1.7 |

### 11.8 Spread of validation scores

Standard deviation over all validation rows, mean over seeds. A big drop compared with famo means a head is flattening toward one value.

| Task | famo | famo_consf1_m5 | famo_consq2_m5 | famo_consq3f2_m5 | famo_tc_drop | famo_tc_ema | famo_tc_mt |
|---|---|---|---|---|---|---|---|
| Propaganda | 0.247 | 0.253 | 0.271 | 0.283 | 0.351 | 0.221 | 0.212 |
| Disagree/Agree | 0.241 | 0.240 | 0.251 | 0.248 | 0.315 | 0.226 | 0.209 |
| Emotion/Fact | 0.304 | 0.327 | 0.327 | 0.333 | 0.372 | 0.293 | 0.295 |
| Attacking/Respectful | 0.340 | 0.337 | 0.350 | 0.350 | 0.395 | 0.312 | 0.299 |
| Nasty/Nice | 0.353 | 0.351 | 0.363 | 0.363 | 0.405 | 0.327 | 0.311 |
| Personal/Audience | 0.286 | 0.241 | 0.254 | 0.295 | 0.353 | 0.271 | 0.228 |
| Defeater/Undercutter | 0.196 | 0.191 | 0.202 | 0.209 | 0.294 | 0.177 | 0.154 |
| Negotiate/Attack | 0.359 | 0.343 | 0.349 | 0.356 | 0.406 | 0.338 | 0.315 |
| Questioning/Asserting | 0.212 | 0.181 | 0.186 | 0.212 | 0.296 | 0.206 | 0.177 |
| Argument Quality | 0.353 | 0.314 | 0.346 | 0.353 | 0.373 | 0.345 | 0.306 |

---

## 12. Conclusions

1. **No new setup beats famo_consf1_m5 (69.9).**
2. **Using more cells didn't help.** Doubling the cells (q 0.2) gave −0.2 F1, and the floor rule gave −0.4. So too few cells wasn't what held the gain back. A likelier limit is that many confident targets are wrong on the imbalanced heads (section 6). That part is an interpretation, not something these runs tested.
3. **Dropout isn't a useful source of noise here.** All three dropout and teacher arms match FAMO alone on F1 (+0.1 to +0.2). The consistency loss stayed tiny (about 0.003–0.005). So **the 5% masking was the useful part of famo_consf1_m5**, not consistency as such.
4. **An averaged (EMA) target helps ranking, not F1.** Mean Teacher vs dropout views: AUC +2.0 [+1.4, +2.6], F1 +0.0. Most of it is dropout views *losing* AUC (70.4 vs 71.2 for famo). Dropout views also spread every head's scores out by 25–50%: more confident, not better ranked.
5. **Noise on the student adds nothing clear** (Mean Teacher vs EMA only: F1 −0.1, AUC +0.8, interval crosses 0).
6. **Scoring the EMA teacher doesn't beat the student** (ΔF1 −0.0 and −0.5).
7. **Mean Teacher hurt Disagree/Agree** (61.6 vs 66.9 for famo) and flattened scores by about 10–20%. This is a single per-task observation among many.

Possible follow-ups, offered but not run: an EMA teacher with the 5% masking kept, or a filter based on each head's tuned threshold instead of plain score rank.

---

## 13. Thesis plan (deadline 25 November)

### What the project has established, as one story

1. **Replication:** reproduced Farzam et al. (2024) and found three problems. The encoder was never fine-tuned on Modal (Keras 3 bug), the paper's stopping rule stops too early, and the warm-up was 20% of training instead of 5%. After the fixes, multi-task reaches 69.0–69.9 test F1 against the paper's 66.74.
2. **Does multi-task learning help?** Under the old settings, single-task beat multi-task by 3.3 F1 (significant).
3. **Can training tricks close the gap?** About a dozen methods (loss balancing, word-guessing loss, stratified batches, label zero-filling, several consistency variants including this session's) all land at 69–70, within seed noise.
4. **Why the plateau:**
   - source and length shortcuts;
   - only one technique head learns inside the propaganda max-pool;
   - the IAC heads collapse onto about 2 directions;
   - only 24.6% of label cells are filled;
   - confident answers on imbalanced heads are often wrong.

### The one experiment left (as of 3 November)

Rerun single-task with the current settings (learning rate 1e-4, stop on validation F1, patience 3), so both sides of the multi-task vs single-task comparison use the same settings. Then freeze experiments.

### Proposed timeline

| Dates | What |
|---|---|
| 3–5 Nov | Single-task rerun; build thesis tables and figures from saved runs. |
| **6 Nov** | **Experiments frozen.** |
| 6–16 Nov | Core chapters: Methods, Replication, Multi-task vs single-task, Improvement attempts, Analysis. |
| 17–20 Nov | Introduction, Discussion, Conclusion, Abstract. |
| 20–23 Nov | Supervisor feedback and revisions. |
| 24–25 Nov | Buffer and submission. |

---

## 14. Terms used

- **FAMO:** a method that adjusts each task's loss weight during training so that all tasks improve at a similar rate.
- **Consistency loss:** a penalty for the model's answer on a changed ("perturbed") input moving away from a target answer. Here it's the binary KL divergence, which is 0 when the two answers match.
- **Teacher / student:** the student is the model being trained. The teacher supplies the target answers and gets no gradient.
- **EMA (exponential moving average) teacher:** a copy whose weights are a running average of the student's recent weights. With decay 0.99, each step keeps 99% of the old teacher and adds 1% of the student.
- **Mean Teacher:** an EMA teacher combined with noise on the student (Tarvainen & Valpola, 2017).
- **Dropout views:** the same input run twice with different random dropout, giving two answers that should agree.
- **Confidence filter (q):** per head and per batch, the consistency term uses only the q share of cells with the lowest scores and the q share with the highest.
- **Floor:** an extra rule that a kept score must be ≤ floor or ≥ 1 − floor.
- **Option 2:** the consistency term from the label-aware study, where a masked copy of each sentence is pulled toward the full-sentence answers.
- **Score spread:** the standard deviation of a head's validation scores. A drop means the head is flattening toward one value.
- **Paired Δ with a 95% interval:** the difference between two setups computed seed by seed. If the interval crosses 0, the difference isn't distinguishable from seed noise.

---

## 15. Commands to reproduce

```bash
cd multitask-forked

# tests (on Modal)
modal run modal_replicate.py::masked_lm_tests

# smoke test, then the full teacher study
modal run --detach modal_replicate.py::teacher_consistency --seeds 42 --epochs 1 --smoke
modal run --detach modal_replicate.py::teacher_consistency

# wider confidence filters
modal run --detach modal_replicate.py::filter_sweep

# reports
modal run modal_replicate.py::label_aware_report --baseline famo --prefixes famo_tc,famo_consf1_m5,famo_consq
modal run modal_replicate.py::label_aware_report --baseline famo_consf1_m5 --prefixes famo_tc,famo_consq
modal run modal_replicate.py::label_aware_report --baseline famo_tc_drop --prefixes famo_tc_mt
modal run modal_replicate.py::label_aware_report --baseline famo_tc_ema --prefixes famo_tc_mt
```

Results: `multitask-outputs` volume, folder `paper_replication_retune_famo/`, files `<config>_s<seed>.json` and `<config>_s<seed>_predictions.npz`.

Launch tips learned in this session:
- Always use `--detach`.
- Launch independent studies in separate commands, not chained with `;`.
- Don't stop a launcher after it has spawned its runs. The runs keep going, and relaunching creates duplicates.
