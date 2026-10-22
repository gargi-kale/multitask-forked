# Hyperparameter and network-size search: what we found

*20 October 2026*

Follow-up to `replication_findings_2026-10-15.md`. That write-up fixed the propaganda loss ("pooled loss") and found that dropout hurts, but it never tuned anything: every run used the paper's settings. This search tunes the settings and the network size.

---

## 1. The short version

- We ran **99 training runs** on Modal: 84 settings once each, then the 5 best on 3 more seeds.
- **Best setup:** the paper's own training settings (learning rate 3e-4, weight decay 0.01, 5% warm-up, batch 256), with **dropout 0** and a **wider network** (4× as many neurons per layer as the paper describes).
- It scores **66.2 ± 1.0** test F1 over 4 seeds, against **66.7** in the paper and **64.1 ± 2.2** for our earlier best. AUC rises from 66.5 to **68.3**.
- The paper's 66.7 is the best of 72 single runs. We now match it as an **average over seeds**, not only on a lucky run.
- **Dropout matters most.** Dropout 0.4, the paper's stated value, is the worst choice by far.
- **More neurons per layer help; more layers don't.** The legacy code's 4-layer blocks did no better than 2-layer blocks of the same width.

---

## 2. What we searched

Every run uses the library with the pooled propaganda loss and the same train/val/test split as before. Settings were **picked on the validation split only**. Test scores are reported but were never used to choose.

### Training settings

The paper's grid (from the commented-out loop in the legacy `main.py`) is in **bold**. We added the other values.

| Setting | Values tried |
|---|---|
| learning rate | 5e-5, 1e-4, **1.5e-4**, **3e-4**, 5e-4, 1e-3 |
| weight decay | 0, **0.01**, **0.05**, **0.1** |
| warm-up (% of training) | **5**, 10, **20** |
| dropout | 0, **0.1**, **0.25**, **0.4** |
| batch size | **32**, **64**, 128, 256 |

### Network size

The network is a tree. BERT's encoding of the text goes into **shared** layers used by every task. It then splits into one branch per **task family** (propaganda, argument quality, discourse: 3 branches), and each of those splits into one branch per **task** (18 propaganda techniques, 1 argument-quality task, 8 discourse tasks: 27 branches). Each task branch ends in its own yes/no output.

```
BERT encoding (128 numbers)
        │
     Shared            1 block, used by every task
        │
  ┌─────┼─────┐
 Prop   WA   Discourse  Task-family blocks (3)
  │     │     │
 18     1     8         Task blocks (27), each → yes/no output
```

Each cell below lists the number of neurons in each layer of one block, in order.

| Size | Shared | Task family (×3) | Task (×27) | Layers per block | Weights in these layers |
|---|---|---|---|---|---|
| paper | 64 → 32 | 32 → 16 | 16 → 8 | 2 | ~27k |
| medium | 128 → 64 | 64 → 32 | 32 → 16 | 2 | ~87k |
| wide | 256 → 128 | 128 → 64 | 64 → 32 | 2 | ~310k |
| legacy | 256 → 256 → 128 → 128 | 128 → 128 → 64 → 64 | 64 → 64 → 32 → 32 | 4 | ~595k |

- **paper** is what the paper describes, and what the library used until now.
- **medium** and **wide** keep the paper's 2 layers per block but put more neurons in each layer (2× and 4×).
- **legacy** is what the original research code actually builds: wide's neuron counts, with each layer repeated, so 4 layers per block.
- Weight counts are computed from the library's layer definitions. The paper-size figure (~27k) is a little below the ~32,500 quoted in the 15 October write-up, which was likely based on the paper's description of the network.

### How the runs were organised

The full grid of settings × sizes would be 4,608 runs, so we sampled it.

| Stage | Runs | What |
|---|---|---|
| 1 | 84 | The paper's settings (with dropout 0) at each of the 4 sizes, plus 80 random points from the tables above. Seed 42. |
| 2 | 15 | The 5 best stage-1 settings by validation F1, each re-run on seeds 1, 2 and 3. |

---

## 3. Results

### The five finalists (4 seeds each)

Mean ± standard deviation over seeds 42, 1, 2 and 3. Ranked by validation F1.

| Settings | Size | Val F1 | Test F1 | Test F1 range | Test AUC |
|---|---|---|---|---|---|
| **lr 3e-4, wd 0.01, warm-up 5, dropout 0, batch 256** | **wide** | **67.1 ± 1.0** | **66.2 ± 1.0** | 64.8 – 67.2 | **68.3 ± 0.7** |
| lr 1.5e-4, wd 0.05, warm-up 5, dropout 0.25, batch 32 | wide | 65.4 ± 1.8 | 64.9 ± 1.6 | 62.6 – 66.1 | 66.3 ± 0.4 |
| lr 1e-3, wd 0.1, warm-up 10, dropout 0, batch 256 | wide | 65.3 ± 1.2 | 64.7 ± 1.1 | 63.4 – 66.1 | 67.8 ± 0.4 |
| lr 3e-4, wd 0, warm-up 5, dropout 0, batch 64 | paper | 65.3 ± 1.6 | 64.7 ± 1.9 | 62.7 – 67.1 | 67.2 ± 0.8 |
| lr 1.5e-4, wd 0, warm-up 20, dropout 0, batch 32 | paper | 65.0 ± 2.3 | 64.4 ± 2.3 | 61.4 – 66.6 | 66.9 ± 1.0 |

The winner is the paper's own training settings with dropout 0, at the wide size. It's the best of the five on every column, and the steadiest across seeds.

### Everything side by side

Mean weighted F1 over the 10 tasks in the paper's Table 3, on the test split. Rows from "earlier work" come from the 15 October write-up.

| Setup | Runs | Test F1 | Test AUC | Gap to paper |
|---|---|---|---|---|
| **Paper** (best of a 72-run grid search) | — | **66.7** | — | — |
| *Earlier work* | | | | |
| Legacy code, paper settings | 2 | 58.2 / 57.3 | — | −8.5 |
| Library, original loss, dropout 0.4 (paper settings) | 1 | 39.2 | 50.7 | −27.5 |
| Library, original loss, dropout 0 / 0.1 | 1 each | 57.6 / 58.7 | 63.5 / 63.6 | about −8.5 |
| Library, pooled loss, dropout 0.4 | 1 | 56.7 | 59.9 | −10.0 |
| Library, pooled loss, dropout 0.1, paper size | 3 seeds | 64.2 ± 0.7 | 65.7 ± 0.7 | −2.5 |
| Library, pooled loss, dropout 0, paper size (earlier best) | 5 seeds | 64.1 ± 2.2 | 66.5 ± 0.5 | −2.6 |
| *This search* | | | | |
| All 84 stage-1 runs | 84 | median 63.0 (43.7 – 67.1) | — | — |
| Paper settings, dropout 0: paper / medium / wide / legacy size | 1 each | 65.3 / 64.7 / 66.5 / 64.5 | 66.1 / 66.9 / 68.1 / 67.1 | — |
| Best single stage-1 run (paper size, batch 64, wd 0) | 1 | 67.1 | 67.9 | +0.4 |
| … the same settings over 4 seeds | 4 seeds | 64.7 ± 1.9 | 67.2 ± 0.8 | −2.0 |
| **Winner: wide size, paper settings, dropout 0** | **4 seeds** | **66.2 ± 1.0** | **68.3 ± 0.7** | **−0.5** |

### The winner, task by task (4 seeds)

| Task | Paper | Test F1 (mean ± sd) | Range | Test AUC |
|---|---|---|---|---|
| Propaganda | 61.7 | **65.6 ± 0.8** | 64.7 – 66.7 | 68.4 |
| Disagree/Agree | 71.4 | 58.5 ± 8.2 | 46.3 – 63.7 | 61.3 |
| Emotion/Fact | 63.9 | **67.9 ± 1.9** | 66.0 – 69.8 | 76.9 |
| Attacking/Respectful | 68.1 | 67.6 ± 1.8 | 64.9 – 69.0 | 69.7 |
| Nasty/Nice | 73.7 | 71.1 ± 1.5 | 69.0 – 72.1 | 72.9 |
| Personal/Audience | 65.7 | 62.6 ± 3.4 | 58.3 – 66.5 | 60.8 |
| Defeater/Undercutter | 55.7 | **59.7 ± 2.1** | 56.9 – 61.9 | 59.3 |
| Negotiate/Attack | 64.8 | **69.2 ± 0.8** | 68.5 – 70.3 | 74.6 |
| Questioning/Asserting | 63.2 | 59.8 ± 4.3 | 55.1 – 64.9 | 60.5 |
| Argument Quality (WA) | 79.2 | **80.1 ± 2.0** | 78.3 – 82.8 | 78.8 |

The winner beats the paper on 5 of 10 tasks (in bold). The biggest remaining gap is **Disagree/Agree**. As in the earlier seed check, its score swings widely between seeds (46 to 64) while its AUC stays low, because the tuned yes/no cut-off is unstable for this lopsided task (about 79% "agree").

Compared with the earlier best (paper size, 5 seeds), Disagree/Agree improved from 47.9 ± 17.5 to 58.5 ± 8.2: higher on average and half as variable.

---

## 4. What each setting does

Median test F1 of the 84 stage-1 runs (one seed each), grouped by each setting's value.

| Setting | Value: median test F1 (number of runs) |
|---|---|
| **dropout** | 0: **65.0** (29) · 0.1: 64.0 (15) · 0.25: 62.2 (20) · 0.4: 57.7 (20) |
| **size** | paper: 62.1 (24) · medium: 63.8 (22) · wide: 63.4 (20) · legacy: 63.5 (18) |
| size, median test AUC | paper: 64.4 · medium: 65.6 · **wide: 66.7** · legacy: 65.5 |

Learning rate, weight decay, warm-up and batch size each moved the median by about 1 point or less, apart from very low learning rates, which did worse. The paper's learning rate of 3e-4 had the best median of the six values tried.

**What this tells us:**

1. **Dropout matters most.** Every step up in dropout costs F1, and 0.4 costs about 7 points on its own. The three worst runs (44–52 F1) all used dropout 0.4. This supports the 15 October finding that the paper's numbers were produced without dropout, because the legacy code never applied it.
2. **More neurons per layer help.** At the paper's settings, the wide network beat the paper size (66.5 vs 65.3 on one seed, then 66.2 ± 1.0 over 4 seeds against the earlier 64.1 ± 2.2). Across all runs, wide had the best median AUC, and the bigger sizes were more forgiving of poor settings.
3. **More layers don't help.** The legacy size has the same neuron counts as wide but twice as many layers, and it did no better (64.5 vs 66.5 at the paper's settings). So if the legacy code's larger network matters, it's the width, not the depth.
4. **The paper's training settings were already close to the best.** Once dropout is 0, the winner uses exactly the paper's learning rate, weight decay, warm-up and batch size.

---

## 5. Why single runs mislead

- **The stage-1 leaders fell back.** The best single stage-1 run scored 67.1 on test. Over 4 seeds, the same settings averaged 64.7. The other finalists also dropped 1–2 points. Picking the best of many single runs reliably overstates a setup.
- **The same seed doesn't give the same result.** The paper-size run at the paper's settings and seed 42 scored 66.4 on 15 October and 65.3 in this search, with identical code paths and settings. TensorFlow on a GPU isn't fully deterministic even with a fixed seed, so single runs carry about 1 point of noise from this alone.
- **So the paper's 66.7 is probably optimistic.** It's the best of 72 single runs. Our winner gets within 0.5 points of it as a 4-seed average, and its best seed (67.2) is above it.

---

## 6. Caveats

- **Sampled, not exhaustive.** Stage 1 covered 84 of 4,608 possible combinations. Every value of every setting was tried 7–37 times, but there could be a better combination we didn't sample.
- **Four seeds per finalist.** That's enough to separate the winner (66.2 ± 1.0) from the other finalists (64.4–64.9), but the ± values are themselves rough.
- **Selection on validation.** The winner was chosen on validation F1 and its test score was never used to choose, so the test numbers are a fair estimate. The cut-offs for each task are also tuned on validation, which makes validation F1 a little optimistic. It's still a fair basis for comparing settings, because every run is treated the same way.
- **The legacy code itself wasn't re-tuned.** All runs used the library. The legacy rows in the comparison are the single runs from the 15 October write-up.

---

## 7. Reproducing this

New command-line flags in `replication/test_no_prop_overlap.py` (all for `paper_replication` only):

- `--arch {paper,medium,wide,legacy}`: network size (the table in section 2).
- `--weight-decay`, `--warmup-percent`: override the paper's 0.01 and 5.
- `--learning-rate` now takes effect. Before, the hardcoded paper settings silently overrode it. Earlier results aren't affected, because they all used 3e-4 either way.

Running the search on Modal:

```bash
modal run --detach modal_replicate.py::sweep --stage 1 --n-random 80
modal run --detach modal_replicate.py::sweep --stage 2 --top 5 --seeds 1,2,3
modal run modal_replicate.py::sweep_report
```

- Each run saves `paper_replication_sweep/<settings>_s<seed>.json` on the `multitask-outputs` volume. Relaunching a command skips runs that already have results, so an interrupted search can be resumed.
- `sweep_report` ranks settings by mean validation F1 and writes every run to `replication/sweep_results.csv`.
- Always use `--detach`. A run started without it stops when the local machine loses its connection, which happened once during this search.

Running the winning setup on its own:

```bash
modal run modal_replicate.py::replicate --pooled-loss --arch wide --dropout 0 \
    --learning-rate 3e-4 --weight-decay 0.01 --warmup-percent 5 --batch-size 256 \
    --data-dir /data/paper --seed 42 --tag wide_best_seed42
```

**Cost:** about 100 runs on L4 GPUs, 10 at a time. Runs at batch 256 took about 25 minutes each, and runs at batch 32 about 80–100 minutes.

## 8. Suggested next steps

1. **Make the wide network the default** for the library's paper replication, with dropout 0. Report results as an average over at least 3–5 seeds.
2. **Stabilise Disagree/Agree.** Most of the remaining gap to the paper, and most of the seed-to-seed spread, comes from this one task's cut-off. Choosing the cut-off on a smoother criterion, or averaging it across seeds, could help.
3. **Try even wider networks** (for example 512 → 256 at the shared level). Width helped and we didn't test past wide.
4. **Run the paper's own 72-point grid** on the legacy code with the dropout bug fixed, to see what the network the paper describes would actually have scored.
