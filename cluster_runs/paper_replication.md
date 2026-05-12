# Paper Replication Experiment Results

**Source run:** `cluster_runs/outputs/nopo-paper-replication-task-specific-with-weights.out`  
**Experiment:** `paper_replication` — hierarchical multi-task model with **per-task class weights** and **task-type weights** (ν_k ∝ 1/|D_k|, scaled to mean 1).

## Model Structure

- **Encoder:** HuggingFace (`EncoderIntegration.HUGGINGFACE`); `bert-en-uncased-l-2-h-128-a-2`.
- **Shared layers:** 2 dense layers.
- **Hierarchical branching:** level 0 → 3 branches; level 1 → [18, 1, 8] sub-branches (18 propaganda binaries, 1 WA_bi, 8 discourse binaries).
- **Total outputs:** 27.
- **Output pool:** `propaganda_max_pool` — max over 18 propaganda task logits → binary.

### Code-defined setup (`run_paper_replication` in `tests/test_no_prop_overlap.py`)

**Task typing for loss:** `task_type_mapping` assigns each propaganda column to `propaganda_binary`, `WA_bi` to `wa_bi`, and each discourse binary to `discourse` (used when task-type weights ν_k are enabled).

**`ModelConfig` (architecture):**

| Field | Value |
|--------|--------|
| `task_structure` | `[[3], [18, 1, 8]]` — three top-level branches; children: 18 propaganda heads, 1 WA head, 8 discourse heads. |
| `shared_layer_sizes` | `[64, 32]` — two dense layers on the encoder representation (code comment: 128→64→32 for the default small BERT width). |
| `branch_layer_sizes` | `[[32, 16], [16, 8]]` — first pair: per task-type branches; second pair: per-task heads. |
| `task_types` | `"binary"` for every task. |
| `num_classes_per_task` | `2` for every task. |
| `output_pools["propaganda_max_pool"]` | `tasks`: all 18 propaganda names; `strategy`: `"max"`; `task_type` / `num_classes`: binary with 2 classes. |

**Optimization settings** (`PAPER_TRAINING_CONFIG` in `test_no_prop_overlap.py`, plus `TrainingConfig` defaults for any field not overridden):

| Setting | Value |
|---------|--------|
| Optimizer | `adamw` |
| Learning rate | `3e-4` |
| Weight decay | `0.01` |
| Dropout | `0.4` |
| Training batch size | `256` |
| Warmup | `5%` (`warmup_percent`) |
| Epochs (max) | `20` (dataclass default) |

## Task-Type and Class Weighting (this run)

- **Class weights:** computed from training labels for all 27 tasks.
- **Task-type weights ν_k** (inverse dataset size per task type, mean-normalized to 1):

| Task type           | Tasks | \|D_k\| (printed) | ν_k (approx.) | ν_k · \|D_k\| |
|---------------------|------:|------------------:|--------------:|--------------:|
| propaganda_binary   |    18 |            10,112 |      2.97e-4 |          ≈3.0 |
| wa_bi               |     1 |                 0 |         2.999 |             0 |
| discourse           |     8 |             9,610 |      3.12e-4 |          ≈3.0 |

**WA_bi** (task 18) receives the large `wa_bi` ν_k while propaganda tasks share `propaganda_binary` ν and discourse tasks share `discourse` ν.

## Training Configuration

| Parameter        | Value  |
|------------------|--------|
| Epochs (max)     | 20     |
| Batch size       | 256    |
| Learning rate    | 3e-4   |
| Weight decay     | 0.01   |
| Dropout rate     | 0.4    |
| Steps per epoch  | 1712   |

**Early stopping:** triggered at **epoch 10**; best weights restored from **epoch 7** (best `val_loss` ≈ 35.68). Training `loss` decreased steadily (≈61.8 → ≈39.0).

**Threshold search:** average validation F1 before per-task thresholding ≈ **0.589**.

## Overall Results (macro-averaged over classification tasks)

| Metric             | Value   |
|--------------------|--------:|
| Accuracy           | 0.548   |
| Precision          | 0.221   |
| Recall             | 0.582   |
| F1                 | 0.245   |
| AUC                | 0.496   |
| Average Precision  | 0.212   |

## Pool Metric (`propaganda_max_pool`)

| Metric              | Value   |
|---------------------|--------:|
| Accuracy            | 0.563   |
| Precision           | 0.499   |
| Recall              | 0.387   |
| F1                  | 0.436   |
| AUC                 | 0.542   |
| Average Precision   | 0.472   |

## Task-Specific Results

### Propaganda techniques (18 tasks, branch 0)

| Task | Acc | Prec | Recall | F1 | AUC | AP |
|------|----:|-----:|-------:|---:|----:|---:|
| Loaded_Language | 0.742 | 0.169 | 0.189 | 0.179 | 0.496 | 0.154 |
| Slogans | 0.501 | 0.009 | 0.533 | 0.018 | 0.485 | 0.009 |
| Black-and-White_Fallacy | 0.773 | 0.008 | 0.273 | 0.015 | 0.418 | 0.007 |
| Flag-Waving | 0.040 | 0.014 | 0.960 | 0.027 | 0.430 | 0.012 |
| Name_Calling | 0.802 | 0.113 | 0.241 | 0.154 | 0.522 | 0.101 |
| Whataboutism | 0.731 | 0.002 | 0.500 | 0.004 | 0.815 | 0.005 |
| Causal_Oversimplification | 0.475 | 0.019 | 0.621 | 0.037 | 0.538 | 0.018 |
| Exaggeration_Minimisation | 0.629 | 0.046 | 0.417 | 0.083 | 0.510 | 0.042 |
| Doubt | 0.416 | 0.039 | 0.695 | 0.073 | 0.537 | 0.044 |
| Appeal_to_Authority | 0.887 | 0.016 | 0.150 | 0.029 | 0.347 | 0.010 |
| Repetition | 0.898 | 0.056 | 0.049 | 0.052 | 0.433 | 0.051 |
| Appeal_to_fear-prejudice | 0.603 | 0.031 | 0.550 | 0.059 | 0.581 | 0.034 |
| Red_Herring | 0.039 | 0.005 | 0.889 | 0.009 | 0.320 | 0.004 |
| Thought-terminating_Cliches | 0.532 | 0.006 | 0.714 | 0.012 | 0.623 | 0.007 |
| Reductio_ad_hitlerum | 0.497 | 0.006 | 0.625 | 0.011 | 0.534 | 0.006 |
| Bandwagon | 0.869 | 0.004 | 0.333 | 0.009 | 0.590 | 0.006 |
| Obfusc./Intentional_Vagueness | 0.444 | 0.000 | 0.000 | 0.000 | 0.356 | 0.001 |
| Straw_Men | 0.107 | 0.001 | 0.500 | 0.001 | 0.144 | 0.001 |

### Argument quality (1 task, branch 1)

| Task | Acc | Prec | Recall | F1 | AUC | AP |
|------|----:|-----:|-------:|---:|----:|---:|
| WA_bi | 0.665 | 0.975 | 0.660 | 0.787 | 0.729 | 0.971 |

### Discourse dimensions (8 tasks, branch 2)

| Task | Acc | Prec | Recall | F1 | AUC | AP |
|------|----:|-----:|-------:|---:|----:|---:|
| disagree_agree_bi | 0.613 | 0.801 | 0.676 | 0.733 | 0.535 | 0.800 |
| attacking_respectful_bi | 0.349 | 0.348 | 0.998 | 0.516 | 0.444 | 0.305 |
| emotion_fact_bi | 0.606 | 0.606 | 0.998 | 0.754 | 0.397 | 0.531 |
| nasty_nice_bi | 0.281 | 0.276 | 0.992 | 0.432 | 0.434 | 0.232 |
| personal_audience_bi | 0.664 | 0.808 | 0.727 | 0.766 | 0.617 | 0.816 |
| defeater_undercutter_bi | 0.627 | 0.631 | 0.964 | 0.763 | 0.488 | 0.609 |
| negotiate_attack_bi | 0.605 | 0.645 | 0.560 | 0.599 | 0.629 | 0.642 |
| questioning_asserting_bi | 0.387 | 0.335 | 0.913 | 0.490 | 0.452 | 0.299 |