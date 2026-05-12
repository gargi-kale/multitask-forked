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
| discourse           |     8 |             9,599 |      3.12e-4 |          ≈3.0 |

**WA_bi** (task 18) receives the large `wa_bi` ν_k while propaganda tasks share `propaganda_binary` ν and discourse tasks share `discourse` ν.

### Per-task class weights

| Task | class_0 | class_1 |
|------|--------:|--------:|
| Loaded_Language | 0.764902 | 1.44375 |
| Slogans | 0.509575 | 26.6105 |
| Black-and-White_Fallacy | 0.508959 | 28.4045 |
| Flag-Waving | 0.523233 | 11.2606 |
| Name_Calling | 0.621359 | 2.56 |
| Whataboutism | 0.505398 | 46.8148 |
| Causal_Oversimplification | 0.516604 | 15.5569 |
| Exaggeration_Minimisation | 0.542839 | 6.33584 |
| Doubt | 0.546417 | 5.88591 |
| Appeal_to_Authority | 0.511017 | 23.1927 |
| Repetition | 0.552448 | 5.26667 |
| Appeal_to_fear-prejudice | 0.521291 | 12.2421 |
| Red_Herring | 0.502734 | 91.9273 |
| Thought-terminating_Cliches | 0.506055 | 41.7851 |
| Reductio_ad_hitlerum | 0.50454 | 55.5604 |
| Bandwagon | 0.500842 | 297.412 |
| Obfuscation_Intentional_Vagueness_Confusion | 0.500892 | 280.889 |
| Straw_Men | 0.500594 | 421.333 |
| WA_bi | 1 | 1 |
| disagree_agree_bi | 0.001 | 0.5 |
| attacking_respectful_bi | 0.001 | 0.5 |
| emotion_fact_bi | 0.001 | 0.5 |
| nasty_nice_bi | 0.001 | 0.5 |
| personal_audience_bi | 0.76087 | 1.45833 |
| defeater_undercutter_bi | 0.5 | 0.001 |
| negotiate_attack_bi | 0.551858 | 5.32083 |
| questioning_asserting_bi | 0.766667 | 1.4375 |

## Training Configuration

| Parameter        | Value  |
|------------------|--------|
| Epochs (max)     | 20     |
| Batch size       | 256    |
| Learning rate    | 3e-4   |
| Weight decay     | 0.01   |
| Dropout rate     | 0.4    |
| Steps per epoch  | 1712   |

**Early stopping:** triggered at **epoch 8**; best weights restored from **epoch 5** (best `val_loss` ≈ 36.08). Training `loss` decreased from ≈59.5 to ≈39.6.

**Threshold search:** average validation F1 before per-task thresholding ≈ **0.659**.

## Overall Results (macro-averaged over classification tasks)

| Metric             | Value   |
|--------------------|--------:|
| Accuracy           | 0.623   |
| Precision          | 0.199   |
| Recall             | 0.484   |
| F1                 | 0.217   |
| AUC                | 0.541   |
| Average Precision  | 0.213   |

## Pool Metric (`propaganda_max_pool`)

| Metric              | Value   |
|---------------------|--------:|
| Accuracy            | 0.553   |
| Precision           | 0.477   |
| Recall              | 0.252   |
| F1                  | 0.330   |
| AUC                 | 0.492   |
| Average Precision   | 0.448   |

## Task-Specific Results

### Propaganda techniques (18 tasks, branch 0)

| Task | Acc | Prec | Recall | F1 | AUC | AP |
|------|----:|-----:|-------:|---:|----:|---:|
| Loaded_Language | 0.693 | 0.173 | 0.284 | 0.215 | 0.506 | 0.157 |
| Slogans | 0.731 | 0.011 | 0.333 | 0.020 | 0.502 | 0.011 |
| Black-and-White_Fallacy | 0.413 | 0.008 | 0.727 | 0.015 | 0.514 | 0.007 |
| Flag-Waving | 0.945 | 0.013 | 0.040 | 0.020 | 0.433 | 0.014 |
| Name_Calling | 0.472 | 0.085 | 0.624 | 0.150 | 0.535 | 0.081 |
| Whataboutism | 0.842 | 0.004 | 0.500 | 0.007 | 0.904 | 0.012 |
| Causal_Oversimplification | 0.447 | 0.022 | 0.759 | 0.043 | 0.592 | 0.028 |
| Exaggeration_Minimisation | 0.920 | 0.062 | 0.069 | 0.066 | 0.482 | 0.047 |
| Doubt | 0.744 | 0.046 | 0.339 | 0.080 | 0.558 | 0.057 |
| Appeal_to_Authority | 0.822 | 0.022 | 0.350 | 0.042 | 0.514 | 0.017 |
| Repetition | 0.889 | 0.064 | 0.069 | 0.066 | 0.433 | 0.051 |
| Appeal_to_fear-prejudice | 0.347 | 0.026 | 0.775 | 0.051 | 0.561 | 0.026 |
| Red_Herring | 0.955 | 0.000 | 0.000 | 0.000 | 0.332 | 0.005 |
| Thought-terminating_Cliches | 0.162 | 0.004 | 0.857 | 0.008 | 0.455 | 0.004 |
| Reductio_ad_hitlerum | 0.854 | 0.012 | 0.375 | 0.022 | 0.698 | 0.012 |
| Bandwagon | 0.506 | 0.002 | 0.667 | 0.005 | 0.675 | 0.005 |
| Obfuscation_Intentional_Vagueness_Confusion | 0.649 | 0.000 | 0.000 | 0.000 | 0.649 | 0.002 |
| Straw_Men | 0.842 | 0.000 | 0.000 | 0.000 | 0.562 | 0.003 |

### Argument quality (1 task, branch 1)

| Task | Acc | Prec | Recall | F1 | AUC | AP |
|------|----:|-----:|-------:|---:|----:|---:|
| WA_bi | 0.648 | 0.974 | 0.643 | 0.775 | 0.734 | 0.972 |

### Discourse dimensions (8 tasks, branch 2)

| Task | Acc | Prec | Recall | F1 | AUC | AP |
|------|----:|-----:|-------:|---:|----:|---:|
| disagree_agree_bi | 0.581 | 0.812 | 0.606 | 0.695 | 0.531 | 0.796 |
| attacking_respectful_bi | 0.382 | 0.349 | 0.892 | 0.501 | 0.445 | 0.303 |
| emotion_fact_bi | 0.394 | 0.000 | 0.000 | 0.000 | 0.391 | 0.527 |
| nasty_nice_bi | 0.318 | 0.278 | 0.924 | 0.428 | 0.434 | 0.233 |
| personal_audience_bi | 0.690 | 0.801 | 0.784 | 0.792 | 0.610 | 0.814 |
| defeater_undercutter_bi | 0.615 | 0.627 | 0.938 | 0.751 | 0.482 | 0.611 |
| negotiate_attack_bi | 0.608 | 0.639 | 0.593 | 0.615 | 0.642 | 0.678 |
| questioning_asserting_bi | 0.357 | 0.326 | 0.932 | 0.483 | 0.423 | 0.278 |