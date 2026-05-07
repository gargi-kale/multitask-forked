# Paper Replication Experiment Results

## Model Structure

- **Encoder:** HuggingFace `bert-en-uncased-l-2-h-128-a-2`
- **Shared Layers:** 2 dense layers
- **Hierarchical Branching:** Level 0 → 3 branches; Level 1 → [18, 1, 8] sub-branches
- **Total Outputs:** 27
- **Output Pool:** `propaganda_max_pool` — max over 18 propaganda tasks → binary

## Training Configuration

| Parameter        | Value  |
|------------------|--------|
| Epochs           | 20     |
| Batch size       | 256    |
| Learning rate    | 3e-4   |
| Weight decay     | 0.01   |
| Dropout rate     | 0.4    |
| Steps per epoch  | 1712   |
| Optimizer        | adamw  |
| Warmup Rate      | 5%     |

**Early stopping** triggered at epoch 4; best weights restored from epoch 1.

## Overall Results

| Metric             | Value  |
|--------------------|--------|
| Accuracy           | 0.511  |
| Precision          | 0.224  |
| Recall             | 0.572  |
| F1                 | 0.230  |
| AUC                | 0.481  |
| Average Precision  | 0.217  |

**Pool metric** (`propaganda_max_pool`): accuracy=0.564, AUC=0.407, AP=0.377, F1=0.000 (no positive predictions — threshold was `inf`).

## Task-Specific Results

### Propaganda Techniques (18 tasks, branch 0)

| Task | Acc | Prec | Recall | F1 | AUC | AP |
|------|-----|------|--------|----|-----|----|
| Loaded_Language | 0.158 | 0.149 | 0.996 | 0.260 | 0.457 | 0.131 |
| Name_Calling | 0.077 | 0.074 | 0.992 | 0.138 | 0.415 | 0.060 |
| Repetition | 0.747 | 0.074 | 0.294 | 0.118 | 0.531 | 0.064 |
| Exaggeration_Minimisation | 0.662 | 0.045 | 0.361 | 0.080 | 0.496 | 0.042 |
| Doubt | 0.157 | 0.035 | 0.915 | 0.067 | 0.482 | 0.032 |
| Slogans | 0.945 | 0.023 | 0.133 | 0.039 | 0.526 | 0.022 |
| Appeal_to_Authority | 0.714 | 0.020 | 0.500 | 0.038 | 0.594 | 0.023 |
| Flag-Waving | 0.015 | 0.013 | 0.960 | 0.027 | 0.351 | 0.010 |
| Causal_Oversimplification | 0.018 | 0.016 | 0.966 | 0.031 | 0.323 | 0.011 |
| Black-and-White_Fallacy | 0.946 | 0.011 | 0.091 | 0.020 | 0.293 | 0.007 |
| Reductio_ad_hitlerum | 0.792 | 0.008 | 0.375 | 0.016 | 0.578 | 0.013 |
| Red_Herring | 0.519 | 0.007 | 0.667 | 0.014 | 0.572 | 0.007 |
| Thought-terminating_Cliches | 0.699 | 0.004 | 0.286 | 0.007 | 0.499 | 0.005 |
| Bandwagon | 0.787 | 0.003 | 0.333 | 0.005 | 0.594 | 0.010 |
| Straw_Men | 0.297 | 0.001 | 0.500 | 0.002 | 0.429 | 0.001 |
| Whataboutism | 0.114 | 0.001 | 0.500 | 0.001 | 0.197 | 0.001 |
| Appeal_to_fear-prejudice | 0.978 | 0.000 | 0.000 | 0.000 | 0.440 | 0.020 |
| Obfusc./Intentional_Vagueness | 0.295 | 0.000 | 0.000 | 0.000 | 0.294 | 0.001 |

### Argument Quality (1 task, branch 1)

| Task | Acc | Prec | Recall | F1 | AUC | AP |
|------|-----|------|--------|----|-----|----|
| WA_bi | 0.730 | 0.968 | 0.737 | 0.837 | 0.733 | 0.974 |

### Discourse Dimensions (8 tasks, branch 2)

| Task | Acc | Prec | Recall | F1 | AUC | AP |
|------|-----|------|--------|----|-----|----|
| personal_audience_bi | 0.727 | 0.757 | 0.939 | 0.838 | 0.435 | 0.712 |
| emotion_fact_bi | 0.607 | 0.607 | 0.995 | 0.754 | 0.431 | 0.562 |
| negotiate_attack_bi | 0.625 | 0.621 | 0.745 | 0.678 | 0.646 | 0.662 |
| defeater_undercutter_bi | 0.585 | 0.658 | 0.691 | 0.674 | 0.536 | 0.633 |
| attacking_respectful_bi | 0.457 | 0.370 | 0.795 | 0.505 | 0.533 | 0.364 |
| questioning_asserting_bi | 0.592 | 0.410 | 0.606 | 0.489 | 0.629 | 0.452 |
| nasty_nice_bi | 0.287 | 0.278 | 0.992 | 0.434 | 0.452 | 0.244 |
| disagree_agree_bi | 0.268 | 0.905 | 0.077 | 0.143 | 0.509 | 0.802 |

## Key Observations

- **Early stopping at epoch 4** (best epoch 1) suggests the model overfit quickly; validation loss never improved past the first epoch.
- **Propaganda tasks** have very low F1 across the board, largely due to extreme class imbalance (task weights range from 0.003 to 26.7). Most thresholds collapsed to near-zero, producing high recall but near-zero precision.
- **Argument Quality** (`WA_bi`) is the strongest single task (F1=0.837, AP=0.974).
- **Discourse dimensions** perform moderately, with `personal_audience_bi` and `emotion_fact_bi` achieving the best F1 (0.838 and 0.754).
- **`propaganda_max_pool`** received a threshold of `inf`, meaning no positive predictions were ever made (F1=0.000).
