# Testing Notes for IRL Data

Things to pay attention to
- Accuracy metrics
- Start small, work up, and see how the metrics change
- Make sure fine-tuning is working
  - `val_loss` should trend down overall
  - Final metrics show non-trivial behavior
  - Could also compare against a frozen encoder run and verify that the tuned run is better
- Make sure early stopping/best-checkpoint is working: look for "restoring model weights from the end of the best epoch"
- Make sure output is working
  - Look for optimal thresholds and generating predictions
  - Make sure evaluation results section prints expected metrics
    - Classification: accuracy/precision/recall/F1/AUC/AP
    - Regression: MSE/MAE/R2/Pearson

## Propaganda Dataset (`no_prop_overlap`)

### Single Branch

Establish single-task and flat-baseline performance before adding hierarchical sharing.

#### Propaganda Multilabel

A single multilabel output head predicts all 18 propaganda techniques with no hierarchical branching. This allows us to isolate the multilabel head to see base signal quality and threshold behavior without cross-task effects.

- Source: `cluster_runs/outputs/nopo-propaganda.out`
- Validation F1 during threshold search: 0.0469

| Metric | Value |
|---|---:|
| Accuracy | 0.0045 |
| Precision | 0.0851 |
| Recall | 0.5374 |
| F1 | 0.1357 |
| AUC | 0.4820 |
| Average Precision | 0.0277 |
| MSE | None |
| MAE | None |
| R2 | None |
| Pearson r | None |

#### Discourse Regression

Eight independent regression heads are trained in a flat single-level multitask setup. This allows us to measure continuous discourse signal quality and regression fit independently from classification losses.

- Source: `cluster_runs/outputs/nopo-discourse-dims.out`
- Validation F1 during threshold search: None (threshold optimization disabled)

| Metric | Value |
|---|---:|
| Accuracy | None |
| Precision | None |
| Recall | None |
| F1 | None |
| AUC | None |
| Average Precision | None |
| MSE | 2.7409 |
| MAE | 1.2550 |
| R2 | 0.1167 |
| Pearson r | 0.3575 |

#### Discourse Binary

Eight independent binary heads are trained in a flat single-level multitask setup. Benchmark easier classification variants of discourse dimensions and confirm threshold optimization works well.

- Source: `cluster_runs/outputs/nopo-discourse-binary.out`
- Validation F1 during threshold search: 0.5265
| Metric | Value |
|---|---:|
| Accuracy | 0.5584 |
| Precision | 0.5965 |
| Recall | 0.6965 |
| F1 | 0.5986 |
| AUC | 0.5227 |
| Average Precision | 0.5640 |
| MSE | None |
| MAE | None |
| R2 | None |
| Pearson r | None |

#### All Task Types (Flat Baseline)

All 17 heads (1 propaganda + 8 regression + 8 binary) share the same trunk and split only at the final output layer. Provides the reference multitask architecture where all heads share one path and only split at outputs.

- Experiment key: `all_task_types`
- Structure: `[[17]]` (1 propaganda multilabel + 8 regression + 8 binary)
- Source: `cluster_runs/outputs/nopo-all-task-types.out`
- Validation F1 during threshold search: 0.4821

| Metric | Value |
|---|---:|
| Accuracy | 0.4956 |
| Precision | 0.5106 |
| Recall | 0.5004 |
| F1 | 0.4335 |
| AUC | 0.6125 |
| Average Precision | 0.5763 |
| MSE | 2.7754 |
| MAE | 1.2688 |
| R2 | 0.1037 |
| Pearson r | 0.3235 |

- Notes:
  - Early stopping triggered at epoch 6; best epoch restored to 3.

### Multiple Branches

Test whether structured sharing (instead of fully flat sharing) improves transfer and reduces negative interference.

#### All Task Types (Grouped by Family)

The model first splits into three family branches (propaganda, regression, binary), then each family expands to its task heads.

- Experiment key: `all_task_types_grouped`
- Structure: `[[3], [1, 8, 8]]`
- Source: `cluster_runs/outputs/nopo-all-task-types-grouped.out`
- Validation F1 during threshold search: 0.4754

| Metric | Value |
|---|---:|
| Accuracy | 0.4923 |
| Precision | 0.4933 |
| Recall | 0.5156 |
| F1 | 0.4360 |
| AUC | 0.6264 |
| Average Precision | 0.5909 |
| MSE | 2.7429 |
| MAE | 1.2669 |
| R2 | 0.1142 |
| Pearson r | 0.3404 |

- Notes:
  - Best AUC/AP among all all-task structures so far, with slightly improved regression quality vs flat.

#### All Task Types (Deep Grouped: All -> Some -> Tasks)

The model splits into propaganda vs discourse, then discourse splits into regression vs binary, then each branch expands to task heads.

- Experiment key: `all_task_types_deep_grouped`
- Structure: `[[2], [1, 2], [1, 8, 8]]`
- Source: `cluster_runs/outputs/nopo-all-task-types-deep-grouped.out`
- Validation F1 during threshold search: 0.4991

| Metric | Value |
|---|---:|
| Accuracy | 0.5364 |
| Precision | 0.4677 |
| Recall | 0.5665 |
| F1 | 0.4607 |
| AUC | 0.5153 |
| Average Precision | 0.4998 |
| MSE | 3.0726 |
| MAE | 1.3497 |
| R2 | 0.0124 |
| Pearson r | 0.1574 |

- Notes:
  - Better recall than flat/grouped, but consistently worse probability-quality (AUC/AP) and regression metrics.

#### All Task Types (Pair-by-Dimension Hierarchy)

One propaganda branch stays separate while each discourse dimension forms a paired mini-branch of regression + binary sibling heads.

- Experiment key: `all_task_types_pair_by_dimension`
- Structure: `[[9], [1, 2, 2, 2, 2, 2, 2, 2, 2]]`
- Source: `cluster_runs/outputs/nopo-all-task-types-pair-by-dim.out`
- Validation F1 during threshold search: 0.5312

| Metric | Value |
|---|---:|
| Accuracy | 0.5276 |
| Precision | 0.5482 |
| Recall | 0.6238 |
| F1 | 0.5395 |
| AUC | 0.5779 |
| Average Precision | 0.5268 |
| MSE | 3.0417 |
| MAE | 1.3457 |
| R2 | -0.0001 |
| Pearson r | 0.1824 |

- Notes:
  - Strongest classification F1 so far, but regression quality drops substantially vs flat/grouped.

#### Quick Comparison Checklist

- Classification side: does F1/AP improve vs `all_task_types` without major AUC drop?
- Regression side: does R2/Pearson improve while MSE/MAE do not degrade badly?
- Training behavior: did `val_loss` remain stable and did best-epoch restore trigger?