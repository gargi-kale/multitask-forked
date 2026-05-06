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
- Structure: `[[1]]` (single multilabel propaganda head)
- Validation F1 during threshold search: 0.0469
- Run metadata:
  - Best epoch: 1
  - Training stopped at epoch: 4 (early stopping)
  - Train time: ~474s (~7m54s)
  - Checkpoint restored: Yes
  - Thresholding mode: Enabled (validation F1 reported)
  - Primary takeaway: Simple multilabel baseline; low precision/F1 and weak probability quality (AUC/AP)

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

#### Propaganda Columns as Binary Heads

Each propaganda technique column is treated as its own binary head (18 heads total) rather than one multilabel head. This tests whether independent binary losses train more stably than a single multilabel head.

- Experiment key: `propaganda_cols_binary`
- Structure: `[[18]]`
- Source: `cluster_runs/outputs/nopo-propaganda-cols-binary.out`
- Validation F1 during threshold search: 0.6496
- Run metadata:
  - Best epoch: 1
  - Training stopped at epoch: 4 (early stopping)
  - Train time: ~794s (~13m14s)
  - Checkpoint restored: Yes
  - Thresholding mode: Enabled (validation F1 reported)
  - Primary takeaway: Per-technique binary heads improve over multilabel baseline, but aggregate final F1 remains low

| Metric | Value |
|---|---:|
| Accuracy | 0.5396 |
| Precision | 0.0343 |
| Recall | 0.5175 |
| F1 | 0.0500 |
| AUC | 0.5421 |
| Average Precision | 0.0335 |
| MSE | None |
| MAE | None |
| R2 | None |
| Pearson r | None |

#### Discourse Regression

Eight independent regression heads are trained in a flat single-level multitask setup. This allows us to measure continuous discourse signal quality and regression fit independently from classification losses.

- Source: `cluster_runs/outputs/nopo-discourse-dims.out`
- Structure: `[[8]]` (8 regression heads)
- Validation F1 during threshold search: None (threshold optimization disabled)
- Run metadata:
  - Best epoch: 3
  - Training stopped at epoch: 6 (early stopping)
  - Train time: ~833s (~13m53s)
  - Checkpoint restored: Yes
  - Thresholding mode: Default thresholds (optimization disabled)
  - Primary takeaway: Best standalone regression-only baseline in this set

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
- Structure: `[[8]]` (8 binary heads)
- Validation F1 during threshold search: 0.5265
- Run metadata:
  - Best epoch: 1
  - Training stopped at epoch: 4 (early stopping)
  - Train time: ~585s (~9m45s)
  - Checkpoint restored: Yes
  - Thresholding mode: Enabled (validation F1 reported)
  - Primary takeaway: Strongest standalone classification metrics (F1/precision/recall)

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
- Run metadata:
  - Best epoch: 3
  - Training stopped at epoch: 6
  - Train time: ~1115s (~18m35s)
  - Checkpoint restored: Yes
  - Thresholding mode: Enabled (classification heads), disabled for regression heads
  - Primary takeaway: Flat multitask baseline for all-task comparisons

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

#### All Task Types (Binary Propaganda Flat Baseline)

All outputs are binary/regression heads: 18 binary propaganda heads + 8 regression discourse heads + 8 binary discourse heads. This isolates the effect of replacing multilabel propaganda with binary heads while keeping a flat topology.

- Experiment key: `all_task_types_binary_prop`
- Structure: `[[34]]` (18 propaganda binary + 8 regression + 8 binary discourse)
- Source: `cluster_runs/outputs/nopo-all-task-types-bin-prop.out`
- Validation F1 during threshold search: 0.6284
- Run metadata:
  - Best epoch: 10
  - Training stopped at epoch: 10 (full run)
  - Train time: ~2455s (~40m55s)
  - Checkpoint restored: Yes
  - Thresholding mode: Enabled (classification heads), disabled for regression heads
  - Primary takeaway: Binary propaganda variant of flat all-task baseline

| Metric | Value |
|---|---:|
| Accuracy | 0.5626 |
| Precision | 0.2265 |
| Recall | 0.5916 |
| F1 | 0.2302 |
| AUC | 0.5939 |
| Average Precision | 0.2321 |
| MSE | 2.7137 |
| MAE | 1.2471 |
| R2 | 0.1363 |
| Pearson r | 0.3608 |

### Multiple Branches

Test whether structured sharing (instead of fully flat sharing) improves transfer and reduces negative interference.

#### All Task Types (Grouped by Family)

The model first splits into three family branches (propaganda, regression, binary), then each family expands to its task heads.

- Experiment key: `all_task_types_grouped`
- Structure: `[[3], [1, 8, 8]]`
- Source: `cluster_runs/outputs/nopo-all-task-types-grouped.out`
- Validation F1 during threshold search: 0.4754
- Run metadata:
  - Best epoch: 10
  - Training stopped at epoch: 10 (full run)
  - Train time: ~1829s (~30m29s)
  - Checkpoint restored: Yes
  - Thresholding mode: Enabled (classification heads), disabled for regression heads
  - Primary takeaway: Family-grouped hierarchy with best AUC/AP so far

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

#### All Task Types (Binary Propaganda Grouped)

The model branches into three families (propaganda binary, discourse regression, discourse binary), then expands to task heads. This checks whether grouped sharing still helps when propaganda is no longer multilabel.

- Experiment key: `all_task_types_binary_prop_grouped`
- Structure: `[[3], [18, 8, 8]]`
- Source: `cluster_runs/outputs/nopo-all-task-bin-prop-grouped.out`
- Validation F1 during threshold search: 0.6335
- Run metadata:
  - Best epoch: 10
  - Training stopped at epoch: 10 (full run)
  - Train time: ~2134s (~35m34s)
  - Checkpoint restored: Yes
  - Thresholding mode: Enabled (classification heads), disabled for regression heads
  - Primary takeaway: Grouped hierarchy with binary propaganda heads

| Metric | Value |
|---|---:|
| Accuracy | 0.5554 |
| Precision | 0.2265 |
| Recall | 0.5568 |
| F1 | 0.2246 |
| AUC | 0.5689 |
| Average Precision | 0.2313 |
| MSE | 2.7361 |
| MAE | 1.2532 |
| R2 | 0.1269 |
| Pearson r | 0.3526 |

#### All Task Types (Deep Grouped: All -> Some -> Tasks)

The model splits into propaganda vs discourse, then discourse splits into regression vs binary, then each branch expands to task heads.

- Experiment key: `all_task_types_deep_grouped`
- Structure: `[[2], [1, 2], [1, 8, 8]]`
- Source: `cluster_runs/outputs/nopo-all-task-types-deep-grouped.out`
- Validation F1 during threshold search: 0.4991
- Run metadata:
  - Best epoch: 1
  - Training stopped at epoch: 4 (early stopping)
  - Train time: ~805s (~13m25s)
  - Checkpoint restored: Yes
  - Thresholding mode: Enabled (classification heads), disabled for regression heads
  - Primary takeaway: Deeper hierarchy boosts recall but hurts AUC/AP and regression quality

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

#### All Task Types (Binary Propaganda Deep Grouped)

The model splits into propaganda vs discourse, then discourse splits into regression vs binary, and finally expands to 18/8/8 task heads. This tests deeper staged specialization with binary propaganda heads.

- Experiment key: `all_task_types_binary_prop_deep_grouped`
- Structure: `[[2], [1, 2], [18, 8, 8]]`
- Source: `cluster_runs/outputs/nopo-all-task-bin-prop-deep.out`
- Validation F1 during threshold search: 0.6001
- Run metadata:
  - Best epoch: 10
  - Training stopped at epoch: 10 (full run)
  - Train time: ~2128s (~35m28s)
  - Checkpoint restored: Yes
  - Thresholding mode: Enabled (classification heads), disabled for regression heads
  - Primary takeaway: Deep grouped hierarchy with binary propaganda heads

| Metric | Value |
|---|---:|
| Accuracy | 0.5254 |
| Precision | 0.2255 |
| Recall | 0.5554 |
| F1 | 0.2133 |
| AUC | 0.5600 |
| Average Precision | 0.2269 |
| MSE | 2.7556 |
| MAE | 1.2610 |
| R2 | 0.1161 |
| Pearson r | 0.3371 |

#### All Task Types (Pair-by-Dimension Hierarchy)

One propaganda branch stays separate while each discourse dimension forms a paired mini-branch of regression + binary sibling heads.

- Experiment key: `all_task_types_pair_by_dimension`
- Structure: `[[9], [1, 2, 2, 2, 2, 2, 2, 2, 2]]`
- Source: `cluster_runs/outputs/nopo-all-task-types-pair-by-dim.out`
- Validation F1 during threshold search: 0.5312
- Run metadata:
  - Best epoch: 1
  - Training stopped at epoch: 4 (early stopping)
  - Train time: ~827s (~13m47s)
  - Checkpoint restored: Yes
  - Thresholding mode: Enabled (classification heads), disabled for regression heads
  - Primary takeaway: Strong classification F1, weaker regression performance

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

#### Paper replication (hierarchical TFHub, 27 heads)

Hierarchical multitask layout aligned with the paper-replication experiment: TFHub encoder, two dense shared layers, first split into three branches (`[3]`), then 18 propaganda + 1 WA + 8 discourse-binary heads (`[18, 1, 8]`). Trains all 27 classification outputs (18 propaganda techniques, `WA_bi`, and eight pairwise discourse binaries).

- Experiment key: `paper_replication`
- Structure: `[[3], [18, 1, 8]]`
- Source: `cluster_runs/outputs/nopo-paper-replication.out`
- Validation F1 during threshold search: 0.5277
- Run metadata:
  - Best epoch: 1 (lowest `val_loss` in log)
  - Training stopped at epoch: 4 (early stopping)
  - Train time: ~689s (~11m30s) across logged epochs (1712 steps/epoch, batch 128)
  - Checkpoint restored: Yes (`Restoring model weights from the end of the best epoch: 1`)
  - Hyperparameters (from log): 10 epochs max, lr `4e-05`, weight decay `0.01`, dropout `0.1`
  - Thresholding mode: Enabled (per-task thresholds printed)
  - Primary takeaway: Training overfits quickly (`val_loss` rises after epoch 1); aggregate classification metrics are weak (F1 ~0.23, AUC ~0.52) with high recall and low precision. Task-weight line shows one propaganda head (`Straw_Men`, index 17) with a very large weight (~26.7) vs ~0.003 for most others—worth checking imbalance handling vs intended paper setup.

| Metric | Value |
|---|---:|
| Accuracy | 0.4531 |
| Precision | 0.2232 |
| Recall | 0.6361 |
| F1 | 0.2285 |
| AUC | 0.5232 |
| Average Precision | 0.2189 |
| MSE | None |
| MAE | None |
| R2 | None |
| Pearson r | None |

#### Quick Comparison Checklist

- Classification side: does F1/AP improve vs `all_task_types` without major AUC drop?
- Regression side: does R2/Pearson improve while MSE/MAE do not degrade badly?
- Training behavior: did `val_loss` remain stable and did best-epoch restore trigger?

## Cross-Experiment Summary Table

Legend: `HIGH` marks the highest numeric value in a row, `LOW` marks the lowest numeric value in a row (ignoring `None`).

| Metric | propaganda | propaganda_cols_binary | discourse_dims | discourse_binary | all_task_types | all_task_types_binary_prop | all_task_types_grouped | all_task_types_binary_prop_grouped | all_task_types_deep_grouped | all_task_types_binary_prop_deep_grouped | all_task_types_pair_by_dimension | paper_replication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Structure | [[1]] | [[18]] | [[8]] | [[8]] | [[17]] | [[34]] | [[3], [1, 8, 8]] | [[3], [18, 8, 8]] | [[2], [1, 2], [1, 8, 8]] | [[2], [1, 2], [18, 8, 8]] | [[9], [1, 2, 2, 2, 2, 2, 2, 2, 2]] | [[3], [18, 1, 8]] |
| Accuracy | **LOW 0.0045** | 0.5396 | None | 0.5584 | 0.4956 | **HIGH 0.5626** | 0.4923 | 0.5554 | 0.5364 | 0.5254 | 0.5276 | 0.4531 |
| Precision | 0.0851 | **LOW 0.0343** | None | **HIGH 0.5965** | 0.5106 | 0.2265 | 0.4933 | 0.2265 | 0.4677 | 0.2255 | 0.5482 | 0.2232 |
| Recall | 0.5374 | 0.5175 | None | **HIGH 0.6965** | **LOW 0.5004** | 0.5916 | 0.5156 | 0.5568 | 0.5665 | 0.5554 | 0.6238 | 0.6361 |
| F1 | 0.1357 | **LOW 0.0500** | None | **HIGH 0.5986** | 0.4335 | 0.2302 | 0.4360 | 0.2246 | 0.4607 | 0.2133 | 0.5395 | 0.2285 |
| AUC | **LOW 0.4820** | 0.5421 | None | 0.5227 | 0.6125 | 0.5939 | **HIGH 0.6264** | 0.5689 | 0.5153 | 0.5600 | 0.5779 | 0.5232 |
| Average Precision | **LOW 0.0277** | 0.0335 | None | 0.5640 | 0.5763 | 0.2321 | **HIGH 0.5909** | 0.2313 | 0.4998 | 0.2269 | 0.5268 | 0.2189 |
| MSE | None | None | 2.7409 | None | 2.7754 | **LOW 2.7137** | 2.7429 | 2.7361 | **HIGH 3.0726** | 2.7556 | 3.0417 | None |
| MAE | None | None | 1.2550 | None | 1.2688 | **LOW 1.2471** | 1.2669 | 1.2532 | **HIGH 1.3497** | 1.2610 | 1.3457 | None |
| R2 | None | None | 0.1167 | None | 0.1037 | **HIGH 0.1363** | 0.1142 | 0.1269 | 0.0124 | 0.1161 | **LOW -0.0001** | None |
| Pearson r | None | None | 0.3575 | None | 0.3235 | **HIGH 0.3608** | 0.3404 | 0.3526 | **LOW 0.1574** | 0.3371 | 0.1824 | None |