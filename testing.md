# Testing Plan

This document outlines all testing tasks for the `multitask` library, covering toy data, real-world data, all functionality, all encoder types, and error handling.

**Checklist:** `[x]` = covered by automated tests under [`tests/`](tests/) (run `pytest tests/`). Unchecked items are not yet covered by that suite or are manual / real-data only.

---

## Where to Find Things

| Item | Location |
|------|----------|
| Main package | [multitask/](multitask/) |
| Model class | [multitask/models/hierarchical.py](multitask/models/hierarchical.py) |
| Config classes | [multitask/config.py](multitask/config.py) |
| Trainer | [multitask/trainers/trainer.py](multitask/trainers/trainer.py) |
| Encoders (API) | [multitask/models/encoders.py](multitask/models/encoders.py) |
| Persistence | [multitask/models/persistence.py](multitask/models/persistence.py) |
| Distributed utils | [multitask/utils/distributed.py](multitask/utils/distributed.py) |
| Existing tests | [tests/](tests/) |
| Examples | [examples/](examples/) |
| Docs | [docs/](docs/) |
| Real-world data | [data/](data/) |

### Data Files

| File | Description | Task types |
|------|-------------|------------|
| `data/CMP_train/val/test.csv` | Comparative Manifesto Project — ~75 policy category columns (0/1) | Multilabel |
| `data/fallacy_train_aug/val_clean/test_clean.csv` | Fallacy detection — `updated_label` column, `aug_type` column | Multiclass |
| `data/no_prop_overlap_train/val/test.csv` | Propaganda + discourse — propaganda techniques (multilabel), continuous discourse dimensions, binary discourse dimensions | Multilabel + regression + binary |

---

## Part 1 — Toy Data Tests

Use small synthetic datasets (e.g., random embeddings, `np.random.randn(N, D)`) so tests run fast without GPU or API keys. Suggested sizes: N=256 samples, embedding_dim=64, batch_size=32, epochs=3.

### 1.1 Individual Task Types

Test each task type in isolation (flat `task_structure=[[1]]`). Confirm output shape, loss decreases, and metrics are returned correctly.

- [x] **Binary** — single binary output; check sigmoid output ∈ (0,1), accuracy and F1 returned (`test_task_types.py::TestSingleBinary`)
- [x] **Multiclass (3-class)** — check softmax output shape `[batch, 3]`, accuracy returned (`test_task_types.py::TestSingleMulticlass`; also 4- and 5-class)
- [x] **Multiclass (10-class)** — verify nothing breaks with many classes (`test_task_types.py::TestSingleMulticlass::test_multiclass_10_classes_smoke`)
- [x] **Multilabel (3-label)** — check sigmoid per-label output, threshold application (`test_task_types.py::TestSingleMultilabel`)
- [x] **Regression** — check linear output, MSE/MAE/R² returned; verify no thresholds applied (`test_task_types.py::TestSingleRegression`)
- [x] **Regression with standardization** — set `standardize_regression=True` (if applicable), verify outputs rescaled (`test_regression_std.py`)

### 1.2 Mixed Task Type Combinations

Use `task_structure=[[N]]` (flat) unless noted. Verify combined loss, combined metrics dict, and that masking works across heterogeneous types.

- [x] Binary + Binary (`test_task_types.py::TestPairwiseCombinations`)
- [x] Binary + Multiclass
- [x] Binary + Multilabel
- [x] Binary + Regression
- [x] Multiclass + Multiclass
- [x] Multiclass + Multilabel
- [x] Multiclass + Regression
- [x] Multilabel + Multilabel
- [x] Multilabel + Regression
- [x] Regression + Regression
- [x] Binary + Multiclass + Multilabel — (`test_task_types.py::TestTripleCombinations::test_binary_multiclass_multilabel`)
- [x] Binary + Multiclass + Regression (`test_task_types.py::TestTripleCombinations`)
- [x] Binary + Multilabel + Regression
- [x] Multiclass + Multilabel + Regression
- [x] **All four types together** — one binary, one multiclass, one multilabel, one regression task (`test_task_types.py::TestAllFourTypes`)

### 1.3 Hierarchical Branching Structures

Test non-trivial `task_structure` values. Confirm model builds without error and `summary_tree()` output matches the expected topology.

- [x] Flat structure: `[[3]]` (3 tasks, no branching) — e.g. `test_task_types.py` triple combinations use `[[3]]`; flat multi-task tests use `[[N]]`
- [x] 2-level branching — e.g. `[[3], [2, 5, 3]]` in `test_hierarchical.py`, `test_hierarchical_encoder.py`, `test_pooling.py` (exact `[[2], [1, 2]]` not required in tests)
- [x] 3-level — `test_hierarchical.py::TestThreeLevelBranching`, `test_pooling_hierarchical.py::TestDifferentTaskStructures`
- [x] Single branch with many tasks: `[[5]]` — `tests/conftest.py::mismatched_config`; `test_optimizers.py` five-task model
- [x] Asymmetric branches — `test_hierarchical.py::test_asymmetric_3level_structure`; `test_pooling.py` / `test_pooling_hierarchical.py`
- [x] Run `model.summary_tree()` and verify it prints without error and reflects structure — `test_hierarchical.py::TestSummaryTree`, `test_pooling_hierarchical.py::TestSummaryTreeWithPools`
- [x] Run `model.get_task_info()` and check returned dicts contain expected keys — `test_hierarchical.py::TestGetTaskInfo`, `test_hierarchical_encoder.py`

### 1.4 Output Pooling

Use a flat model with all binary tasks, then configure `output_pools`.

- [x] Pool with `strategy='max'` over 3 binary tasks — verify pool output ∈ (0,1) (`test_pooling_hierarchical.py::TestPooledOutputsMax`)
- [x] Pool with `strategy='mean'` (`test_pooling_hierarchical.py::TestPooledOutputsMean`)
- [x] Pool with `strategy='min'` (`test_pooling_hierarchical.py::TestPooledOutputsMin`)
- [x] Verify pool output appears as a key in `model(inputs)` dict (`test_pooling_hierarchical.py`, `test_pooling_full.py`)
- [x] Run `model.get_pool_info()` and confirm correct tasks listed (`test_pooling_hierarchical.py::TestGetPoolInfo`)
- [x] Evaluate pool metrics via `trainer.evaluate()` and confirm pool metrics in returned dict (`test_pooling_trainer.py`)
- [x] Verify that adding a pooled output does not change individual task outputs (`test_pooling_trainer.py::test_task_metrics_unchanged`, `test_pooling.py::TestPoolsWithExistingFeatures`)

### 1.5 Label Masking

- [x] Introduce `mask_value=-1.0` labels for some samples on some tasks; verify loss is not affected by masked positions (`test_masking.py`, `test_trainer.py::test_masking_and_normalization_logic`)
- [x] Mask 50% of labels for one task; verify the non-masked task still trains normally (`test_masking.py::TestCrossTaskMaskingIsolation::test_masking_independent_tasks_do_not_interfere`)
- [x] Fully mask a task for a batch; verify no NaN in loss (`test_masking.py::TestFullyMaskedBatch`, `test_fully_masked_task_does_not_prevent_other_from_training`)
- [x] Custom `mask_value` (e.g., `mask_value=-999.0`); verify masking still works (`test_masking.py::TestCustomMaskValue`)

### 1.6 Training Configuration Options

Use toy data with a binary task. Swap one setting at a time.

- [x] **Optimizer:** Adam, AdamW, SGD, Nadam — `test_optimizers.py` (also exercises additional Keras optimizers)
- [x] **LR schedules:** `polynomial_decay`, `exponential_decay`, `cosine_decay`, `constant` — `test_optimizers.py`, `test_training_features.py::TestLRSchedules`
- [x] **LR warmup:** `use_warmup=True` with each schedule type — `test_optimizers.py::test_optimizer_with_warmup`, `test_trainer.py::test_warmup_call_compatibility`
- [x] **Early stopping:** set patience=2, min_delta=0.01; confirm training stops early — `test_training_features.py::TestEarlyStopping`
- [x] **Threshold optimization:** `optimize_thresholds=True`; verify returned thresholds are floats ∈ (0,1) — `test_pooling_trainer.py::test_fit_with_optimize_thresholds`, `test_task_types.py`
- [x] **Class weights:** pass `class_weights` dict to `trainer.fit()`; verify loss changes — `test_trainer.py::test_class_weights_scaling`, `test_masking.py`
- [x] **Task weights:** pass `task_weights` dict; verify higher-weighted task dominates loss — `test_trainer.py::test_task_weights_importance`, `test_masking.py`
- [x] **Dropout:** `dropout_rate=0.3`; verify model builds and trains — `test_training_features.py::TestDropoutBehavior`
- [x] **Checkpointing:** `checkpoint_dir=...`; verify checkpoint files are written — `test_training_features.py::TestCheckpointing`
- [x] **Verbose modes:** `verbose=0`, `verbose=1`, `verbose=2` — `test_training_features.py::TestVerboseModes`

### 1.7 Model Persistence

- [x] `model.save(path)` then `load_model(path)` — verify predictions match before/after (`test_training_features.py::TestPredictionConsistency::test_model_save_load_identical_predictions`, `test_persistence.py`)
- [x] Save/load with thresholds; verify thresholds are restored (`test_training_features.py::test_thresholds_round_trip`)
- [x] `trainer.save_checkpoint()` then `Trainer.load_checkpoint()` — verify model loads (`test_training_features.py::TestSaveLoadCheckpoint`)
- [x] Load model with `encoder=None` (no encoder required for pre-computed embeddings) — covered by embedding-only save/load paths in persistence tests
- [x] Verify `model.get_config()` is JSON-serializable — `test_hierarchical.py::TestGetConfig`, `test_pooling.py::TestPoolSerialization`

### 1.8 Dataset Validation

- [x] `model.validate_dataset(dataset, verbose=True)` on a well-formed dataset — no errors (`test_training_features.py::TestDatasetValidation::test_valid_dataset_returns_true`)
- [x] Pass a dataset with wrong label shape; verify informative error (`test_training_features.py::test_invalid_binary_label_values_raises`, `test_wrong_task_name_in_labels_raises`)
- [x] Pass a dataset with wrong feature shape; verify informative error (`test_training_features.py::test_wrong_embedding_dim_raises`)

### 1.9 Distributed / Device Utilities

- [x] `get_distribution_strategy('auto')` returns a strategy and info string without error (`test_distributed.py`)
- [x] `setup_gpu_memory_growth()` runs without error (`test_distributed.py`)
- [x] `check_distributed_compatibility()` returns a dict with expected keys (`test_distributed.py`)
- [x] `print_device_info()` prints without error (`test_distributed.py::TestPrintDeviceInfo`)

---

## Part 2 — Encoder Tests (Toy Data)

For each encoder type, build a minimal model (1 binary task), train for 1–2 epochs on toy data (or mock inputs where applicable), and verify forward pass shape and that training runs without error.

### 2.1 Pre-computed Embeddings (No Encoder)
- [x] Pass `EncoderConfig(embedding_dim=64)` — no encoder specified (`test_encoder_config.py`, `test_masking.py` helpers)
- [x] Input: `np.random.randn(N, 64)` — verify passes through without error (`test_hierarchical.py`, `test_task_types.py`, `test_masking.py`)
- [x] Vary embedding_dim (32, 128, 512) — (`test_task_types.py::TestPrecomputedEmbeddingDims::test_training_runs_with_embedding_dim`)

### 2.2 Custom Keras Layer
- [x] `EncoderIntegration.KERAS` with a simple `tf.keras.layers.Dense` projection (`test_hierarchical_encoder.py`)
- [x] `trainable_encoder=True`; verify encoder weights appear in `model.trainable_variables` (`test_hierarchical_encoder.py`)
- [x] `trainable_encoder=False`; verify encoder weights are frozen (`test_hierarchical_encoder.py::test_trainable_encoder_false`)

### 2.3 TensorFlow Hub — Universal Sentence Encoder
- [x] `encoder_identifier='https://tfhub.dev/google/universal-sentence-encoder/4'`, `encoder_input_type=EncoderInputType.RAW_STRING` (`test_encoder_integration.py::TestTfHubUniversalSentenceEncoderIntegration`)
- [x] Input: `tf.constant(["hello world", "test sentence"])` — verify embedding shape
- [x] `trainable_encoder=True` vs `False` — (`test_encoder_integration.py::TestTfHubUniversalSentenceEncoderIntegration::test_tfhub_use_trainable_encoder_true`, `::test_tfhub_use_trainable_encoder_false`)

### 2.4 TensorFlow Hub — BERT-style (tokens)
- [x] `encoder_input_type=EncoderInputType.TFHUB_TOKENS` (`test_encoder_integration.py::TestTfHubBertIntegration`)
- [x] Input: dict with `input_word_ids`, `input_mask`, `input_type_ids`
- [x] Verify correct output shape

### 2.5 HuggingFace Transformers (local)
- [x] `encoder_identifier='distilbert-base-uncased'`, `encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS` (`test_encoder_integration.py::TestHuggingFaceEncoderIntegration`)
- [x] Tokenize with `AutoTokenizer.from_pretrained('distilbert-base-uncased')`
- [x] Input: dict with `input_ids`, `attention_mask`, `token_type_ids`
- [x] `trainable_encoder=True` vs `False` — (`test_encoder_integration.py::TestHuggingFaceEncoderIntegration::test_distilbert_trainable_encoder_true`, `::test_distilbert_trainable_encoder_false`)
- [x] Try `bert-base-uncased` as well (and other HF models in `test_encoder_integration.py`)

### 2.6 API Encoders (requires API keys)

For each, verify: (a) correct embedding dimension, (b) caching works (second call is faster), (c) model trains with returned embeddings.

- [x] **OpenAI:** `encoder_identifier='openai:text-embedding-3-small'` (`test_encoder_integration.py::TestOpenAIEncoderIntegration`; requires `OPENAI_API_KEY` + `openai` package)
  - [x] Also test `text-embedding-3-large` and `text-embedding-ada-002` (`test_encoder_integration.py::TestOpenAIEncoderIntegration::test_openai_extra_models_forward`)
- [x] **Cohere:** `encoder_identifier='cohere:embed-english-v3.0'` (`test_encoder_integration.py::TestCohereEncoderIntegration`; requires `COHERE_API_KEY` + `cohere` package)
  - [x] Also test `embed-english-light-v3.0` (`test_encoder_integration.py::TestCohereEncoderIntegration::test_cohere_embed_english_light_forward`)
- [x] **Voyage:** `encoder_identifier='voyage:voyage-3'` (`test_encoder_integration.py::TestVoyageEncoderIntegration`; requires `VOYAGE_API_KEY` + `voyageai` package)
- [x] **HuggingFace Inference API:** `encoder_identifier='huggingface_api:sentence-transformers/all-MiniLM-L6-v2'` (`test_encoder_integration.py::TestHuggingFaceInferenceEncoderIntegration`; requires `HF_API_TOKEN`)
  - [x] Also test `BAAI/bge-small-en-v1.5` (`test_encoder_integration.py::TestHuggingFaceInferenceEncoderIntegration::test_hf_inference_bge_small_forward`)
- [x] **Missing API key:** verify informative error when key env var is not set (`test_error_handling.py::TestApiEncoderErrors`)
- [x] **Unknown model name:** verify error for unsupported model string (`test_error_handling.py::TestApiEncoderErrors::test_openai_unknown_model_raises`, `::test_hf_inference_unknown_model_raises`)

### 2.7 Encoder Identifier Auto-detection
- [x] `https://tfhub.dev/...` → resolves to `TFHUB` without specifying integration (`test_encoder_config.py`, `test_model_config_from_flat_dict_roundtrip`)
- [x] `openai:...` → resolves to `OPENAI` (`test_encoder_config.py::test_with_prefixed_identifier`)
- [x] `cohere:...` → resolves to `COHERE` (`test_encoder_config.py::TestEncoderConfigDefaults::test_cohere_prefix_resolves`)
- [x] `voyage:...` → resolves to `VOYAGE` (`test_encoder_config.py::TestEncoderConfigDefaults::test_voyage_prefix_resolves`)
- [x] Bare model name (e.g., `bert-base-uncased`) → resolves to `HUGGINGFACE` (`test_model_config_encoder_config` / `ModelConfig` with HF identifiers)

---

## Part 3 — Error Handling Tests

Verify that errors are raised promptly with descriptive messages (not silent failures or cryptic TF errors).

### 3.1 ModelConfig Validation
- [x] `task_structure` inconsistent with number of tasks — expect `ValueError` (`test_error_handling.py::TestModelConfigTaskStructure`, `test_pooling.py::TestRegressionModelConfigValidation`)
- [x] `task_types` contains an unknown string — expect `ValueError` (`test_error_handling.py::TestModelConfigTasks`)
- [x] `num_classes_per_task` wrong length — expect `ValueError` (`test_error_handling.py::test_num_classes_wrong_length_raises`)
- [x] `num_classes_per_task=1` for a multiclass task — expect `ValueError` (`test_error_handling.py::test_multiclass_with_1_class_raises`)
- [x] `task_names` wrong length — expect `ValueError` (`test_error_handling.py::test_task_names_wrong_length_raises`)
- [x] Pool references a task that doesn't exist — expect `ValueError` (`test_error_handling.py::TestModelConfigOutputPools`, `test_pooling.py::TestPoolValidationTaskReferences`)
- [x] Pool name collides with a task name — expect `ValueError` (`test_error_handling.py::test_pool_name_equals_task_name_raises`, `test_pooling.py::TestPoolValidationNameCollisions`)
- [x] Pool with only 1 task — expect `ValueError` (`test_error_handling.py::test_pool_with_single_task_raises`, `test_pooling.py::TestPoolValidationTaskReferences::test_single_task_pool_rejected`)
- [x] Regression task included in a pool — expect `ValueError` (`test_error_handling.py::test_regression_task_in_pool_raises`)
- [x] Invalid pooling strategy (e.g., `'product'`) — expect `ValueError` (`test_error_handling.py::test_pool_invalid_strategy_raises`, `test_pooling.py::TestPoolValidationStrategy`)

### 3.2 EncoderConfig Validation
- [x] `encoder` is not a Keras Layer — expect `ValueError` (`test_error_handling.py::TestEncoderConfigValidation::test_custom_encoder_not_keras_layer_raises`)
- [x] Custom encoder with no `encoder_input_type` — expect `ValueError` (`test_error_handling.py::test_custom_encoder_without_input_type_raises`)
- [x] Unrecognized encoder identifier format — expect `ValueError` (`test_error_handling.py::test_unrecognized_integration_prefix_raises`)

### 3.3 TrainingConfig Validation
- [x] `epochs=0` — expect `ValueError` (`test_error_handling.py::TestTrainingConfigValidation`)
- [x] `batch_size=-1` — expect `ValueError`
- [x] `learning_rate=0` — expect `ValueError`
- [x] `dropout_rate=1.5` — expect `ValueError` (also `dropout_rate=1.0` / negative in `test_error_handling.py::TestMultiTaskModelErrors`)
- [x] `max_sequence_length=1024` — expect `ValueError`

### 3.4 Runtime / Input Errors
- [x] Wrong input shape for pre-computed embeddings — expect informative error (`test_training_features.py::TestDatasetValidation::test_wrong_embedding_dim_raises`)
- [x] String inputs to a pre-computed model — expect error (`test_training_features.py::TestDatasetValidation::test_string_input_to_embedding_model_raises`)
- [x] Missing key in HuggingFace token dict — expect error (`test_encoder_integration.py::TestHuggingFaceEncoderIntegration::test_forward_dict_missing_input_ids_raises`)
- [x] `multilabel` task with partially masked observations (not all-or-nothing masking per label) — expect `ValueError` (per trainer validation) (`test_error_handling.py::TestTrainerRuntimeValidation::test_multilabel_partial_mask_raises`)

### 3.5 Persistence Errors
- [x] `load_model(path)` where path doesn't exist — expect `ValueError` (`test_error_handling.py::TestPersistenceErrors::test_load_nonexistent_path_raises`)
- [x] `load_model(path)` missing config file — expect `ValueError` (`test_error_handling.py::TestPersistenceErrors::test_load_missing_config_raises`)

---

## Part 4 — Real-World Data Tests

Goal: verify end-to-end pipelines on actual data and inspect accuracy metrics.

For all experiments: use pre-computed embeddings from an API encoder (recommended: `openai:text-embedding-3-small` or `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace Inference API) or a lightweight HuggingFace model (e.g., `distilbert-base-uncased`). Train for a meaningful number of epochs (e.g., 10–20 with early stopping). Report metrics from `trainer.evaluate()`.

### 4.1 Fallacy Detection (`data/fallacy_*.csv`)

**Relevant columns:** `text`, `updated_label` (multiclass), `aug_type` (original/augmented)

- [ ] **Baseline — multiclass only:** one multiclass task on `updated_label`
  - Record accuracy and macro F1 on test set
  - Check that `aug_type='original'` vs augmented splits perform as expected
- [ ] **Outputs to check:** softmax probabilities for each fallacy class; confirm all sum to ~1
- [ ] **With threshold optimization** (N/A for pure multiclass, but verify no error thrown)
- [ ] **Metric readability:** print the returned metrics dict; verify class names match labels

### 4.2 CMP Manifesto (`data/CMP_*.csv`)

**Relevant columns:** `text`, ~75 numeric category columns (0/1 per sentence)

- [ ] **Baseline — multilabel:** all category columns as one multilabel task
  - Record accuracy (all-correct) and macro F1
- [ ] **Hierarchical:** group related CMP codes into branches (e.g., economic policy codes vs. social policy codes); define `task_structure` accordingly
- [ ] **With class imbalance weighting:** many categories will be rare — pass `class_weights`; compare F1 with/without weighting
- [ ] **Output inspection:** for a few held-out sentences, print predicted category probabilities; verify they are interpretable
- [ ] **Threshold optimization:** run with `optimize_thresholds=True`; report optimized thresholds per label

### 4.3 No-Propaganda-Overlap (`data/no_prop_overlap_*.csv`)

**Relevant columns:**
- `text`
- Propaganda technique labels (multilabel, 18 techniques): `Loaded_Language`, `Slogans`, `Black-and-White_Fallacy`, `Flag-Waving`, `Name_Calling`, `Whataboutism`, `Causal_Oversimplification`, `Exaggeration,Minimisation`, `Doubt`, `Appeal_to_Authority`, `Repetition`, `Appeal_to_fear-prejudice`, `Red_Herring`, `Thought-terminating_Cliches`, `Reductio_ad_hitlerum`, `Bandwagon`, `Obfuscation,Intentional_Vagueness,Confusion`, `Straw_Men`
- Discourse dimensions (continuous/regression): `disagree_agree`, `attacking_respectful`, `emotion_fact`, `nasty_nice`, `personal_audience`, `defeater_undercutter`, `negotiate_attack`, `questioning_asserting`
- Binary discourse dimensions: `disagree_agree_bi`, `attacking_respectful_bi`, etc.
- `propaganda` (binary or multiclass — check values)
- `WA`, `WA_bi`
- `dataset` (source — use as stratification, not a label)
- Note: `NA` values in discourse columns indicate those annotations don't apply for a given dataset — use label masking (`mask_value`)

**Experiments:**

- [ ] **Propaganda techniques only (multilabel):** train on the 18 technique columns
  - Report per-technique F1 and macro F1 on test set
  - Compare with/without class weights (imbalanced techniques)
  - Check threshold optimization

- [ ] **Discourse dimensions only (regression):** train on continuous discourse columns
  - Report MSE/MAE/R² per dimension
  - Verify masking handles `NA` rows correctly

- [ ] **Binary discourse only:** train on `*_bi` columns
  - Report accuracy and F1 per dimension

- [ ] **Multi-task: propaganda techniques + binary discourse** (multilabel + binary)
  - Compare individual task metrics vs. single-task baselines above
  - Check that masking correctly handles rows where discourse labels are `NA`

- [ ] **Multi-task: all task types together** (multilabel propaganda + regression discourse + binary discourse + binary `propaganda` column)
  - Full test of mixed-type multi-task setup
  - Report per-task and overall metrics
  - Inspect `overall_metrics` for reasonableness

- [ ] **Output pooling on propaganda:** pool all 18 technique columns with `strategy='max'` into a single `any_propaganda` output; compare pool F1 against the standalone `propaganda` binary column

### 4.4 Cross-Dataset Generalization (Optional)
- [ ] Train on CMP + fallacy jointly (multi-task, flat); test on each test set separately
- [ ] Verify masking allows different label sets per dataset

---

## Part 5 — Output Readability Checks

Verify human-readable formatting across outputs.

- [x] `model.summary_tree()` — readable tree structure, branch labels, pool indicators (`test_hierarchical.py`, `test_pooling_hierarchical.py::TestSummaryTreeWithPools`)
- [x] `model.get_task_info()` — each dict contains `name`, `type`, `num_classes`, `branch` (`test_hierarchical.py::TestGetTaskInfo`, `test_hierarchical_encoder.py`)
- [x] `model.get_pool_info()` — each dict contains `name`, `strategy`, `tasks` (`test_pooling_hierarchical.py::TestGetPoolInfo`)
- [x] `trainer.evaluate()` result — nested dict printable; all keys are task/pool names; values are float metrics (`test_task_types.py`, `test_pooling_trainer.py`)
- [x] Training progress (verbose=1) — loss printed per epoch; no garbled output (`test_training_features.py::TestVerboseModes::test_verbose_1_progress_bar`)
- [x] Error messages — each `ValueError` includes the invalid value and what was expected (spot-check 3–5 error cases) — (`test_error_handling.py::TestValueErrorMessagesReadable`)
- [x] `print_device_info()` — human-readable hardware summary (`test_distributed.py::TestPrintDeviceInfo::test_print_device_info_produces_output`)

---

## Part 6 — Regression Standardization

- [x] Train regression task with standardization enabled; verify outputs are in expected range (`test_regression_std.py`)
- [x] Verify standardization parameters are saved/loaded with the model (`test_regression_std.py` tests 5a/5b)
- [x] Compare predictions before and after standardization to confirm scale — (`test_regression_extended.py::test_regression_predictions_match_label_scale`)
- [x] Check that standardization doesn't affect binary/multiclass/multilabel tasks in a mixed model — (`test_regression_extended.py::test_standardization_mixed_with_classification_heads`)

---

## Notes

- **Running existing tests:** `pytest tests/` from the repo root
- **API keys:** set `OPENAI_API_KEY`, `COHERE_API_KEY`, `VOYAGE_API_KEY`, `HF_API_TOKEN` / `HF_TOKEN` as environment variables before running API encoder tests
- **GPU:** most toy data tests should run on CPU; real-world experiments benefit from GPU
- **Masking convention:** `NA` string values in CSVs must be converted to `mask_value` (default `-1.0`) before building TF datasets; check `docs/DATA_FORMAT.md` for details
- **Metric interpretation:** `overall_metrics` averages across tasks but excludes pool outputs; pool outputs appear separately under `pool_metrics`
