# Testing Plan

This document outlines all testing tasks for the `multitask` library, covering toy data, real-world data, all functionality, all encoder types, and error handling.

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

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | **Binary** — single binary output | Output shape `[batch, 1]`; sigmoid output ∈ (0,1); `accuracy` and `f1` returned; threshold is a float in (0,1) | [tests/test_task_types.py](tests/test_task_types.py) (`TestSingleBinary`) |
| [x] | **Multiclass (3-class)** | Softmax output shape `[batch, num_classes]` sums to 1; accuracy returned; `thresholds[t]` is `None` (shape/sum tests use 4- and 5-class; 3-class covered by `test_loss_finite`) | [tests/test_task_types.py](tests/test_task_types.py) (`TestSingleMulticlass`) |
| [x] | **Multiclass (10-class)** | Nothing breaks with many classes | [tests/test_task_types.py](tests/test_task_types.py) (`TestSingleMulticlass::test_multiclass_10_classes_smoke`) |
| [x] | **Multilabel (3-label)** | Output shape `[batch, num_labels]`; sigmoid per-label outputs ∈ (0,1); `f1` returned; threshold is a float (single float per task, applied to all labels) | [tests/test_task_types.py](tests/test_task_types.py) (`TestSingleMultilabel`) |
| [x] | **Regression** | Linear output shape `[batch, 1]`; `mse`, `mae`, `r2` returned; `thresholds[t]` is `None` | [tests/test_task_types.py](tests/test_task_types.py) (`TestSingleRegression`) |
| [x] | **Regression with standardization** | Set `standardize_regression=True` (if applicable); outputs rescaled | [tests/test_regression_std.py](tests/test_regression_std.py) |

### 1.2 Mixed Task Type Combinations

Use `task_structure=[[N]]` (flat) unless noted. Verify combined loss, combined metrics dict, and that masking works across heterogeneous types.

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | Binary + Binary | — | [tests/test_task_types.py](tests/test_task_types.py) (`TestPairwiseCombinations`) |
| [x] | Binary + Multiclass | — | [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | Binary + Multilabel | — | [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | Binary + Regression | — | [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | Multiclass + Multiclass | — | [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | Multiclass + Multilabel | — | [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | Multiclass + Regression | — | [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | Multilabel + Multilabel | — | [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | Multilabel + Regression | — | [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | Regression + Regression | — | [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | Binary + Multiclass + Multilabel | — | [tests/test_task_types.py](tests/test_task_types.py) (`TestTripleCombinations`) |
| [x] | Binary + Multiclass + Regression | — | [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | Binary + Multilabel + Regression | — | [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | Multiclass + Multilabel + Regression | — | [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | **All four types together** — one binary, one multiclass, one multilabel, one regression task | — | [tests/test_task_types.py](tests/test_task_types.py) (`TestAllFourTypes`) |

### 1.3 Hierarchical Branching Structures

Test non-trivial `task_structure` values. Confirm model builds without error and `summary_tree()` output matches the expected topology.

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | Flat structure: `[[3]]` (3 tasks, no branching) — e.g. triple combinations use `[[3]]`; flat multi-task tests use `[[N]]` | — | [tests/test_task_types.py](tests/test_task_types.py), [tests/test_hierarchical.py](tests/test_hierarchical.py), etc. |
| [x] | 2-level branching — e.g. `[[3], [2, 5, 3]]` (exact `[[2], [1, 2]]` not required in tests) | — | [tests/test_hierarchical.py](tests/test_hierarchical.py), [tests/test_hierarchical_encoder.py](tests/test_hierarchical_encoder.py), [tests/test_pooling.py](tests/test_pooling.py) |
| [x] | 3-level branching | — | [tests/test_hierarchical.py](tests/test_hierarchical.py) (`TestThreeLevelBranching`), [tests/test_pooling_hierarchical.py](tests/test_pooling_hierarchical.py) (`TestDifferentTaskStructures`) |
| [x] | Single branch with many tasks: `[[5]]` | — | [tests/conftest.py](tests/conftest.py) (`mismatched_config`); [tests/test_optimizers.py](tests/test_optimizers.py) |
| [x] | Asymmetric branches | — | [tests/test_hierarchical.py](tests/test_hierarchical.py) (`test_asymmetric_3level_structure`); [tests/test_pooling.py](tests/test_pooling.py), [tests/test_pooling_hierarchical.py](tests/test_pooling_hierarchical.py) |
| [x] | Run `model.summary_tree()` — prints without error; reflects structure | — | [tests/test_hierarchical.py](tests/test_hierarchical.py) (`TestSummaryTree`), [tests/test_pooling_hierarchical.py](tests/test_pooling_hierarchical.py) (`TestSummaryTreeWithPools`) |
| [x] | Run `model.get_task_info()` — returned dicts contain expected keys | — | [tests/test_hierarchical.py](tests/test_hierarchical.py) (`TestGetTaskInfo`), [tests/test_hierarchical_encoder.py](tests/test_hierarchical_encoder.py) |

### 1.4 Output Pooling

Use a flat model with all binary tasks, then configure `output_pools`.

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | Pool with `strategy='max'` over 3 binary tasks | Pool output equals element-wise max of constituents; values ∈ [0,1]; pool shape matches constituents `[batch, 1]` | [tests/test_pooling_hierarchical.py](tests/test_pooling_hierarchical.py) (`TestPooledOutputsMax`, `TestPooledOutputsShape`) |
| [x] | Pool with `strategy='mean'` | — | [tests/test_pooling_hierarchical.py](tests/test_pooling_hierarchical.py) (`TestPooledOutputsMean`) |
| [x] | Pool with `strategy='min'` | — | [tests/test_pooling_hierarchical.py](tests/test_pooling_hierarchical.py) (`TestPooledOutputsMin`) |
| [x] | Verify pool output appears as a key in `model(inputs)` dict | — | [tests/test_pooling_hierarchical.py](tests/test_pooling_hierarchical.py), [tests/test_pooling_full.py](tests/test_pooling_full.py) |
| [x] | Run `model.get_pool_info()` — correct tasks listed | — | [tests/test_pooling_hierarchical.py](tests/test_pooling_hierarchical.py) (`TestGetPoolInfo`) |
| [x] | Evaluate pool metrics via `trainer.evaluate()` — pool metrics in returned dict | — | [tests/test_pooling_trainer.py](tests/test_pooling_trainer.py) |
| [x] | Adding a pooled output does not change individual task outputs | — | [tests/test_pooling_trainer.py](tests/test_pooling_trainer.py) (`test_task_metrics_unchanged`), [tests/test_pooling.py](tests/test_pooling.py) (`TestPoolsWithExistingFeatures`) |

### 1.5 Label Masking

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | Introduce `mask_value=-1.0` labels for some samples on some tasks | Loss not affected by masked positions | [tests/test_masking.py](tests/test_masking.py), [tests/test_trainer.py](tests/test_trainer.py) (`test_masking_and_normalization_logic`) |
| [x] | Mask 50% of labels for one task | Non-masked task still trains normally | [tests/test_masking.py](tests/test_masking.py) (`TestCrossTaskMaskingIsolation::test_masking_independent_tasks_do_not_interfere`) |
| [x] | Fully mask a task for a batch | No NaN in loss | [tests/test_masking.py](tests/test_masking.py) (`TestFullyMaskedBatch`, `test_fully_masked_task_does_not_prevent_other_from_training`) |
| [x] | Custom `mask_value` (e.g., `mask_value=-999.0`) | Masking still works | [tests/test_masking.py](tests/test_masking.py) (`TestCustomMaskValue`) |

### 1.6 Training Configuration Options

Use toy data with a binary task. Swap one setting at a time.

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | **Optimizer:** Adam, AdamW, SGD, Nadam | — | [tests/test_optimizers.py](tests/test_optimizers.py) (also additional Keras optimizers) |
| [x] | **LR schedules:** `polynomial_decay`, `exponential_decay`, `cosine_decay`, `constant` | — | [tests/test_optimizers.py](tests/test_optimizers.py), [tests/test_training_features.py](tests/test_training_features.py) (`TestLRSchedules`) |
| [x] | **LR warmup:** `use_warmup=True` with each schedule type | — | [tests/test_optimizers.py](tests/test_optimizers.py) (`test_optimizer_with_warmup`), [tests/test_trainer.py](tests/test_trainer.py) (`test_warmup_call_compatibility`) |
| [x] | **Early stopping:** patience=2, min_delta=0.01 | Training stops early | [tests/test_training_features.py](tests/test_training_features.py) (`TestEarlyStopping`) |
| [x] | **Threshold optimization:** `optimize_thresholds=True` | Returned thresholds are floats ∈ (0,1) | [tests/test_pooling_trainer.py](tests/test_pooling_trainer.py) (`test_fit_with_optimize_thresholds`), [tests/test_task_types.py](tests/test_task_types.py) |
| [x] | **Class weights:** pass `class_weights` dict to `trainer.fit()` | Loss changes | [tests/test_trainer.py](tests/test_trainer.py) (`test_class_weights_scaling`), [tests/test_masking.py](tests/test_masking.py) |
| [x] | **Task weights:** pass `task_weights` dict | Higher-weighted task dominates loss | [tests/test_trainer.py](tests/test_trainer.py) (`test_task_weights_importance`), [tests/test_masking.py](tests/test_masking.py) |
| [x] | **Dropout:** `dropout_rate=0.3` | Model builds and trains | [tests/test_training_features.py](tests/test_training_features.py) (`TestDropoutBehavior`) |
| [x] | **Checkpointing:** `checkpoint_dir=...` | Checkpoint files written | [tests/test_training_features.py](tests/test_training_features.py) (`TestCheckpointing`) |
| [x] | **Verbose modes:** `verbose=0`, `verbose=1`, `verbose=2` | — | [tests/test_training_features.py](tests/test_training_features.py) (`TestVerboseModes`) |

### 1.7 Model Persistence

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `model.save(path)` then `load_model(path)` | Predictions match before/after | [tests/test_training_features.py](tests/test_training_features.py) (`TestPredictionConsistency::test_model_save_load_identical_predictions`), [tests/test_persistence.py](tests/test_persistence.py) |
| [x] | Save/load with thresholds | Thresholds restored | [tests/test_training_features.py](tests/test_training_features.py) (`test_thresholds_round_trip`) |
| [x] | `trainer.save_checkpoint()` then `Trainer.load_checkpoint()` | Model loads | [tests/test_training_features.py](tests/test_training_features.py) (`TestSaveLoadCheckpoint`) |
| [x] | Load model with `encoder=None` (no encoder for pre-computed embeddings) | — | Embedding-only save/load paths in [tests/test_persistence.py](tests/test_persistence.py) |
| [x] | Verify `model.get_config()` is JSON-serializable | — | [tests/test_hierarchical.py](tests/test_hierarchical.py) (`TestGetConfig`), [tests/test_pooling.py](tests/test_pooling.py) (`TestPoolSerialization`) |

### 1.8 Dataset Validation

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `model.validate_dataset(dataset, verbose=True)` on a well-formed dataset | No errors | [tests/test_training_features.py](tests/test_training_features.py) (`TestDatasetValidation::test_valid_dataset_returns_true`) |
| [x] | Pass a dataset with wrong label shape | Informative error | [tests/test_training_features.py](tests/test_training_features.py) (`test_invalid_binary_label_values_raises`, `test_wrong_task_name_in_labels_raises`) |
| [x] | Pass a dataset with wrong feature shape | Informative error | [tests/test_training_features.py](tests/test_training_features.py) (`test_wrong_embedding_dim_raises`) |

### 1.9 Distributed / Device Utilities

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `get_distribution_strategy('auto')` | Returns strategy and info string without error | [tests/test_distributed.py](tests/test_distributed.py) |
| [x] | `setup_gpu_memory_growth()` | Runs without error | [tests/test_distributed.py](tests/test_distributed.py) |
| [x] | `check_distributed_compatibility()` | Returns dict with expected keys | [tests/test_distributed.py](tests/test_distributed.py) |
| [x] | `print_device_info()` | Prints without error | [tests/test_distributed.py](tests/test_distributed.py) (`TestPrintDeviceInfo`) |

---

## Part 2 — Encoder Tests (Toy Data)

For each encoder type, build a minimal model (1 binary task), train for 1–2 epochs on toy data (or mock inputs where applicable), and verify forward pass shape and that training runs without error.

### 2.1 Pre-computed Embeddings (No Encoder)

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | Pass `EncoderConfig(embedding_dim=64)` — no encoder specified | — | [tests/test_encoder_config.py](tests/test_encoder_config.py), [tests/test_masking.py](tests/test_masking.py) helpers |
| [x] | Input: `np.random.randn(N, 64)` | Passes through without error | [tests/test_hierarchical.py](tests/test_hierarchical.py), [tests/test_task_types.py](tests/test_task_types.py), [tests/test_masking.py](tests/test_masking.py) |
| [x] | Vary embedding_dim (32, 128, 512) | Training runs | [tests/test_task_types.py](tests/test_task_types.py) (`TestPrecomputedEmbeddingDims::test_training_runs_with_embedding_dim`) |

### 2.2 Custom Keras Layer

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `EncoderIntegration.KERAS` with a simple `tf.keras.layers.Dense` projection | — | [tests/test_hierarchical_encoder.py](tests/test_hierarchical_encoder.py) |
| [x] | `trainable_encoder=True` | `model.encoder.trainable is True` (encoder weights participate in training) | [tests/test_hierarchical_encoder.py](tests/test_hierarchical_encoder.py), [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`test_tfhub_use_trainable_encoder_true`, `test_distilbert_trainable_encoder_true`) |
| [x] | `trainable_encoder=False` | `model.encoder.trainable is False` (encoder weights frozen) | [tests/test_hierarchical_encoder.py](tests/test_hierarchical_encoder.py) (`test_trainable_encoder_false`), [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`test_tfhub_use_trainable_encoder_false`, `test_distilbert_trainable_encoder_false`) |

### 2.3 TensorFlow Hub — Universal Sentence Encoder

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `encoder_identifier='https://tfhub.dev/google/universal-sentence-encoder/4'`, `encoder_input_type=EncoderInputType.RAW_STRING` | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestTfHubUniversalSentenceEncoderIntegration`) |
| [x] | Input: `tf.constant(["hello world", "test sentence"])` | Correct embedding shape | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) |
| [x] | `trainable_encoder=True` vs `False` | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestTfHubUniversalSentenceEncoderIntegration::test_tfhub_use_trainable_encoder_true`, `::test_tfhub_use_trainable_encoder_false`) |

### 2.4 TensorFlow Hub — BERT-style (tokens)

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `encoder_input_type=EncoderInputType.TFHUB_TOKENS` | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestTfHubBertIntegration`) |
| [x] | Input: dict with `input_word_ids`, `input_mask`, `input_type_ids` | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) |
| [x] | Verify correct output shape | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) |

### 2.5 HuggingFace Transformers (local)

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `encoder_identifier='distilbert-base-uncased'`, `encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS` | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestHuggingFaceEncoderIntegration`) |
| [x] | Tokenize with `AutoTokenizer.from_pretrained('distilbert-base-uncased')` | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) |
| [x] | Input: dict with `input_ids`, `attention_mask`, `token_type_ids` | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) |
| [x] | `trainable_encoder=True` vs `False` | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestHuggingFaceEncoderIntegration::test_distilbert_trainable_encoder_true`, `::test_distilbert_trainable_encoder_false`) |
| [x] | Try `bert-base-uncased` as well (and other HF models) | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) |

### 2.6 API Encoders (requires API keys)

For each, verify: (a) correct embedding dimension, (b) caching works (second call is faster), (c) model trains with returned embeddings.

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | **OpenAI:** `encoder_identifier='openai:text-embedding-3-small'` | Requires `OPENAI_API_KEY` + `openai` package | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestOpenAIEncoderIntegration`) |
| [x] | Also test `text-embedding-3-large` and `text-embedding-ada-002` | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestOpenAIEncoderIntegration::test_openai_extra_models_forward`) |
| [x] | **Cohere:** `encoder_identifier='cohere:embed-english-v3.0'` | Requires `COHERE_API_KEY` + `cohere` package | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestCohereEncoderIntegration`) |
| [x] | Also test `embed-english-light-v3.0` | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestCohereEncoderIntegration::test_cohere_embed_english_light_forward`) |
| [x] | **Voyage:** `encoder_identifier='voyage:voyage-3'` | Requires `VOYAGE_API_KEY` + `voyageai` package | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestVoyageEncoderIntegration`) |
| [x] | **HuggingFace Inference API:** `encoder_identifier='huggingface_api:sentence-transformers/all-MiniLM-L6-v2'` | Requires `HF_API_TOKEN` | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestHuggingFaceInferenceEncoderIntegration`) |
| [x] | Also test `BAAI/bge-small-en-v1.5` | — | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestHuggingFaceInferenceEncoderIntegration::test_hf_inference_bge_small_forward`) |
| [x] | **Missing API key:** key env var not set | Informative error | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestApiEncoderErrors`) |
| [x] | **Unknown model name:** unsupported model string | Error | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestApiEncoderErrors::test_openai_unknown_model_raises`, `::test_hf_inference_unknown_model_raises`) |

### 2.7 Encoder Identifier Auto-detection

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `https://tfhub.dev/...` → resolves to `TFHUB` without specifying integration | — | [tests/test_encoder_config.py](tests/test_encoder_config.py) (`test_model_config_from_flat_dict_roundtrip`) |
| [x] | `openai:...` → resolves to `OPENAI` | — | [tests/test_encoder_config.py](tests/test_encoder_config.py) (`test_with_prefixed_identifier`) |
| [x] | `cohere:...` → resolves to `COHERE` | — | [tests/test_encoder_config.py](tests/test_encoder_config.py) (`TestEncoderConfigDefaults::test_cohere_prefix_resolves`) |
| [x] | `voyage:...` → resolves to `VOYAGE` | — | [tests/test_encoder_config.py](tests/test_encoder_config.py) (`TestEncoderConfigDefaults::test_voyage_prefix_resolves`) |
| [x] | Bare model name (e.g., `bert-base-uncased`) → resolves to `HUGGINGFACE` | — | [tests/test_encoder_config.py](tests/test_encoder_config.py) (`TestModelConfigEncoderConfig` / `ModelConfig` with HF identifiers) |

---

## Part 3 — Error Handling Tests

Verify that errors are raised promptly with descriptive messages (not silent failures or cryptic TF errors).

### 3.1 ModelConfig Validation

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `task_structure` inconsistent with number of tasks | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestModelConfigTaskStructure`), [tests/test_pooling.py](tests/test_pooling.py) (`TestRegressionModelConfigValidation`) |
| [x] | `task_types` contains an unknown string | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestModelConfigTasks`) |
| [x] | `num_classes_per_task` wrong length | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`test_num_classes_wrong_length_raises`) |
| [x] | `num_classes_per_task=1` for a multiclass task | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`test_multiclass_with_1_class_raises`) |
| [x] | `task_names` wrong length | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`test_task_names_wrong_length_raises`) |
| [x] | Pool references a task that doesn't exist | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestModelConfigOutputPools`), [tests/test_pooling.py](tests/test_pooling.py) (`TestPoolValidationTaskReferences`) |
| [x] | Pool name collides with a task name | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`test_pool_name_equals_task_name_raises`), [tests/test_pooling.py](tests/test_pooling.py) (`TestPoolValidationNameCollisions`) |
| [x] | Pool with only 1 task | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`test_pool_with_single_task_raises`), [tests/test_pooling.py](tests/test_pooling.py) (`TestPoolValidationTaskReferences::test_single_task_pool_rejected`) |
| [x] | Regression task included in a pool | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`test_regression_task_in_pool_raises`) |
| [x] | Invalid pooling strategy (e.g., `'product'`) | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`test_pool_invalid_strategy_raises`), [tests/test_pooling.py](tests/test_pooling.py) (`TestPoolValidationStrategy`) |

### 3.2 EncoderConfig Validation

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `encoder` is not a Keras Layer | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestEncoderConfigValidation::test_custom_encoder_not_keras_layer_raises`) |
| [x] | Custom encoder with no `encoder_input_type` | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`test_custom_encoder_without_input_type_raises`) |
| [x] | Unrecognized encoder identifier format | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`test_unrecognized_integration_prefix_raises`) |

### 3.3 TrainingConfig Validation

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `epochs=0` | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestTrainingConfigValidation`) |
| [x] | `batch_size=-1` | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestTrainingConfigValidation`) |
| [x] | `learning_rate=0` | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestTrainingConfigValidation`) |
| [x] | `dropout_rate=1.5` | `ValueError` (also `dropout_rate=1.0` / negative in `TestMultiTaskModelErrors`) | [tests/test_error_handling.py](tests/test_error_handling.py) |
| [x] | `max_sequence_length=1024` | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestTrainingConfigValidation`) |

### 3.4 Runtime / Input Errors

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | Wrong input shape for pre-computed embeddings | Informative error | [tests/test_training_features.py](tests/test_training_features.py) (`TestDatasetValidation::test_wrong_embedding_dim_raises`) |
| [x] | String inputs to a pre-computed model | Error | [tests/test_training_features.py](tests/test_training_features.py) (`TestDatasetValidation::test_string_input_to_embedding_model_raises`) |
| [x] | Missing key in HuggingFace token dict | Error | [tests/test_encoder_integration.py](tests/test_encoder_integration.py) (`TestHuggingFaceEncoderIntegration::test_forward_dict_missing_input_ids_raises`) |
| [x] | `multilabel` task with partially masked observations (not all-or-nothing per label) | `ValueError` (per trainer validation) | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestTrainerRuntimeValidation::test_multilabel_partial_mask_raises`) |

### 3.5 Persistence Errors

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `load_model(path)` where path doesn't exist | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestPersistenceErrors::test_load_nonexistent_path_raises`) |
| [x] | `load_model(path)` missing config file | `ValueError` | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestPersistenceErrors::test_load_missing_config_raises`) |

---

## Part 4 — Real-World Data Tests

Goal: verify end-to-end pipelines on actual data and inspect accuracy metrics.

For all experiments: use a lightweight HuggingFace model (e.g., `distilbert-base-uncased`). Train for a meaningful number of epochs (e.g., 10–20 with early stopping). Make sure output and fine-tuning are working. Report metrics from `trainer.evaluate()`. Test on Midway cluster.

### 4.1 Fallacy Detection (`data/fallacy_*.csv`)

**Relevant columns:** `text`, `updated_label` (multiclass), `aug_type` (original/augmented)

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [ ] | **Baseline — multiclass only:** one multiclass task on `updated_label` | Record accuracy and macro F1 on test set; check that `aug_type='original'` vs augmented splits perform as expected | — |
| [ ] | **Outputs to check:** softmax probabilities for each fallacy class | All sum to ~1 | — |
| [ ] | **With threshold optimization** | N/A for pure multiclass, but verify no error thrown | — |
| [ ] | **Metric readability:** print the returned metrics dict | Class names match labels | — |

### 4.2 CMP Manifesto (`data/CMP_*.csv`)

**Relevant columns:** `text`, ~75 numeric category columns (0/1 per sentence)

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [ ] | **Baseline — multilabel:** all category columns as one multilabel task | Record accuracy (all-correct) and macro F1 | — |
| [ ] | **Hierarchical:** group related CMP codes into branches; define `task_structure` accordingly | — | — |
| [ ] | **With class imbalance weighting:** many categories rare — pass `class_weights` | Compare F1 with/without weighting | — |
| [ ] | **Output inspection:** for a few held-out sentences, print predicted category probabilities | Interpretable | — |
| [ ] | **Threshold optimization:** `optimize_thresholds=True` | Report optimized thresholds per label | — |

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

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [ ] | **Propaganda techniques only (multilabel):** train on the 18 technique columns | Per-technique F1 and macro F1 on test set; compare with/without class weights; check threshold optimization | — |
| [ ] | **Discourse dimensions only (regression):** train on continuous discourse columns | MSE/MAE/R² per dimension; masking handles `NA` rows correctly | — |
| [ ] | **Binary discourse only:** train on `*_bi` columns | Accuracy and F1 per dimension | — |
| [ ] | **Multi-task: propaganda techniques + binary discourse** (multilabel + binary) | Individual task metrics vs. single-task baselines; masking handles rows where discourse labels are `NA` | — |
| [ ] | **Multi-task: all task types together** (multilabel propaganda + regression discourse + binary discourse + binary `propaganda` column) | Per-task and overall metrics; `overall_metrics` reasonable | — |
| [ ] | **Output pooling on propaganda:** pool all 18 technique columns with `strategy='max'` into `any_propaganda` | Compare pool F1 against standalone `propaganda` binary column | — |

### 4.4 Cross-Dataset Generalization (Optional)

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [ ] | Train on CMP + fallacy jointly (multi-task, flat); test on each test set separately | — | — |
| [ ] | Verify masking allows different label sets per dataset | — | — |

---

## Part 5 — Output Readability Checks

Verify human-readable formatting across outputs.

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | `model.summary_tree()` | Readable tree structure, branch labels, pool indicators | [tests/test_hierarchical.py](tests/test_hierarchical.py), [tests/test_pooling_hierarchical.py](tests/test_pooling_hierarchical.py) (`TestSummaryTreeWithPools`) |
| [x] | `model.get_task_info()` | Each dict contains `output_idx`, `name`, `parent_branch_path` (a list of branch indices from root to leaf) | [tests/test_hierarchical.py](tests/test_hierarchical.py) (`TestGetTaskInfo`), [tests/test_hierarchical_encoder.py](tests/test_hierarchical_encoder.py) |
| [x] | `model.get_pool_info()` | Each dict contains `name`, `strategy`, `task_type`, `num_classes`, `constituent_tasks` | [tests/test_pooling_hierarchical.py](tests/test_pooling_hierarchical.py) (`TestGetPoolInfo`) |
| [x] | `trainer.evaluate()` result | Nested dict printable; keys are task/pool names; values are float metrics | [tests/test_task_types.py](tests/test_task_types.py), [tests/test_pooling_trainer.py](tests/test_pooling_trainer.py) |
| [x] | Training progress (verbose=1) | Loss printed per epoch; no garbled output | [tests/test_training_features.py](tests/test_training_features.py) (`TestVerboseModes::test_verbose_1_progress_bar`) |
| [x] | Error messages — each `ValueError` includes invalid value and what was expected (spot-check 3–5 cases) | — | [tests/test_error_handling.py](tests/test_error_handling.py) (`TestValueErrorMessagesReadable`) |
| [x] | `print_device_info()` | Human-readable hardware summary | [tests/test_distributed.py](tests/test_distributed.py) (`TestPrintDeviceInfo::test_print_device_info_produces_output`) |

---

## Part 6 — Regression Standardization

| Completed | Test description | Expected output or error | Test file |
|-----------|------------------|-------------------------|-----------|
| [x] | Train regression task with standardization enabled | Outputs in expected range | [tests/test_regression_std.py](tests/test_regression_std.py) |
| [x] | Standardization parameters saved/loaded with model | — | [tests/test_regression_std.py](tests/test_regression_std.py) (tests 5a/5b) |
| [x] | Compare predictions before and after standardization to confirm scale | — | [tests/test_regression_extended.py](tests/test_regression_extended.py) (`test_regression_predictions_match_label_scale`) |
| [x] | Standardization doesn't affect binary/multiclass/multilabel tasks in a mixed model | — | [tests/test_regression_extended.py](tests/test_regression_extended.py) (`test_standardization_mixed_with_classification_heads`) |

---

## Notes

- **Running existing tests:** `pytest tests/` from the repo root
- **API keys:** set `OPENAI_API_KEY`, `COHERE_API_KEY`, `VOYAGE_API_KEY`, `HF_API_TOKEN` / `HF_TOKEN` as environment variables before running API encoder tests
- **GPU:** most toy data tests should run on CPU; real-world experiments benefit from GPU
- **Masking convention:** `NA` string values in CSVs must be converted to `mask_value` (default `-1.0`) before building TF datasets; check `docs/DATA_FORMAT.md` for details
- **Metric interpretation:** `overall_metrics` averages across tasks but excludes pool outputs; pool outputs appear separately under `pool_metrics`
