"""
Comprehensive tests for error handling across all configuration classes and runtime.

Covers:
- ModelConfig: task_structure, task_types, num_classes, task_names, output_pools
- TrainingConfig: epochs, batch_size, learning_rate, dropout_rate, max_sequence_length
- EncoderConfig: custom encoder validation, identifier parsing
- MultiTaskModel: missing embedding_dim, invalid dropout, num_classes
- Trainer: multilabel partial mask, invalid dataset shapes
- Persistence: load_model with missing path
- API encoders: missing env keys, unknown model ids (no live API calls)
- ValueError messages: spot-check that messages name invalid values / expectations
"""

import pytest
import tensorflow as tf

from multitask.config import (
    ModelConfig,
    TrainingConfig,
    EncoderConfig,
    EncoderInputType,
    EncoderIntegration,
)
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer


# ---------------------------------------------------------------------------
# ModelConfig — task_structure validation
# ---------------------------------------------------------------------------

class TestModelConfigTaskStructure:
    def test_empty_task_structure_raises(self):
        with pytest.raises(ValueError):
            ModelConfig(task_structure=[])

    def test_level0_with_two_elements_raises(self):
        with pytest.raises(ValueError, match="First level"):
            ModelConfig(task_structure=[[2, 3]])

    def test_level1_too_short_raises(self):
        # Level 0: 3 branches → level 1 must have 3 elements, not 2
        with pytest.raises(ValueError, match="Level 1"):
            ModelConfig(task_structure=[[3], [1, 1]])

    def test_level1_too_long_raises(self):
        # Level 0: 2 branches → level 1 must have 2 elements, not 4
        with pytest.raises(ValueError, match="Level 1"):
            ModelConfig(task_structure=[[2], [1, 1, 1, 1]])

    def test_zero_in_level0_raises(self):
        with pytest.raises(ValueError, match="invalid values"):
            ModelConfig(task_structure=[[0]])

    def test_negative_in_task_structure_raises(self):
        with pytest.raises(ValueError, match="invalid values"):
            ModelConfig(task_structure=[[-1]])

    def test_zero_in_level1_raises(self):
        with pytest.raises(ValueError, match="invalid values"):
            ModelConfig(task_structure=[[2], [0, 1]])


# ---------------------------------------------------------------------------
# ModelConfig — task_types, task_names, num_classes
# ---------------------------------------------------------------------------

class TestModelConfigTasks:
    def test_invalid_task_type_string_raises(self):
        with pytest.raises(ValueError, match="Invalid task_type"):
            ModelConfig(task_structure=[[1]], task_types=["ordinal"])

    def test_task_types_wrong_length_raises(self):
        with pytest.raises(ValueError, match="task_types length"):
            ModelConfig(task_structure=[[3]], task_types=["binary", "binary"])

    def test_task_names_wrong_length_raises(self):
        with pytest.raises(ValueError, match="task_names length"):
            ModelConfig(task_structure=[[3]], task_names=["a", "b"])

    def test_num_classes_wrong_length_raises(self):
        with pytest.raises(ValueError, match="num_classes_per_task length"):
            ModelConfig(task_structure=[[2]], num_classes_per_task=[2])

    def test_binary_task_with_3_classes_raises(self):
        with pytest.raises(ValueError, match="Binary tasks should have num_classes=2"):
            ModelConfig(
                task_structure=[[1]],
                task_types=["binary"],
                num_classes_per_task=[3],
            )

    def test_multiclass_with_2_classes_raises(self):
        with pytest.raises(ValueError, match="Multiclass tasks should have num_classes >= 3"):
            ModelConfig(
                task_structure=[[1]],
                task_types=["multiclass"],
                num_classes_per_task=[2],
            )

    def test_multiclass_with_1_class_raises(self):
        with pytest.raises(ValueError, match="Multiclass tasks should have num_classes >= 3"):
            ModelConfig(
                task_structure=[[1]],
                task_types=["multiclass"],
                num_classes_per_task=[1],
            )


# ---------------------------------------------------------------------------
# ModelConfig — output_pools validation
# ---------------------------------------------------------------------------

class TestModelConfigOutputPools:
    def _cfg(self, output_pools):
        return ModelConfig(
            task_structure=[[3]],
            task_names=["a", "b", "c"],
            task_types=["binary", "binary", "binary"],
            output_pools=output_pools,
        )

    def test_pools_without_task_names_raises(self):
        with pytest.raises(ValueError, match="task_names must be provided"):
            ModelConfig(
                task_structure=[[2]],
                output_pools={"p": {"tasks": ["task_0", "task_1"], "strategy": "max"}},
            )

    def test_pool_name_equals_task_name_raises(self):
        with pytest.raises(ValueError, match="conflicts with an existing task name"):
            self._cfg({"a": {"tasks": ["b", "c"], "strategy": "max"}})

    def test_pool_references_nonexistent_task_raises(self):
        with pytest.raises(ValueError, match="not in task_names"):
            self._cfg({"p": {"tasks": ["a", "xyz"], "strategy": "max"}})

    def test_pool_with_single_task_raises(self):
        with pytest.raises(ValueError, match="at least 2 task names"):
            self._cfg({"p": {"tasks": ["a"], "strategy": "max"}})

    def test_pool_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            self._cfg({"p": {"tasks": ["a", "b"], "strategy": "product"}})

    def test_task_assigned_to_two_pools_raises(self):
        with pytest.raises(ValueError, match="already assigned to pool"):
            self._cfg({
                "p1": {"tasks": ["a", "b"], "strategy": "max"},
                "p2": {"tasks": ["a", "c"], "strategy": "max"},
            })

    def test_regression_task_in_pool_raises(self):
        with pytest.raises(ValueError, match="Regression tasks cannot be included"):
            ModelConfig(
                task_structure=[[3]],
                task_names=["a", "b", "r"],
                task_types=["binary", "binary", "regression"],
                num_classes_per_task=[2, 2, 1],
                output_pools={"p": {"tasks": ["a", "r"], "strategy": "max"}},
            )

    def test_pool_missing_tasks_field_raises(self):
        with pytest.raises(ValueError, match="missing required field 'tasks'"):
            self._cfg({"p": {"strategy": "max"}})

    def test_pool_missing_strategy_field_raises(self):
        with pytest.raises(ValueError, match="missing required field 'strategy'"):
            self._cfg({"p": {"tasks": ["a", "b"]}})

    def test_pool_num_classes_less_than_2_raises(self):
        with pytest.raises(ValueError, match="num_classes must be an integer >= 2"):
            self._cfg({"p": {"tasks": ["a", "b"], "strategy": "max", "num_classes": 1}})

    def test_binary_pool_with_wrong_num_classes_raises(self):
        with pytest.raises(ValueError, match="Binary pools should have num_classes=2"):
            self._cfg({
                "p": {
                    "tasks": ["a", "b"],
                    "strategy": "max",
                    "task_type": "binary",
                    "num_classes": 4,
                }
            })


# ---------------------------------------------------------------------------
# TrainingConfig validation
# ---------------------------------------------------------------------------

class TestTrainingConfigValidation:
    def test_epochs_zero_raises(self):
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=0)

    def test_epochs_negative_raises(self):
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=-5)

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=0)

    def test_batch_size_negative_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=-1)

    def test_learning_rate_zero_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=0.0)

    def test_learning_rate_negative_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=-1e-4)

    def test_dropout_rate_equal_one_raises(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            TrainingConfig(dropout_rate=1.0)

    def test_dropout_rate_above_one_raises(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            TrainingConfig(dropout_rate=1.5)

    def test_dropout_rate_negative_raises(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            TrainingConfig(dropout_rate=-0.1)

    def test_max_sequence_length_exceeds_512_raises(self):
        with pytest.raises(ValueError, match="max_sequence_length"):
            TrainingConfig(max_sequence_length=1024)

    def test_max_sequence_length_exactly_512_is_valid(self):
        cfg = TrainingConfig(max_sequence_length=512)
        assert cfg.max_sequence_length == 512

    def test_valid_config_does_not_raise(self):
        cfg = TrainingConfig(
            epochs=5,
            batch_size=16,
            learning_rate=1e-3,
            dropout_rate=0.1,
            max_sequence_length=256,
        )
        assert cfg.epochs == 5


# ---------------------------------------------------------------------------
# EncoderConfig validation
# ---------------------------------------------------------------------------

class TestEncoderConfigValidation:
    def test_custom_encoder_not_keras_layer_raises(self):
        with pytest.raises(ValueError, match="Keras Layer"):
            EncoderConfig(
                encoder="not_a_layer",
                encoder_input_type=EncoderInputType.RAW_STRING,
            )

    def test_custom_encoder_without_input_type_raises(self):
        layer = tf.keras.layers.Dense(64)
        with pytest.raises(ValueError, match="encoder_input_type"):
            EncoderConfig(encoder=layer, encoder_input_type=None)

    def test_custom_encoder_with_input_type_is_valid(self):
        layer = tf.keras.layers.Dense(64)
        cfg = EncoderConfig(
            encoder=layer,
            encoder_input_type=EncoderInputType.RAW_STRING,
        )
        assert cfg.encoder is layer

    def test_unrecognized_integration_prefix_raises(self):
        with pytest.raises(ValueError, match="Unrecognized integration"):
            EncoderConfig(encoder_identifier="notreal:some-model")

    def test_api_integration_forces_trainable_false(self):
        cfg = EncoderConfig(
            encoder_identifier="openai:text-embedding-3-small",
            trainable_encoder=True,
        )
        assert cfg.trainable_encoder is False

    def test_huggingface_allows_trainable_true(self):
        cfg = EncoderConfig(
            encoder_identifier="bert-base-uncased",
            trainable_encoder=True,
        )
        assert cfg.trainable_encoder is True


# ---------------------------------------------------------------------------
# API encoders (OpenAI, Cohere, Voyage, HF Inference)
# ---------------------------------------------------------------------------


class TestApiEncoderErrors:
    """Missing API keys and unsupported model strings; no live API calls."""

    def test_openai_encoder_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        pytest.importorskip("openai")
        from multitask.models.encoders import OpenAIEncoder

        enc = OpenAIEncoder(model_name="text-embedding-3-small")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            enc._embed_texts(["hello"])

    def test_cohere_encoder_raises_without_api_key(self, monkeypatch):
        pytest.importorskip("cohere")
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        from multitask.models.encoders import CohereEncoder

        enc = CohereEncoder(model_name="embed-english-v3.0")
        with pytest.raises(ValueError, match="COHERE_API_KEY"):
            enc._embed_texts(["hello"])

    def test_voyage_encoder_raises_without_api_key(self, monkeypatch):
        pytest.importorskip("voyageai")
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        from multitask.models.encoders import VoyageEncoder

        enc = VoyageEncoder(model_name="voyage-3")
        with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
            enc._embed_texts(["hello"])

    def test_openai_unknown_model_raises(self):
        from multitask.models.encoders import OpenAIEncoder

        with pytest.raises(ValueError, match="Unknown OpenAI model"):
            OpenAIEncoder(model_name="not-a-real-openai-model")

    def test_hf_inference_unknown_model_raises(self):
        from multitask.models.encoders import HFInferenceEncoder

        with pytest.raises(ValueError, match="Unknown HuggingFace model"):
            HFInferenceEncoder(model_name="not-in-embedding-table")


# ---------------------------------------------------------------------------
# MultiTaskModel construction errors
# ---------------------------------------------------------------------------

class TestMultiTaskModelErrors:
    def test_no_encoder_and_no_embedding_dim_raises(self):
        config = ModelConfig(task_structure=[[2]])
        with pytest.raises(ValueError, match="embedding_dim"):
            MultiTaskModel(config=config)

    def test_dropout_rate_one_raises(self):
        config = ModelConfig(task_structure=[[2]])
        with pytest.raises(ValueError, match="dropout_rate"):
            MultiTaskModel(config=config, embedding_dim=32, dropout_rate=1.0)

    def test_dropout_rate_negative_raises(self):
        config = ModelConfig(task_structure=[[2]])
        with pytest.raises(ValueError, match="dropout_rate"):
            MultiTaskModel(config=config, embedding_dim=32, dropout_rate=-0.1)

    def test_num_classes_zero_raises(self):
        config = ModelConfig(task_structure=[[2]])
        with pytest.raises(ValueError, match="num_classes_per_task"):
            MultiTaskModel(config=config, embedding_dim=32, num_classes_per_task=[2, 0])


# ---------------------------------------------------------------------------
# Trainer runtime validation
# ---------------------------------------------------------------------------

class TestTrainerRuntimeValidation:
    def _model(self, task_names, task_types, num_classes=None):
        if num_classes is None:
            num_classes = [2] * len(task_types)
        config = ModelConfig(
            task_structure=[[len(task_types)]],
            task_names=task_names,
            task_types=task_types,
            num_classes_per_task=num_classes,
        )
        return MultiTaskModel(config=config, embedding_dim=32)

    def test_multilabel_partial_mask_raises(self):
        """A multilabel task with one label masked but not all must raise ValueError."""
        model = self._model(["ml"], ["multilabel"], [3])
        trainer = Trainer(model=model)
        y_true = {"ml": tf.constant([[1.0, -1.0, 0.0]], dtype=tf.float32)}
        with pytest.raises(ValueError, match="partially missing"):
            trainer._validate_multilabel(y_true)

    def test_multilabel_all_labels_masked_does_not_raise(self):
        model = self._model(["ml"], ["multilabel"], [3])
        trainer = Trainer(model=model)
        y_true = {"ml": tf.constant([[-1.0, -1.0, -1.0]], dtype=tf.float32)}
        # Should not raise
        trainer._validate_multilabel(y_true)

    def test_multilabel_all_labels_present_does_not_raise(self):
        model = self._model(["ml"], ["multilabel"], [3])
        trainer = Trainer(model=model)
        y_true = {"ml": tf.constant([[1.0, 0.0, 1.0]], dtype=tf.float32)}
        trainer._validate_multilabel(y_true)

    def test_non_multilabel_task_passes_validation(self):
        """_validate_multilabel skips non-multilabel tasks without error."""
        model = self._model(["b"], ["binary"])
        trainer = Trainer(model=model)
        y_true = {"b": tf.constant([[1.0], [-1.0]], dtype=tf.float32)}
        trainer._validate_multilabel(y_true)  # must not raise


# ---------------------------------------------------------------------------
# Persistence errors
# ---------------------------------------------------------------------------

class TestPersistenceErrors:
    def test_load_nonexistent_path_raises(self):
        from multitask.models.persistence import load_model
        with pytest.raises(ValueError, match="[Nn]ot found"):
            load_model("/tmp/multitask_definitely_does_not_exist_xyz987")

    def test_load_missing_config_raises(self):
        import tempfile

        from multitask.models.persistence import load_model

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="[Cc]onfig file not found"):
                load_model(tmp)


# ---------------------------------------------------------------------------
# ValueError message readability (spot-checks)
# ---------------------------------------------------------------------------


class TestValueErrorMessagesReadable:
    """Common ValueError messages should name the bad value and what was expected."""

    @pytest.mark.parametrize(
        "build_fn,expected_fragments",
        [
            (
                lambda: ModelConfig(task_structure=[[1]], task_types=["not_a_type"]),
                ["not_a_type", "task_type", "binary", "multiclass"],
            ),
            (
                lambda: TrainingConfig(epochs=0),
                ["epochs", "1"],
            ),
            (
                lambda: EncoderConfig(
                    encoder=tf.keras.layers.Dense(8),
                    encoder_input_type=None,
                ),
                ["encoder_input_type"],
            ),
        ],
    )
    def test_valueerror_contains_invalid_value_or_expectation(self, build_fn, expected_fragments):
        with pytest.raises(ValueError) as excinfo:
            build_fn()
        msg = str(excinfo.value).lower()
        for frag in expected_fragments:
            assert frag.lower() in msg, f"Expected {frag!r} in message: {excinfo.value!r}"

    def test_unrecognized_encoder_prefix_message(self):
        with pytest.raises(ValueError) as excinfo:
            EncoderConfig(encoder_identifier="badprefix:model")
        msg = str(excinfo.value)
        assert "badprefix" in msg or "Unrecognized" in msg
