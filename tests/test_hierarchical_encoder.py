import os
import sys
import pytest
from unittest.mock import MagicMock, patch

ENCODER_OUTPUT_DIM = 64
BATCH_SIZE = 4

try:
    import tensorflow as tf
    from multitask.config import EncoderConfig, EncoderInputType, EncoderIntegration, ModelConfig
    from multitask.models.hierarchical import MultiTaskModel
    HAS_TF = True

except ImportError:
    HAS_TF = False
    tf = None
    ModelConfig = None
    MultiTaskModel = None
    EncoderConfig = None
    EncoderInputType = None

try:
    import transformers  # noqa: F401
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

pytestmark = pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")

RUN_HF_FROM_PT_INTEGRATION = os.getenv("MULTITASK_RUN_HF_FROM_PT_INTEGRATION", "0") == "1"
HF_PT_ONLY_MODEL_ID = "sentence-transformers/multi-qa-distilbert-cos-v1"

class FakeEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def call(self, inputs, training=None):
        return inputs

def make_encoder_inputs(batch_size=BATCH_SIZE, encoder_dim=ENCODER_OUTPUT_DIM):
    """Inputs matching the custom encoder contract: (batch, encoder_output_dim)"""
    return tf.random.normal((batch_size, encoder_dim))

# --- Fixtures (all use EncoderConfig) ---

@pytest.fixture
def config_two_tasks():
    """ModelConfig with 2 tasks: [[2]], EncoderConfig with trainable_encoder=False."""
    return ModelConfig(
        task_structure=[[2]],
        shared_layer_sizes=[64, 32],
        default_layer_sizes=[32, 16],
        encoder_config=EncoderConfig(trainable_encoder=False),
    )

@pytest.fixture
def config_ten_tasks():
    """ModelConfig with 10 tasks: [[3], [2, 5, 3]], EncoderConfig with trainable_encoder=False."""
    return ModelConfig(
        task_structure=[[3], [2, 5, 3]],
        task_names=[f"task_{i}" for i in range(10)],
        shared_layer_sizes=[64, 32],
        default_layer_sizes=[32, 16],
        encoder_config=EncoderConfig(trainable_encoder=False),
    )

@pytest.fixture
def config_multiclass():
    """ModelConfig with mixed binary and multiclass tasks, EncoderConfig."""
    return ModelConfig(
        task_structure=[[2]],
        task_types=["binary", "multiclass"],
        num_classes_per_task=[2, 5],
        task_names=["binary_task", "multiclass_task"],
        shared_layer_sizes=[64],
        default_layer_sizes=[32],
        encoder_config=EncoderConfig(),
    )

@pytest.fixture
def config_two_tasks_with_encoder_in_config(encoder_instance):
    """ModelConfig with custom encoder inside encoder_config (no encoder= override)."""
    return ModelConfig(
        task_structure=[[2]],
        shared_layer_sizes=[64, 32],
        default_layer_sizes=[32, 16],
        encoder_config=EncoderConfig(
            encoder=encoder_instance,
            encoder_input_type=EncoderInputType.RAW_STRING,
            trainable_encoder=False,
        ),
    )

@pytest.fixture
def config_ten_tasks_with_encoder_in_config(encoder_instance):
    """ModelConfig with 10 tasks and custom encoder inside encoder_config."""
    return ModelConfig(
        task_structure=[[3], [2, 5, 3]],
        task_names=[f"task_{i}" for i in range(10)],
        shared_layer_sizes=[64, 32],
        default_layer_sizes=[32, 16],
        encoder_config=EncoderConfig(
            encoder=encoder_instance,
            encoder_input_type=EncoderInputType.RAW_STRING,
            trainable_encoder=False,
        ),
    )

@pytest.fixture
def encoder_instance():
    """Fresh custom encoder per test."""
    return FakeEncoderLayer(output_dim=ENCODER_OUTPUT_DIM)

@pytest.fixture
def encoder_inputs():
    """Sample inputs matching the encoder contract: (batch_size, encoder_output_dim)"""
    return make_encoder_inputs()

def _make_model(config, encoder=None, num_classes_per_task=None, trainable_encoder=None):
    """Build MultiTaskModel; pass encoder= for a custom Keras encoder."""
    return MultiTaskModel(
        config=config,
        encoder=encoder,
        num_classes_per_task=num_classes_per_task,
        trainable_encoder=trainable_encoder,
    )

@pytest.fixture
def model_with_encoder_two_tasks(config_two_tasks, encoder_instance):
    """MultiTaskModel built with custom encoder; no embedding_dim."""
    return _make_model(config_two_tasks, encoder=encoder_instance)

@pytest.fixture
def model_with_encoder_ten_tasks(config_ten_tasks, encoder_instance):
    """MultiTaskModel built with custom encoder; no embedding_dim."""
    return _make_model(config_ten_tasks, encoder=encoder_instance)

# Models built with encoder inside encoder_config (no encoder= to constructor)
@pytest.fixture
def model_two_tasks_encoder_in_config(config_two_tasks_with_encoder_in_config):
    """MultiTaskModel with custom encoder inside config.encoder_config."""
    return MultiTaskModel(config=config_two_tasks_with_encoder_in_config)

@pytest.fixture
def model_ten_tasks_encoder_in_config(config_ten_tasks_with_encoder_in_config):
    """MultiTaskModel with 10 tasks and encoder inside config.encoder_config."""
    return MultiTaskModel(config=config_ten_tasks_with_encoder_in_config)

class TestMultiTaskModelWithCustomEncoder:
    """MultiTaskModel with a custom Keras layer as encoder (no TF Hub / HuggingFace)."""
    def test_model_uses_encoder_and_custom_type(self, model_with_encoder_two_tasks):
        """With custom encoder: uses_encoder=True, encoder_type='custom', no embedding_dim."""
        assert model_with_encoder_two_tasks.uses_encoder is True
        assert model_with_encoder_two_tasks.encoder_type == "keras"
        assert model_with_encoder_two_tasks.embedding_dim is None

    def test_forward_two_tasks_with_encoder(self, model_with_encoder_two_tasks, encoder_inputs):
        out = model_with_encoder_two_tasks(encoder_inputs, training=False)
        assert isinstance(out, dict)
        assert out["task_0"].shape == (BATCH_SIZE, 1)
        assert out["task_1"].shape == (BATCH_SIZE, 1)

    def test_forward_ten_tasks_with_encoder(self, model_with_encoder_ten_tasks, encoder_inputs):
        out = model_with_encoder_ten_tasks(encoder_inputs, training=False)
        assert len(out) == 10
        for i in range(10):
            assert out[f"task_{i}"].shape == (BATCH_SIZE, 1)

    def test_forward_multiclass_with_encoder(self, config_multiclass, encoder_instance, encoder_inputs):
        model = _make_model(config_multiclass, encoder=encoder_instance)
        out = model(encoder_inputs, training=False)
        assert out["binary_task"].shape == (BATCH_SIZE, 1)
        assert out["multiclass_task"].shape == (BATCH_SIZE, 5)

    def test_get_config_with_encoder(self, model_with_encoder_two_tasks):
        config = model_with_encoder_two_tasks.get_config()
        assert config["uses_encoder"] is True
        assert config["embedding_dim"] is None
        assert "config" in config
        assert config["config"]["task_structure"] == [[2]]

    def test_summary_tree_mentions_encoder(self, model_with_encoder_two_tasks):
        summary = model_with_encoder_two_tasks.summary_tree()
        assert "Encoder" in summary

    def test_trainable_encoder_false(self, config_two_tasks, encoder_instance):
        """Encoder is non-trainable when trainable_encoder=False."""
        model = _make_model(config_two_tasks, encoder=encoder_instance, trainable_encoder=False)
        assert model.encoder.trainable is False

    def test_get_task_info_with_encoder(self, model_with_encoder_two_tasks):
        info = model_with_encoder_two_tasks.get_task_info()
        assert len(info) == 2
        assert info[0]["name"] == "task_0"
        assert info[0]["parent_branch_path"] == [0]
        assert info[1]["parent_branch_path"] == [1]

    def test_flatten_unflatten_roundtrip_with_encoder(self, model_with_encoder_two_tasks, encoder_inputs):
        out = model_with_encoder_two_tasks(encoder_inputs, training=False)
        flat, indices = model_with_encoder_two_tasks.flatten_outputs(out)
        restored = model_with_encoder_two_tasks.unflatten_outputs(flat)
        assert set(restored.keys()) == set(out.keys())
        for name in out:
            tf.debugging.assert_near(restored[name], out[name])


class TestMultiTaskModelWithEncoderInConfig:
    """MultiTaskModel with custom encoder provided via config.encoder_config (no encoder= to constructor)."""

    def test_model_uses_encoder_from_config(self, model_two_tasks_encoder_in_config):
        """Encoder is taken from config.encoder_config.encoder."""
        assert model_two_tasks_encoder_in_config.uses_encoder is True
        assert model_two_tasks_encoder_in_config.embedding_dim is None
        assert model_two_tasks_encoder_in_config.encoder_type == "keras"

    def test_forward_two_tasks_encoder_in_config(self, model_two_tasks_encoder_in_config, encoder_inputs):
        out = model_two_tasks_encoder_in_config(encoder_inputs, training=False)
        assert isinstance(out, dict)
        assert out["task_0"].shape == (BATCH_SIZE, 1)
        assert out["task_1"].shape == (BATCH_SIZE, 1)

    def test_forward_ten_tasks_encoder_in_config(self, model_ten_tasks_encoder_in_config, encoder_inputs):
        out = model_ten_tasks_encoder_in_config(encoder_inputs, training=False)
        assert len(out) == 10
        for i in range(10):
            assert out[f"task_{i}"].shape == (BATCH_SIZE, 1)

    def test_get_config_encoder_in_config(self, model_two_tasks_encoder_in_config):
        config = model_two_tasks_encoder_in_config.get_config()
        assert config["uses_encoder"] is True
        assert config["embedding_dim"] is None

    def test_summary_tree_encoder_in_config(self, model_two_tasks_encoder_in_config):
        summary = model_two_tasks_encoder_in_config.summary_tree()
        assert "Encoder" in summary

    def test_trainable_encoder_false_in_config(self, model_two_tasks_encoder_in_config):
        """Encoder in encoder_config with trainable_encoder=False is non-trainable."""
        assert model_two_tasks_encoder_in_config.encoder.trainable is False

    def test_get_task_info_encoder_in_config(self, model_two_tasks_encoder_in_config):
        info = model_two_tasks_encoder_in_config.get_task_info()
        assert len(info) == 2
        assert info[0]["name"] == "task_0"
        assert info[1]["name"] == "task_1"

    def test_flatten_unflatten_roundtrip_encoder_in_config(self, model_two_tasks_encoder_in_config, encoder_inputs):
        out = model_two_tasks_encoder_in_config(encoder_inputs, training=False)
        flat, indices = model_two_tasks_encoder_in_config.flatten_outputs(out)
        restored = model_two_tasks_encoder_in_config.unflatten_outputs(flat)
        assert set(restored.keys()) == set(out.keys())
        for name in out:
            tf.debugging.assert_near(restored[name], out[name])


_PT_ONLY_ERROR = OSError(
    "bert-base-uncased does not appear to have a file named tf_model.h5, "
    "model.safetensors.index.json, model.safetensors, tf_model.h5.index.json or tf_model.h5"
)


def _mock_transformers(from_pretrained_side_effect):
    """Return a sys.modules patch that replaces transformers with a mock TFAutoModel."""
    mock_tf_auto = MagicMock()
    mock_tf_auto.from_pretrained.side_effect = from_pretrained_side_effect
    return patch.dict(sys.modules, {"transformers": MagicMock(TFAutoModel=mock_tf_auto)})


class TestHuggingFaceFromPt:
    """from_pt error handling when a HuggingFace model only ships PyTorch weights."""
    def test_raises_informative_error_without_from_pt(self):
        """OSError with no TF weights becomes a clear message pointing to from_pt=True."""
        config = ModelConfig(
            task_structure=[[2]],
            encoder_config=EncoderConfig(
                encoder_identifier="bert-base-uncased",
                encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
                encoder_integration=EncoderIntegration.HUGGINGFACE,
                from_pt=False,
            ),
        )
        with _mock_transformers(_PT_ONLY_ERROR):
            with pytest.raises(OSError, match="from_pt=True"):
                MultiTaskModel(config=config)

    def test_original_error_raised_when_from_pt_true(self):
        """When from_pt=True an OSError is not wrapped — the original is re-raised."""
        config = ModelConfig(
            task_structure=[[2]],
            encoder_config=EncoderConfig(
                encoder_identifier="bert-base-uncased",
                encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
                encoder_integration=EncoderIntegration.HUGGINGFACE,
                from_pt=True,
            ),
        )
        with _mock_transformers(OSError("network timeout")):
            with pytest.raises(OSError, match="network timeout"):
                MultiTaskModel(config=config)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
@pytest.mark.skipif(
    not RUN_HF_FROM_PT_INTEGRATION,
    reason="set MULTITASK_RUN_HF_FROM_PT_INTEGRATION=1 to run networked HF from_pt integration tests",
)
class TestHuggingFaceFromPtIntegration:
    """Networked integration test for models that may only publish PyTorch weights."""

    def test_real_model_without_from_pt_raises_guidance(self):
        """Real HF fetch with from_pt=False should surface guidance for PyTorch-only models."""
        config = ModelConfig(
            task_structure=[[2]],
            encoder_config=EncoderConfig(
                encoder_identifier=HF_PT_ONLY_MODEL_ID,
                encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
                encoder_integration=EncoderIntegration.HUGGINGFACE,
                from_pt=False,
            ),
        )
        try:
            MultiTaskModel(config=config)
        except OSError as exc:
            assert "from_pt=True" in str(exc)
        except ValueError as exc:
            if "Unrecognized configuration class" in str(exc) and "TFAutoModel" in str(exc):
                pytest.skip(
                    f"Model '{HF_PT_ONLY_MODEL_ID}' has no TensorFlow architecture in transformers; "
                    "choose a model with TF architecture support via MULTITASK_HF_PT_ONLY_MODEL_ID."
                )
            raise
        else:
            pytest.skip(
                f"Model '{HF_PT_ONLY_MODEL_ID}' loaded TensorFlow weights directly; "
                "pick a PyTorch-only model via MULTITASK_HF_PT_ONLY_MODEL_ID."
            )

    def test_real_model_with_from_pt_true_loads_without_error(self):
        """Real HF fetch with from_pt=True should initialize without wrapping TF-weight errors."""
        config = ModelConfig(
            task_structure=[[2]],
            encoder_config=EncoderConfig(
                encoder_identifier=HF_PT_ONLY_MODEL_ID,
                encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
                encoder_integration=EncoderIntegration.HUGGINGFACE,
                from_pt=True,
            ),
        )
        try:
            model = MultiTaskModel(config=config)
        except ValueError as exc:
            if "Unrecognized configuration class" in str(exc) and "TFAutoModel" in str(exc):
                pytest.skip(
                    f"Model '{HF_PT_ONLY_MODEL_ID}' has no TensorFlow architecture in transformers; "
                    "cannot validate from_pt conversion for this model family."
                )
            raise
        assert model.uses_encoder is True