import pytest

ENCODER_OUTPUT_DIM = 64
BATCH_SIZE = 4

try:
    import tensorflow as tf
    from multitask.config import ModelConfig
    from multitask.models.hierarchical import MultiTaskModel
    HAS_TF = True

except ImportError:
    HAS_TF = False
    tf = None
    ModelConfig = None
    MultiTaskModel = None
    encoder = None

pytestmark = pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")

class FakeEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def call(self, inputs, training=None):
        return inputs

def make_encoder_inputs(batch_size=BATCH_SIZE, encoder_dim=ENCODER_OUTPUT_DIM):
    """Inputs matching the custom encoder contract: (batch, encoder_output_dim)"""
    return tf.random.normal((batch_size, encoder_dim))

# --- Fixtures ---

@pytest.fixture
def config_two_tasks():
    """ModelConfig with 2 tasks: [[2]]."""
    return ModelConfig(
        task_structure=[[2]],
        shared_layer_sizes=[64, 32],
        default_layer_sizes=[32, 16],
        trainable_encoder=False
    )

@pytest.fixture
def config_ten_tasks():
    """ModelConfig with 10 tasks: [[3], [2, 5, 3]]."""
    return ModelConfig(
        task_structure=[[3], [2, 5, 3]],
        task_names=[f"task_{i}" for i in range(10)],
        shared_layer_sizes=[64, 32],
        default_layer_sizes=[32, 16],
        trainable_encoder=False
    )


@pytest.fixture
def config_multiclass():
    """ModelConfig with mixed binary and multiclass tasks."""
    return ModelConfig(
        task_structure=[[2]],
        task_types=["binary", "multiclass"],
        num_classes_per_task=[2, 5],
        task_names=["binary_task", "multiclass_task"],
        shared_layer_sizes=[64],
        default_layer_sizes=[32],
    )

@pytest.fixture
def encoder_instance():
    """Fresh custom encoder per test."""
    return FakeEncoderLayer(output_dim=ENCODER_OUTPUT_DIM)

@pytest.fixture
def encoder_inputs():
    """Sample inputs matching the encoder contract: (batch_size, encoder_output_dim)"""
    return make_encoder_inputs()

def _make_model(config, encoder, num_classes=None, dropout_rate=0.1, trainable_encoder=True):
    """Build MultiTaskModel with custom Keras layer as the encoder."""
    return MultiTaskModel(
        config=config,
        encoder=encoder,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        trainable_encoder=trainable_encoder,
    )

@pytest.fixture
def model_with_encoder_two_tasks(config_two_tasks, encoder_instance):
    """MultiTaskModel built with custom encoder; no embedding_dim."""
    return _make_model(config_two_tasks, encoder_instance)

@pytest.fixture
def model_with_encoder_ten_tasks(config_ten_tasks, encoder_instance):
    """MultiTaskModel built with custom encoder; no embedding_dim."""
    return _make_model(config_ten_tasks, encoder_instance)

class TestMultiTaskModelWithCustomEncoder:
    """MultiTaskModel with a custom Keras layer as encoder (no TF Hub / HuggingFace)."""
    def test_model_uses_encoder_and_custom_type(self, model_with_encoder_two_tasks):
        """With custom encoder: uses_encoder=True, encoder_type='custom', no embedding_dim."""
        assert model_with_encoder_two_tasks.uses_encoder is True
        assert model_with_encoder_two_tasks.encoder_type == "custom"
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
        model = _make_model(config_multiclass, encoder_instance)
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
        model = _make_model(config_two_tasks, encoder_instance, trainable_encoder=False)
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