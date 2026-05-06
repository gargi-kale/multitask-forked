"""Unit tests for hierarchical multi-task model."""

import pytest

try:
    import tensorflow as tf
    from multitask.config import EncoderConfig, ModelConfig
    from multitask.models.hierarchical import MultiTaskModel
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None
    ModelConfig = None
    MultiTaskModel = None
    EncoderConfig = None

pytestmark = pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")


# --- Fixtures ---


@pytest.fixture
def config_two_tasks():
    """ModelConfig with 2 tasks: [[2]], no encoder (pre-computed embeddings)."""
    return ModelConfig(
        task_structure=[[2]],
        shared_layer_sizes=[64, 32],
        default_layer_sizes=[32, 16],
        encoder_config=EncoderConfig(),
    )


@pytest.fixture
def config_ten_tasks():
    """ModelConfig with 10 tasks: [[3], [2, 5, 3]], no encoder."""
    return ModelConfig(
        task_structure=[[3], [2, 5, 3]],
        task_names=[f"task_{i}" for i in range(10)],
        shared_layer_sizes=[64, 32],
        default_layer_sizes=[32, 16],
        encoder_config=EncoderConfig(),
    )


@pytest.fixture
def config_multiclass():
    """ModelConfig with mixed binary and multiclass tasks, no encoder."""
    return ModelConfig(
        task_structure=[[2]],
        task_types=["binary", "multiclass"],
        num_classes_per_task=[2, 5],
        task_names=["binary_task", "multiclass_task"],
        shared_layer_sizes=[64],
        default_layer_sizes=[32],
        encoder_config=EncoderConfig(),
    )


def _make_model(config, embedding_dim=64, num_classes_per_task=None):
    """Build MultiTaskModel with embeddings (no encoder) for testing."""
    return MultiTaskModel(
        config=config,
        embedding_dim=embedding_dim,
        num_classes_per_task=num_classes_per_task
    )


# --- Initialization and validation ---


class TestMultiTaskModelInit:
    """Tests for MultiTaskModel initialization and validation."""

    def test_build_with_embeddings_two_tasks(self, config_two_tasks):
        """Model builds with embedding input and 2 tasks."""
        model = _make_model(config_two_tasks, embedding_dim=64)
        assert model.uses_encoder is False
        assert model.embedding_dim == 64
        assert len(model.num_classes_per_task) == 2
        assert model.num_classes_per_task == [2, 2]

    def test_build_with_embeddings_ten_tasks(self, config_ten_tasks):
        """Model builds with 10 tasks and default num_classes_per_task."""
        model = _make_model(config_ten_tasks, embedding_dim=128)
        assert model.uses_encoder is False
        assert len(model.num_classes_per_task) == 10
        assert all(n == 2 for n in model.num_classes_per_task)

    def test_num_classes_per_task_override(self, config_two_tasks):
        """Passed num_classes_per_task overrides config default."""
        model = _make_model(
            config_two_tasks,
            embedding_dim=64,
            num_classes_per_task=[2, 5],
        )
        assert model.num_classes_per_task == [2, 5]

    def test_raises_when_binary_task_wrong_num_classes(self):
        """ModelConfig rejects binary task with num_classes != 2."""
        with pytest.raises(ValueError, match="Binary tasks should have num_classes=2"):
            ModelConfig(
                task_structure=[[2]],
                task_types=["binary", "binary"],
                num_classes_per_task=[2, 1],
                encoder_config=EncoderConfig(),
            )

    def test_raises_when_num_classes_per_task_zero(self, config_two_tasks):
        """ValueError when num_classes_per_task contains 0."""
        with pytest.raises(ValueError, match="num_classes_per_task must be at least 1"):
            _make_model(
                config_two_tasks,
                embedding_dim=64,
                num_classes_per_task=[2, 0],
            )

    def test_raises_when_no_encoder_and_no_embedding_dim(self, config_two_tasks):
        """ValueError when encoder is None and embedding_dim is None."""
        with pytest.raises(ValueError, match="Must specify embedding_dim"):
            MultiTaskModel(config=config_two_tasks)


# --- Forward pass and output structure ---


class TestMultiTaskModelCall:
    """Tests for call() and output structure."""

    def test_call_returns_dict(self, config_two_tasks):
        """call() returns a dict mapping task names to tensors."""
        model = _make_model(config_two_tasks, embedding_dim=64)
        batch_size = 4
        x = tf.random.normal((batch_size, 64))
        out = model(x, training=False)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"task_0", "task_1"}
        assert out["task_0"].shape == (batch_size, 1)
        assert out["task_1"].shape == (batch_size, 1)

    def test_call_with_task_names(self, config_ten_tasks):
        """call() keys match config task_names."""
        model = _make_model(config_ten_tasks, embedding_dim=64)
        x = tf.random.normal((4, 64))
        out = model(x, training=False)
        assert set(out.keys()) == {f"task_{i}" for i in range(10)}
        for i in range(10):
            assert out[f"task_{i}"].shape == (4, 1)

    def test_call_multiclass_output_dims(self, config_multiclass):
        """Multiclass task has K output dimensions."""
        model = _make_model(config_multiclass, embedding_dim=64)
        x = tf.random.normal((4, 64))
        out = model(x, training=False)
        assert out["binary_task"].shape == (4, 1)
        assert out["multiclass_task"].shape == (4, 5)

    def test_call_training_true(self, config_two_tasks):
        """call() works with training=True."""
        model = _make_model(config_two_tasks, embedding_dim=64)
        x = tf.random.normal((2, 64))
        out = model(x, training=True)
        assert isinstance(out, dict)
        assert len(out) == 2


# --- get_task_info ---


class TestGetTaskInfo:
    """Tests for get_task_info()."""

    def test_task_info_two_tasks(self, config_two_tasks):
        """get_task_info() for [[2]] has two entries and correct paths."""
        model = _make_model(config_two_tasks, embedding_dim=64)
        info = model.get_task_info()
        assert len(info) == 2
        assert info[0]["output_idx"] == 0
        assert info[0]["name"] == "task_0"
        assert info[0]["parent_branch_path"] == [0]
        assert info[1]["output_idx"] == 1
        assert info[1]["parent_branch_path"] == [1]

    def test_task_info_ten_tasks_with_names(self, config_ten_tasks):
        """get_task_info() for [[3], [2,5,3]] has 10 entries and correct paths."""
        model = _make_model(config_ten_tasks, embedding_dim=64)
        info = model.get_task_info()
        assert len(info) == 10
        # First 2 from branch 0, next 5 from branch 1, last 3 from branch 2
        assert info[0]["parent_branch_path"] == [0, 0]
        assert info[1]["parent_branch_path"] == [0, 1]
        assert info[2]["parent_branch_path"] == [1, 0]
        assert info[6]["parent_branch_path"] == [1, 4]
        assert info[7]["parent_branch_path"] == [2, 0]
        assert info[9]["parent_branch_path"] == [2, 2]
        assert info[0]["name"] == "task_0"
        assert info[9]["name"] == "task_9"


# --- summary_tree ---


class TestSummaryTree:
    """Tests for summary_tree()."""

    def test_summary_tree_contains_structure(self, config_two_tasks):
        """summary_tree() includes structure and total outputs."""
        model = _make_model(config_two_tasks, embedding_dim=64)
        summary = model.summary_tree()
        assert "Pre-computed Embeddings" in summary or "Embeddings" in summary
        assert "Level 0" in summary
        assert "Total Outputs: 2" in summary

    def test_summary_tree_ten_tasks(self, config_ten_tasks):
        """summary_tree() shows correct total for 10 tasks."""
        model = _make_model(config_ten_tasks, embedding_dim=64)
        summary = model.summary_tree()
        assert "Total Outputs: 10" in summary


# --- flatten_outputs / unflatten_outputs ---


class TestFlattenUnflatten:
    """Tests for flatten_outputs and unflatten_outputs."""

    def test_flatten_outputs_shape(self, config_two_tasks):
        """flatten_outputs returns correct concatenated shape."""
        model = _make_model(config_two_tasks, embedding_dim=64)
        x = tf.random.normal((4, 64))
        out_dict = model(x, training=False)
        flat, indices = model.flatten_outputs(out_dict)
        assert flat.shape == (4, 2)  # two binary tasks -> 1 + 1
        assert indices["task_0"] == (0, 1)
        assert indices["task_1"] == (1, 2)

    def test_flatten_multiclass_dims(self, config_multiclass):
        """flatten_outputs accounts for multiclass output dims."""
        model = _make_model(config_multiclass, embedding_dim=64)
        x = tf.random.normal((4, 64))
        out_dict = model(x, training=False)
        flat, indices = model.flatten_outputs(out_dict)
        assert flat.shape == (4, 1 + 5)  # binary + 5-class
        assert indices["binary_task"] == (0, 1)
        assert indices["multiclass_task"] == (1, 6)

    def test_unflatten_roundtrip(self, config_two_tasks):
        """unflatten_outputs(flatten_outputs(out)) matches original structure."""
        model = _make_model(config_two_tasks, embedding_dim=64)
        x = tf.random.normal((4, 64))
        out_dict = model(x, training=False)
        flat, indices = model.flatten_outputs(out_dict)
        restored = model.unflatten_outputs(flat)
        assert set(restored.keys()) == set(out_dict.keys())
        for name in out_dict:
            tf.debugging.assert_near(restored[name], out_dict[name])

    def test_unflatten_roundtrip_multiclass(self, config_multiclass):
        """Roundtrip with mixed binary and multiclass."""
        model = _make_model(config_multiclass, embedding_dim=64)
        x = tf.random.normal((4, 64))
        out_dict = model(x, training=False)
        flat, _ = model.flatten_outputs(out_dict)
        restored = model.unflatten_outputs(flat)
        assert set(restored.keys()) == set(out_dict.keys())
        for name in out_dict:
            tf.debugging.assert_near(restored[name], out_dict[name])


# --- get_config ---


class TestGetConfig:
    """Tests for get_config() serialization."""

    def test_get_config_contains_task_structure(self, config_two_tasks):
        """get_config() includes task_structure and uses_encoder."""
        model = _make_model(config_two_tasks, embedding_dim=64)
        config = model.get_config()
        assert "config" in config
        assert config["config"]["task_structure"] == [[2]]
        assert config["uses_encoder"] is False
        assert config["embedding_dim"] == 64
        assert config["num_classes_per_task"] == [2, 2]
