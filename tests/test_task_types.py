"""
End-to-end tests for each individual task type and all mixed-type combinations.

Covers the full pipeline: build → train (few epochs) → evaluate → inspect metrics.
Uses small synthetic datasets (pre-computed embeddings) so tests run fast without
GPU or API keys.

The goal is to make sure mixed heads don't break training, loss aggregation, or
evaluation; and that each task still gets the right kind of metrics and threshold
handling for its type. 

Outcomes measured
* Finite loss: makes sure combined forward/backward path is consistent
* Per-task metrics and keys: catches routing bugs
* Thresholds: Catches misclassification of task type in trainer
* Accuracy, F1, MSE: Makes sure evaluation uses the intended metric for the task
* Shapes
"""

import pytest
import numpy as np
import tensorflow as tf

from multitask.config import ModelConfig, TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

EMB_DIM = 32
N = 64
BATCH = 16
EPOCHS = 2


def _make_dataset(task_names, task_types, num_classes, n=N, seed=0, emb_dim=EMB_DIM):
    """Synthetic (embeddings, labels_dict) dataset."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n, emb_dim).astype(np.float32)
    labels = {}
    for name, ttype, nc in zip(task_names, task_types, num_classes):
        if ttype == "binary":
            labels[name] = rng.randint(0, 2, n).astype(np.float32)
        elif ttype == "multiclass":
            labels[name] = rng.randint(0, nc, n).astype(np.float32)
        elif ttype == "multilabel":
            labels[name] = rng.randint(0, 2, (n, nc)).astype(np.float32)
        elif ttype == "regression":
            labels[name] = rng.randn(n).astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((x, labels)).batch(BATCH)


def _run(task_types, num_classes, task_names=None, emb_dim=EMB_DIM):
    """Build, train, evaluate; return (model, history, thresholds, results)."""
    n = len(task_types)
    if task_names is None:
        task_names = [f"task_{i}" for i in range(n)]
    config = ModelConfig(
        task_structure=[[n]],
        task_names=task_names,
        task_types=task_types,
        num_classes_per_task=num_classes,
        shared_layer_sizes=[32],
        default_layer_sizes=[16],
    )
    model = MultiTaskModel(config=config, embedding_dim=emb_dim)
    training_config = TrainingConfig(
        epochs=EPOCHS,
        early_stopping=False,
        use_lr_schedule=False,
        use_warmup=False,
    )
    trainer = Trainer(model=model, config=training_config)
    ds = _make_dataset(task_names, task_types, num_classes, emb_dim=emb_dim)
    history, thresholds = trainer.fit(ds, ds, num_tasks=n, verbose=0)
    results = trainer.evaluate(ds)
    return model, history, thresholds, results


# ---------------------------------------------------------------------------
# Section 1: Single task type — isolated end-to-end
# ---------------------------------------------------------------------------

class TestSingleBinary:
    def test_output_shape(self):
        model, _, _, _ = _run(["binary"], [2], ["s"])
        out = model(tf.random.normal((4, EMB_DIM)), training=False)
        assert out["s"].shape == (4, 1)

    def test_output_is_probability(self):
        model, _, _, _ = _run(["binary"], [2], ["s"])
        out = model(tf.random.normal((8, EMB_DIM)), training=False)["s"].numpy()
        assert np.all(out > 0) and np.all(out < 1)

    def test_classification_metrics_present(self):
        _, _, _, results = _run(["binary"], [2], ["s"])
        m = results["task_metrics"]["s"]
        for key in ("accuracy", "f1"):
            assert key in m, f"Expected metric '{key}' missing from binary task"

    def test_threshold_is_float(self):
        _, _, thresholds, _ = _run(["binary"], [2], ["s"])
        assert isinstance(thresholds["s"], float)
        assert 0.0 < thresholds["s"] < 1.0

    def test_loss_finite(self):
        _, history, _, _ = _run(["binary"], [2], ["s"])
        assert all(np.isfinite(v) for v in history.history["loss"])


class TestSingleMulticlass:
    def test_output_shape(self):
        model, _, _, _ = _run(["multiclass"], [4], ["t"])
        out = model(tf.random.normal((4, EMB_DIM)), training=False)
        assert out["t"].shape == (4, 4)

    def test_softmax_sums_to_one(self):
        model, _, _, _ = _run(["multiclass"], [4], ["t"])
        out = model(tf.random.normal((8, EMB_DIM)), training=False)["t"].numpy()
        np.testing.assert_allclose(out.sum(axis=1), np.ones(8), atol=1e-5)

    def test_accuracy_metric_present(self):
        _, _, _, results = _run(["multiclass"], [4], ["t"])
        assert "accuracy" in results["task_metrics"]["t"]

    def test_threshold_is_none(self):
        _, _, thresholds, _ = _run(["multiclass"], [5], ["t"])
        assert thresholds["t"] is None

    def test_loss_finite(self):
        _, history, _, _ = _run(["multiclass"], [3], ["t"])
        assert all(np.isfinite(v) for v in history.history["loss"])

    def test_multiclass_10_classes_smoke(self):
        """Many-class softmax path: shapes, metrics, and training stay consistent."""
        model, _, _, results = _run(["multiclass"], [10], ["many"])
        out = model(tf.random.normal((4, EMB_DIM)), training=False)
        assert out["many"].shape == (4, 10)
        assert "accuracy" in results["task_metrics"]["many"]
        np.testing.assert_allclose(
            out["many"].numpy().sum(axis=1), np.ones(4), atol=1e-5
        )


class TestSingleMultilabel:
    def test_output_shape(self):
        model, _, _, _ = _run(["multilabel"], [3], ["ml"])
        out = model(tf.random.normal((4, EMB_DIM)), training=False)
        assert out["ml"].shape == (4, 3)

    def test_sigmoid_outputs_in_unit_interval(self):
        model, _, _, _ = _run(["multilabel"], [3], ["ml"])
        out = model(tf.random.normal((8, EMB_DIM)), training=False)["ml"].numpy()
        assert np.all(out > 0) and np.all(out < 1)

    def test_f1_metric_present(self):
        _, _, _, results = _run(["multilabel"], [3], ["ml"])
        assert "f1" in results["task_metrics"]["ml"]

    def test_threshold_is_float(self):
        _, _, thresholds, _ = _run(["multilabel"], [3], ["ml"])
        assert isinstance(thresholds["ml"], float)

    def test_loss_finite(self):
        _, history, _, _ = _run(["multilabel"], [4], ["ml"])
        assert all(np.isfinite(v) for v in history.history["loss"])


class TestSingleRegression:
    def test_output_shape(self):
        model, _, _, _ = _run(["regression"], [1], ["r"])
        out = model(tf.random.normal((4, EMB_DIM)), training=False)
        assert out["r"].shape == (4, 1)

    def test_regression_metrics_present(self):
        _, _, _, results = _run(["regression"], [1], ["r"])
        m = results["task_metrics"]["r"]
        for key in ("mse", "mae", "r2"):
            assert key in m, f"Expected metric '{key}' missing from regression task"

    def test_threshold_is_none(self):
        _, _, thresholds, _ = _run(["regression"], [1], ["r"])
        assert thresholds["r"] is None

    def test_loss_finite(self):
        _, history, _, _ = _run(["regression"], [1], ["r"])
        assert all(np.isfinite(v) for v in history.history["loss"])


# ---------------------------------------------------------------------------
# Section 2: Pairwise mixed-type combinations
# ---------------------------------------------------------------------------

class TestPairwiseCombinations:
    """One test per unordered pair; checks metrics present and loss finite."""

    def _check(self, types, nc, names):
        _, history, thresholds, results = _run(types, nc, names)
        assert all(np.isfinite(v) for v in history.history["loss"])
        assert set(results["task_metrics"].keys()) == set(names)
        return thresholds, results

    def test_binary_binary(self):
        thresholds, _ = self._check(["binary", "binary"], [2, 2], ["b0", "b1"])
        assert isinstance(thresholds["b0"], float)
        assert isinstance(thresholds["b1"], float)

    def test_binary_multiclass(self):
        thresholds, results = self._check(
            ["binary", "multiclass"], [2, 4], ["b", "mc"]
        )
        assert isinstance(thresholds["b"], float)
        assert thresholds["mc"] is None
        assert "accuracy" in results["task_metrics"]["b"]
        assert "accuracy" in results["task_metrics"]["mc"]

    def test_binary_multilabel(self):
        _, results = self._check(
            ["binary", "multilabel"], [2, 3], ["b", "ml"]
        )
        assert "f1" in results["task_metrics"]["b"]
        assert "f1" in results["task_metrics"]["ml"]

    def test_binary_regression(self):
        thresholds, results = self._check(
            ["binary", "regression"], [2, 1], ["b", "r"]
        )
        assert isinstance(thresholds["b"], float)
        assert thresholds["r"] is None
        assert "accuracy" in results["task_metrics"]["b"]
        assert "mse" in results["task_metrics"]["r"]

    def test_multiclass_multiclass(self):
        _, results = self._check(["multiclass", "multiclass"], [3, 5], ["mc0", "mc1"])
        assert "accuracy" in results["task_metrics"]["mc0"]
        assert "accuracy" in results["task_metrics"]["mc1"]

    def test_multiclass_multilabel(self):
        self._check(["multiclass", "multilabel"], [4, 3], ["mc", "ml"])

    def test_multiclass_regression(self):
        _, results = self._check(["multiclass", "regression"], [4, 1], ["mc", "r"])
        assert "accuracy" in results["task_metrics"]["mc"]
        assert "mse" in results["task_metrics"]["r"]

    def test_multilabel_multilabel(self):
        _, results = self._check(["multilabel", "multilabel"], [3, 4], ["ml0", "ml1"])
        assert "f1" in results["task_metrics"]["ml0"]
        assert "f1" in results["task_metrics"]["ml1"]

    def test_multilabel_regression(self):
        _, results = self._check(["multilabel", "regression"], [3, 1], ["ml", "r"])
        assert "f1" in results["task_metrics"]["ml"]
        assert "mse" in results["task_metrics"]["r"]

    def test_regression_regression(self):
        _, results = self._check(["regression", "regression"], [1, 1], ["r0", "r1"])
        assert "mse" in results["task_metrics"]["r0"]
        assert "mse" in results["task_metrics"]["r1"]


# ---------------------------------------------------------------------------
# Section 3: Triple combinations (3 types at once)
# ---------------------------------------------------------------------------

class TestTripleCombinations:
    def test_binary_multiclass_regression(self):
        _, history, thresholds, results = _run(
            ["binary", "multiclass", "regression"], [2, 3, 1], ["b", "mc", "r"]
        )
        assert all(np.isfinite(v) for v in history.history["loss"])
        assert isinstance(thresholds["b"], float)
        assert thresholds["mc"] is None
        assert thresholds["r"] is None

    def test_binary_multilabel_regression(self):
        _, history, _, results = _run(
            ["binary", "multilabel", "regression"], [2, 3, 1], ["b", "ml", "r"]
        )
        assert all(np.isfinite(v) for v in history.history["loss"])
        assert "accuracy" in results["task_metrics"]["b"]
        assert "f1" in results["task_metrics"]["ml"]
        assert "mse" in results["task_metrics"]["r"]

    def test_multiclass_multilabel_regression(self):
        _, history, _, _ = _run(
            ["multiclass", "multilabel", "regression"], [4, 3, 1], ["mc", "ml", "r"]
        )
        assert all(np.isfinite(v) for v in history.history["loss"])

    def test_binary_multiclass_multilabel(self):
        """Single triple: binary + multiclass + multilabel (no regression)."""
        _, history, thresholds, results = _run(
            ["binary", "multiclass", "multilabel"], [2, 5, 3], ["b", "mc", "ml"]
        )
        assert all(np.isfinite(v) for v in history.history["loss"])
        assert isinstance(thresholds["b"], float)
        assert thresholds["mc"] is None
        assert isinstance(thresholds["ml"], float)
        assert "accuracy" in results["task_metrics"]["mc"]
        assert "f1" in results["task_metrics"]["ml"]


# ---------------------------------------------------------------------------
# Section 4: All four types together
# ---------------------------------------------------------------------------

class TestAllFourTypes:
    TYPES = ["binary", "multiclass", "multilabel", "regression"]
    NC = [2, 4, 3, 1]
    NAMES = ["sentiment", "topic", "tags", "score"]

    def test_model_builds(self):
        config = ModelConfig(
            task_structure=[[4]],
            task_names=self.NAMES,
            task_types=self.TYPES,
            num_classes_per_task=self.NC,
            shared_layer_sizes=[32],
            default_layer_sizes=[16],
        )
        model = MultiTaskModel(config=config, embedding_dim=EMB_DIM)
        assert model is not None

    def test_output_shapes(self):
        model, _, _, _ = _run(self.TYPES, self.NC, self.NAMES)
        x = tf.random.normal((4, EMB_DIM))
        out = model(x, training=False)
        assert out["sentiment"].shape == (4, 1)
        assert out["topic"].shape == (4, 4)
        assert out["tags"].shape == (4, 3)
        assert out["score"].shape == (4, 1)

    def test_all_tasks_have_metrics(self):
        _, _, _, results = _run(self.TYPES, self.NC, self.NAMES)
        assert set(results["task_metrics"].keys()) == set(self.NAMES)

    def test_correct_threshold_types(self):
        _, _, thresholds, _ = _run(self.TYPES, self.NC, self.NAMES)
        assert isinstance(thresholds["sentiment"], float)
        assert thresholds["topic"] is None
        assert isinstance(thresholds["tags"], float)
        assert thresholds["score"] is None

    def test_loss_finite(self):
        _, history, _, _ = _run(self.TYPES, self.NC, self.NAMES)
        assert all(np.isfinite(v) for v in history.history["loss"])

    def test_overall_metrics_present(self):
        _, _, _, results = _run(self.TYPES, self.NC, self.NAMES)
        assert "overall_metrics" in results
        assert isinstance(results["overall_metrics"], dict)
        assert len(results["overall_metrics"]) > 0

    def test_task_specific_metric_keys(self):
        _, _, _, results = _run(self.TYPES, self.NC, self.NAMES)
        assert "accuracy" in results["task_metrics"]["sentiment"]
        assert "accuracy" in results["task_metrics"]["topic"]
        assert "f1" in results["task_metrics"]["tags"]
        assert "mse" in results["task_metrics"]["score"]


# ---------------------------------------------------------------------------
# Pre-computed embeddings: embedding dimension sweep
# ---------------------------------------------------------------------------


class TestPrecomputedEmbeddingDims:
    @pytest.mark.parametrize("emb_dim", [32, 128, 512])
    def test_training_runs_with_embedding_dim(self, emb_dim):
        _, history, _, results = _run(["binary"], [2], ["s"], emb_dim=emb_dim)
        assert all(np.isfinite(v) for v in history.history["loss"])
        assert "accuracy" in results["task_metrics"]["s"]
