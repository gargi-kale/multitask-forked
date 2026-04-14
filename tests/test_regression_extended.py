"""Extra regression standardization checks: prediction scale and mixed task types."""

import numpy as np
import pytest
import tensorflow as tf

from multitask.config import ModelConfig, TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer


@pytest.fixture
def rng():
    return np.random.RandomState(7)


def test_regression_predictions_match_label_scale(rng):
    """After training with standardization, regression head outputs stay in original target scale."""
    emb = 32
    n = 128
    batch = 16
    x = rng.randn(n, emb).astype(np.float32)
    labels = {
        "r": (rng.randn(n) * 50 + 300).astype(np.float32),
    }
    ds = tf.data.Dataset.from_tensor_slices((x, labels)).batch(batch)

    config = ModelConfig(
        task_structure=[[1]],
        task_names=["r"],
        task_types=["regression"],
        standardize_regression=True,
        shared_layer_sizes=[32],
        default_layer_sizes=[16],
    )
    model = MultiTaskModel(config=config, embedding_dim=emb)
    trainer = Trainer(
        model,
        TrainingConfig(epochs=3, early_stopping=False, use_lr_schedule=False, use_warmup=False),
    )
    trainer.fit(ds, ds, num_tasks=1, verbose=0)

    preds = model(x[: batch], training=False)["r"].numpy().ravel()
    y = labels["r"][: batch]
    assert np.mean(np.abs(preds - y)) < np.std(y) * 2.0 + 1e-2


def test_standardization_mixed_with_classification_heads(rng):
    """Regression standardization must not break softmax/sigmoid heads in the same model."""
    emb = 32
    n = 96
    batch = 16
    x = rng.randn(n, emb).astype(np.float32)
    labels = {
        "b": rng.randint(0, 2, n).astype(np.float32),
        "mc": rng.randint(0, 4, n).astype(np.float32),
        "ml": rng.randint(0, 2, (n, 3)).astype(np.float32),
        "r": (rng.randn(n) * 40 + 200).astype(np.float32),
    }
    ds = tf.data.Dataset.from_tensor_slices((x, labels)).batch(batch)

    config = ModelConfig(
        task_structure=[[4]],
        task_names=["b", "mc", "ml", "r"],
        task_types=["binary", "multiclass", "multilabel", "regression"],
        num_classes_per_task=[2, 4, 3, 1],
        standardize_regression=True,
        shared_layer_sizes=[48],
        default_layer_sizes=[16],
    )
    model = MultiTaskModel(config=config, embedding_dim=emb)
    trainer = Trainer(
        model,
        TrainingConfig(epochs=2, early_stopping=False, use_lr_schedule=False, use_warmup=False),
    )
    history, _ = trainer.fit(ds, ds, num_tasks=4, verbose=0)
    assert all(np.isfinite(v) for v in history.history["loss"])

    out = model(x[:batch], training=False)
    assert out["b"].shape == (batch, 1)
    assert out["mc"].shape == (batch, 4)
    assert out["ml"].shape == (batch, 3)
    assert out["r"].shape == (batch, 1)
    assert np.all((out["b"].numpy() > 0) & (out["b"].numpy() < 1))
    np.testing.assert_allclose(out["mc"].numpy().sum(axis=1), 1.0, atol=1e-4)
    assert np.all((out["ml"].numpy() > 0) & (out["ml"].numpy() < 1))

    results = trainer.evaluate(ds)
    for name, keys in [
        ("b", ("accuracy",)),
        ("mc", ("accuracy",)),
        ("ml", ("f1",)),
        ("r", ("mse",)),
    ]:
        for k in keys:
            assert k in results["task_metrics"][name], (
                f"missing {k} for {name}: {results['task_metrics'][name]}"
            )
