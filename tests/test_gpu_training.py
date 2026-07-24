"""
GPU training smoke test for Apple Silicon (Metal backend).

Confirms that:
1. tensorflow-metal is installed and TF detects the GPU.
2. get_distribution_strategy('auto') selects a Metal GPU strategy.
3. A multi-task model built and trained under that strategy completes
   without error and returns sane outputs.

Skip conditions: no GPU detected (e.g. CI without Metal).
"""

import os
import pytest
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from multitask.config import ModelConfig, TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer
from multitask.utils.distributed import get_distribution_strategy, setup_gpu_memory_growth


GPUS = tf.config.list_physical_devices("GPU")
pytestmark = pytest.mark.skipif(
    len(GPUS) == 0,
    reason="No GPU detected — skipping Metal training test",
)


def make_dataset(num_samples=64, embedding_dim=32, seed=7):
    rng = np.random.RandomState(seed)
    x = rng.randn(num_samples, embedding_dim).astype(np.float32)
    labels = {
        "task_binary": rng.randint(0, 2, size=(num_samples,)).astype(np.float32),
        "task_multiclass": rng.randint(0, 3, size=(num_samples,)).astype(np.float32),
    }
    return tf.data.Dataset.from_tensor_slices((x, labels)).batch(16)


def test_metal_gpu_detected():
    """At least one GPU device must be visible to TF."""
    assert len(GPUS) >= 1, f"Expected >=1 GPU, got: {GPUS}"
    assert GPUS[0].device_type == "GPU"


def test_distribution_strategy_selects_gpu():
    """get_distribution_strategy('auto') should pick a GPU-based strategy on Apple Silicon."""
    strategy, info = get_distribution_strategy("auto", verbose=False)
    assert "GPU" in info or "Metal" in info, (
        f"Expected GPU/Metal strategy on Apple Silicon, got: {info!r}"
    )
    assert not isinstance(strategy, tf.distribute.OneDeviceStrategy) or "/gpu:0" in (
        strategy._device.lower()
        if hasattr(strategy, "_device")
        else "/gpu:0"
    )


def test_gpu_training_completes():
    """Build and train a small model under the Metal GPU strategy for 1 epoch.

    The model and trainer must NOT be created inside strategy.scope() — Trainer.fit()
    enters the scope internally during compilation. Pre-scoping causes Keras to
    mis-route dict-output tensors to the loss function on single-GPU (OneDevice) strategies.
    """
    setup_gpu_memory_growth()
    strategy, info = get_distribution_strategy("auto", verbose=False)

    config = ModelConfig(
        task_structure=[[2]],
        task_names=["task_binary", "task_multiclass"],
        task_types=["binary", "multiclass"],
        num_classes_per_task=[2, 3],
    )
    training_config = TrainingConfig(epochs=1, batch_size=16)

    model = MultiTaskModel(config=config, embedding_dim=32)
    trainer = Trainer(model=model, config=training_config, strategy=strategy)

    ds = make_dataset()
    history, thresholds = trainer.fit(ds, ds, num_tasks=2, verbose=0)

    assert "task_binary" in thresholds
    assert "task_multiclass" in thresholds
    assert len(history.history.get("loss", [])) == 1


def test_gpu_predict_returns_correct_shape():
    """Model predictions under Metal strategy should have correct output shape."""
    setup_gpu_memory_growth()
    strategy, _ = get_distribution_strategy("auto", verbose=False)

    config = ModelConfig(
        task_structure=[[2]],
        task_names=["task_binary", "task_multiclass"],
        task_types=["binary", "multiclass"],
        num_classes_per_task=[2, 3],
    )

    model = MultiTaskModel(config=config, embedding_dim=32)

    rng = np.random.RandomState(0)
    x = rng.randn(8, 32).astype(np.float32)
    preds = model.predict(x, verbose=0)

    assert "task_binary" in preds
    assert "task_multiclass" in preds
    assert preds["task_binary"].shape == (8, 1)
    assert preds["task_multiclass"].shape == (8, 3)
