"""Minimal end-to-end multitask pipeline (precomputed embeddings, no encoder).

Run locally:
    python examples/minimal_pipeline_example.py

Steps: ModelConfig → MultiTaskModel → tf.data.Dataset → Trainer.fit → evaluate → save/load.
"""

from __future__ import annotations

import tempfile

import numpy as np
import tensorflow as tf

from multitask import MultiTaskModel, Trainer, TrainingConfig, load_model
from multitask.config import EncoderConfig, EncoderInputType, ModelConfig


def main() -> None:
    # Create synthetic training and validation data
    rng = np.random.default_rng(0)
    n_train, n_val = 256, 64
    dim = 128

    # Model: three heads from one split (binary, 4-class, binary)
    task_names = ["sentiment", "topic", "toxicity"]
    config = ModelConfig(
        task_structure=[[3]],
        task_names=task_names,
        task_types=["binary", "multiclass", "binary"],
        num_classes_per_task=[2, 4, 2],
        shared_layer_sizes=[64, 32],
        default_layer_sizes=[32],
        encoder_config=EncoderConfig(
            encoder_identifier=None,
            encoder_input_type=EncoderInputType.PRECOMPUTED,
            embedding_dim=dim,
        ),
    )

    model = MultiTaskModel(config, embedding_dim=dim, dropout_rate=0.2)
    _ = model(tf.zeros((1, dim)), training=False)

    # Synthetic (x, labels_dict) datasets
    def make_arrays(n: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x = rng.standard_normal((n, dim)).astype(np.float32)
        labels = {
            "sentiment": rng.integers(0, 2, size=n, dtype=np.int32),
            "topic": rng.integers(0, 4, size=n, dtype=np.int32),
            "toxicity": rng.integers(0, 2, size=n, dtype=np.int32),
        }
        return x, labels

    x_tr, y_tr = make_arrays(n_train)
    x_va, y_va = make_arrays(n_val)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .batch(16)
        .repeat()
        .prefetch(2)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((x_va, y_va)).batch(32).prefetch(2)

    training = TrainingConfig(epochs=3, batch_size=16, learning_rate=3e-4)
    trainer = Trainer(model, training)

    print("Training...")
    _, thresholds = trainer.fit(
        train_ds,
        val_ds,
        num_tasks=3,
        steps_per_epoch=n_train // 16,
        validation_steps=(n_val + 31) // 32,
        optimize_thresholds=True,
        verbose=1,
    )
    print("Thresholds:", thresholds)

    print("\nEvaluating...")
    results = trainer.evaluate(val_ds, thresholds=thresholds)
    overall = results.get("overall_metrics", {})
    print("Overall:", {k: round(v, 4) for k, v in overall.items() if v is not None})

    with tempfile.TemporaryDirectory(prefix="multitask_minimal_") as d:
        model.save(d, thresholds=thresholds)
        loaded, _loaded_thr = load_model(d)
        x_test = x_va[:8]
        o1 = model(x_test, training=False)
        o2 = loaded(x_test, training=False)
        for name in task_names:
            diff = float(tf.reduce_max(tf.abs(o1[name] - o2[name])))
            assert diff < 1e-5, name
        print("Save/load check passed (max abs diff < 1e-5).")


if __name__ == "__main__":
    main()
