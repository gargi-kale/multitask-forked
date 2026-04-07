"""End-to-end pipeline with a small hierarchical task graph (two levels).

Run locally:
    python examples/hierarchical_pipeline_example.py

Structure ``[[2], [2, 1]]``: two trunks from the encoder, then 2 + 1 = 3 leaf tasks.
Leaf order follows the tree: first trunk’s children, then the second trunk’s child.
"""

from __future__ import annotations

import tempfile

import numpy as np
import tensorflow as tf

from multitask import MultiTaskModel, Trainer, TrainingConfig, load_model
from multitask.config import EncoderConfig, EncoderInputType, ModelConfig


def main() -> None:
    rng = np.random.default_rng(1)
    dim = 96
    n_train, n_val = 200, 48

    task_names = ["task_a", "task_b", "task_c"]
    config = ModelConfig(
        task_structure=[[2], [2, 1]],
        task_names=task_names,
        task_types=["binary", "binary", "multiclass"],
        num_classes_per_task=[2, 2, 3],
        shared_layer_sizes=[64],
        default_layer_sizes=[48],
        encoder_config=EncoderConfig(
            encoder_identifier=None,
            encoder_input_type=EncoderInputType.PRECOMPUTED,
            embedding_dim=dim,
        ),
    )

    model = MultiTaskModel(config, embedding_dim=dim)
    _ = model(tf.zeros((1, dim)), training=False)

    def make_arrays(n: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x = rng.standard_normal((n, dim)).astype(np.float32)
        labels = {
            "task_a": rng.integers(0, 2, size=n, dtype=np.int32),
            "task_b": rng.integers(0, 2, size=n, dtype=np.int32),
            "task_c": rng.integers(0, 3, size=n, dtype=np.int32),
        }
        return x, labels

    x_tr, y_tr = make_arrays(n_train)
    x_va, y_va = make_arrays(n_val)

    bs = 16
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .batch(bs)
        .repeat()
        .prefetch(2)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((x_va, y_va)).batch(bs).prefetch(2)

    trainer = Trainer(model, TrainingConfig(epochs=4, batch_size=bs, learning_rate=1e-3))

    print("Training hierarchical model...")
    _, thresholds = trainer.fit(
        train_ds,
        val_ds,
        num_tasks=3,
        steps_per_epoch=max(1, n_train // bs),
        validation_steps=max(1, (n_val + bs - 1) // bs),
        optimize_thresholds=True,
        verbose=1,
    )
    print("Thresholds:", thresholds)

    results = trainer.evaluate(val_ds, thresholds=thresholds)
    print("Overall metrics:", results.get("overall_metrics", {}))

    with tempfile.TemporaryDirectory(prefix="multitask_hier_") as d:
        model.save(d, thresholds=thresholds)
        loaded, _ = load_model(d)
        t = tf.constant(x_va[:4], dtype=tf.float32)
        d1 = model(t, training=False)
        d2 = loaded(t, training=False)
        for name in task_names:
            assert float(tf.reduce_max(tf.abs(d1[name] - d2[name]))) < 1e-5
        print("Save/load OK.")


if __name__ == "__main__":
    main()
