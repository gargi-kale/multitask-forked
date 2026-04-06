"""End-to-end pipeline starting from a pandas DataFrame (common local workflow).

Run locally:
    python examples/dataframe_pipeline_example.py

Builds embeddings + label columns in a DataFrame, converts to tf.data, trains,
evaluates, and saves the model. Uses the default TensorFlow distribution strategy
(single machine, no MirroredStrategy setup).
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

from multitask import MultiTaskModel, Trainer, TrainingConfig, load_model
from multitask.config import EncoderConfig, EncoderInputType, ModelConfig


TASK_NAMES = ["rating_binary", "stars_multiclass"]


def build_dataframe(n: int, dim: int, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    return pd.DataFrame(
        {
            "embedding": list(emb),
            "rating_binary": rng.integers(0, 2, size=n, dtype=np.int32),
            "stars_multiclass": rng.integers(0, 5, size=n, dtype=np.int32),
        }
    )


def dataframe_to_dataset(
    df: pd.DataFrame, batch_size: int, *, shuffle: bool, repeat: bool
) -> tf.data.Dataset:
    x = np.stack(df["embedding"].values)
    labels = {k: df[k].values.astype(np.int32) for k in TASK_NAMES}
    ds = tf.data.Dataset.from_tensor_slices((x, labels))
    if shuffle:
        ds = ds.shuffle(min(len(df), 512))
    ds = ds.batch(batch_size)
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(2)


def main() -> None:
    dim = 128
    train_df = build_dataframe(400, dim)
    val_df = build_dataframe(80, dim, seed=3)

    config = ModelConfig(
        task_structure=[[2]],
        task_names=TASK_NAMES,
        task_types=["binary", "multiclass"],
        num_classes_per_task=[2, 5],
        shared_layer_sizes=[128, 64],
        default_layer_sizes=[48],
        encoder_config=EncoderConfig(
            encoder_identifier=None,
            encoder_input_type=EncoderInputType.PRECOMPUTED,
            embedding_dim=dim,
        ),
    )

    model = MultiTaskModel(config, embedding_dim=dim)
    _ = model(tf.zeros((1, dim)), training=False)

    batch_size = 32
    train_ds = dataframe_to_dataset(train_df, batch_size, shuffle=True, repeat=True)
    val_ds = dataframe_to_dataset(val_df, batch_size, shuffle=False, repeat=False)

    trainer = Trainer(model, TrainingConfig(epochs=5, batch_size=batch_size, learning_rate=2e-3))

    print("Training...")
    _, thresholds = trainer.fit(
        train_ds,
        val_ds,
        num_tasks=2,
        optimize_thresholds=True,
        verbose=1,
    )

    print("\nEvaluation...")
    results = trainer.evaluate(val_ds, thresholds=thresholds)
    for k, v in results.get("overall_metrics", {}).items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    sample_x = np.stack(val_df.head(8)["embedding"].values)
    preds_before = model.predict(sample_x, verbose=0)

    with tempfile.TemporaryDirectory(prefix="multitask_df_") as d:
        model.save(d, thresholds=thresholds)
        loaded, _ = load_model(d)
        preds_after = loaded.predict(sample_x, verbose=0)
        for name in TASK_NAMES:
            diff = np.abs(preds_before[name] - preds_after[name]).max()
            assert diff < 1e-5, (name, diff)
    print("DataFrame → train → evaluate → save/load completed successfully.")


if __name__ == "__main__":
    main()
