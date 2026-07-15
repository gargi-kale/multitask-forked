"""Example: BERT fine-tuning with hierarchical multi-task branching.

Demonstrates end-to-end training with a HuggingFace BERT encoder shared
across a branching multi-task architecture, using GPU acceleration where
available.

Architecture:
    bert-base-uncased (shared, fine-tuned)
        └── shared dense layers
            ├── Branch A
            │   ├── political  (binary)
            │   └── sentiment  (multiclass: neg / neutral / pos)
            └── Branch B
                ├── agreement  (binary)
                └── confidence (binary)

Usage:
    pip install 'multitask-py[huggingface]'
    pip install tensorflow-metal==1.2.0  # Apple Silicon only
    python examples/bert_multitask_example.py
"""

import os, shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

from multitask import MultiTaskModel, Trainer, TrainingConfig, load_model
from multitask.config import ModelConfig, EncoderConfig, EncoderInputType, EncoderIntegration
from multitask.utils.distributed import get_distribution_strategy, setup_gpu_memory_growth

SEED = 77
BERT_ID = "bert-base-uncased"
MAX_LEN = 32
N_TRAIN, N_VAL = 160, 40
BATCH_SIZE = 8
EPOCHS = 2
SAVE_PATH = "/tmp/bert_multitask_model"

rng = np.random.RandomState(SEED)

SUBJECTS = ["the government", "researchers", "the media", "experts", "officials"]
VERBS    = ["claim", "argue", "suggest", "confirm", "announce"]
OBJECTS  = [
    "that the policy is effective",
    "a new scientific finding",
    "the latest economic data",
    "significant disagreement exists",
    "the situation has improved",
]


def make_texts(n):
    subj = rng.choice(SUBJECTS, n)
    verb = rng.choice(VERBS,    n)
    obj  = rng.choice(OBJECTS,  n)
    return [f"{s} {v} {o}" for s, v, o in zip(subj, verb, obj)]


def make_labels(n):
    return {
        "political":  rng.randint(0, 2, n).astype(np.int32),
        "sentiment":  rng.randint(0, 3, n).astype(np.int32),
        "agreement":  rng.randint(0, 2, n).astype(np.int32),
        "confidence": rng.randint(0, 2, n).astype(np.int32),
    }


def tokenize(texts, tokenizer):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="tf",
    )
    return {k: enc[k] for k in ["input_ids", "attention_mask", "token_type_ids"]}


def make_dataset(texts, labels, tokenizer):
    tokens   = tokenize(texts, tokenizer)
    token_ds = tf.data.Dataset.from_tensor_slices(tokens)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((token_ds, label_ds)).batch(BATCH_SIZE)


def main():
    # device setup
    print("\n" + "=" * 60)
    print("Device setup")
    print("=" * 60)
    setup_gpu_memory_growth()
    strategy, info = get_distribution_strategy("auto", verbose=True)
    print(f"Strategy: {info}\n")

    # data
    print("=" * 60)
    print("Generating toy data")
    print("=" * 60)
    train_texts  = make_texts(N_TRAIN)
    val_texts    = make_texts(N_VAL)
    train_labels = make_labels(N_TRAIN)
    val_labels   = make_labels(N_VAL)
    print(f"Train: {N_TRAIN} samples | Val: {N_VAL} samples")
    print(f"Example: '{train_texts[0]}'")

    print("\nLoading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_ID)
    train_ds  = make_dataset(train_texts, train_labels, tokenizer)
    val_ds    = make_dataset(val_texts,   val_labels,   tokenizer)
    print("Datasets ready\n")

    # model config — two-level branching: [[2], [2, 2]] → 4 task heads
    encoder_cfg = EncoderConfig(
        encoder_identifier=BERT_ID,
        encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
        encoder_integration=EncoderIntegration.HUGGINGFACE,
        trainable_encoder=True,
    )
    model_cfg = ModelConfig(
        task_structure=[[2], [2, 2]],
        task_names=["political", "sentiment", "agreement", "confidence"],
        task_types=["binary", "multiclass", "binary", "binary"],
        num_classes_per_task=[2, 3, 2, 2],
        shared_layer_sizes=[128],
        default_layer_sizes=[64],
        encoder_config=encoder_cfg,
    )
    train_cfg = TrainingConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        early_stopping=False,
    )

    # training — do NOT wrap in strategy.scope(); Trainer.fit() handles it internally
    print("=" * 60)
    print("Training")
    print("=" * 60)
    model   = MultiTaskModel(model_cfg)
    trainer = Trainer(model=model, config=train_cfg, strategy=strategy)
    history, thresholds = trainer.fit(train_ds, val_ds, num_tasks=4)

    print(f"\nLoss curve: {[round(v, 4) for v in history.history['loss']]}")
    print(f"Thresholds: {thresholds}")

    # predict
    print("\n" + "=" * 60)
    print("Predict")
    print("=" * 60)
    sample_tokens = tokenize(val_texts[:4], tokenizer)
    preds = model.predict(sample_tokens)
    for task, tensor in preds.items():
        print(f"  {task:12s} shape: {tensor.shape}")

    # evaluate
    print("\n" + "=" * 60)
    print("Evaluate")
    print("=" * 60)
    metrics = trainer.evaluate(val_ds)
    overall = metrics.get("overall_metrics", {})
    for metric, value in overall.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")

    # save / load
    print("\n" + "=" * 60)
    print("Save / Load")
    print("=" * 60)
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
    model.save(SAVE_PATH, thresholds=thresholds)
    print(f"Saved to {SAVE_PATH}")

    loaded_model, loaded_thresholds = load_model(SAVE_PATH)
    loaded_preds = loaded_model.predict(sample_tokens)
    for task in preds:
        diff = np.abs(preds[task] - loaded_preds[task]).max()
        print(f"  {task:12s} max prediction diff after reload: {diff:.2e}")

    print("\n✓ Example complete")


if __name__ == "__main__":
    main()
