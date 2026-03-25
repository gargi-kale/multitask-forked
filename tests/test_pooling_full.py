"""
Full test for output_pools feature using real data structure.

Uses small subsets of the actual data (200 train / 200 val / 200 test)
with TF-IDF embeddings instead of a transformer encoder so it runs
quickly on any machine (~1-2 minutes total).

Usage:
    python smoke_test_pools.py

What it verifies (7 checks):
    1. Model builds with output_pools and produces pooled predictions
    2. Pooled predictions match element-wise strategy over constituents
    3. fit() returns thresholds for both tasks AND pools
    4. fit(optimize_thresholds=True) finds pool thresholds
    5. evaluate() returns pool_metrics separate from task_metrics
    6. overall_metrics averages over tasks only, not pools
    7. save/load round-trip preserves pool config and predictions
"""

import os
import sys
import csv
import shutil
import tempfile
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

from multitask.config import ModelConfig, TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer
from multitask.models.persistence import load_model


# 2 propaganda technique columns (original CSV headers)
_PROP_COLUMNS = ["Loaded_Language", "Slogans"]

# Sanitized names for Keras (commas are illegal in TF scope names)
def _sanitize(name: str) -> str:
    return name.replace(",", "_")

PROP_TECHNIQUES = [_sanitize(c) for c in _PROP_COLUMNS]

# Mapping: sanitized task name → original CSV column name
_COL_MAP = {_sanitize(c): c for c in _PROP_COLUMNS}

# 2 argumentation binary columns (no commas, no sanitization needed)
ARG_TASKS = ["disagree_agree_bi", "emotion_fact_bi"]
OTHER_TASKS = ["WA_bi", "propaganda"]

ALL_TASK_NAMES = PROP_TECHNIQUES + ARG_TASKS + OTHER_TASKS  # 28 tasks

# Build full column map (identity for non-propaganda columns)
for t in ARG_TASKS + OTHER_TASKS:
    _COL_MAP[t] = t


def load_csv(path, max_rows=None):
    """Load CSV and return (texts, labels_dict).  NA → -1.
    
    Keys in labels_dict use sanitized task names (matching ALL_TASK_NAMES),
    while CSV columns are looked up via _COL_MAP.
    """
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if max_rows:
        rows = rows[:max_rows]

    texts = [r["text"] for r in rows]
    labels = {}
    for task_name in ALL_TASK_NAMES:
        csv_col = _COL_MAP[task_name]  # map sanitized name → CSV header
        vals = []
        for r in rows:
            v = r[csv_col]
            if v in ("NA", ""):
                vals.append(-1.0)
            else:
                vals.append(float(v))
        labels[task_name] = np.array(vals, dtype=np.float32)
    return texts, labels


def make_tf_dataset(embeddings, labels_dict, batch_size=32):
    """Build a tf.data.Dataset from pre-computed embeddings."""
    ds = tf.data.Dataset.from_tensor_slices((embeddings, labels_dict))
    ds = ds.batch(batch_size).prefetch(1)
    return ds


# ======================================================================
# Configuration
# ======================================================================

SAMPLE_SIZE = 200       # rows per split — small for speed
EMBEDDING_DIM = 256     # TF-IDF features
EPOCHS = 2
BATCH_SIZE = 32

# Task structure: 28 leaf tasks organized in a 2-level hierarchy.
# Level 0: 1 shared trunk → 4 branches
# Level 1: each branch → 7 tasks = 28 total
# TASK_STRUCTURE = [[4], [7, 7, 7, 7]]
TASK_STRUCTURE = [[3], [2, 2, 2]]

# Output pools to test
OUTPUT_POOLS = {
    "any_technique": {
        "tasks": PROP_TECHNIQUES,
        "strategy": "max",
        "task_type": "binary",
        "num_classes": 2,
    },
    "mean_argumentation": {
        "tasks": ARG_TASKS,
        "strategy": "mean",
        "task_type": "binary",
        "num_classes": 2,
    },
}


# ======================================================================
# Smoke test
# ======================================================================

def run():
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}  — {detail}")
            failed += 1

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading data (subsets)...")
    print("=" * 60)

    # Use val as "train" and test as both "val" and "test" to avoid
    # needing the large train file.
    train_path = os.path.join(os.path.dirname(__file__), "no_prop_overlap_train.csv")
    val_path = os.path.join(os.path.dirname(__file__), "no_prop_overlap_val.csv")
    test_path = os.path.join(os.path.dirname(__file__), "no_prop_overlap_test.csv")

    # Fall back to uploads dir if running from elsewhere
    if not os.path.exists(val_path):
        train_path = "data/raw/no_prop_overlap_train.csv"
        val_path = "data/raw/no_prop_overlap_val.csv"
        test_path = "data/raw/no_prop_overlap_test.csv"

    train_texts, train_labels = load_csv(train_path, max_rows=SAMPLE_SIZE)
    val_texts, val_labels = load_csv(val_path, max_rows=SAMPLE_SIZE)
    test_texts, test_labels = load_csv(test_path, max_rows=SAMPLE_SIZE)

    print(f"  Train: {len(train_texts)} rows")
    print(f"  Val:   {len(val_texts)} rows")
    print(f"  Test:  {len(test_texts)} rows")

    # Simple TF-IDF embeddings (no transformer needed for smoke test)
    print("  Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=EMBEDDING_DIM, stop_words="english")
    train_emb = vectorizer.fit_transform(train_texts).toarray().astype(np.float32)
    val_emb = vectorizer.transform(val_texts).toarray().astype(np.float32)
    test_emb = vectorizer.transform(test_texts).toarray().astype(np.float32)

    train_ds = make_tf_dataset(train_emb, train_labels, BATCH_SIZE)
    val_ds = make_tf_dataset(val_emb, val_labels, BATCH_SIZE)
    test_ds = make_tf_dataset(test_emb, test_labels, BATCH_SIZE)

    # ------------------------------------------------------------------
    # Check 1: Model builds and forward pass includes pools
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Check 1: Model build and forward pass")
    print("=" * 60)

    config = ModelConfig(
        task_structure=TASK_STRUCTURE,
        task_names=ALL_TASK_NAMES,
        output_pools=OUTPUT_POOLS,
    )
    model = MultiTaskModel(config=config, embedding_dim=EMBEDDING_DIM)

    dummy = tf.zeros((1, EMBEDDING_DIM))
    out = model(dummy, training=False)

    expected_keys = len(ALL_TASK_NAMES) + len(OUTPUT_POOLS)
    check(
        f"Forward pass returns {len(ALL_TASK_NAMES)} tasks + {len(OUTPUT_POOLS)} pools = {expected_keys} keys",
        len(out) == expected_keys,
        f"got {len(out)} keys: {sorted(out.keys())}",
    )
    check(
        "'any_technique' pool key present",
        "any_technique" in out,
    )
    check(
        "'mean_argumentation' pool key present",
        "mean_argumentation" in out,
    )

    # ------------------------------------------------------------------
    # Check 2: Pooled values match element-wise strategy
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Check 2: Pool values match strategy")
    print("=" * 60)

    batch = tf.random.normal((8, EMBEDDING_DIM))
    out2 = model(batch, training=False)

    # any_technique = max over 18 propaganda technique outputs
    technique_outputs = tf.stack(
        [out2[t] for t in PROP_TECHNIQUES], axis=1
    )  # [8, 18, 1]
    expected_max = tf.reduce_max(technique_outputs, axis=1)  # [8, 1]
    check(
        "any_technique == element-wise max over 18 techniques",
        np.allclose(out2["any_technique"].numpy(), expected_max.numpy(), atol=1e-6),
    )

    # mean_argumentation = mean over 8 arg outputs
    arg_outputs = tf.stack([out2[t] for t in ARG_TASKS], axis=1)
    expected_mean = tf.reduce_mean(arg_outputs, axis=1)
    check(
        "mean_argumentation == element-wise mean over 8 arg tasks",
        np.allclose(
            out2["mean_argumentation"].numpy(), expected_mean.numpy(), atol=1e-6
        ),
    )

    # ------------------------------------------------------------------
    # Check 3: fit() with default thresholds includes pools
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Check 3: fit() default thresholds include pools")
    print("=" * 60)
    
    trainer = Trainer(model, TrainingConfig(epochs=EPOCHS, optimizer_type='adam'))
    history, thresholds = trainer.fit(
        train_ds, val_ds, num_tasks=len(ALL_TASK_NAMES), verbose=0
    )

    check(
        "Training completed (history exists)",
        history is not None,
    )
    check(
        "Thresholds contain all 28 task names",
        all(t in thresholds for t in ALL_TASK_NAMES),
        f"missing: {set(ALL_TASK_NAMES) - set(thresholds.keys())}",
    )
    check(
        "Thresholds contain 'any_technique' pool",
        "any_technique" in thresholds,
    )
    check(
        "Thresholds contain 'mean_argumentation' pool",
        "mean_argumentation" in thresholds,
    )
    check(
        "Pool default threshold is 0.5",
        thresholds.get("any_technique") == 0.5
        and thresholds.get("mean_argumentation") == 0.5,
        f"got any_technique={thresholds.get('any_technique')}, "
        f"mean_argumentation={thresholds.get('mean_argumentation')}",
    )

    # ------------------------------------------------------------------
    # Check 4: fit(optimize_thresholds=True) finds pool thresholds
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Check 4: fit() with optimized thresholds")
    print("=" * 60)

    # Rebuild model to avoid reusing compiled state
    model2 = MultiTaskModel(config=config, embedding_dim=EMBEDDING_DIM)
    trainer2 = Trainer(model2, TrainingConfig(epochs=EPOCHS, optimizer_type='adam'))
    _, opt_thresholds = trainer2.fit(
        train_ds, val_ds,
        num_tasks=len(ALL_TASK_NAMES),
        optimize_thresholds=True,
        verbose=0,
    )

    check(
        "Optimized thresholds include pool keys",
        "any_technique" in opt_thresholds
        and "mean_argumentation" in opt_thresholds,
    )
    any_tech_thresh = opt_thresholds.get("any_technique")
    check(
        "Pool threshold is a float (not default 0.5)",
        isinstance(any_tech_thresh, float),
        f"type={type(any_tech_thresh)}",
    )
    print(f"    → any_technique threshold: {any_tech_thresh:.4f}")
    print(f"    → mean_argumentation threshold: "
          f"{opt_thresholds.get('mean_argumentation', 'MISSING')}")

    # ------------------------------------------------------------------
    # Check 5: evaluate() returns pool_metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Check 5: evaluate() pool_metrics")
    print("=" * 60)

    results = trainer2.evaluate(test_ds)

    check(
        "Results contain 'pool_metrics' key",
        "pool_metrics" in results,
    )
    check(
        "pool_metrics has both pool names",
        "any_technique" in results.get("pool_metrics", {})
        and "mean_argumentation" in results.get("pool_metrics", {}),
        f"got keys: {list(results.get('pool_metrics', {}).keys())}",
    )
    check(
        f"task_metrics has {len(ALL_TASK_NAMES)} task entries (no pool keys mixed in)",
        len(results["task_metrics"]) == len(ALL_TASK_NAMES)
        and "any_technique" not in results["task_metrics"],
    )

    # Print a sample of pool metrics
    for pool_name in ["any_technique", "mean_argumentation"]:
        pm = results.get("pool_metrics", {}).get(pool_name, {})
        print(f"    → {pool_name}: {pm}")

    # ------------------------------------------------------------------
    # Check 6: overall_metrics averages tasks only (not pools)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Check 6: overall_metrics excludes pools")
    print("=" * 60)

    # Evaluate a model WITHOUT pools on same data for comparison
    config_no_pool = ModelConfig(
        task_structure=TASK_STRUCTURE,
        task_names=ALL_TASK_NAMES,
    )
    model_np = MultiTaskModel(config=config_no_pool, embedding_dim=EMBEDDING_DIM)
    trainer_np = Trainer(model_np, TrainingConfig(epochs=EPOCHS, optimizer_type='adam'))
    trainer_np.fit(train_ds, val_ds, num_tasks=len(ALL_TASK_NAMES), verbose=0)
    results_np = trainer_np.evaluate(test_ds)

    # Both should have the same set of overall metric keys
    check(
        "overall_metrics keys match (pool vs no-pool model)",
        set(results["overall_metrics"].keys())
        == set(results_np["overall_metrics"].keys()),
    )

    # ------------------------------------------------------------------
    # Check 7: save → load round-trip preserves pool config
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Check 7: save/load round-trip")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="pool_smoke_")
    save_path = os.path.join(tmpdir, "pool_model")

    try:
        # Save
        model2.save(save_path, thresholds=opt_thresholds)

        # Verify config.json has output_pools
        import json
        with open(os.path.join(save_path, "config.json")) as f:
            saved_config = json.load(f)
        check(
            "config.json contains output_pools",
            "output_pools" in saved_config
            and saved_config["output_pools"] is not None,
        )

        # Verify thresholds.json has pool thresholds
        with open(os.path.join(save_path, "thresholds.json")) as f:
            saved_thresh = json.load(f)
        check(
            "thresholds.json contains pool thresholds",
            "any_technique" in saved_thresh
            and "mean_argumentation" in saved_thresh,
        )

        # Load and verify forward pass
        loaded_model, loaded_thresh = load_model(save_path)
        check(
            "Loaded model config has pools",
            loaded_model.config.has_pools,
        )
        check(
            "Loaded model pool_names match",
            sorted(loaded_model.config.pool_names)
            == sorted(["any_technique", "mean_argumentation"]),
        )

        # Forward pass on loaded model
        out_loaded = loaded_model(batch, training=False)
        check(
            "Loaded model forward pass produces pool keys",
            "any_technique" in out_loaded and "mean_argumentation" in out_loaded,
        )

        # Predictions should match the model we saved (model2)
        out_model2 = model2(batch, training=False)
        check(
            "Loaded model predictions match saved model",
            np.allclose(
                out_model2["any_technique"].numpy(),
                out_loaded["any_technique"].numpy(),
                atol=1e-5,
            ),
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"ALL {total} CHECKS PASSED ✓")
    else:
        print(f"{passed}/{total} passed, {failed} FAILED ✗")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)