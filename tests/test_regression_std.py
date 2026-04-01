import numpy as np
import os
import tensorflow as tf
from multitask.config import ModelConfig, TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer

# Test 1: stats computed correctly
config = ModelConfig(
    task_structure=[[2]],
    task_names=["binary_task", "reg_task"],
    task_types=["binary", "regression"],
)
model = MultiTaskModel(config=config, embedding_dim=32)
trainer = Trainer(model, TrainingConfig(epochs=1))

rng = np.random.RandomState(42)
x = rng.randn(64, 32).astype(np.float32)
labels = {
    "binary_task": rng.randint(0, 2, size=(64,)).astype(np.float32),
    "reg_task": (rng.randn(64) * 100 + 500).astype(np.float32),  # mean~500, std~100
}
ds = tf.data.Dataset.from_tensor_slices((x, labels)).batch(16)

stats = trainer._compute_regression_stats(ds)
assert "reg_task" in stats
assert abs(stats["reg_task"]["mean"] - 500) < 50, f"mean={stats['reg_task']['mean']}"
assert abs(stats["reg_task"]["std"] - 100) < 50, f"std={stats['reg_task']['std']}"
print(f"Test 1 passed: stats mean={stats['reg_task']['mean']:.1f}, std={stats['reg_task']['std']:.1f}")

# Test 2: training works
history, thresholds = trainer.fit(ds, ds, num_tasks=2, verbose=0)
assert hasattr(model, '_regression_stats')
assert "reg_task" in model._regression_stats
print("Test 2 passed: training with standardization works")

# Test 3: evaluate returns original-scale metrics
results = trainer.evaluate(ds)
mse = results["task_metrics"]["reg_task"]["mse"]
# MSE should be in original scale (large), not standardized scale (~1)
assert mse > 10, f"MSE looks standardized, not original scale: {mse}"
print(f"Test 3 passed: MSE in original scale = {mse:.1f}")

# Test 4: standardization disabled
config_off = ModelConfig(
    task_structure=[[2]],
    task_names=["binary_task", "reg_task"],
    task_types=["binary", "regression"],
    standardize_regression=False,
)
model_off = MultiTaskModel(config=config_off, embedding_dim=32)
trainer_off = Trainer(model_off, TrainingConfig(epochs=1))
stats_off = trainer_off._compute_regression_stats(ds)
assert stats_off == {}, f"Expected empty stats, got {stats_off}"
print("Test 4 passed: standardization disabled returns empty stats")

# Test 5: save/load preserves regression_stats
import tempfile, shutil, json
tmpdir = tempfile.mkdtemp()
save_path = tmpdir + "/model"
try:
    model.save(save_path, thresholds=thresholds)
    
    # Check regression_stats.json exists
    assert os.path.exists(save_path + "/regression_stats.json")
    with open(save_path + "/regression_stats.json") as f:
        saved_stats = json.load(f)
    assert "reg_task" in saved_stats
    print(f"Test 5a passed: regression_stats.json saved")
    
    # Load and verify
    from multitask.models.persistence import load_model
    loaded_model, _ = load_model(save_path)
    assert hasattr(loaded_model, '_regression_stats')
    assert "reg_task" in loaded_model._regression_stats
    assert abs(loaded_model._regression_stats["reg_task"]["mean"] - stats["reg_task"]["mean"]) < 0.01
    print("Test 5b passed: regression_stats loaded correctly")
finally:
    shutil.rmtree(tmpdir)

print("\nAll standardization tests passed!")