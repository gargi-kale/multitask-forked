import numpy as np
import tensorflow as tf
from multitask.config import ModelConfig, TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer

# --- Test 1: Model builds with mixed types ---
config = ModelConfig(
    task_structure=[[3]],
    task_names=["sentiment", "toxicity", "intensity"],
    task_types=["binary", "binary", "regression"],
)
model = MultiTaskModel(config=config, embedding_dim=32)
trainer = Trainer(model, TrainingConfig(epochs=2))

# --- Test 2: Forward pass produces correct shapes ---
dummy = tf.random.normal((4, 32))
out = model(dummy, training=False)
assert out["sentiment"].shape == (4, 1), f"Expected (4,1), got {out['sentiment'].shape}"
assert out["intensity"].shape == (4, 1), f"Expected (4,1), got {out['intensity'].shape}"
# Regression output should NOT be bounded to [0,1]
# (sigmoid would clip; linear can go negative or >1)
print(f"  intensity range: [{out['intensity'].numpy().min():.3f}, {out['intensity'].numpy().max():.3f}]")

# --- Test 3: Training works with mixed types ---
rng = np.random.RandomState(42)
x = rng.randn(64, 32).astype(np.float32)
labels = {
    "sentiment": rng.randint(0, 2, size=(64,)).astype(np.float32),
    "toxicity": rng.randint(0, 2, size=(64,)).astype(np.float32),
    "intensity": rng.randn(64).astype(np.float32) * 5,  # continuous, unbounded
}
ds = tf.data.Dataset.from_tensor_slices((x, labels)).batch(16)

history, thresholds = trainer.fit(ds, ds, num_tasks=3, verbose=0)
assert history is not None, "Training failed"
print(f"  Final loss: {history.history['loss'][-1]:.4f}")

# --- Test 4: Thresholds correct ---
assert thresholds["sentiment"] == 0.5, f"Expected 0.5, got {thresholds['sentiment']}"
assert thresholds["toxicity"] == 0.5, f"Expected 0.5, got {thresholds['toxicity']}"
assert thresholds["intensity"] is None, f"Expected None, got {thresholds['intensity']}"

# --- Test 5: Optimized thresholds skip regression ---
model2 = MultiTaskModel(config=config, embedding_dim=32)
trainer2 = Trainer(model2, TrainingConfig(epochs=2))
_, opt_thresh = trainer2.fit(ds, ds, num_tasks=3, optimize_thresholds=True, verbose=0)
assert opt_thresh["intensity"] is None, f"Regression threshold should be None, got {opt_thresh['intensity']}"
assert isinstance(opt_thresh["sentiment"], float), "Binary should have float threshold"

# --- Test 6: Class weights skip regression ---
cw = trainer2._compute_class_weights(ds)
assert cw[0] == {0: 1.0}, f"Regression class weights should be {{0: 1.0}}, got {cw[0]}"

# --- Test 7: mask_value is configurable ---
config_custom = ModelConfig(
    task_structure=[[2]],
    task_names=["a", "b"],
    task_types=["binary", "regression"],
    mask_value=-999.0,
)
assert config_custom.mask_value == -999.0
model_custom = MultiTaskModel(config=config_custom, embedding_dim=32)
trainer_custom = Trainer(model_custom, TrainingConfig(epochs=1))
assert trainer_custom.mask_value == -999.0

print("\nAll Day 2 checks passed!")