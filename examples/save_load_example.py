"""Quick test of model save/load functionality."""

import tensorflow as tf
import numpy as np
from multitask import ModelConfig, MultiTaskModel, load_model, Trainer, TrainingConfig

# Create a simple model configuration
config = ModelConfig(
    task_structure=[2, 1],
    task_types=['binary', 'binary', 'multiclass'],
    num_classes_per_task=[1, 1, 3],
    task_names=['task_0', 'task_1', 'task_2']
)

# Build model with embeddings (no encoder)
print("Creating model...")
model = MultiTaskModel(
    config=config,
    encoder=None,
    embedding_dim=128,
    dropout_rate=0.3
)

# Create dummy dataset
def create_dummy_dataset(num_samples=100):
    def generator():
        for _ in range(num_samples):
            # Embeddings input
            x = np.random.randn(128).astype(np.float32)
            
            # Labels
            labels = {
                'task_0': np.random.randint(0, 2, size=()).astype(np.int32),
                'task_1': np.random.randint(0, 2, size=()).astype(np.int32),
                'task_2': np.random.randint(0, 3, size=()).astype(np.int32),
            }
            yield x, labels
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(128,), dtype=tf.float32),
            {
                'task_0': tf.TensorSpec(shape=(), dtype=tf.int32),
                'task_1': tf.TensorSpec(shape=(), dtype=tf.int32),
                'task_2': tf.TensorSpec(shape=(), dtype=tf.int32),
            }
        )
    )
    return dataset.batch(8)

# Create datasets
train_ds = create_dummy_dataset(80)
val_ds = create_dummy_dataset(20)

# Build model by running it once
print("Building model...")
dummy_input = tf.zeros((1, 128))
_ = model(dummy_input, training=False)

print(f"Model has {model.count_params():,} parameters\n")

# Train for a few epochs
print("Training model...")
trainer = Trainer(model, TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-3))
history, thresholds = trainer.fit(
    train_ds,
    val_ds,
    num_tasks=3,
    steps_per_epoch=10,
    validation_steps=3,
    optimize_thresholds=True
)

print(f"\nOptimized thresholds: {thresholds}")

# Save model and checkpoint
print("\n" + "="*60)
print("Testing save functionality...")
print("="*60)
model.save('test_model')
trainer.save_checkpoint('test_checkpoint', thresholds)

# Test model load
print("\n" + "="*60)
print("Testing load_model()...")
print("="*60)
loaded_model = load_model('test_model')

# Verify outputs match
print("\nVerifying model outputs match...")
test_input = tf.random.normal((4, 128))
original_output = model(test_input, training=False)
loaded_output = loaded_model(test_input, training=False)

for task_name in original_output.keys():
    diff = tf.reduce_mean(tf.abs(original_output[task_name] - loaded_output[task_name]))
    print(f"  {task_name}: Mean absolute difference = {diff.numpy():.6f}")
    if diff.numpy() > 1e-5:
        print(f"    ⚠️  WARNING: Outputs differ significantly!")
    else:
        print(f"    ✓ Outputs match")

# Test checkpoint load
print("\n" + "="*60)
print("Testing Trainer.load_checkpoint()...")
print("="*60)
loaded_trainer, loaded_thresholds = Trainer.load_checkpoint('test_checkpoint')

print(f"\nLoaded thresholds: {loaded_thresholds}")
print(f"Original thresholds: {thresholds}")

# Verify thresholds match
thresholds_match = all(
    abs(loaded_thresholds[k] - thresholds[k]) < 1e-6 if thresholds[k] is not None else loaded_thresholds[k] is None
    for k in thresholds.keys()
)

if thresholds_match:
    print("✓ Thresholds match!")
else:
    print("⚠️  WARNING: Thresholds don't match!")

# Test evaluation with loaded model
print("\n" + "="*60)
print("Testing evaluation with loaded checkpoint...")
print("="*60)
results = loaded_trainer.evaluate(val_ds, thresholds=loaded_thresholds)

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
