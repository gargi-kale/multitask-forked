# Model Persistence Guide

Save and load multi-task models with optimized thresholds and training state.

## Quick Start

### Save After Training

```python
from multitask import Trainer, TrainingConfig, ModelConfig, MultiTaskModel

# Train model
config = ModelConfig(...)
model = MultiTaskModel(config, encoder='bert-base-uncased')
trainer = Trainer(model, TrainingConfig())

history, thresholds = trainer.fit(
    train_ds, val_ds, 
    num_tasks=10,
    optimize_thresholds=True
)

# Save with thresholds (includes everything)
trainer.save_checkpoint('my_model', thresholds)

# Or just save the model with thresholds
model.save('my_model', thresholds=thresholds)
```

### Load for Inference

```python
from multitask import load_model

# Load with thresholds
model, thresholds = load_model('my_model')

# Use for inference
predictions = model.predict(test_data)
```

### Load for Training Resumption

```python
# Load with optimizer state
trainer, thresholds = Trainer.load_checkpoint(
    'my_model',
    load_optimizer=True,
    compile_model=True
)

# Continue training seamlessly
history, new_thresholds = trainer.fit(train_ds, val_ds, num_tasks=10)
```

---

## What Gets Saved?

| Component | model.save() | save_checkpoint() |
|-----------|--------------|------------------|
| Weights | ✅ | ✅ |
| Config | ✅ | ✅ |
| Encoder info | ✅ | ✅ |
| **Thresholds** | ✅ | ✅ |
| Training config | ❌ | ✅ |
| Optimizer state | ❌ | ✅ (optional) |

**Key design**: Thresholds are now always saved, ensuring models are deployment-ready immediately.

## Table of Contents

1. [Usage Patterns](#usage-patterns)
2. [API Reference](#api-reference)
3. [Optimizer State](#optimizer-state)
4. [Automatic Checkpointing](#automatic-checkpointing)
5. [Troubleshooting](#troubleshooting)
6. [File Structure](#file-structure-summary)

---

## Usage Patterns

### Pattern 1: Simple Model Save (Deployment)

```python
# After training
history, thresholds = trainer.fit(train_ds, val_ds, optimize_thresholds=True)

# Save just the model
model.save('deploy/model', thresholds=thresholds)

# Later, load for inference
model, thresholds = load_model('deploy/model')
results = trainer.evaluate(test_ds, thresholds=thresholds)
```

**File structure**: Small, contains only what's needed for inference.

### Pattern 2: Complete Checkpoint (Training Resumption)

```python
# During long experiments, use automatic checkpointing
history, thresholds = trainer.fit(
    train_ds, val_ds, num_tasks=10,
    checkpoint_dir='checkpoints/experiment_1',  # Auto-saves best model
    optimize_thresholds=True
)

# Or save manually with optimizer
trainer.save_checkpoint('checkpoints/v1', thresholds, save_optimizer=True)

# Later, resume training from exact point
trainer, thresholds = Trainer.load_checkpoint(
    'checkpoints/v1',
    load_optimizer=True,
    compile_model=True
)
new_history, _ = trainer.fit(train_ds, val_ds, num_tasks=10)
```

**File structure**: Larger (includes optimizer), enables seamless training resumption.

### Pattern 3: Deployment Without Optimizer

```python
# Save for deployment (smaller package)
trainer.save_checkpoint('deploy/v1', thresholds, save_optimizer=False)

# Load for inference
trainer, thresholds = Trainer.load_checkpoint('deploy/v1')
```

**File structure**: Smaller than full checkpoint (no optimizer state).

---

## API Reference

### model.save()

```python
model.save(
    filepath: str,
    save_format: str = 'tf',
    thresholds: Optional[dict] = None
) -> None
```

Saves model to directory with optional thresholds.

**Args**:
- `filepath`: Directory path for saving
- `save_format`: 'tf' (default) or 'h5'
- `thresholds`: Optional dict of task_name -> threshold values

**Files created**:
```
filepath/
├── config.json       # ModelConfig
├── metadata.json     # Encoder info, dropout rate
├── weights/          # Model weights
└── thresholds.json   # Optimized thresholds (if provided)
```

### load_model()

```python
model, thresholds = load_model(
    filepath: str,
    encoder: Optional[Union[str, Layer]] = None,
    custom_objects: Optional[dict] = None
) -> Tuple[MultiTaskModel, Optional[Dict]]
```

Loads model and thresholds from directory.

**Returns**: Tuple of (model, thresholds_dict or None)

### trainer.save_checkpoint()

```python
trainer.save_checkpoint(
    filepath: str,
    thresholds: Optional[dict] = None,
    save_format: str = 'tf',
    save_optimizer: bool = True
) -> None
```

Saves complete training checkpoint (includes optimizer state).

**Args**:
- `thresholds`: Optional thresholds. If None, uses self.thresholds
- `save_optimizer`: Set to False for deployment-only (saves space)

**Files created** (in addition to model.save()):
```
filepath/
├── optimizer.pkl       # Optimizer state (if save_optimizer=True)
└── training_config.json
```

### Trainer.load_checkpoint()

```python
trainer, thresholds = Trainer.load_checkpoint(
    filepath: str,
    encoder: Optional[Union[str, Layer]] = None,
    custom_objects: Optional[dict] = None,
    load_optimizer: bool = True,
    compile_model: bool = False
) -> Tuple[Trainer, Optional[Dict]]
```

Loads checkpoint and returns ready-to-use Trainer.

**Args**:
- `load_optimizer`: Whether to restore optimizer state
- `compile_model`: Required if load_optimizer=True

---

## Optimizer State

### What is It?

When training with Adam or SGD+momentum, the optimizer maintains state:
- **Adam**: Running averages of gradients and squared gradients
- **SGD+Momentum**: Velocity vector for each weight
- **Learning rate schedule**: Current position in schedule

### When to Save

| Use Case | save_optimizer |
|----------|----------------|
| Long training runs | ✅ True (resume seamlessly) |
| Experiments | ✅ True (checkpoint progress) |
| Production deployment | ❌ False (smaller ~2x) |
| Transfer learning | ❌ False (will retrain) |

### Size Impact

Optimizer adds roughly **2x the model size** for Adam:
- Model: 110 MB
- With optimizer: ~330 MB

For deployment:
```python
trainer.save_checkpoint('deploy/model', thresholds, save_optimizer=False)
# Size: 110 MB instead of 330 MB
```

---

## Automatic Checkpointing

Save the best model during training automatically:

```python
history, thresholds = trainer.fit(
    train_ds, val_ds, num_tasks=10,
    checkpoint_dir='checkpoints/experiment',
    checkpoint_monitor='val_loss',     # Metric to monitor
    checkpoint_mode='min',              # 'min' or 'max'
    optimize_thresholds=True
)
```

**How it works**:
1. After each epoch, monitors the specified metric
2. If metric improves, saves checkpoint automatically
3. Only saves when performance improves (no redundant saves)
4. Includes optimizer state by default

---

## Troubleshooting

### "Thresholds not loaded"

Ensure you save thresholds when calling `model.save()`:

```python
# ✅ Correct
model.save('my_model', thresholds=thresholds)

# ❌ Wrong - thresholds lost
model.save('my_model')
```

### "Cannot load optimizer state"

Optimizer state is optional and only needed for training resumption:

```python
# For inference, optimizer isn't needed
trainer, thresholds = Trainer.load_checkpoint('model')  # Works fine

# For resumed training, must compile first
trainer, thresholds = Trainer.load_checkpoint(
    'model',
    load_optimizer=True,
    compile_model=True  # Required
)
```

### "Model was saved with custom encoder"

Custom Keras layers can't be auto-loaded. Provide the encoder:

```python
custom_encoder = MyCustomLayer()
model, thresholds = load_model('my_model', encoder=custom_encoder)
```

### "Different TensorFlow/HuggingFace versions"

Versions mismatch can cause subtle issues. Pin versions in requirements.txt:

```
tensorflow==2.15.0
transformers==4.30.0
```

---

## File Structure Summary

### Model Save

```
my_model/
├── config.json           # Task structure, types, classes
├── metadata.json         # Encoder type/ID, embedding dim
├── thresholds.json       # Optimized thresholds (if provided)
└── weights/              # Model weights (TF format)
    ├── variables/
    └── saved_model.pb
```

### Checkpoint Save

```
my_model/
├── config.json
├── metadata.json
├── thresholds.json
├── training_config.json  # Learning rate, epochs, etc.
├── optimizer.pkl         # Optimizer state (if save_optimizer=True)
└── weights/
```

---

## Summary

**Use model.save()** for simple deployments where you just need weights and thresholds:
```python
model.save('my_model', thresholds=thresholds)
model, thresholds = load_model('my_model')
```

**Use trainer.save_checkpoint()** for:
- Long training runs (preserve optimizer progress)
- Training resumption after interruptions
- Production deployments with full reproducibility
- Automatic checkpointing during training

```python
trainer.save_checkpoint('checkpoints/model', thresholds, save_optimizer=True)
trainer, thresholds = Trainer.load_checkpoint('checkpoints/model')
```

Both approaches save thresholds automatically, ensuring models are deployment-ready.
