# Model Persistence Guide

Save and load multi-task models with optimized thresholds and training state.

## Quick Start

### Save After Training

```python
from multitask import Trainer, TrainingConfig, ModelConfig, MultiTaskModel, EncoderConfig

# Train model
encoder_config = EncoderConfig(
    encoder_identifier="bert-base-uncased",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE
)
config = ModelConfig(
    ...
    encoder_config=encoder_config
)
model = MultiTaskModel(config)
trainer = Trainer(model, TrainingConfig())

history, thresholds = trainer.fit(
    train_ds, val_ds, 
    num_tasks=10,
    optimize_thresholds=True
)

# Save with thresholds (recommended for deployment)
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
| **Thresholds** | ✅ (if provided) | ✅ (if provided or available on trainer) |
| Training config | ❌ | ✅ |
| Optimizer state | ❌ | ✅ (optional) |

**Key design**: Thresholds are saved when provided (or when available on the trainer during checkpoint save), so pass thresholds explicitly when packaging for deployment.

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
    thresholds: Optional[Dict[str, Optional[float]]] = None
) -> None
```

Saves model to directory with optional thresholds.

**Args**:
- `filepath`: Directory path to save model artifacts
- `save_format`: Format to save weights ('tf' or 'h5')
- `thresholds`: Optional dict of task_name -> threshold values for classification.
                Essential for binary/multilabel tasks. If None, thresholds must be
                provided during inference via evaluate() or predict().

**Files created**:
```
filepath/
├── config.json       # ModelConfig
├── metadata.json     # Encoder info, dropout rate
├── weights.weights.h5  # Model weights (Keras 3 format)
└── thresholds.json     # Optimized thresholds (if provided)
```

### load_model()

```python
model, thresholds = load_model(
    filepath: str,
    encoder: Optional[Union[str, Layer]] = None
) -> Tuple[MultiTaskModel, Optional[Dict]]
```

Loads model and thresholds from directory.

**Returns**: Tuple of (model, thresholds_dict or None)

### trainer.save_checkpoint()

```python
trainer.save_checkpoint(
    filepath: str,
    thresholds: Optional[Dict[str, Optional[float]]] = None,
    save_format: str = 'tf',
    save_optimizer: bool = True
) -> None
```

Saves complete training checkpoint (includes optimizer state).

**Args**:
- `filepath`: Directory path to save checkpoint
- `thresholds`: Optional dict of task_name -> threshold values. If None, uses trainer-stored thresholds when available (prefers `self.optimal_thresholds`, then `self.thresholds`).
- `save_format`: Save-format argument accepted by API (current implementation writes Keras 3 weights to `weights.weights.h5`)
- `save_optimizer`: Whether to save optimizer state for training resumption. Use False for deployment-only to save space (~2x reduction).

**Files created** (in addition to `model.save()`):
```
filepath/
├── optimizer.pkl       # Optimizer state (if save_optimizer=True)
└── training_config.json
```

### Trainer.load_checkpoint()

```python
trainer, thresholds = Trainer.load_checkpoint(
    filepath: str,
    encoder: Optional[Union[str, tf.keras.layers.Layer]] = None,
    custom_objects: Optional[dict] = None,
    load_optimizer: bool = True,
    compile_model: bool = False
) -> Tuple['Trainer', Optional[Dict[str, Optional[float]]]]
```

Loads checkpoint and returns ready-to-use Trainer.

**Args**:
- `filepath`: Directory path containing saved checkpoint
- `encoder`: Optional encoder override (see load_model documentation)
- `custom_objects`: Present in the API signature, but currently not used by `load_model()`
- `load_optimizer`: Whether to restore optimizer state for training resumption. Only works if model is compiled first.
- `compile_model`: Whether to compile the model after loading. Required if load_optimizer=True. Uses saved training config.

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

You can also pass thresholds explicitly to checkpoint save:

```python
trainer.save_checkpoint('my_model', thresholds=thresholds)
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
└── weights.weights.h5    # Model weights (Keras 3 format)
```

### Checkpoint Save

```
my_model/
├── config.json
├── metadata.json
├── thresholds.json
├── training_config.json  # Learning rate, epochs, etc.
├── optimizer.pkl         # Optimizer state (if save_optimizer=True)
└── weights.weights.h5
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

Both approaches support threshold persistence when thresholds are available; pass thresholds explicitly to guarantee deployment-ready behavior.
