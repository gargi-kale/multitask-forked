# Distributed Training

This guide covers distributed training with the multitask library, including cross-platform support for macOS (Apple Silicon/Intel), Linux, and Windows.

## Quick Start

```python
from multitask import (
    ModelConfig, TrainingConfig, MultiTaskModel, Trainer,
    get_distribution_strategy, setup_gpu_memory_growth
)

# Setup: enable memory growth and get optimal strategy
setup_gpu_memory_growth()
strategy, info = get_distribution_strategy('auto', verbose=True)

# Create model and trainer within strategy scope
with strategy.scope():
    model = MultiTaskModel(model_config)
    trainer = Trainer(model, training_config, strategy=strategy)

# Train - distribution handled automatically
history, thresholds = trainer.fit(train_data, val_data)
```

## Platform Compatibility

| Platform | Multi-GPU | Single GPU | CPU | Notes |
|----------|-----------|------------|-----|-------|
| **Linux/Windows with NVIDIA** | ✅ | ✅ | ✅ | Best for multi-GPU training |
| **Apple Silicon (M1/M2/M3)** | ⚠️ N/A | ✅ Metal | ✅ | Single GPU only |
| **Intel Mac** | ⚠️ N/A | ⚠️ Limited | ✅ | Most lack discrete GPU |

**Check your setup:**
```python
from multitask import check_distributed_compatibility, print_device_info

compat = check_distributed_compatibility()
print(f"{compat['platform']}: {compat['num_gpus']} GPUs - {compat['recommendation']}")

print_device_info()  # Detailed hardware info
```

## Strategy Selection

**Auto (Recommended)** - Automatically selects best strategy:
```python
strategy, info = get_distribution_strategy('auto', verbose=True)
# Apple Silicon → Single GPU (Metal)
# 2+ NVIDIA GPUs → MirroredStrategy
# 1 GPU → Single GPU
# No GPUs → CPU
```

**Manual Options:**
```python
# Force multi-GPU (falls back if <2 GPUs available)
strategy, info = get_distribution_strategy('mirrored')

# Single device (GPU if available, else CPU)
strategy, info = get_distribution_strategy('one_device')

# CPU only
strategy, info = get_distribution_strategy('cpu')
```

## Usage Example

```python
from multitask import (
    ModelConfig, TrainingConfig, MultiTaskModel, Trainer,
    get_distribution_strategy, setup_gpu_memory_growth
)
import pandas as pd

# Setup
setup_gpu_memory_growth()
strategy, info = get_distribution_strategy('auto', verbose=True)

# Load data
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')

# Create model within strategy scope
with strategy.scope():
    model_config = ModelConfig(
        tasks={
            'sentiment': {'type': 'multiclass', 'num_classes': 3},
            'toxicity': {'type': 'binary', 'num_classes': 1},
        },
        encoder_type='tfhub',
        encoder_name='https://tfhub.dev/google/universal-sentence-encoder/4',
    )
    
    training_config = TrainingConfig(batch_size=32, epochs=10)
    
    model = MultiTaskModel(model_config)
    trainer = Trainer(model, training_config, strategy=strategy)

# Train
history, thresholds = trainer.fit(train_data, val_data)

# Save
model.save('models/my_model', thresholds=thresholds)

# Load (also in strategy scope)
with strategy.scope():
    from multitask import load_model
    loaded_model, loaded_thresholds = load_model('models/my_model')
```

## Platform-Specific Notes

### macOS (Apple Silicon - M1/M2/M3)

**Setup:**
```bash
# Install TensorFlow with Metal support
pip install tensorflow-macos tensorflow-metal
```

**Characteristics:**
- Uses Metal backend instead of CUDA
- Consumer Macs have unified memory architecture (single GPU)
- Metal backend can be buggy with some TensorFlow operations
- MirroredStrategy not applicable (single GPU)

**Recommended Strategy:**
```python
# Auto mode will correctly select single GPU on Apple Silicon
strategy, info = get_distribution_strategy('auto', verbose=True)
# Output: "Using Metal GPU on Apple Silicon (single device)"
```

**Common Issues:**
- **Metal ops not implemented**: Some TensorFlow operations may not be supported by Metal
  - Solution: Fall back to CPU with `strategy_type='cpu'`
- **Memory pressure**: Unified memory shared with system
  - Solution: Use smaller batch sizes or models

### macOS (Intel)

Most Intel Macs don't have discrete GPUs. The library will automatically use CPU training.

```python
strategy, info = get_distribution_strategy('auto')
# Output: "Using CPU-only training on Darwin (no GPUs detected)"
```

### Linux/Windows with NVIDIA GPUs

**Setup:**
```bash
# Ensure CUDA and cuDNN are installed
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]
```

**Multi-GPU Training:Setup

**Apple Silicon (M1/M2/M3):**
```bash
pip install tensorflow-macos tensorflow-metal
```
- Uses Metal backend (not CUDA)
- Single GPU only (unified memory architecture)
- Some TF ops may not be supported - fall back to CPU if errors occur

**Linux/Windows with NVIDIA:**
```bash
pip install tensorflow[and-cuda]
```
- Requires CUDA and cuDNN installed
- Supports multi-GPU with MirroredStrategy
- Scale batch size with GPU count: `batch_size = 32 * num_gpus
2. Update TensorFlow and tensorflow-metal:
   ```bash
   pip install --upgrade tensorflow-macos tensorflow-metal
   ```

3. Check TensorFlow Metal compatibility:
   https://developer.apple.com/metal/tensorflow-plugin/

### "Out of memory" errors

**Reduce memory usage:**
1. Enable memory growth:
**No GPUs detected:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # Check if TF sees GPUs
```
- Verify GPU drivers and CUDA/cuDNN versions
- On Apple Silicon: install `tensorflow-metal`

**Out of memory:**
```python
setup_gpu_memory_growth()  # Enable incremental allocation
config = TrainingConfig(batch_size=16)  # Reduce batch size
```

**Metal backend errors (Apple Silicon):**
```python
strategy = get_distribution_strategy('cpu')  # Fall back to CPU
```
Update TensorFlow: `pip install --upgrade tensorflow-macos tensorflow-metal`

**Multi-GPU slower than single GPU:**
- Batch size should be ≥ `num_gpus * 32`
- Model may be too small for parallelization overhead
- Test single GPU first with `get_distribution_strategy('one_device')`

Verify your setup with the example script:
```bash
python examples/distributed_training.py
```

This tests platform detection, strategy selection, training, and save/load functionality.

## Best Practices

1. Create models inside `strategy.scope()`
2. Call `setup_gpu_memory_growth()` before creating models
3. Use `'auto'` mode for automatic platform detection
4. Scale batch size with GPU count for multi-GPU
5. Test on CPU first before scaling to multi-GPU