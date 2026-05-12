# multitask

A flexible multi-task learning library for NLP supporting mixed task types (binary/multiclass/multilabel/regression), multiple encoders, and distributed training.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Encoder Types](#encoder-types)
- [Training](#training)
- [Model Persistence](#model-persistence)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Citation and License](#citation)

## Features

- Mixed task types in single model (binary, multiclass, multilabel, regression)
- Multiple encoders: TensorFlow Hub, HuggingFace, pre-computed embeddings
- Automatic class imbalance and task weighting
- Automatic threshold optimization for binary/multilabel tasks
- Cross-platform distributed training with automatic GPU detection
- Model persistence with threshold saving
- Comprehensive type hints

## Installation

### From PyPI

```bash
pip install multitask
```

This installs the core library and its dependencies:

- TensorFlow (+ TF Hub, TF Text, tf-models-official)
- NumPy
- scikit-learn
- pandas
- PyYAML

To install with all optional backends and features:

```bash
pip install 'multitask[huggingface,openai,cohere,voyageai,regression]'
```

Or install extras individually as needed:

| Extra | What it adds |
|-------|-------------|
| `huggingface` | `transformers`, `torch`, `huggingface-hub` |
| `openai` | OpenAI embeddings API client |
| `cohere` | Cohere embeddings API client |
| `voyageai` | Voyage AI embeddings API client |
| `regression` | `scipy` (Pearson r in evaluation) |

### From source

```bash
git clone https://github.com/yourusername/multitask.git
cd multitask
pip install .                  # core only
pip install '.[huggingface]'   # core + HuggingFace support
```

### GPU support

- Apple Silicon: `pip install tensorflow-macos tensorflow-metal`
- NVIDIA: `pip install tensorflow[and-cuda]`

### Developers

The Conda environment pins every dependency for reproducibility:

```bash
conda env create -f environment.yml
conda activate multitask
pip install -e .
```

This includes all optional extras, dev tools (pytest, black, mypy), and system libraries (MPI, BLAS, HDF5) that pip cannot provide.

### Package management overview

| File | Audience | Purpose |
|------|----------|---------|
| `pyproject.toml` | Users | Declares runtime deps (loose bounds) and optional extras |
| `environment.yml` | Developers | Pins exact versions for a reproducible conda environment |

## Quick Start

```python
from multitask import ModelConfig, TrainingConfig, MultiTaskModel, Trainer, EncoderConfig
from multitask.config import EncoderIntegration, EncoderInputType

# Configure universal sentence encoder
encoder_config = EncoderConfig(
    encoder_integration=EncoderIntegration.TFHUB,
    encoder_input_type=EncoderInputType.RAW_STRING,
    encoder_identifier='https://tfhub.dev/google/universal-sentence-encoder/4',
)

# Three tasks → task_structure must have three outputs (see docs for hierarchical layouts)
config = ModelConfig(
    task_structure=[[3]],
    task_names=['sentiment', 'toxicity', 'emotions'],
    task_types=['multiclass', 'binary', 'multiclass'],
    num_classes_per_task=[3, 2, 5],  # binary tasks use num_classes=2
    encoder_config=encoder_config,
)

model = MultiTaskModel(config)

# Train: build tf.data.Dataset batches of (inputs, labels_dict); see docs/DATA_FORMAT.md
train_dataset = ...  # e.g. tf.data.Dataset.from_tensor_slices(...) then .batch(...)
val_dataset = ...
trainer = Trainer(model, TrainingConfig())
history, thresholds = trainer.fit(train_dataset, val_dataset, num_tasks=3)

# Predict: pass tensors or numpy matching the model input (e.g. text or embeddings)
predictions = model.predict(x_test)  # dict[str, Tensor] per task
```

## Data Format

### Text Input

DataFrame with `text` column and one column per task:

```python
import pandas as pd

data = pd.DataFrame({
    'text': ['great product', 'terrible', ...],
    'sentiment': [2, 0, ...],              # 0-2 for 3-class
    'toxicity': [0, 1, ...],               # 0-1 for binary
    'emotions': ['joy anger', 'fear', ...],  # Space-separated for multilabel
})
```

### Pre-computed Embeddings

Add `embedding` column with numpy arrays:

```python
import numpy as np
from multitask.config import EncoderInputType

data['embedding'] = [np.random.randn(768) for _ in range(len(data))]

encoder_config = EncoderConfig(
    encoder_input_type=EncoderInputType.PRECOMPUTED,
    embedding_dim=768
)

# PRECOMPUTED path: no encoder_identifier is required
# (inputs are embedding vectors provided directly by your dataset)

config = ModelConfig(
    task_structure=[[1]],
    task_names=['sentiment'],
    task_types=['multiclass'],
    num_classes_per_task=[3],
    encoder_config=encoder_config
)
```

### Missing Labels

You can set any float as the missing labels. Here we use `-1` to mark missing labels (excluded from loss):

```python
data = pd.DataFrame({
    'text': ['text1', 'text2', 'text3'],
    'task1': [0, 1, -1],     # task1 missing for text3
    'task2': [1, -1, 0],     # task2 missing for text2
})
```

```python
# Optional: use a custom missing-label value globally
config = ModelConfig(
    task_structure=[[2]],
    task_names=['task1', 'task2'],
    task_types=['binary', 'multiclass'],
    num_classes_per_task=[2, 3],
    mask_value=-999.0,
)
```

See [DATA_FORMAT.md](docs/DATA_FORMAT.md) for complete documentation.

## Encoder Types

### TensorFlow Hub

```python
from multitask.config import EncoderIntegration, EncoderInputType

# Universal Sentence Encoder
encoder_config = EncoderConfig(
    encoder_identifier="https://tfhub.dev/google/universal-sentence-encoder/4",
    encoder_input_type=EncoderInputType.RAW_STRING,
    encoder_integration=EncoderIntegration.TFHUB
)

config = ModelConfig(
    task_structure=[[1]],
    task_names=['sentiment'],
    task_types=['multiclass'],
    num_classes_per_task=[3],
    encoder_config=encoder_config
)
```

### HuggingFace

```python
from multitask.config import EncoderIntegration, EncoderInputType

encoder_config = EncoderConfig(
    encoder_identifier="bert-base-uncased",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE
)

config = ModelConfig(
    task_structure=[[1]],
    task_names=['sentiment'],
    task_types=['multiclass'],
    num_classes_per_task=[3],
    encoder_config=encoder_config
)
```

### Pre-computed Embeddings

```python
from multitask.config import EncoderInputType

encoder_config = EncoderConfig(
    embedding_dim=768,
    encoder_input_type=EncoderInputType.PRECOMPUTED,
)

config = ModelConfig(
    task_structure=[[1]],
    encoder_config=encoder_config
)
```

See [ENCODER_SUPPORT.md](docs/ENCODER_SUPPORT.md) for all options.

## Training

### Basic

```python
config = TrainingConfig(batch_size=32, epochs=10, learning_rate=2e-5)
trainer = Trainer(model, config)
history, thresholds = trainer.fit(train_dataset, val_dataset, num_tasks=N)
```

### Automatic Weight Computation

By default, trainer automatically computes:
- **Task weights**: Inversely proportional to non-masked samples per task
- **Class weights**: Inversely proportional to class frequency, estimated from
  a sampled pass over up to 10,000 training examples

If you want full-dataset class frequencies (recommended for heavily masked or
imbalanced tasks), pass explicit `class_weights` to `trainer.fit(...)`:

Override with explicit weights:

```python
trainer.fit(
    train_data, val_data,
    class_weights=[{0: 1.0, 1: 2.0}, ...],  # Per task
    task_weights=[1.0, 2.0, ...],
)
```

### Threshold Optimization

Find optimal thresholds for binary/multilabel tasks:

```python
history, thresholds = trainer.fit(
    train_dataset, val_dataset, num_tasks=N,
    optimize_thresholds=True,  # Uses Youden's J statistic
)
```

### Distributed Training

Automatic GPU detection and strategy selection:

```python
from multitask import get_distribution_strategy, setup_gpu_memory_growth

setup_gpu_memory_growth()
strategy, info = get_distribution_strategy('auto', verbose=True)

with strategy.scope():
    model = MultiTaskModel(config)
    trainer = Trainer(model, training_config, strategy=strategy)

trainer.fit(train_dataset, val_dataset, num_tasks=N)
```

See [DISTRIBUTED_TRAINING.md](docs/DISTRIBUTED_TRAINING.md).

### Checkpointing

```python
trainer.fit(
    train_dataset, val_dataset, num_tasks=N,
    checkpoint_dir='checkpoints/exp1',
    checkpoint_monitor='val_loss',
)
```

### Verbosity

```python
trainer.fit(train_dataset, val_dataset, num_tasks=N, verbose=0)  # Silent
trainer.fit(train_dataset, val_dataset, num_tasks=N, verbose=1)  # Progress (default)
trainer.fit(train_dataset, val_dataset, num_tasks=N, verbose=2)  # One line per epoch
```

## Model Persistence

### Save

```python
model.save('models/my_model', thresholds=thresholds)
```

### Load

```python
from multitask import load_model

model, thresholds = load_model('models/my_model')
predictions = model.predict(x_test)
```

See [MODEL_PERSISTENCE.md](docs/MODEL_PERSISTENCE.md).

## Configuration

### ModelConfig

```python
ModelConfig(
    task_structure=[[3]],            # Branching layout; sets number of outputs
    task_names=['a', 'b', 'c'],
    task_types=['binary', 'multiclass', 'multilabel'],
    num_classes_per_task=[2, 5, 4],
    shared_layer_sizes=[256, 128],   # Optional dense stack before branches
    default_layer_sizes=[128, 64],   # Per-branch dense stacks (if not using branch_layer_sizes)
    encoder_config=encoder_config,
)
```

`dropout_rate` is not a `ModelConfig` field — pass it to `MultiTaskModel(...)`:

```python
model = MultiTaskModel(config, dropout_rate=training_config.dropout_rate)
```

The default falls back to `0.1` (matching `TrainingConfig.dropout_rate`'s default).

### EncoderConfig
```python
EncoderConfig(
    encoder_integration=EncoderIntegration.TFHUB,
    encoder_input_type=EncoderInputType.TFHUB_TOKENS,
    encoder_identifier='...',
    embedding_dim=768,                # Pre-computed / no encoder
)
```

### TrainingConfig

```python
TrainingConfig(
    batch_size=32,
    epochs=10,
    learning_rate=2e-5,
    weight_decay_rate=0.01,
    dropout_rate=0.1,
    early_stopping=True,
    early_stopping_patience=3,
)
```

## Troubleshooting

**Import errors** — install the missing package or the matching extra:
```bash
pip install tensorflow-hub             # ModuleNotFoundError: tensorflow_hub
pip install transformers               # ModuleNotFoundError: transformers
pip install 'multitask[huggingface]'   # ...or install the extra (includes transformers + torch)
pip install 'multitask[openai]'        # ModuleNotFoundError: openai
pip install 'multitask[regression]'    # ModuleNotFoundError: scipy
```

**GPU issues:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**Out of memory:**
```python
from multitask import setup_gpu_memory_growth
setup_gpu_memory_growth()
# Or: TrainingConfig(batch_size=16)
```

**NaN loss:**
- Check labels are integers in valid range
- Reduce learning rate: `learning_rate=1e-5`

## API Reference

**Package exports** (`from multitask import ...`): `MultiTaskModel`, `Trainer`, `ModelConfig`, `TrainingConfig`, `EncoderConfig`, `load_model`, and distributed helpers above.

**Enums** (import from `multitask.config`): `EncoderIntegration`, `EncoderInputType`.

**Utils:** `load_model()`, `get_distribution_strategy()`, `setup_gpu_memory_growth()`, `check_distributed_compatibility()`, `print_device_info()`

## Examples

- `examples/minimal_pipeline_example.py`: Shortest full pipeline (NumPy + `tf.data`, train, evaluate, save/load)
- `examples/hierarchical_pipeline_example.py`: Two-level `task_structure`, same end-to-end flow
- `examples/dataframe_pipeline_example.py`: Pandas DataFrame → `tf.data` → train → evaluate
- `examples/save_load_example.py`: Model persistence
- `examples/distributed_training.py`: Distributed training

## Citation

```bibtex
@software{multitask,
  title = {multitask: A Flexible Multi-Task Learning Library for NLP},
  author = {Mehlhaff, Isaac D. and Morucci, Marco},
  year = {2025},
}
```

See [LICENSE](LICENSE).

Pull requests welcome.
