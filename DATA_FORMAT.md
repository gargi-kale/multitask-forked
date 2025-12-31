# Data Format Requirements

This document describes the required data formats for training and evaluating multi-task models with the `multitask` library.

## Table of Contents
- [Overview](#overview)
- [Dataset Format](#dataset-format)
- [Input Formats](#input-formats)
- [Label Formats](#label-formats)
- [Model Output Format](#model-output-format)
- [Preprocessing Examples](#preprocessing-examples)
- [Common Pitfalls](#common-pitfalls)
- [Validation](#validation)

## Overview

The `multitask` library uses TensorFlow `tf.data.Dataset` objects for training and evaluation. Each dataset must yield tuples of `(inputs, labels_dict)` where:
- **inputs**: Text input (format depends on encoder type)
- **labels_dict**: Dictionary mapping task names to label tensors

## Dataset Format

### Basic Structure

```python
import tensorflow as tf

# Create a dataset that yields (inputs, labels_dict) tuples
def dataset_generator():
    for sample in data:
        inputs = ...  # See "Input Formats" section
        labels_dict = {
            'task_0': tf.constant([1], dtype=tf.float32),
            'task_1': tf.constant([2], dtype=tf.int32),
            'task_2': tf.constant([0, 1, 0], dtype=tf.float32)
        }
        yield inputs, labels_dict

dataset = tf.data.Dataset.from_generator(
    dataset_generator,
    output_signature=(
        input_spec,  # See "Input Formats" section
        {
            'task_0': tf.TensorSpec(shape=(1,), dtype=tf.float32),
            'task_1': tf.TensorSpec(shape=(), dtype=tf.int32),
            'task_2': tf.TensorSpec(shape=(3,), dtype=tf.float32)
        }
    )
).batch(32)
```

### Key Requirements

1. **Batching**: Apply `.batch()` to your dataset with desired batch size
2. **Prefetching**: Use `.prefetch(tf.data.AUTOTUNE)` for performance
3. **Shuffling**: Use `.shuffle()` for training data
4. **Task names**: Must match `ModelConfig.task_names` (or use default `task_0`, `task_1`, etc.)

## Input Formats

The input format depends on whether you're using a text encoder or pre-computed embeddings.

### Format 1: Text Encoder Input (BERT, RoBERTa, etc.)

When using TensorFlow Hub encoders, inputs should be a **dictionary** with keys expected by the encoder.

#### BERT/RoBERTa/DistilBERT
```python
inputs = {
    'input_word_ids': tf.constant([[101, 2054, 2003, ...]], dtype=tf.int32),      # Token IDs
    'input_mask': tf.constant([[1, 1, 1, ...]], dtype=tf.int32),                  # Attention mask
    'input_type_ids': tf.constant([[0, 0, 0, ...]], dtype=tf.int32)               # Segment IDs
}
# Shape: [batch_size, sequence_length] for each field
```

#### Universal Sentence Encoder
```python
inputs = tf.constant(["This is a sentence.", "Another sentence."])
# Shape: [batch_size] (list of strings)
```

#### How to Find Required Keys

```python
import tensorflow_hub as hub

# Load encoder and check signature
encoder = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
print(encoder.signatures)  # Shows required input keys
```

### Format 2: Pre-computed Embedding Input

When using pre-computed embeddings (`encoder_url=None`), inputs should be a **tensor**.

```python
inputs = tf.constant([[0.1, 0.2, ..., 0.768]], dtype=tf.float32)
# Shape: [batch_size, embedding_dim]
```

**Requirements:**
- Must be 2D tensor: `[batch_size, embedding_dim]`
- `embedding_dim` must match `ModelConfig.embedding_dim`
- Typically float32 dtype

## Label Formats

Labels must be provided as a **dictionary** mapping task names to label tensors. The format depends on the task type.

### Binary Classification (`task_type='binary'`)

**Format**: Single value per sample (0 or 1), or -1 for masked/missing labels

```python
# Option 1: Shape [batch_size]
labels_dict = {
    'is_toxic': tf.constant([1, 0, 1, -1], dtype=tf.float32)  # Last sample masked
}

# Option 2: Shape [batch_size, 1]
labels_dict = {
    'is_toxic': tf.constant([[1], [0], [1], [-1]], dtype=tf.float32)
}
```

**Valid values**: `0`, `1`, `-1` (mask)

**Output**: Model returns probabilities in [0, 1] via sigmoid activation

### Multiclass Classification (`task_type='multiclass'`)

**Format**: Class index per sample (0 to K-1), or -1 for masked

```python
# For 3 classes (e.g., negative=0, neutral=1, positive=2)
labels_dict = {
    'sentiment': tf.constant([2, 0, 1, -1], dtype=tf.int32)  # Last sample masked
}
# Shape: [batch_size]
```

**Valid values**: `0` to `K-1` where K = `num_classes_per_task[i]`, or `-1` (mask)

**Output**: Model returns class probabilities via softmax activation

### Multilabel Classification (`task_type='multilabel'`)

**Format**: Binary indicator for each class (0 or 1), or -1 for masked

```python
# For 4 possible labels (e.g., categories: politics, sports, tech, business)
labels_dict = {
    'categories': tf.constant([
        [1, 0, 1, 0],   # Sample has politics and tech
        [0, 1, 0, 0],   # Sample has only sports
        [1, 1, 1, 1],   # Sample has all categories
        [-1, -1, -1, -1]  # Masked sample
    ], dtype=tf.float32)
}
# Shape: [batch_size, num_classes]
```

**Valid values**: `0`, `1`, `-1` (mask) for each class

**Output**: Model returns per-class probabilities via sigmoid activation

### Label Masking

Use `-1` to indicate missing, invalid, or unlabeled data. Masked labels are:
- **Excluded from loss computation** (don't contribute to training)
- **Excluded from metrics** (don't affect accuracy, F1, etc.)
- **Useful for**: Partially labeled data, different tasks available for different samples

```python
labels_dict = {
    'task_0': tf.constant([1, 0, -1, 1], dtype=tf.float32),    # Binary: 3rd sample masked
    'task_1': tf.constant([2, -1, 1, 0], dtype=tf.int32),       # Multiclass: 2nd sample masked
    'task_2': tf.constant([[1, 0, 1], [-1, -1, -1], [0, 1, 0], [1, 1, 1]], dtype=tf.float32)  # Multilabel: 2nd sample masked
}
```

## Model Output Format

### Dictionary Outputs

The model returns predictions as a **dictionary** mapping task names to output tensors:

```python
outputs = model(inputs)
# Returns: {
#     'task_0': tf.Tensor(shape=[batch_size, 1], dtype=float32),      # Binary
#     'task_1': tf.Tensor(shape=[batch_size, num_classes], dtype=float32),  # Multiclass
#     'task_2': tf.Tensor(shape=[batch_size, num_classes], dtype=float32)   # Multilabel
# }
```

**Benefits**:
- Clean separation between tasks
- Easy to inspect individual task outputs
- Natural handling of variable output dimensions

### Flattening/Unflattening Utilities

For compatibility with external tools that expect concatenated outputs:

```python
# Flatten dictionary to concatenated tensor
flat_tensor = model.flatten_outputs(outputs)
# Shape: [batch_size, total_output_dim]

# Unflatten back to dictionary
outputs_dict = model.unflatten_outputs(flat_tensor)
# Returns: {'task_0': ..., 'task_1': ..., 'task_2': ...}
```

## Preprocessing Examples

### Example 1: BERT with Binary Task

```python
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

# Load preprocessing model
preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

def preprocess_data(texts, labels):
    """Convert raw texts to BERT inputs."""
    # Tokenize using BERT preprocessor
    inputs = preprocessor(texts)
    
    # Create labels dict
    labels_dict = {
        'is_spam': tf.cast(labels, tf.float32)
    }
    
    return inputs, labels_dict

# Create dataset
raw_texts = ["Buy now!", "Meeting at 3pm", "Free money!!!"]
raw_labels = [1, 0, 1]

dataset = tf.data.Dataset.from_tensor_slices((raw_texts, raw_labels))
dataset = dataset.map(preprocess_data)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```

### Example 2: Pre-computed Embeddings with Mixed Tasks

```python
import tensorflow as tf
import numpy as np

def load_embeddings_and_labels():
    """Load pre-computed embeddings with multiple task labels."""
    # Assume embeddings are pre-computed (e.g., from sentence-transformers)
    embeddings = np.load('embeddings.npy')  # Shape: [num_samples, 768]
    
    # Load labels for multiple tasks
    binary_labels = np.load('binary_labels.npy')       # Shape: [num_samples]
    multiclass_labels = np.load('multiclass_labels.npy')  # Shape: [num_samples]
    multilabel_labels = np.load('multilabel_labels.npy')  # Shape: [num_samples, 5]
    
    def generator():
        for i in range(len(embeddings)):
            inputs = embeddings[i]
            labels_dict = {
                'sentiment': binary_labels[i],
                'category': multiclass_labels[i],
                'topics': multilabel_labels[i]
            }
            yield inputs, labels_dict
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(768,), dtype=tf.float32),
            {
                'sentiment': tf.TensorSpec(shape=(), dtype=tf.float32),
                'category': tf.TensorSpec(shape=(), dtype=tf.int32),
                'topics': tf.TensorSpec(shape=(5,), dtype=tf.float32)
            }
        )
    )
    
    return dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```

### Example 3: Handling Partially Labeled Data

```python
import tensorflow as tf

def create_partially_labeled_dataset(texts, task1_labels, task2_labels):
    """Some samples only have labels for task1, some for task2, some for both."""
    
    def preprocess(text, label1, label2):
        # Convert text to BERT inputs (pseudo-code)
        inputs = tokenize_text(text)
        
        # Use -1 to mask unavailable labels
        labels_dict = {
            'task_1': tf.where(tf.math.is_nan(label1), -1.0, label1),  # Mask NaN
            'task_2': tf.where(tf.math.is_nan(label2), -1.0, label2)
        }
        
        return inputs, labels_dict
    
    dataset = tf.data.Dataset.from_tensor_slices((texts, task1_labels, task2_labels))
    dataset = dataset.map(preprocess)
    return dataset.batch(32)
```

## Common Pitfalls

### 1. **Wrong Input Format for Encoder Type**

❌ **Wrong**: Using tensor input with BERT encoder
```python
inputs = tf.constant([[0.1, 0.2, ...]])  # Wrong for BERT!
```

✅ **Correct**: Using dict input with BERT encoder
```python
inputs = {
    'input_word_ids': token_ids,
    'input_mask': attention_mask,
    'input_type_ids': segment_ids
}
```

### 2. **Labels Not in Dictionary**

❌ **Wrong**: Passing labels as tensor or list
```python
labels = tf.constant([1, 0, 1])  # Wrong!
```

✅ **Correct**: Labels in dictionary
```python
labels_dict = {'task_0': tf.constant([1, 0, 1], dtype=tf.float32)}
```

### 3. **Wrong Label Shape for Task Type**

❌ **Wrong**: Binary labels with multiple columns
```python
labels_dict = {'binary_task': tf.constant([[1, 0], [0, 1]], dtype=tf.float32)}  # Wrong!
```

✅ **Correct**: Binary labels as single column
```python
labels_dict = {'binary_task': tf.constant([1, 0], dtype=tf.float32)}
```

❌ **Wrong**: Multiclass labels as one-hot encoded
```python
labels_dict = {'multiclass_task': tf.constant([[0, 0, 1], [1, 0, 0]], dtype=tf.float32)}  # Wrong!
```

✅ **Correct**: Multiclass labels as class indices
```python
labels_dict = {'multiclass_task': tf.constant([2, 0], dtype=tf.int32)}
```

### 4. **Wrong Label Data Type**

❌ **Wrong**: Using wrong dtype
```python
labels_dict = {
    'binary_task': tf.constant([1, 0], dtype=tf.int32),      # Should be float32
    'multiclass_task': tf.constant([2, 0], dtype=tf.float32)  # Should be int32
}
```

✅ **Correct**: Using correct dtype
```python
labels_dict = {
    'binary_task': tf.constant([1, 0], dtype=tf.float32),
    'multiclass_task': tf.constant([2, 0], dtype=tf.int32)
}
```

### 5. **Mismatched Task Names**

❌ **Wrong**: Label keys don't match config
```python
# Config has task_names=['sentiment', 'category']
labels_dict = {'task_0': ..., 'task_1': ...}  # Wrong names!
```

✅ **Correct**: Label keys match config
```python
# Config has task_names=['sentiment', 'category']
labels_dict = {'sentiment': ..., 'category': ...}
```

### 6. **Embedding Dimension Mismatch**

❌ **Wrong**: Embeddings don't match config
```python
# Config has embedding_dim=768
inputs = tf.constant([[0.1] * 512], dtype=tf.float32)  # Wrong dimension!
```

✅ **Correct**: Embeddings match config
```python
# Config has embedding_dim=768
inputs = tf.constant([[0.1] * 768], dtype=tf.float32)
```

### 7. **Forgot to Batch Dataset**

❌ **Wrong**: Unbatched dataset
```python
dataset = tf.data.Dataset.from_generator(...)  # Missing .batch()!
```

✅ **Correct**: Batched dataset
```python
dataset = tf.data.Dataset.from_generator(...).batch(32)
```

### 8. **Invalid Label Values**

❌ **Wrong**: Using values outside valid range
```python
# Binary task with values outside {0, 1, -1}
labels_dict = {'binary': tf.constant([0.5, 2, -2], dtype=tf.float32)}  # Wrong!

# Multiclass task with out-of-range indices
labels_dict = {'multiclass': tf.constant([5, -2], dtype=tf.int32)}  # Wrong if only 3 classes!
```

✅ **Correct**: Using valid values
```python
# Binary: 0, 1, or -1
labels_dict = {'binary': tf.constant([0, 1, -1], dtype=tf.float32)}

# Multiclass with 3 classes: 0, 1, 2, or -1
labels_dict = {'multiclass': tf.constant([2, 0, -1], dtype=tf.int32)}
```

## Validation

### Pre-Training Validation

Use `model.validate_dataset()` to catch format errors before training:

```python
from multitask import MultiTaskModel
from multitask.config import ModelConfig

# Define model
config = ModelConfig(
    task_structure=[[2, 1]],
    task_types=['binary', 'binary', 'multiclass'],
    num_classes_per_task=[1, 1, 3],
    task_names=['is_spam', 'is_urgent', 'category']
)

model = MultiTaskModel(
    config=config,
    encoder_url="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
)

# Validate dataset format
try:
    model.validate_dataset(train_dataset, num_batches=3, verbose=True)
    print("Dataset validation passed!")
except ValueError as e:
    print(f"Dataset validation failed: {e}")
    # Fix your data based on the error message
```

### What validate_dataset() Checks

- ✅ Dataset yields (inputs, labels_dict) tuples
- ✅ Input format matches encoder type (dict vs tensor)
- ✅ Input shapes are valid
- ✅ Label dictionary has correct task names
- ✅ Label shapes match task types
- ✅ Label values are in valid ranges
- ✅ Batch consistency across samples
- ✅ Warns about all-masked tasks

### Manual Inspection

```python
# Inspect first batch
for inputs, labels_dict in dataset.take(1):
    print("Inputs:", inputs)
    print("Labels:", labels_dict)
    
    # Check shapes
    if isinstance(inputs, dict):
        for key, val in inputs.items():
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    else:
        print(f"  shape={inputs.shape}, dtype={inputs.dtype}")
    
    for task_name, labels in labels_dict.items():
        print(f"  {task_name}: shape={labels.shape}, dtype={labels.dtype}")
        print(f"    unique values: {tf.unique(tf.reshape(labels, [-1]))[0].numpy()}")
```

## Summary Checklist

Before training, verify:

- [ ] Dataset yields `(inputs, labels_dict)` tuples
- [ ] Inputs are **dict** for text encoders or **tensor** for embeddings
- [ ] Input dimensions match encoder requirements
- [ ] Labels are in a **dictionary** with task names as keys
- [ ] Task names match `ModelConfig.task_names`
- [ ] Label shapes match task types:
  - Binary: `[batch]` or `[batch, 1]`
  - Multiclass: `[batch]`
  - Multilabel: `[batch, num_classes]`
- [ ] Label values are valid:
  - Binary: `{0, 1, -1}`
  - Multiclass: `{0..K-1, -1}`
  - Multilabel: `{0, 1, -1}` per class
- [ ] Label dtypes are correct (float32 for binary/multilabel, int32 for multiclass)
- [ ] Dataset is batched with `.batch(batch_size)`
- [ ] Used `model.validate_dataset()` to verify format

---

**Need Help?** Run `model.validate_dataset(your_dataset, verbose=True)` for detailed diagnostic output!
