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

`inputs` are passed into `MultiTaskModel` on the forward pass, which then forwards it to the encoder. 

Note that API encoders do not require these as the remote service owns tokenization and the model. API encoders are also not trainable for the same reason.

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

The input format depends on whether you're using a text encoder, API encoder, or pre-computed embeddings.

### Format 1: Text Encoder Input

The library supports both **TensorFlow Hub** and **HuggingFace** encoders. Each has different input requirements.

#### TensorFlow Hub Models

TensorFlow Hub encoders expect specific dictionary keys.

**BERT/RoBERTa/DistilBERT (TF Hub):**
```python
inputs = {
    'input_word_ids': tf.constant([[101, 2054, 2003, ...]], dtype=tf.int32),      # Token IDs
    'input_mask': tf.constant([[1, 1, 1, ...]], dtype=tf.int32),                  # Attention mask
    'input_type_ids': tf.constant([[0, 0, 0, ...]], dtype=tf.int32)               # Segment IDs
}
# Shape: [batch_size, sequence_length] for each field
```

**Universal Sentence Encoder:**
```python
inputs = tf.constant(["This is a sentence.", "Another sentence."])
# Shape: [batch_size] (list of strings)
```

**How to Find Required Keys:**
```python
import tensorflow_hub as hub

# Load encoder and check signature
encoder = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
print(encoder.signatures)  # Shows required input keys
```

#### HuggingFace Models

HuggingFace transformers use standardized input keys across all models.

**BERT/RoBERTa/DistilBERT/etc (HuggingFace):**
```python
inputs = {
    'input_ids': tf.constant([[101, 2054, 2003, ...]], dtype=tf.int32),           # Token IDs
    'attention_mask': tf.constant([[1, 1, 1, ...]], dtype=tf.int32),              # Attention mask
    'token_type_ids': tf.constant([[0, 0, 0, ...]], dtype=tf.int32)               # Segment IDs (optional for some models)
}
# Shape: [batch_size, sequence_length] for each field
```

**Note**: Some HuggingFace models (like RoBERTa, DistilBERT) don't use `token_type_ids`. Check model documentation.

**How to Tokenize for HuggingFace:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(
    ["Text 1", "Text 2"],
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='tf'
)
# Returns dict with input_ids, attention_mask, token_type_ids
```

### Format 2: API Encoder Input

When using API-based encoders (e.g. OpenAI, Cohere, Voyage, or HuggingFace Inference), inputs should be raw string tensors.

You **do not** pass token dictionaries for API encoders.

**How It Works**

1. ** Create model with an API encoder**
```python
from multitask import MultiTaskModel
from multitask.config import ModelConfig, EncoderConfig

config = ModelConfig(
    task_structure=[[2]],
    task_names=['task_0', 'task_1'],
    encoder_config=EncoderConfig(
        encoder_identifier='openai:text-embedding-3-small' # example API model
    )
)

model = MultiTaskModel(config=config)
```

2. **Dataset yields raw text + labels**
```python
import tensorflow as tf

texts = ["Example sentence one.", "Example sentence two."]
labels_task0 = [1,0]
labels_task1 = [0,1]

dataset = tf.data.Dataset.from_tensor_slices((
    tf.constant(texts, dtype=tf.string),
    {
        'task_0': tf.constant(labels_task0, dtype=tf.float32),
        'task_1': tf.constant(labels_task1, dtype=tf.float32)
    }
))
dataset = dataset.shuffle(len(texts)).batch(32).prefetch(tf.data.AUTOTUNE)
```

3. **Train Normally**
```python
from multitask.trainers import Trainer
from multitask.config import TrainingConfig

training_config = TrainingConfig(epochs=10, batch_size=32)
trainer = Trainer(model, config=training_config)

# num_tasks should match the number of architectural task outputs
history, thresholds = trainer.fit(train_dataset, val_dataset, num_tasks=2)
```

**Key Points**
- Input format is a 1D `tf.string` tensor per batch: shape `[batch_size]`
- No local tokenization keys required (`input_ids`, `attention_mask`, etc)
- API encoders are **not trainable**, i.e. the encoder's embedding behavior does not change, but training learns how to map embeddings to task labels
- Embeddings cached by the API encoder layer to reduce repeated API calls

**Requirements**
- ✅ Shape: [batch_size] for text batches
- ✅ Dtype: tf.string for inputs
- ✅ Labels still follow standard labels_dict format
- ✅ Same (inputs, labels_dict) contract as other encoder types

### Format 3: Pre-computed Embedding Input

When using pre-computed embeddings (no `encoder_identifier` in `EncoderConfig`, only `embedding_dim`), inputs should be a **tensor** passed directly through the dataset.

**How It Works:**

1. **Create model without encoder:**
```python
from multitask import MultiTaskModel
from multitask.config import ModelConfig, EncoderConfig

config = ModelConfig(
    task_structure=[[2]],
    task_names=['task_0', 'task_1'],
    encoder_config=EncoderConfig(embedding_dim=768)
)

model = MultiTaskModel(
    config=config
)
```

2. **Dataset yields embeddings directly:**
```python
import tensorflow as tf
import numpy as np

# Your pre-computed embeddings (e.g., from sentence-transformers, OpenAI, etc.)
embeddings = np.array([
    [0.1, 0.2, ..., 0.768],  # Sample 1 embedding (768-dim)
    [0.3, 0.4, ..., 0.768],  # Sample 2 embedding
    # ...
])

labels_task0 = np.array([1, 0, ...])
labels_task1 = np.array([0, 1, ...])

def generator():
    for i in range(len(embeddings)):
        # Yield (embedding_tensor, labels_dict)
        yield embeddings[i], {
            'task_0': labels_task0[i],
            'task_1': labels_task1[i]
        }

dataset = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=(768,), dtype=tf.float32),  # Embedding tensor
        {
            'task_0': tf.TensorSpec(shape=(), dtype=tf.float32),
            'task_1': tf.TensorSpec(shape=(), dtype=tf.float32)
        }
    )
)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```

3. **Train normally:**
```python
from multitask.trainers import Trainer
from multitask.config import TrainingConfig

training_config = TrainingConfig(
    epochs=10
)

trainer = Trainer(
    model,
    config=training_config
)

# Assuming you've split your data into training and validation datasets
history, thresholds = trainer.fit(train_dataset, val_dataset, num_tasks=2)
```

**Key Points:**
- Embeddings bypass the encoder and go directly to the shared/branch layers
- Must be 2D tensor: `[batch_size, embedding_dim]` after batching
- `embedding_dim` overrides `ModelConfig.encoder_config.embedding_dim`
- Typically float32 dtype
- Can use embeddings from any source: sentence-transformers, OpenAI, Cohere, etc.

**Requirements:**
- ✅ Shape: `[batch_size, embedding_dim]` (after batching)
- ✅ Dtype: `tf.float32`
- ✅ No tokenization needed - embeddings computed externally
- ✅ Faster training - no encoder forward pass

## Label Formats

Labels must be provided as a **dictionary** mapping task names to label tensors. The format depends on the task type.

### Binary Classification (`task_type='binary'`)

***TODO: Update this after regression support***
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

***TODO: Update this after regression support***

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
concatenated_tensor, task_indices = model.flatten_outputs(outputs)
# Shape: [batch_size, total_output_dim]

# Unflatten back to dictionary
outputs_dict = model.unflatten_outputs(concatenated_tensor)
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

### Example 2: HuggingFace BERT with Binary Task

```python
import tensorflow as tf
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data_hf(texts, labels):
    """Convert raw texts to HuggingFace BERT inputs."""
    # Tokenize using HuggingFace tokenizer
    inputs = tokenizer(
        texts.numpy().tolist() if hasattr(texts, 'numpy') else texts,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )
    
    # Create labels dict
    labels_dict = {
        'is_spam': tf.cast(labels, tf.float32)
    }
    
    return inputs, labels_dict

# Create dataset
raw_texts = ["Buy now!", "Meeting at 3pm", "Free money!!!"]
raw_labels = [1, 0, 1]

dataset = tf.data.Dataset.from_tensor_slices((raw_texts, raw_labels))
dataset = dataset.map(lambda t, l: tf.py_function(
    preprocess_data_hf, [t, l], 
    ({
        'input_ids': tf.int32,
        'attention_mask': tf.int32,
        'token_type_ids': tf.int32
    }, {'is_spam': tf.float32})
))
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```

### Example 3: Pre-computed Embeddings with Mixed Tasks

```python
import tensorflow as tf
import numpy as np
from multitask import MultiTaskModel
from multitask.config import ModelConfig, EncoderConfig, TrainingConfig
from multitask.trainers import Trainer

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

training_config = TrainingConfig(
    epochs=10
)

# Create model and trainer
config = ModelConfig(
    task_structure=[[3]],
    task_types=['binary', 'multiclass', 'multilabel'],
    num_classes_per_task=[2, 5, 5],
    task_names=['sentiment', 'category', 'topics'],
    encoder_config=EncoderConfig(
        embedding_dim=768      # Match your embedding dimension
    )
)

model = MultiTaskModel(
    config=config
)

# Note: train_dataset and val_dataset are split off from your dataset above.
trainer = Trainer(model, config=training_config)
history, thresholds = trainer.fit(train_dataset, val_dataset, num_tasks=3)
```

### Example 4: Computing Embeddings from Scratch

```python
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from multitask import MultiTaskModel
from multitask.config import ModelConfig, TrainingConfig
from multitask.trainers import Trainer

# Load embedding model (runs once, before creating dataset)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim

# Compute embeddings for all texts
texts = ["Text 1", "Text 2", "Text 3", ...]
embeddings = embedding_model.encode(texts, convert_to_tensor=False)  # numpy array

# Save embeddings for reuse (optional but recommended)
# np.save('embeddings.npy', embeddings)

# Create dataset
labels = [1, 0, 1, ...]

def generator():
    for emb, label in zip(embeddings, labels):
        yield emb, {'is_spam': label}

dataset = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=(384,), dtype=tf.float32),  # sentence-transformers uses 384-dim
        {'is_spam': tf.TensorSpec(shape=(), dtype=tf.float32)}
    )
).batch(32).prefetch(tf.data.AUTOTUNE)

# Create model
config = ModelConfig(task_structure=[[1]], task_names=['is_spam'])
model = MultiTaskModel(
    config=config,
    embedding_dim=384  # Match sentence-transformers dimension
)

# Train
training_config = TrainingConfig(
    epochs=10
)
trainer = Trainer(model, config=training_config)
history, thresholds = trainer.fit(dataset, val_dataset, num_tasks=1)
```

**Why Use Pre-computed Embeddings?**
- ✅ **Faster training**: No encoder forward pass needed
- ✅ **Lower memory**: Don't need to store encoder weights
- ✅ **Flexibility**: Use any embedding source (OpenAI, Cohere, sentence-transformers, etc.)
- ✅ **Caching**: Compute embeddings once, reuse for multiple experiments

### Example 6: Handling Partially Labeled Data

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

❌ **Wrong**: Using tensor input with text encoder
```python
inputs = tf.constant([[0.1, 0.2, ...]])  # Wrong for BERT!
```

✅ **Correct**: Using dict input with TF Hub BERT
```python
inputs = {
    'input_word_ids': token_ids,
    'input_mask': attention_mask,
    'input_type_ids': segment_ids
}
```

✅ **Correct**: Using dict input with HuggingFace BERT
```python
inputs = {
    'input_ids': token_ids,
    'attention_mask': attention_mask,
    'token_type_ids': segment_ids
}
```

### 2. **Wrong Dictionary Keys for Encoder**

❌ **Wrong**: Using TF Hub keys with HuggingFace model
```python
# Using HuggingFace model but TF Hub keys
inputs = {
    'input_word_ids': token_ids,  # Wrong! Should be 'input_ids'
    'input_mask': attention_mask,  # Wrong! Should be 'attention_mask'
}
```

✅ **Correct**: Match keys to encoder type
```python
# TF Hub: input_word_ids, input_mask, input_type_ids
# HuggingFace: input_ids, attention_mask, token_type_ids
```

### 3. **Labels Not in Dictionary**

❌ **Wrong**: Passing labels as tensor or list
```python
labels = tf.constant([1, 0, 1])  # Wrong!
```

✅ **Correct**: Labels in dictionary
```python
labels_dict = {'task_0': tf.constant([1, 0, 1], dtype=tf.float32)}
```

### 4. **Wrong Label Shape for Task Type**

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

### 5. **Wrong Label Data Type**

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

### 6. **Mismatched Task Names**

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

### 7. **Embedding Dimension Mismatch**

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

### 8. **Forgot to Batch Dataset**

❌ **Wrong**: Unbatched dataset
```python
dataset = tf.data.Dataset.from_generator(...)  # Missing .batch()!
```

✅ **Correct**: Batched dataset
```python
dataset = tf.data.Dataset.from_generator(...).batch(32)
```

### 9. **Invalid Label Values**

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

### 10. **Missing HuggingFace Installation**

❌ **Wrong**: Using HuggingFace model without installing transformers
```python
# This will fail if transformers not installed
model = MultiTaskModel(
    config=config, 
    encoder_config=EncoderConfig(
        encoder_identifier="bert-base-uncased",
        encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
        encoder_integration=EncoderIntegration.HUGGINGFACE
    )
)
```

✅ **Correct**: Install HuggingFace support first
```bash
pip install multitask[huggingface]
# or
pip install transformers>=4.30.0
```

## Validation

### Pre-Training Validation

Use `model.validate_dataset()` to catch format errors before training:

```python
from multitask import MultiTaskModel
from multitask.config import ModelConfig, EncoderConfig, EncoderInputType, EncoderIntegration

# Define model with TensorFlow Hub
config = ModelConfig(
    task_structure=[[3]],
    task_types=['binary', 'binary', 'multiclass'],
    num_classes_per_task=[2, 2, 3],
    task_names=['is_spam', 'is_urgent', 'category'],
    encoder_config=EncoderConfig(
        encoder_identifier="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
        encoder_input_type=EncoderInputType.TFHUB_TOKENS,
        encoder_integration=EncoderIntegration.TFHUB
    )
)

model = MultiTaskModel(
    config=config,
)

# Or use HuggingFace
# Requires: pip install multitask[huggingface]
config.encoder_config = EncoderConfig(
    encoder_identifier="bert-base-uncased",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE,
)
model = MultiTaskModel(
    config=config
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
- [ ] **For text encoders**: Inputs match model type
  - TF Hub BERT-style: dict keys `input_word_ids`, `input_mask`, `input_type_ids`
  - TF Hub USE: 1D `tf.string` tensor (`[batch_size]`)
  - HuggingFace: dict keys `input_ids`, `attention_mask`, `token_type_ids` (if required by model)
- [ ] **For API encoders**: Inputs are **string tensors**
  - Use `EncoderConfig(encoder_identifier=...)` with the model name
- [ ] **For embeddings**: Inputs are **tensor** with shape `[batch_size, embedding_dim]`
  - Use `ModelConfig(embedding_dim=...)` / `EncoderConfig(embedding_dim=...)` with no `encoder_identifier`, and pass `embedding_dim` to `MultiTaskModel` if needed
  - Dataset yields pre-computed embeddings directly
- [ ] Input dimensions match encoder/embedding requirements
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
