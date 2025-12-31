# Encoder Support

The `multitask` library supports multiple encoder backends for maximum flexibility.

## Supported Encoders

### 1. TensorFlow Hub Models

Any TensorFlow Hub model can be used by providing its URL.

**Example:**
```python
from multitask import MultiTaskModel
from multitask.config import ModelConfig

config = ModelConfig(task_structure=[[2]])

# BERT from TF Hub
model = MultiTaskModel(
    config=config,
    encoder="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
)

# Universal Sentence Encoder
model = MultiTaskModel(
    config=config,
    encoder="https://tfhub.dev/google/universal-sentence-encoder/4"
)
```

**Advantages:**
- No extra dependencies (TensorFlow Hub comes with the library)
- Optimized for TensorFlow
- Easy to use

**Input Format:** Varies by model (check model documentation on TF Hub)

### 2. HuggingFace Transformers

Any HuggingFace model with a TensorFlow implementation can be used.

**Installation:**
```bash
pip install multitask[huggingface]
# or
pip install transformers>=4.30.0
```

**Example:**
```python
# BERT base uncased
model = MultiTaskModel(
    config=config,
    encoder="bert-base-uncased"
)

# RoBERTa large
model = MultiTaskModel(
    config=config,
    encoder="roberta-large"
)

# DistilBERT
model = MultiTaskModel(
    config=config,
    encoder="distilbert-base-uncased"
)

# Any HuggingFace model
model = MultiTaskModel(
    config=config,
    encoder="microsoft/deberta-v3-base"
)
```

**Advantages:**
- Access to 1000s of pre-trained models
- Latest model architectures
- Easy model switching for experimentation
- Community-supported

**Input Format:** All HuggingFace models use the same keys:
- `input_ids`: Token IDs
- `attention_mask`: Attention mask
- `token_type_ids`: Segment IDs (optional for some models)

### 3. Custom Keras Layers

Pass any custom Keras layer as the encoder.

**Example:**
```python
import tensorflow as tf

# Custom encoder
class MyEncoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(768, activation='relu')
    
    def call(self, inputs):
        return self.dense(inputs)

# Use custom encoder
model = MultiTaskModel(
    config=config,
    encoder=MyEncoder()
)
```

### 4. Pre-computed Embeddings

Use pre-computed embeddings without any encoder.

**Example:**
```python
# Use 768-dimensional embeddings
model = MultiTaskModel(
    config=config,
    encoder=None,
    embedding_dim=768
)
```

**Advantages:**
- Fastest training (no encoder forward pass)
- Useful when embeddings are pre-computed
- Reduces memory usage

## Choosing an Encoder

### By Use Case

| Use Case | Recommended Encoder |
|----------|-------------------|
| General English NLP | `bert-base-uncased` (HF) or TF Hub BERT |
| Faster inference | `distilbert-base-uncased` (HF) |
| Best accuracy | `roberta-large` (HF) or `deberta-v3-large` (HF) |
| Multilingual | `bert-base-multilingual-cased` (HF) |
| Sentence embeddings | Universal Sentence Encoder (TF Hub) |
| Domain-specific | Find domain-specific model on HuggingFace |
| Custom architecture | Build custom Keras layer |

### By Speed

**Fastest → Slowest:**
1. Pre-computed embeddings
2. DistilBERT (HF)
3. BERT base (HF/TF Hub)
4. BERT large (HF/TF Hub)
5. RoBERTa large (HF)

### By Accuracy

**Generally (task-dependent):**
1. DeBERTa v3 large
2. RoBERTa large
3. BERT large
4. BERT base
5. DistilBERT

## Input Preprocessing

### TensorFlow Hub

Use TF Hub preprocessing models:

```python
import tensorflow_hub as hub
import tensorflow_text as text

# Load preprocessor
preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

# Tokenize
texts = ["Hello world", "Another sentence"]
inputs = preprocessor(texts)
# Returns dict with input_word_ids, input_mask, input_type_ids
```

### HuggingFace

Use `transformers` tokenizers:

```python
from transformers import AutoTokenizer

# Load tokenizer (must match encoder)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize
texts = ["Hello world", "Another sentence"]
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='tf'
)
# Returns dict with input_ids, attention_mask, token_type_ids
```

## Fine-tuning Control

Control whether encoder weights are updated during training:

```python
# Fine-tune encoder (default)
model = MultiTaskModel(
    config=config,
    encoder="bert-base-uncased",
    trainable_encoder=True  # Encoder weights will be updated
)

# Freeze encoder (feature extraction only)
model = MultiTaskModel(
    config=config,
    encoder="bert-base-uncased",
    trainable_encoder=False  # Encoder weights frozen
)
```

**When to freeze:**
- Small training dataset (prevent overfitting)
- Faster training
- Lower memory usage
- Encoder already well-suited to task

**When to fine-tune:**
- Large training dataset
- Domain-specific data (medical, legal, etc.)
- Maximum accuracy needed
- Sufficient GPU memory

## Model Detection

The library automatically detects encoder type:

```python
model = MultiTaskModel(config=config, encoder="https://tfhub.dev/...")
print(model.encoder_type)  # 'tfhub'

model = MultiTaskModel(config=config, encoder="bert-base-uncased")
print(model.encoder_type)  # 'huggingface'

model = MultiTaskModel(config=config, encoder=custom_layer)
print(model.encoder_type)  # 'custom'

model = MultiTaskModel(config=config, encoder=None, embedding_dim=768)
print(model.encoder_type)  # None
```

## Troubleshooting

### "Module not found: transformers"

**Problem:** Trying to use HuggingFace encoder without installing transformers.

**Solution:**
```bash
pip install multitask[huggingface]
```

### Wrong input format

**Problem:** Using TF Hub input format with HuggingFace encoder (or vice versa).

**Solution:** Match tokenizer to encoder:
- TF Hub → TF Hub preprocessor
- HuggingFace → HuggingFace tokenizer

### Model not found

**Problem:** HuggingFace model name incorrect.

**Solution:** Check model name on [HuggingFace Model Hub](https://huggingface.co/models?library=tf)

Make sure model has TensorFlow implementation (look for `TF` badge).

## Examples

See [DATA_FORMAT.md](DATA_FORMAT.md) for complete preprocessing examples with both encoder types.
