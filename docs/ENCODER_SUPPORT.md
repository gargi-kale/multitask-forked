# Encoder Support

The `multitask` library supports multiple encoder backends for maximum flexibility.

## EncoderConfig Class

This class handles encoder configurations and has the following attributes
- encoder: actual encoder object
- encoder_identifier: TF Hub URL, HuggingFace model id, or `integration:model` (e.g. `openai:text-embedding-3-small`); bare names default to HuggingFace
- encoder_input_type: type of input accepted by the encoder
- encoder_integration: how we access the encoder
- trainable_encoder
- embedding_dim

## Supported Encoders

### 1. TensorFlow Hub Models

Any TensorFlow Hub model can be used by providing its URL.

**Example:**
```python
from multitask import MultiTaskModel
from multitask.config import (
    ModelConfig,
    EncoderConfig,
    EncoderInputType,
    EncoderIntegration,
)

# BERT from TF Hub
encoder_config = EncoderConfig(
    encoder_identifier="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
    encoder_input_type=EncoderInputType.TFHUB_TOKENS,
    encoder_integration=EncoderIntegration.TFHUB
)

config = ModelConfig(
    task_structure=[[2]],
    encoder_config=encoder_config
)

# Universal Sentence Encoder
encoder_config = EncoderConfig(
    encoder_identifier="https://tfhub.dev/google/universal-sentence-encoder/4",
    encoder_input_type=EncoderInputType.RAW_STRING,
    encoder_integration=EncoderIntegration.TFHUB
)

config = ModelConfig(
    task_structure=[[2]],
    encoder_config=encoder_config
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
config.encoder_config = EncoderConfig(
    encoder_identifier="bert-base-uncased",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE
)
model = MultiTaskModel(
    config=config
)

# RoBERTa large
config.encoder_config = EncoderConfig(
    encoder_identifier="roberta-large",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE
)
model = MultiTaskModel(
    config=config
)

# DistilBERT
config.encoder_config = EncoderConfig(
    encoder_identifier="distilbert-base-uncased",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE
)
model = MultiTaskModel(
    config=config
)

# Any HuggingFace model
config.encoder_config = EncoderConfig(
    encoder_identifier="microsoft/deberta-v3-base",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE
)
model = MultiTaskModel(
    config=config
)
```

You can also access encoders via the HuggingFace Inference API (model must be one supported in `HFInferenceEncoder.EMBEDDING_DIMS` in `models/encoders.py`):
```python
config.encoder_config=EncoderConfig(
    encoder_identifier="huggingface_api:sentence-transformers/all-MiniLM-L6-v2",
    encoder_input_type=EncoderInputType.RAW_STRING,
    encoder_integration=EncoderIntegration.HUGGINGFACE_API
)
model = MultiTaskModel(
    config=config
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
from multitask import MultiTaskModel
from multitask.config import ModelConfig, EncoderConfig, EncoderInputType

# Custom encoder
class MyEncoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(768, activation='relu')
    
    def call(self, inputs):
        return self.dense(inputs)

# Use custom encoder (set encoder_input_type to match your dataset; Keras layers are built as EncoderIntegration.KERAS)
config = ModelConfig(
    task_structure=[[2]],
    encoder_config=EncoderConfig(
        encoder=MyEncoder(),
        encoder_input_type=EncoderInputType.PRECOMPUTED,
    ),
)
model = MultiTaskModel(config=config)
```

### 4. API Encoders

Use identifiers of the form `integration:model_name`. The prefix must match `EncoderIntegration` (`openai`, `cohere`, `voyage`, `huggingface_api`).

Environment variables (see `models/encoders.py`):
* HuggingFace Inference API (`HF_API_TOKEN` or `HF_TOKEN`)
* OpenAI (`OPENAI_API_KEY`)
* Cohere (`COHERE_API_KEY`)
* Voyage (`VOYAGE_API_KEY`)

See `models/encoders.py` for supported model names per provider.

HuggingFace
```python
config.encoder_config=EncoderConfig(
    encoder_identifier="huggingface_api:sentence-transformers/all-MiniLM-L6-v2",
    encoder_input_type=EncoderInputType.RAW_STRING,
    encoder_integration=EncoderIntegration.HUGGINGFACE_API
)
model = MultiTaskModel(
    config=config
)
```

OpenAI
```python
config.encoder_config=EncoderConfig(
    encoder_identifier="openai:text-embedding-3-small",
    encoder_input_type=EncoderInputType.RAW_STRING,
    encoder_integration=EncoderIntegration.OPENAI
)
model = MultiTaskModel(
    config=config
)
```

Cohere
```python
config.encoder_config=EncoderConfig(
    encoder_identifier="cohere:embed-english-v3.0",
    encoder_input_type=EncoderInputType.RAW_STRING,
    encoder_integration=EncoderIntegration.COHERE
)
model = MultiTaskModel(
    config=config
)
```

Voyage
```python
config.encoder_config=EncoderConfig(
    encoder_identifier="voyage:voyage-2",
    encoder_input_type=EncoderInputType.RAW_STRING,
    encoder_integration=EncoderIntegration.VOYAGE
)
model = MultiTaskModel(
    config=config
)
```

In some cases, using precomputed embeddings might be faster.

### 5. Pre-computed Embeddings

Use pre-computed embeddings without any encoder.

**Example:**
```python
from multitask import MultiTaskModel
from multitask.config import ModelConfig

# Pre-computed path: no encoder_identifier, only embedding dimension
config = ModelConfig(
    task_structure=[[2]],
    embedding_dim=768,
)
model = MultiTaskModel(config=config, embedding_dim=768)
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
config.encoder_config = EncoderConfig(
    encoder_identifier="bert-base-uncased",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE,
    trainable_encoder=True # Encoder weights will be updated
)
model = MultiTaskModel(
    config=config
)

# Freeze encoder (feature extraction only)
config.encoder_config = EncoderConfig(
    encoder_identifier="bert-base-uncased",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE,
    trainable_encoder=False # Encoder weights frozen
)
model = MultiTaskModel(
    config=config,
    encoder="bert-base-uncased",  # optional override of encoder_identifier
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
config.encoder_config=EncoderConfig(
    encoder_identifier="https://tfhub.dev/..."
)
model = MultiTaskModel(config=config)
# BERT-like TF Hub models: EncoderInputType.TFHUB_TOKENS; Universal Sentence Encoder: EncoderInputType.RAW_STRING
print(model.encoder_input_type)
print(model.encoder_integration)  # EncoderIntegration.TFHUB

config.encoder_config=EncoderConfig(
    encoder=custom_layer,
    encoder_input_type=EncoderInputType.RAW_STRING,  # must match your data
)
model = MultiTaskModel(config=config, encoder=custom_layer)
print(model.encoder_input_type)  # EncoderInputType.RAW_STRING
print(model.encoder_integration)  # EncoderIntegration.KERAS

model = MultiTaskModel(config=config, embedding_dim=768)
print(model.encoder_input_type)  # None
print(model.encoder_integration)  # None
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
