"""
Integration tests for the encoding step with BERT and TF models from ENCODER_SUPPORT.md.

These tests build MultiTaskModel with each supported encoder, feed correctly
preprocessed inputs, and assert that the forward pass produces expected output shapes.
They require network access (to download models on first run) and optional deps
for HuggingFace (transformers).

Run all tests (including integration):
  pytest tests/test_encoder_integration.py -v

Run only integration tests (e.g. from another suite):
  pytest -m integration -v

Skip integration tests (e.g. in CI without network):
  pytest -m "not integration" -v

TF Hub BERT tests require tensorflow_text (CaseFoldUTF8 op). If Universal Sentence
Encoder fails with "contains neither saved_model.pb", clear TF Hub cache:
  rm -rf $TFHUB_CACHE_DIR  or  rm -rf /var/folders/.../T/tfhub_modules
"""

import pytest

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None

try:
    import tensorflow_hub as hub
    HAS_TFHUB = True
except ImportError:
    HAS_TFHUB = False
    hub = None

try:
    import tensorflow_text  # noqa: F401 - registers CaseFoldUTF8 etc. for TF Hub BERT preprocessor
    HAS_TF_TEXT = True
except ImportError:
    HAS_TF_TEXT = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None

from multitask.config import EncoderConfig, EncoderInputType, EncoderIntegration, ModelConfig

pytestmark = [
    pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed"),
    pytest.mark.integration,
]

# Shared test config: 2 binary tasks
TASK_STRUCTURE = [[2]]
BATCH_SIZE = 2
SAMPLE_TEXTS = ["Hello world", "Another sentence for encoding."]


def _get_base_config():
    return ModelConfig(
        task_structure=TASK_STRUCTURE,
        shared_layer_sizes=[32],
        default_layer_sizes=[16]
    )


# --- TF Hub: BERT ---

@pytest.mark.skipif(not HAS_TFHUB, reason="tensorflow_hub not installed")
@pytest.mark.skipif(not HAS_TF_TEXT, reason="tensorflow_text required for BERT preprocessor (CaseFoldUTF8 op)")
class TestTfHubBertIntegration:
    """TF Hub BERT encoder: build model, preprocess with TF Hub preprocessor, run forward."""

    def test_tfhub_bert_uncased_forward(self):
        from multitask.models.hierarchical import MultiTaskModel
        config = _get_base_config()
        encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
        preprocessor_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

        model = MultiTaskModel(config=config, encoder=encoder_url)
        assert model.encoder_type == "tfhub"
        assert model.uses_encoder is True

        preprocessor = hub.load(preprocessor_url)
        inputs = preprocessor(SAMPLE_TEXTS)

        out = model(inputs, training=False)
        assert isinstance(out, dict)
        assert len(out) == 2
        assert out["task_0"].shape == (BATCH_SIZE, 1)
        assert out["task_1"].shape == (BATCH_SIZE, 1)


# --- TF Hub: Universal Sentence Encoder ---

@pytest.mark.skipif(not HAS_TFHUB, reason="tensorflow_hub not installed")
class TestTfHubUniversalSentenceEncoderIntegration:
    """TF Hub Universal Sentence Encoder: raw string input, run forward."""

    def test_tfhub_use_forward(self):
        from multitask.models.hierarchical import MultiTaskModel
        config = _get_base_config()
        encoder_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

        try:
            model = MultiTaskModel(config=config, encoder=encoder_url)
        except (ValueError, OSError) as e:
            err_msg = str(e).lower()
            if "saved_model.pb" in err_msg or "incompatible" in err_msg or "unknown type" in err_msg:
                pytest.skip(
                    "USE model failed to load (often due to corrupted/incompatible TF Hub cache). "
                    "Clear cache and re-run: rm -rf $TFHUB_CACHE_DIR or unset TFHUB_CACHE_DIR"
                )
            raise

        assert model.encoder_type == "tfhub"
        assert model.uses_encoder is True

        inputs = tf.constant(SAMPLE_TEXTS)
        out = model(inputs, training=False)
        assert isinstance(out, dict)
        assert len(out) == 2
        assert out["task_0"].shape == (BATCH_SIZE, 1)
        assert out["task_1"].shape == (BATCH_SIZE, 1)


# --- HuggingFace encoders (ENCODER_SUPPORT.md) ---

@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed (pip install multitask[huggingface])")
class TestHuggingFaceEncoderIntegration:
    """HuggingFace encoders: build model, tokenize with AutoTokenizer, run forward."""

    def _hf_forward(self, encoder_id: str, max_length: int = 128):
        from multitask.models.hierarchical import MultiTaskModel
        config = _get_base_config()
        model = MultiTaskModel(config=config, encoder=encoder_id)
        assert model.encoder_type == "huggingface"
        assert model.uses_encoder is True
        assert model.encoder_config.trainable_encoder is True

        # AutoTokenizer automatically selects, loads, and instantiates the 
        # correct tokenizer for a specific pre-trained model.
        tokenizer = AutoTokenizer.from_pretrained(encoder_id)
        inputs = tokenizer(
            SAMPLE_TEXTS,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="tf",
        )
        out = model(inputs=inputs, training=False)
        assert isinstance(out, dict)
        assert len(out) == 2
        assert out["task_0"].shape == (BATCH_SIZE, 1)
        assert out["task_1"].shape == (BATCH_SIZE, 1)

    def test_bert_base_uncased(self):
        self._hf_forward("bert-base-uncased")

    def test_distilbert_base_uncased(self):
        self._hf_forward("distilbert-base-uncased")

    def test_roberta_large(self):
        self._hf_forward("roberta-large", max_length=64)

    def test_deberta_v3_base(self):
        self._hf_forward("microsoft/deberta-v3-base")

    def test_bert_base_multilingual_cased(self):
        self._hf_forward("bert-base-multilingual-cased")
