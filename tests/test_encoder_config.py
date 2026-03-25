"""Tests for EncoderConfig and ModelConfig encoder_config integration."""

import pytest

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    tf = None
    HAS_TF = False

from multitask.config import (
    EncoderConfig,
    EncoderInputType,
    EncoderIntegration,
    ModelConfig,
)


# --- EncoderConfig: construction and defaults ---


class TestEncoderConfigDefaults:
    """EncoderConfig with default / minimal arguments."""

    def test_default_construction(self):
        """Default EncoderConfig has no encoder, no identifier, trainable=True."""
        cfg = EncoderConfig()
        assert cfg.encoder is None
        assert cfg.encoder_identifier is None
        assert cfg.encoder_input_type is None
        assert cfg.encoder_integration is None
        assert cfg.trainable_encoder is True
        assert cfg.api_key is None

    def test_precomputed_style(self):
        """EncoderConfig with encoder_identifier=None represents pre-computed embeddings path."""
        cfg = EncoderConfig(encoder_identifier=None)
        assert cfg.encoder_identifier is None
        assert cfg.encoder is None

    def test_with_encoder_identifier_only(self):
        """EncoderConfig with only encoder_identifier (e.g. HuggingFace name)."""
        cfg = EncoderConfig(encoder_identifier="bert-base-uncased")
        assert cfg.encoder_identifier == "bert-base-uncased"
        assert cfg.encoder is None
        assert cfg.trainable_encoder is True

    def test_with_tfhub_identifier(self):
        """EncoderConfig with TF Hub URL."""
        url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
        cfg = EncoderConfig(encoder_identifier=url)
        assert cfg.encoder_identifier == url

    def test_with_prefixed_identifier(self):
        """EncoderConfig with integration prefix (e.g. openai:model-name)."""
        cfg = EncoderConfig(encoder_identifier="openai:text-embedding-3-small")
        assert cfg.encoder_identifier == "openai:text-embedding-3-small"

    def test_trainable_encoder_false(self):
        """EncoderConfig with trainable_encoder=False."""
        cfg = EncoderConfig(encoder_identifier="bert-base-uncased", trainable_encoder=False)
        assert cfg.trainable_encoder is False

    def test_api_key(self):
        """EncoderConfig stores api_key and get_api_key returns it."""
        cfg = EncoderConfig(encoder_identifier="openai:text-embedding-3-small", api_key="sk-fake")
        assert cfg.api_key == "sk-fake"
        assert cfg.get_api_key() == "sk-fake"

    def test_get_api_key_none(self):
        """get_api_key returns None when api_key not set."""
        cfg = EncoderConfig()
        assert cfg.get_api_key() is None


# --- EncoderConfig: to_dict_for_save / from_dict ---


class TestEncoderConfigPersistence:
    """EncoderConfig serialization for save/load."""

    def test_to_dict_for_save_keys(self):
        """to_dict_for_save returns expected keys and does not include encoder or api_key."""
        cfg = EncoderConfig(
            encoder_identifier="bert-base-uncased",
            encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
            encoder_integration=EncoderIntegration.HUGGINGFACE,
            trainable_encoder=False,
            api_key="sk-secret",
        )
        d = cfg.to_dict_for_save()
        assert "encoder_identifier" in d
        assert "encoder_input_type" in d
        assert "encoder_integration" in d
        assert "trainable_encoder" in d
        assert d["encoder_identifier"] == "bert-base-uncased"
        assert d["encoder_input_type"] == EncoderInputType.HUGGINGFACE_TOKENS.value
        assert d["encoder_integration"] == EncoderIntegration.HUGGINGFACE.value
        assert d["trainable_encoder"] is False
        assert "encoder" not in d
        assert "api_key" not in d

    def test_to_dict_for_save_none_values(self):
        """to_dict_for_save with all None produces serializable dict."""
        cfg = EncoderConfig()
        d = cfg.to_dict_for_save()
        assert d["encoder_identifier"] is None
        assert d["encoder_input_type"] is None
        assert d["encoder_integration"] is None
        assert d["trainable_encoder"] is True

    def test_from_dict_empty(self):
        """from_dict with empty dict gives default-like EncoderConfig."""
        cfg = EncoderConfig.from_dict({})
        assert cfg.encoder is None
        assert cfg.encoder_identifier is None
        assert cfg.encoder_input_type is None
        assert cfg.encoder_integration is None
        assert cfg.trainable_encoder is True
        assert cfg.api_key is None

    def test_from_dict_with_string_values(self):
        """from_dict accepts string values (e.g. from JSON) and converts enums."""
        d = {
            "encoder_identifier": "bert-base-uncased",
            "encoder_input_type": "huggingface_tokens",
            "encoder_integration": "huggingface",
            "trainable_encoder": False,
        }
        cfg = EncoderConfig.from_dict(d)
        assert cfg.encoder_identifier == "bert-base-uncased"
        assert cfg.encoder_input_type == EncoderInputType.HUGGINGFACE_TOKENS
        assert cfg.encoder_integration == EncoderIntegration.HUGGINGFACE
        assert cfg.trainable_encoder is False
        assert cfg.encoder is None
        assert cfg.api_key is None

    def test_from_dict_roundtrip(self):
        """to_dict_for_save then from_dict preserves identifier and trainable_encoder."""
        original = EncoderConfig(
            encoder_identifier="https://tfhub.dev/google/universal-sentence-encoder/4",
            encoder_input_type=EncoderInputType.RAW_STRING,
            encoder_integration=EncoderIntegration.TFHUB,
            trainable_encoder=False,
        )
        d = original.to_dict_for_save()
        restored = EncoderConfig.from_dict(d)
        assert restored.encoder_identifier == original.encoder_identifier
        assert restored.encoder_input_type == original.encoder_input_type
        assert restored.encoder_integration == original.encoder_integration
        assert restored.trainable_encoder == original.trainable_encoder
        assert restored.encoder is None
        assert restored.api_key is None

    def test_from_dict_with_enum_values(self):
        """from_dict accepts existing enum values (e.g. from asdict) without re-converting."""
        d = {
            "encoder_identifier": "cohere:embed-english-v3",
            "encoder_input_type": EncoderInputType.RAW_STRING,
            "encoder_integration": EncoderIntegration.COHERE,
            "trainable_encoder": True,
        }
        cfg = EncoderConfig.from_dict(d)
        assert cfg.encoder_input_type == EncoderInputType.RAW_STRING
        assert cfg.encoder_integration == EncoderIntegration.COHERE
        # API integrations (e.g. COHERE) force trainable_encoder=False in __post_init__
        assert cfg.trainable_encoder is False


# --- EncoderConfig: validation ---


class TestEncoderConfigValidation:
    """EncoderConfig __post_init__ validation."""

    def test_api_integration_forces_trainable_false(self):
        """When encoder_integration is an API (e.g. OPENAI), trainable_encoder becomes False."""
        cfg = EncoderConfig(
            encoder_identifier="openai:text-embedding-3-small",
            encoder_integration=EncoderIntegration.OPENAI,
            trainable_encoder=True,
        )
        assert cfg.trainable_encoder is False

    def test_tfhub_keeps_trainable_true(self):
        """TF Hub integration does not force trainable_encoder to False."""
        cfg = EncoderConfig(
            encoder_identifier="https://tfhub.dev/...",
            encoder_integration=EncoderIntegration.TFHUB,
            trainable_encoder=True,
        )
        assert cfg.trainable_encoder is True

    def test_huggingface_keeps_trainable_true(self):
        """HuggingFace integration does not force trainable_encoder to False."""
        cfg = EncoderConfig(
            encoder_identifier="bert-base-uncased",
            encoder_integration=EncoderIntegration.HUGGINGFACE,
            trainable_encoder=True,
        )
        assert cfg.trainable_encoder is True

    @pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
    def test_custom_encoder_requires_encoder_input_type(self):
        """Custom encoder (Keras layer) without encoder_input_type raises."""
        layer = tf.keras.layers.Dense(64)
        with pytest.raises(ValueError, match="encoder_input_type"):
            EncoderConfig(encoder=layer, encoder_input_type=None)

    @pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
    def test_custom_encoder_with_encoder_input_type_ok(self):
        """Custom encoder with encoder_input_type set is valid."""
        layer = tf.keras.layers.Dense(64)
        cfg = EncoderConfig(
            encoder=layer,
            encoder_input_type=EncoderInputType.RAW_STRING,
        )
        assert cfg.encoder is layer
        assert cfg.encoder_input_type == EncoderInputType.RAW_STRING


# --- ModelConfig + EncoderConfig integration ---


class TestModelConfigEncoderConfig:
    """ModelConfig building or accepting encoder_config."""

    def test_model_config_builds_encoder_config_from_flat_kwargs(self):
        """When encoder_config is None, ModelConfig builds it from flat encoder kwargs."""
        config = ModelConfig(
            task_structure=[[2]],
            encoder_identifier="bert-base-uncased",
            trainable_encoder=False,
        )
        assert config.encoder_config is not None
        assert config.encoder_config.encoder_identifier == "bert-base-uncased"
        assert config.encoder_config.trainable_encoder is False

    def test_model_config_with_explicit_encoder_config(self):
        """ModelConfig accepts explicit encoder_config and uses it."""
        enc_cfg = EncoderConfig(
            encoder_identifier="openai:text-embedding-3-small",
            trainable_encoder=False,
        )
        config = ModelConfig(task_structure=[[2]], encoder_config=enc_cfg)
        assert config.encoder_config is enc_cfg
        assert config.encoder_config.encoder_identifier == "openai:text-embedding-3-small"

    def test_model_config_no_encoder_flat(self):
        """ModelConfig with no encoder args builds encoder_config with all None."""
        config = ModelConfig(task_structure=[[2]])
        assert config.encoder_config is not None
        assert config.encoder_config.encoder is None
        assert config.encoder_config.encoder_identifier is None
        assert config.encoder_config.trainable_encoder is True

    def test_model_config_from_flat_dict_roundtrip(self):
        """ModelConfig(**dict) with flat encoder keys builds valid encoder_config."""
        d = {
            "task_structure": [[2]],
            "encoder_identifier": "https://tfhub.dev/google/universal-sentence-encoder/4",
            "encoder_input_type": "raw_string",
            "encoder_integration": "tfhub",
            "trainable_encoder": False,
        }
        config = ModelConfig(**d)
        assert config.encoder_config.encoder_identifier == d["encoder_identifier"]
        assert config.encoder_config.encoder_input_type == EncoderInputType.RAW_STRING
        assert config.encoder_config.encoder_integration == EncoderIntegration.TFHUB
        assert config.encoder_config.trainable_encoder is False

    def test_model_config_explicit_encoder_config_overrides_flat(self):
        """When both encoder_config and flat encoder kwargs are passed, encoder_config is used."""
        enc_cfg = EncoderConfig(
            encoder_identifier="openai:text-embedding-3-small",
            trainable_encoder=False,
        )
        config = ModelConfig(
            task_structure=[[2]],
            encoder_config=enc_cfg,
            encoder_identifier="bert-base-uncased",
            trainable_encoder=True,
        )
        assert config.encoder_config is enc_cfg
        assert config.encoder_config.encoder_identifier == "openai:text-embedding-3-small"
        assert config.encoder_config.trainable_encoder is False
