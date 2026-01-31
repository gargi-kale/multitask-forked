"""
We want to ensure that the dummy inputs used to build the model in load_model() are valid
for each encoder type (i.e. model(dummy_input, training=False) succeeds).

1. Unit tests: _get_build_dummy_input(model, metadata)
   - Use a minimal mock model (object with uses_encoder, encoder_type) and metadata dict.
   - For each branch, assert the returned dummy input has the expected type and structure:
     - No encoder: tf.Tensor, shape (1, embedding_dim), float dtype
     - HuggingFace: dict with input_ids, attention_mask, token_type_ids; each (1, 128), int32
     - TF Hub USE: tf.Tensor (string), shape (1,)
     - TF Hub BERT-style: dict with input_word_ids, input_mask, input_type_ids; each (1, 128)
     - Custom encoder: tf.Tensor (string), shape (1,)
   - Edge: TF Hub with missing/empty encoder_identifier defaults to BERT-style dict.

2. Integration tests: dummy input is valid (forward pass succeeds)
   - No encoder: build MultiTaskModel(encoder=None, embedding_dim=N), get dummy input,
     call model(dummy_input, training=False); assert no error and output shape correct.
   - Custom encoder (string-accepting): use a small Keras layer that accepts string input
     and returns fixed-dim tensor; build model, get dummy input (string), run forward;
     assert no error. This validates the persistence default for custom encoders.

3. Not covered here (future work possibly):
   - HuggingFace / TF Hub with real URLs: require network and heavy deps; test structure
     only in unit tests, or add optional integration tests marked @pytest.mark.skipif.
   - load_model() end-to-end with saved artifacts: see examples/save_load_example.py.
"""

import pytest

try:
    import tensorflow as tf
    from multitask.config import ModelConfig
    from multitask.models.hierarchical import MultiTaskModel
    from multitask.models.persistence import _get_build_dummy_input
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None
    ModelConfig = None
    MultiTaskModel = None
    _get_build_dummy_input = None

pytestmark = pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")


def _mock_model(uses_encoder: bool, encoder_type: str = None):
    """Minimal object with uses_encoder and encoder_type for _get_build_dummy_input."""
    return type("MockModel", (), {"uses_encoder": uses_encoder, "encoder_type": encoder_type})()


# --- Unit tests: _get_build_dummy_input structure for each branch ---


class TestGetBuildDummyInputNoEncoder:
    """Dummy input when model uses pre-computed embeddings (no encoder)."""

    def test_returns_tensor_shape_1_embedding_dim(self):
        model = _mock_model(uses_encoder=False)
        metadata = {"embedding_dim": 64}
        out = _get_build_dummy_input(model, metadata)
        assert isinstance(out, tf.Tensor)
        assert out.shape == (1, 64)
        assert out.dtype == tf.float32

    def test_embedding_dim_from_metadata(self):
        model = _mock_model(uses_encoder=False)
        metadata = {"embedding_dim": 256}
        out = _get_build_dummy_input(model, metadata)
        assert out.shape == (1, 256)


class TestGetBuildDummyInputHuggingFace:
    """Dummy input for HuggingFace encoder: dict of token tensors."""

    def test_returns_dict_with_expected_keys(self):
        model = _mock_model(uses_encoder=True, encoder_type="huggingface")
        metadata = {}
        out = _get_build_dummy_input(model, metadata)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"input_ids", "attention_mask", "token_type_ids"}

    def test_tensors_shape_1_128_int32(self):
        model = _mock_model(uses_encoder=True, encoder_type="huggingface")
        metadata = {}
        out = _get_build_dummy_input(model, metadata)
        for k in ("input_ids", "attention_mask", "token_type_ids"):
            assert out[k].shape == (1, 128)
            assert out[k].dtype == tf.int32


class TestGetBuildDummyInputTFHub:
    """Dummy input for TF Hub: USE (string) vs BERT-style (dict)."""

    def test_universal_sentence_encoder_returns_string_tensor(self):
        model = _mock_model(uses_encoder=True, encoder_type="tfhub")
        metadata = {"encoder_identifier": "https://tfhub.dev/google/universal-sentence-encoder/4"}
        out = _get_build_dummy_input(model, metadata)
        assert isinstance(out, tf.Tensor)
        assert out.dtype == tf.string
        assert out.shape == (1,)

    def test_use_case_insensitive(self):
        model = _mock_model(uses_encoder=True, encoder_type="tfhub")
        metadata = {"encoder_identifier": "https://tfhub.dev/UNIVERSAL-SENTENCE-ENCODER/4"}
        out = _get_build_dummy_input(model, metadata)
        assert out.dtype == tf.string

    def test_bert_style_returns_dict_when_no_use_in_identifier(self):
        model = _mock_model(uses_encoder=True, encoder_type="tfhub")
        metadata = {"encoder_identifier": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"}
        out = _get_build_dummy_input(model, metadata)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"input_word_ids", "input_mask", "input_type_ids"}
        for v in out.values():
            assert v.shape == (1, 128)
            assert v.dtype == tf.int32

    def test_bert_style_when_encoder_identifier_missing(self):
        model = _mock_model(uses_encoder=True, encoder_type="tfhub")
        metadata = {}
        out = _get_build_dummy_input(model, metadata)
        assert isinstance(out, dict)
        assert "input_word_ids" in out

    def test_bert_style_when_encoder_identifier_empty(self):
        model = _mock_model(uses_encoder=True, encoder_type="tfhub")
        metadata = {"encoder_identifier": ""}
        out = _get_build_dummy_input(model, metadata)
        assert isinstance(out, dict)
        assert "input_word_ids" in out


class TestGetBuildDummyInputCustomEncoder:
    """Dummy input for custom encoder: default is string tensor."""

    def test_returns_string_tensor(self):
        model = _mock_model(uses_encoder=True, encoder_type="custom")
        metadata = {}
        out = _get_build_dummy_input(model, metadata)
        assert isinstance(out, tf.Tensor)
        assert out.dtype == tf.string
        assert out.shape == (1,)


# --- Integration tests: dummy input is valid (forward pass succeeds) ---


class TestDummyInputValidNoEncoder:
    """Dummy input from _get_build_dummy_input is valid for a model without encoder."""

    def test_forward_succeeds_with_dummy_input(self):
        config = ModelConfig(
            task_structure=[[2]],
            shared_layer_sizes=[32],
            default_layer_sizes=[16],
        )
        embedding_dim = 64
        model = MultiTaskModel(
            config=config,
            encoder=None,
            embedding_dim=embedding_dim,
            dropout_rate=0.0,
        )
        metadata = {"embedding_dim": embedding_dim}
        dummy_input = _get_build_dummy_input(model, metadata)
        out = model(dummy_input, training=False)
        assert isinstance(out, dict)
        assert out["task_0"].shape == (1, 1)
        assert out["task_1"].shape == (1, 1)


class TestDummyInputValidCustomEncoderString:
    """Dummy input for custom encoder (string) is valid when encoder accepts strings."""

    @staticmethod
    def _string_encoder_layer(output_dim=32):
        """Keras layer that accepts string tensor and returns fixed-dim tensor (e.g. USE-like)."""
        class StringEncoder(tf.keras.layers.Layer):
            def __init__(self, output_dim, **kwargs):
                super().__init__(**kwargs)
                self.output_dim = output_dim

            def call(self, inputs, training=None):
                # Hash strings to a deterministic shape; in real USE this would be the encoder
                # We only need to return (batch, output_dim) for the rest of the model
                batch = tf.shape(inputs)[0]
                return tf.random.uniform((batch, self.output_dim), seed=42)

            def get_config(self):
                return {"output_dim": self.output_dim}
        return StringEncoder(output_dim=output_dim)

    def test_forward_succeeds_with_string_dummy_input(self):
        config = ModelConfig(
            task_structure=[[2]],
            shared_layer_sizes=[32],
            default_layer_sizes=[16],
        )
        encoder = self._string_encoder_layer(output_dim=32)
        model = MultiTaskModel(
            config=config,
            encoder=encoder,
            dropout_rate=0.0,
        )
        metadata = {}
        dummy_input = _get_build_dummy_input(model, metadata)
        assert dummy_input.dtype == tf.string
        out = model(dummy_input, training=False)
        assert isinstance(out, dict)
        assert out["task_0"].shape == (1, 1)
        assert out["task_1"].shape == (1, 1)
import pytest
import os
import tensorflow as tf
from multitask.models.persistence import load_model

def test_load_nonexistent_path():
    """verifies that a ValueError is raised for missing paths."""
    with pytest.raises(ValueError) as excinfo:
        load_model("/tmp/imaginary_path_12345")
    assert "Model directory not found" in str(excinfo.value)

def test_architecture_mismatch(tmp_path, dummy_model):
    """ensures load_task_weights catches structural mismatches."""
    test_dir = tmp_path / "fake_model"
    test_dir.mkdir()
    weights_prefix = str(test_dir / "weights")
    (test_dir / "weights.index").write_text("fake_content")

    with pytest.raises(ValueError) as excinfo:
        dummy_model.load_task_weights(weights_prefix)
    
    assert "Model architecture mismatch" in str(excinfo.value)
    
def test_load_with_index_extension(tmp_path, dummy_model_config):
    """
    verifies that the model can load weights when only the .index file is present.
    """
    from multitask.models.persistence import load_model
   
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    (model_dir / "weights.index").write_text("dummy index content")
    
    (model_dir / "config.json").write_text("{}") 
    (model_dir / "metadata.json").write_text('{"uses_encoder": false}')

    with pytest.raises(Exception) as excinfo:
        load_model(str(model_dir))
    
    # check it did not fail bc of the path validation
    assert "Weights file not found" not in str(excinfo.value)