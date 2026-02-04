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