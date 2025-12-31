"""
multitask - A flexible multi-task learning library for NLP

This library provides easy-to-use tools for training multi-task neural networks
with hierarchical task structures. It supports various encoders (TensorFlow Hub,
HuggingFace) and allows users to bring their own pre-computed embeddings.

Main components:
    - models: Multi-task model architectures
    - trainers: Training and evaluation logic
    - data: Data handling utilities
    - layers: Custom neural network layers
    - utils: Helper functions
"""

__version__ = "0.1.0"
__author__ = "Isaac D. Mehlhaff and Marco Morucci"

# Import public API components
from multitask.config import ModelConfig, TrainingConfig
from multitask.models import MultiTaskModel
from multitask.models.persistence import load_model
from multitask.trainers import Trainer

__all__ = [
    "__version__",
    "__author__",
    # Configuration
    "ModelConfig",
    "TrainingConfig",
    # Models
    "MultiTaskModel",
    "load_model",
    # Training
    "Trainer",
]
