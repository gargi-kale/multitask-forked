"""
multitask - A flexible multi-task learning library for NLP

This library provides easy-to-use tools for training multi-task neural networks
with hierarchical task structures. It supports various encoders (TensorFlow Hub,
HuggingFace) and allows users to bring their own pre-computed embeddings.

Main components:
    - models: Multi-task model architectures
    - trainers: Training and evaluation logic
    - layers: Custom neural network layers
    - utils: Helper functions
"""

__version__ = "1.1.0"
__author__ = "Isaac D. Mehlhaff, Marco Morucci, Josephene Ginting, and Eddie Ye Tian"

# Import public API components
from multitask.config import ModelConfig, TrainingConfig, EncoderConfig
from multitask.models import MultiTaskModel
from multitask.models.persistence import load_model
from multitask.trainers import Trainer
from multitask.utils.distributed import (
    get_distribution_strategy,
    setup_gpu_memory_growth,
    check_distributed_compatibility,
    print_device_info,
)

__all__ = [
    "__version__",
    "__author__",
    # Configuration
    "ModelConfig",
    "TrainingConfig",
    "EncoderConfig",
    # Models
    "MultiTaskModel",
    "load_model",
    # Training
    "Trainer",
    # Distributed training
    "get_distribution_strategy",
    "setup_gpu_memory_growth",
    "check_distributed_compatibility",
    "print_device_info",
]
