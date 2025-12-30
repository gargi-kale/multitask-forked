"""
Multi-task model architectures

This module contains the core model classes for multi-task learning.
Models can be configured with different task hierarchies, encoders, and
architectural parameters.
"""

from multitask.models.hierarchical import MultiTaskModel

__all__ = [
    'MultiTaskModel',
]
