"""
Multi-task learning model with hierarchical branching.

This module implements a flexible hierarchical branching architecture for multi-task learning.
"""

import copy
import tensorflow as tf
from typing import Optional, Union, Dict, List, Any, Tuple
from multitask.config import ModelConfig


class MultiTaskModel(tf.keras.Model):
    """Multi-task learning model with flexible hierarchical branching.
    
    This model supports arbitrary hierarchical task structures where each branch
    can have a different number of children. The architecture consists of:
    
    1. Optional encoder (BERT, etc.) or pre-computed embeddings
    2. Optional shared layers (common to all tasks)
    3. Hierarchical branching layers (configurable depth and branching factor)
    4. Output layers (one per final task)
    
    Args:
        config: ModelConfig object defining the architecture
        encoder: Optional encoder (TensorFlow Hub URL or Keras layer)
        trainable_encoder: Whether encoder weights are trainable
        embedding_dim: If using embeddings, their dimension
        num_classes: Number of classes per task (default: 2 for binary)
        dropout: Dropout rate (overrides config if provided)
        name: Optional model name
    
    Example:
        >>> config = ModelConfig(
        ...     task_structure=[[3], [2, 5, 3]],  # 3 branches -> 2,5,3 children
        ...     shared_layer_sizes=[256, 128],
        ...     default_layer_sizes=[128, 64]
        ... )
        >>> model = MultiTaskModel(config=config, encoder='https://tfhub.dev/...')
    """
    
    def __init__(
        self,
        config: ModelConfig,
        encoder: Optional[Union[str, tf.keras.layers.Layer]] = None,
        trainable_encoder: bool = True,
        embedding_dim: Optional[int] = None,
        num_classes: int = 2,
        dropout: Optional[float] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        # Use dropout from config if not explicitly provided
        dropout = dropout if dropout is not None else 0.1
        
        super().__init__(name=name or "MultiTaskModel", **kwargs)
        
        self.config = config
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.task_names = config.task_names
        
        # Validate configuration
        self._validate_config()
        
        # Build encoder
        self._build_encoder(encoder, trainable_encoder, embedding_dim)
        
        # Build architecture
        self._build_shared_layers()
        self._build_branch_layers()
        self._build_output_layers()
    
    def _validate_config(self):
        """Validate model configuration."""
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout must be in range [0, 1)")
    
    def _build_encoder(
        self,
        encoder: Optional[Union[str, tf.keras.layers.Layer]] = None,
        trainable_encoder: bool = True,
        embedding_dim: Optional[int] = None
    ):
        """Build the encoder component.
        
        Args:
            encoder: Either a TensorFlow Hub URL, a Keras layer, or None for embedding input
            trainable_encoder: Whether encoder weights should be trainable
            embedding_dim: If using pre-computed embeddings, the dimension size
        """
        if encoder is not None:
            # Using a pre-trained encoder (e.g., BERT from TensorFlow Hub)
            if isinstance(encoder, str):
                import tensorflow_hub as hub
                self.encoder = hub.KerasLayer(encoder, trainable=trainable_encoder, name='encoder')
            else:
                self.encoder = encoder
            self.uses_encoder = True
            self.embedding_dim = None
        else:
            # Using pre-computed embeddings
            if embedding_dim is None:
                raise ValueError("Must specify embedding_dim when not using an encoder")
            self.encoder = None
            self.uses_encoder = False
            self.embedding_dim = embedding_dim
    
    def _encode_input(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Encode input through encoder or pass through embeddings.
        
        Args:
            inputs: Raw inputs (text tokens) or pre-computed embeddings
            training: Whether in training mode
            
        Returns:
            Encoded representation
        """
        if self.uses_encoder:
            encoder_output = self.encoder(inputs, training=training)
            # Handle different encoder output formats
            if isinstance(encoder_output, dict):
                return encoder_output.get('pooled_output', encoder_output.get('default', encoder_output))
            return encoder_output
        else:
            # Already embeddings, just pass through
            return inputs
    
    def _build_shared_layers(self):
        """Build optional shared layers before branching."""
        self.shared_layers = []
        
        if self.config.shared_layer_sizes is not None:
            for i, size in enumerate(self.config.shared_layer_sizes):
                self.shared_layers.append(
                    tf.keras.layers.Dense(size, activation='relu', name=f'shared_dense_{i}')
                )
                self.shared_layers.append(
                    tf.keras.layers.Dropout(self.dropout_rate, name=f'shared_dropout_{i}')
                )
    
    def _build_branch_layers(self):
        """Build hierarchical branching layers based on task_structure."""
        self.branch_layers = []
        
        # Build layers for each branching level
        for level_idx in range(self.config.num_branch_levels):
            level_branches = self.config.get_branches_at_level(level_idx)
            layer_sizes = self.config.get_layer_sizes(level_idx)
            
            # Create layers for each branch at this level
            branch_level_layers = []
            for branch_idx in range(level_branches):
                branch_layers = []
                for layer_idx, size in enumerate(layer_sizes):
                    branch_layers.append(
                        tf.keras.layers.Dense(
                            size, 
                            activation='relu',
                            name=f'branch_L{level_idx}_B{branch_idx}_dense_{layer_idx}'
                        )
                    )
                    branch_layers.append(
                        tf.keras.layers.Dropout(
                            self.dropout_rate,
                            name=f'branch_L{level_idx}_B{branch_idx}_dropout_{layer_idx}'
                        )
                    )
                branch_level_layers.append(branch_layers)
            
            self.branch_layers.append(branch_level_layers)
    
    def _build_output_layers(self):
        """Build output layers for final predictions."""
        num_outputs = self.config.total_outputs
        self.output_layers = [
            tf.keras.layers.Dense(
                1, 
                activation='sigmoid',
                name=f'output_{i}'
            )
            for i in range(num_outputs)
        ]
    
    def call(
        self, 
        inputs: Union[tf.Tensor, Dict[str, tf.Tensor]], 
        training: Optional[bool] = None
    ) -> tf.Tensor:
        """Forward pass through the hierarchical model.
        
        Args:
            inputs: Either preprocessed text tokens (if using encoder) or 
                    pre-computed embeddings
            training: Whether in training mode
            
        Returns:
            Concatenated predictions for all tasks, shape [batch_size, num_outputs]
        """
        # Encode input
        x = self._encode_input(inputs, training=training)
        
        # Apply shared layers
        for layer in self.shared_layers:
            x = layer(x, training=training)
        
        # Store representation from shared layers
        shared_rep = x
        
        # Process hierarchical branches
        branch_reps = [shared_rep]  # Start with shared representation
        
        for level_idx, level_layers in enumerate(self.branch_layers):
            next_level_reps = []
            
            # Determine which parent representations to use
            if level_idx == 0:
                # First level: all branches use shared representation
                num_parent_branches = 1
                parent_start_idx = 0
            else:
                # Subsequent levels: each branch uses its parent's representation
                num_parent_branches = self.config.get_branches_at_level(level_idx - 1)
                parent_start_idx = 0
            
            # Process each branch at this level
            branch_idx_in_level = 0
            for parent_idx in range(num_parent_branches):
                parent_rep = branch_reps[parent_start_idx + parent_idx] if level_idx > 0 else shared_rep
                num_children = self.config.task_structure[level_idx][parent_idx]
                
                # Create representation for each child branch
                for child_idx in range(num_children):
                    x = parent_rep
                    for layer in level_layers[branch_idx_in_level]:
                        x = layer(x, training=training)
                    next_level_reps.append(x)
                    branch_idx_in_level += 1
            
            branch_reps = next_level_reps
        
        # Apply output layers to final branch representations
        outputs = []
        for i, branch_rep in enumerate(branch_reps):
            output = self.output_layers[i](branch_rep, training=training)
            outputs.append(output)
        
        # Concatenate all outputs
        final_output = tf.keras.layers.Concatenate(axis=1)(outputs)
        
        return final_output
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'dropout': self.dropout_rate,
            'config': {
                'task_structure': self.config.task_structure,
                'shared_layer_sizes': self.config.shared_layer_sizes,
                'branch_layer_sizes': self.config.branch_layer_sizes,
                'default_layer_sizes': self.config.default_layer_sizes,
                'trainable_encoder': self.config.trainable_encoder,
                'pooling_strategy': self.config.pooling_strategy,
            },
            'uses_encoder': self.uses_encoder,
            'embedding_dim': self.embedding_dim,
        })
        return config
    
    def summary_tree(self) -> str:
        """Generate a human-readable summary of the branching structure.
        
        Returns:
            String representation of the hierarchical structure
        """
        lines = ["Hierarchical Multi-Task Model Structure:"]
        lines.append("=" * 50)
        
        if self.uses_encoder:
            lines.append("Encoder: TensorFlow Hub Layer")
        else:
            lines.append(f"Input: Pre-computed Embeddings (dim={self.embedding_dim})")
        
        if self.shared_layers:
            lines.append(f"\nShared Layers: {len(self.shared_layers) // 2} dense layers")
        
        lines.append("\nHierarchical Branching:")
        for level_idx, level_structure in enumerate(self.config.task_structure):
            indent = "  " * level_idx
            lines.append(f"{indent}Level {level_idx}: {level_structure}")
        
        lines.append(f"\nTotal Outputs: {self.config.total_outputs}")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def get_task_info(self) -> List[Dict[str, Any]]:
        """Get information about each task output.
        
        Returns a list where each element corresponds to a model output and contains:
        - 'output_idx': The index of this output in the model's output tensor
        - 'name': Task name if task_names provided, otherwise 'task_{idx}'
        - 'parent_branch_path': List of branch indices tracing from root to this output
        
        The parent_branch_path shows the hierarchical path through the branching structure.
        For example, [1, 2] means this output came from the 3rd child (index 2) of the 
        2nd parent branch (index 1).
        
        Returns:
            List of dictionaries with task information
            
        Example:
            >>> config = ModelConfig(
            ...     task_structure=[[3], [2, 5, 3]],
            ...     task_names=['P1', 'P2', 'I1', 'I2', 'I3', 'I4', 'I5', 'W1', 'W2', 'W3']
            ... )
            >>> model = MultiTaskModel(config)
            >>> info = model.get_task_info()
            >>> info[0]
            {'output_idx': 0, 'name': 'P1', 'parent_branch_path': [0, 0]}
            >>> info[7]
            {'output_idx': 7, 'name': 'W1', 'parent_branch_path': [2, 0]}
        """
        task_info = []
        output_idx = 0
        
        # Get the final level of the hierarchy (the one that produces outputs)
        final_level = self.config.task_structure[-1]
        
        # Trace back through the hierarchy to find parent paths
        # We need to work through the structure to map each output to its parent path
        
        if len(self.config.task_structure) == 1:
            # Simple case: only one level, each output is a direct branch
            for branch_idx in range(final_level[0]):
                task_info.append({
                    'output_idx': output_idx,
                    'name': self.task_names[output_idx] if self.task_names else f'task_{output_idx}',
                    'parent_branch_path': [branch_idx]
                })
                output_idx += 1
        else:
            # Multi-level hierarchy: trace through the structure
            # Build paths by traversing the hierarchy level by level
            parent_paths = self._build_parent_paths()
            
            for path in parent_paths:
                task_info.append({
                    'output_idx': output_idx,
                    'name': self.task_names[output_idx] if self.task_names else f'task_{output_idx}',
                    'parent_branch_path': path
                })
                output_idx += 1
        
        return task_info
    
    def _build_parent_paths(self) -> List[List[int]]:
        """Build the hierarchical paths for all outputs.
        
        Returns:
            List of paths, where each path is a list of branch indices from root to leaf
        """
        paths = []
        
        # Start with paths at level 0 (initial branches)
        num_initial_branches = self.config.task_structure[0][0]
        current_paths = [[i] for i in range(num_initial_branches)]
        
        # Traverse through each level
        for level_idx in range(1, len(self.config.task_structure)):
            next_paths = []
            level_structure = self.config.task_structure[level_idx]
            
            # For each parent path, add children according to level_structure
            structure_idx = 0
            for parent_path in current_paths:
                num_children = level_structure[structure_idx]
                
                # Create a path for each child
                for child_idx in range(num_children):
                    child_path = parent_path + [child_idx]
                    next_paths.append(child_path)
                
                structure_idx += 1
            
            current_paths = next_paths
        
        return current_paths
