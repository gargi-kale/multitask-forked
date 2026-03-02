"""
Configuration and default hyperparameters

This module defines the default configuration for models, training, and
evaluation. Users can override these defaults by passing arguments to
the relevant functions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingConfig:
    """Default configuration for training multi-task models."""
    
    # Training hyperparameters
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay_rate: float = 0.01
    warmup_percent: int = 5
    
    # Model hyperparameters
    dropout_rate: float = 0.1
    
    # Sequence/data parameters
    max_sequence_length: int = 512
    
    # Distributed training
    num_workers: int = 1
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    
    # Optimizer settings
    optimizer_type: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    
    # optimizer hyperparameters
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    
    # Learning rate schedule settings
    use_lr_schedule: bool = True
    lr_schedule_type: str = 'polynomial_decay'  # 'polynomial_decay', 'exponential_decay', 'cosine_decay', 'constant'
    use_warmup: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in range [0, 1)")
        if self.max_sequence_length > 512:
            raise ValueError("max_sequence_length cannot exceed 512 for BERT-based models")


@dataclass
class ModelConfig:
    """Default configuration for multi-task model architecture.
    
    This configuration supports arbitrary hierarchical branching structures
    where each branch can have a different number of children.
    
    Task Structure Format:
        A list of lists, where each inner list specifies the number of children
        for each branch at that level.
        
        - Level 0: List with one element (number of initial branches from encoder)
        - Level N: List with one element per branch from level N-1
        
    Example branching structures:
        - [[2]]: 2 branches (minimum structure, 2 outputs)
        
        - [[3], [2, 5, 3]]: 
            * Level 0: 3 branches
            * Level 1: branch 0 - 2 children, branch 1 - 5 children, branch 2 - 3 children
            * Total outputs: 2 + 5 + 3 = 10
        
        - [[2], [3, 4], [1, 2, 1, 1, 1, 2]]:
            * Level 0: 2 branches
            * Level 1: branch 0 - 3 children, branch 1 - 4 children (7 total branches)
            * Level 2: each of 7 branches splits as [1, 2, 1, 1, 1, 2]
            * Outputs on each of 7 branches: 1 + 2 + 1 + 1 + 1 + 2 = 8
    
    Layer sizes can be specified per branch level, or use defaults for all levels.
    
    Output Pools: 
        Optional grouping of final task outputs into aggregate predictions.
        Pooling is applied at inference time only and training uses individual task
        losses. This is useful when multiple tasks should be summarized
        into a coarser prediction
    Format: Dict mapping pool name to pool configuration dict:
            {
                "pool_name": {
                    "tasks": ["task_a", "task_b", ...],
                    "strategy": "max",                    # pooling op: 'max', 'mean', 'min' (required)
                    "task_type": "binary",                # type of pooled output (default: 'binary')
                    "num_classes": 2,                     # classes for pooled output (default: 2)
                }
            }
    """
    # Model hyperparameters
    dropout_rate: float = 0.1
    
    # Task structure: nested list defining branching hierarchy
    # Format: [[n_initial], [n_children_0, n_children_1, ...], [...], ...]
    task_structure: List[List[int]] = field(default_factory=lambda: [[3], [2, 5, 3]])
    
    # Layer sizes for each branch level
    # If None, defaults are used. Can specify different sizes per level.
    # Format: List of layer size lists, one per branch level
    # e.g., [[256, 128], [128, 64], [64, 32]] for a 3-level hierarchy
    branch_layer_sizes: Optional[List[List[int]]] = None
    
    # Default layer sizes if branch_layer_sizes is not specified
    default_layer_sizes: List[int] = field(default_factory=lambda: [128, 64])
    
    # Optional shared encoder layers before branching
    shared_layer_sizes: Optional[List[int]] = field(default_factory=lambda: [256, 128])
    
    # Encoder settings
    trainable_encoder: bool = True
    
    # Pooling strategy for hierarchical outputs (if applicable)
    # Options: 'concat', 'max', 'mean', 'sum', None (no pooling)
    pooling_strategy: Optional[str] = None
    
    # Optional task names for model outputs
    # If provided, must be a flat list with length equal to total_outputs
    # These correspond to the final level outputs in order
    task_names: Optional[List[str]] = None
    
    # Task types: 'binary', 'multiclass', or 'multilabel'
    # If provided, must be a flat list with length equal to total_tasks (final level outputs)
    # Defaults to 'binary' for all tasks if not provided
    task_types: Optional[List[str]] = None
    
    # Number of classes per task (for multiclass tasks)
    # Only needed for 'multiclass' tasks. Binary tasks default to 2, multiclass needs explicit count.
    # Must be a list of length equal to total_tasks
    # Example: [2, 2, 5, 2, 2, 3, 2, 2, 2, 2] for 10 tasks (some binary, some multiclass)
    num_classes_per_task: Optional[List[int]] = None
    
    # Output pooling: group final task outputs into aggregate predictions.
    # Pooled outputs are derived at inference from individual task outputs.
    # They are NOT included in the training loss.
    output_pools: Optional[Dict[str, Dict[str, Any]]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate task structure
        if not self.task_structure or len(self.task_structure) == 0:
            raise ValueError("task_structure must contain at least one branching level")
        
        # Level 0 should have exactly one element
        if len(self.task_structure[0]) != 1:
            raise ValueError("First level of task_structure must have exactly one element (number of initial branches)")
        
        # Validate all values are positive
        for level_idx, level in enumerate(self.task_structure):
            if not all(isinstance(n, int) and n > 0 for n in level):
                raise ValueError(f"Level {level_idx} contains invalid values (must be positive integers)")
        
        # Validate consistency: each level must have the right number of elements
        for level_idx in range(1, len(self.task_structure)):
            prev_level = self.task_structure[level_idx - 1]
            curr_level = self.task_structure[level_idx]
            expected_length = sum(prev_level)  # Total branches from previous level
            
            if len(curr_level) != expected_length:
                raise ValueError(
                    f"Level {level_idx} has {len(curr_level)} elements, but should have "
                    f"{expected_length} (sum of level {level_idx - 1}: {prev_level})"
                )
        
        # Validate branch_layer_sizes if provided
        if self.branch_layer_sizes is not None:
            if len(self.branch_layer_sizes) != len(self.task_structure):
                raise ValueError(
                    f"branch_layer_sizes length ({len(self.branch_layer_sizes)}) must match "
                    f"task_structure length ({len(self.task_structure)})"
                )
            for i, layer_sizes in enumerate(self.branch_layer_sizes):
                if not layer_sizes or not all(s > 0 for s in layer_sizes):
                    raise ValueError(f"Invalid layer sizes at branch level {i}")
        
        # Validate shared layers if provided
        if self.shared_layer_sizes is not None:
            if not all(s > 0 for s in self.shared_layer_sizes):
                raise ValueError("All shared_layer_sizes must be positive")
        
        # Validate default layer sizes
        if not all(s > 0 for s in self.default_layer_sizes):
            raise ValueError("All default_layer_sizes must be positive")
        
        # Validate pooling strategy
        valid_pooling = {'concat', 'max', 'mean', 'sum', None}
        if self.pooling_strategy not in valid_pooling:
            raise ValueError(f"pooling_strategy must be one of {valid_pooling}")
        
        # Default num_classes_per_task to binary (2) per task when not provided
        if self.num_classes_per_task is None:
            self.num_classes_per_task = [2] * self.total_outputs
        
        # Validate task_types if provided
        if self.task_types is not None:
            if len(self.task_types) != self.total_outputs:
                raise ValueError(
                    f"task_types length ({len(self.task_types)}) must equal "
                    f"total_outputs ({self.total_outputs})"
                )
            valid_types = {'binary', 'multiclass', 'multilabel'}
            for task_type in self.task_types:
                if task_type not in valid_types:
                    raise ValueError(
                        f"Invalid task_type '{task_type}'. Must be one of {valid_types}"
                    )
        
        # Validate num_classes_per_task (always set after default above)
        if len(self.num_classes_per_task) != self.total_outputs:
            raise ValueError(
                f"num_classes_per_task length ({len(self.num_classes_per_task)}) must equal "
                f"total_outputs ({self.total_outputs})"
            )
        if not all(n >= 2 for n in self.num_classes_per_task):
            raise ValueError("All num_classes_per_task values must be >= 2")
        if self.task_types is not None:
            for task_idx, (task_type, num_classes) in enumerate(zip(self.task_types, self.num_classes_per_task)):
                if task_type == 'binary' and num_classes != 2:
                    raise ValueError(
                        f"Task {task_idx} is type 'binary' but num_classes_per_task[{task_idx}]={num_classes}. "
                        f"Binary tasks should have num_classes=2."
                    )
                elif task_type == 'multiclass' and num_classes < 3:
                    raise ValueError(
                        f"Task {task_idx} is type 'multiclass' but num_classes_per_task[{task_idx}]={num_classes}. "
                        f"Multiclass tasks should have num_classes >= 3."
                    )
        
        # Validate task_names if provided
        if self.task_names is not None:
            if len(self.task_names) != self.total_outputs:
                raise ValueError(
                    f"task_names length ({len(self.task_names)}) must equal "
                    f"total_outputs ({self.total_outputs})"
                )
                
        # validate output_pools if provided
        self._validate_output_pools()
    
    def _validate_output_pools(self):
        """Validate output_pools configuration.
        
        Checks:
            1. output_pools is a dict (if provided)
            2. Each pool has required 'tasks' (list[str]) and 'strategy' (str) fields
            3. Pool names don't collide with task_names
            4. All referenced task names exist in task_names
            5. A task cannot belong to multiple pools
            6. Strategy is one of: 'max', 'mean', 'min'
            7. Optional task_type is valid
            8. Optional num_classes is >= 2
            9. Each pool references at least 2 tasks
        """
        if self.output_pools is None:
            return
        
        if not isinstance(self.output_pools, dict):
            raise ValueError("output_pools must be a dict mapping pool names to pool configs")
        
        # task_names are required when output_pools is used, because we need
        # to verify that constituent task references actually exist.
        if self.task_names is None:
            raise ValueError(
                "task_names must be provided when using output_pools, "
                "so that constituent task references can be validated"
            )
        
        task_name_set = set(self.task_names)
        valid_strategies = {'max', 'mean', 'min'}
        valid_pool_types = {'binary', 'multiclass', 'multilabel'}
        
        # Track which tasks have already been assigned to a pool.
        # A task can only belong to one pool — otherwise the semantics of
        # "what does this pooled output mean" become ambiguous.
        assigned_tasks: Dict[str, str] = {}  # task_name -> pool_name
        
        for pool_name, pool_cfg in self.output_pools.items():
            # --- Basic type checks ---
            if not isinstance(pool_name, str) or not pool_name:
                raise ValueError("Pool names must be non-empty strings")
            
            if not isinstance(pool_cfg, dict):
                raise ValueError(
                    f"Pool '{pool_name}' config must be a dict, "
                    f"got {type(pool_cfg).__name__}"
                )
            
            # --- Pool name must not collide with any architectural task name ---
            if pool_name in task_name_set:
                raise ValueError(
                    f"Pool name '{pool_name}' conflicts with an existing task name. "
                    f"Pool names must be distinct from task names."
                )
            
            # --- Required fields ---
            if 'tasks' not in pool_cfg:
                raise ValueError(
                    f"Pool '{pool_name}' is missing required field 'tasks' "
                    f"(list of constituent task names)"
                )
            if 'strategy' not in pool_cfg:
                raise ValueError(
                    f"Pool '{pool_name}' is missing required field 'strategy' "
                    f"(pooling operation: {valid_strategies})"
                )
            
            # --- Validate 'tasks' field ---
            constituent_tasks = pool_cfg['tasks']
            
            if not isinstance(constituent_tasks, list) or len(constituent_tasks) < 2:
                raise ValueError(
                    f"Pool '{pool_name}': 'tasks' must be a list of at least 2 task names"
                )
            
            for task_name in constituent_tasks:
                if task_name not in task_name_set:
                    raise ValueError(
                        f"Pool '{pool_name}' references task '{task_name}' which is not "
                        f"in task_names. Available tasks: {sorted(task_name_set)}"
                    )
                
                if task_name in assigned_tasks:
                    raise ValueError(
                        f"Task '{task_name}' is referenced by pool '{pool_name}' but is "
                        f"already assigned to pool '{assigned_tasks[task_name]}'. "
                        f"A task cannot belong to multiple pools."
                    )
                assigned_tasks[task_name] = pool_name
            
            # --- Validate 'strategy' field ---
            strategy = pool_cfg['strategy']
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Pool '{pool_name}': strategy '{strategy}' is not supported. "
                    f"Must be one of {valid_strategies}"
                )
            
            # --- Validate optional 'task_type' field (default: 'binary') ---
            pool_task_type = pool_cfg.get('task_type', 'binary')
            if pool_task_type not in valid_pool_types:
                raise ValueError(
                    f"Pool '{pool_name}': task_type '{pool_task_type}' is not valid. "
                    f"Must be one of {valid_pool_types}"
                )
            
            # --- Validate optional 'num_classes' field (default: 2) ---
            pool_num_classes = pool_cfg.get('num_classes', 2)
            if not isinstance(pool_num_classes, int) or pool_num_classes < 2:
                raise ValueError(
                    f"Pool '{pool_name}': num_classes must be an integer >= 2, "
                    f"got {pool_num_classes}"
                )
            
            # --- Cross-validate task_type and num_classes ---
            if pool_task_type == 'binary' and pool_num_classes != 2:
                raise ValueError(
                    f"Pool '{pool_name}': task_type is 'binary' but num_classes={pool_num_classes}. "
                    f"Binary pools should have num_classes=2."
                )
            elif pool_task_type == 'multiclass' and pool_num_classes < 3:
                raise ValueError(
                    f"Pool '{pool_name}': task_type is 'multiclass' but num_classes={pool_num_classes}. "
                    f"Multiclass pools should have num_classes >= 3."
                )
    
    # output pool helpers
    @property
    def pool_names(self) -> List[str]:
        """Names of all configured output pools, sorted alphabetically.
        
        Returns empty list if no pools are configured.
        """
        if self.output_pools is None:
            return []
        return sorted(self.output_pools.keys())
    
    @property
    def all_output_names(self) -> List[str]:
        """All output names: architectural task names followed by pool names.
        
        Task names appear first (in their original order), then pool names
        (sorted alphabetically). This ordering is stable and predictable.
        """
        task_names = self.task_names or [f'task_{i}' for i in range(self.total_outputs)]
        return list(task_names) + self.pool_names
    
    @property
    def has_pools(self) -> bool:
        """Whether any output pools are configured."""
        return self.output_pools is not None and len(self.output_pools) > 0
    
    def get_pool_config(self, pool_name: str) -> Dict[str, Any]:
        """Get the full configuration for a specific pool, with defaults filled in.
        
        Args:
            pool_name: Name of the pool.
            
        Returns:
            Dict with keys: 'tasks', 'strategy', 'task_type', 'num_classes'.
            Optional fields are populated with their defaults so callers
            always receive a complete dict.
            
        Raises:
            KeyError: If pool_name is not a configured pool.
        """
        if self.output_pools is None or pool_name not in self.output_pools:
            raise KeyError(
                f"No pool named '{pool_name}'. "
                f"Available pools: {self.pool_names}"
            )
        
        raw = self.output_pools[pool_name]
        return {
            'tasks': raw['tasks'],
            'strategy': raw['strategy'],
            'task_type': raw.get('task_type', 'binary'),
            'num_classes': raw.get('num_classes', 2),
        }
    
    def get_pool_task_type(self, pool_name: str) -> str:
        """Get the task type of a pooled output ('binary', 'multiclass', etc.)."""
        return self.get_pool_config(pool_name)['task_type']
    
    def get_pool_num_classes(self, pool_name: str) -> int:
        """Get the number of classes for a pooled output."""
        return self.get_pool_config(pool_name)['num_classes']
    
    def is_pool(self, name: str) -> bool:
        """Check whether a name refers to a pooled output vs. an architectural task.
        
        Useful in code that iterates over all_output_names and needs to
        branch on whether an output is a real task or a derived pool.
        """
        if self.output_pools is None:
            return False
        return name in self.output_pools          
                
    # Architectural helpers
    def get_layer_sizes(self, branch_level: int) -> List[int]:
        """Get layer sizes for a specific branch level.
        
        Args:
            branch_level: The branching level (0-indexed)
            
        Returns:
            List of layer sizes for that level
        """
        if self.branch_layer_sizes is not None:
            return self.branch_layer_sizes[branch_level]
        return self.default_layer_sizes
    
    @property
    def num_branch_levels(self) -> int:
        """Total number of branching levels in the hierarchy."""
        return len(self.task_structure)
    
    @property
    def total_outputs(self) -> int:
        """Total number of output predictions (final branches at last level)."""
        return sum(self.task_structure[-1])
    
    def get_branches_at_level(self, level: int) -> int:
        """Get the total number of branches at a specific level.
        
        Args:
            level: The branching level (0-indexed)
            
        Returns:
            Total number of branches at that level
        """
        return sum(self.task_structure[level])    
    def get_task_type(self, task_idx: int) -> str:
        """Get the type of a specific task.
        
        Args:
            task_idx: The task index (0-indexed)
            
        Returns:
            Task type: 'binary', 'multiclass', or 'multilabel'
        """
        if self.task_types is None:
            return 'binary'
        return self.task_types[task_idx]
    
    def get_num_classes_per_task(self, task_idx: int) -> int:
        """Get the number of classes for a specific task.
        
        Args:
            task_idx: The task index (0-indexed)
            
        Returns:
            Number of classes (2 for binary, K for multiclass)
        """
        if self.num_classes_per_task is None:
            return 2  # Default to binary
        return self.num_classes_per_task[task_idx]
    
    def get_output_dim_for_task(self, task_idx: int) -> int:
        """Get the output dimension for a specific task.
        
        Binary tasks output 1 dimension (probability of class 1).
        Multiclass tasks output K dimensions (probability distribution).
        
        Args:
            task_idx: The task index (0-indexed)
            
        Returns:
            Output dimension for this task
        """
        task_type = self.get_task_type(task_idx)
        num_classes = self.get_num_classes_per_task(task_idx)
        
        if task_type == 'binary':
            return 1
        elif task_type == 'multiclass' or task_type == 'multilabel':
            return num_classes
        else:
            raise ValueError(f"Unknown task_type: {task_type}")