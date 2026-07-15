"""Utilities for distributed training across different platforms."""

import os
import platform
import tensorflow as tf
from typing import Optional, Tuple


def get_distribution_strategy(
    strategy_type: str = 'auto',
    verbose: bool = True
) -> Tuple[tf.distribute.Strategy, str]:
    """Get appropriate distribution strategy for the current platform.
    
    Handles platform-specific quirks:
    - macOS: Metal backend on Apple Silicon, no multi-GPU on consumer hardware
    - Linux/Windows: CUDA multi-GPU support
    - Fallback: CPU-only training
    
    Args:
        strategy_type: Strategy to use. Options:
            - 'auto': Automatically detect best strategy for platform
            - 'mirrored': Force MirroredStrategy (multi-GPU)
            - 'one_device': Force single device (CPU or GPU)
            - 'cpu': Force CPU-only
        verbose: Whether to print strategy selection info
        
    Returns:
        Tuple of (strategy, device_info_string)
        
    Example:
        >>> strategy, info = get_distribution_strategy('auto')
        >>> print(info)  # "Using MirroredStrategy with 2 GPUs"
        >>> 
        >>> with strategy.scope():
        ...     model = MultiTaskModel(...)
        ...     trainer = Trainer(model, config, strategy=strategy)
    """
    system = platform.system()
    is_apple_silicon = system == 'Darwin' and platform.machine() == 'arm64'
    
    # Get available devices
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)
    
    # Determine strategy
    if strategy_type == 'cpu':
        strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
        info = "Using CPU-only training"
        
    elif strategy_type == 'one_device':
        if num_gpus > 0:
            strategy = tf.distribute.OneDeviceStrategy('/gpu:0')
            info = f"Using single GPU training on {system}"
        else:
            strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
            info = "Using single CPU training (no GPUs detected)"
            
    elif strategy_type == 'mirrored':
        if num_gpus < 2:
            if verbose:
                print(f"  Warning: Requested 'mirrored' strategy but only {num_gpus} GPU(s) available.")
                print("   Falling back to single device strategy.")
            if num_gpus == 1:
                strategy = tf.distribute.OneDeviceStrategy('/gpu:0')
                info = "Using single GPU (requested mirrored but only 1 GPU available)"
            else:
                strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
                info = "Using CPU (requested mirrored but no GPUs available)"
        else:
            strategy = tf.distribute.MirroredStrategy()
            info = f"Using MirroredStrategy with {num_gpus} GPUs"
            
    else:  # 'auto'
        if is_apple_silicon:
            # Apple Silicon: Metal backend, typically single GPU (M1/M2/M3)
            if num_gpus > 0:
                # Metal backend can be buggy with MirroredStrategy
                strategy = tf.distribute.OneDeviceStrategy('/gpu:0')
                info = f"Using Metal GPU on Apple Silicon (single device)"
                if verbose:
                    print("  Note: Apple Silicon detected. Using single GPU strategy.")
                    print("   Multi-GPU training not supported on consumer Apple Silicon.")
            else:
                strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
                info = "Using CPU on Apple Silicon (Metal not available)"
                
        elif num_gpus >= 2:
            # Multi-GPU available: use MirroredStrategy
            strategy = tf.distribute.MirroredStrategy()
            info = f"Using MirroredStrategy with {num_gpus} GPUs on {system}"
            
        elif num_gpus == 1:
            # Single GPU: use OneDeviceStrategy
            strategy = tf.distribute.OneDeviceStrategy('/gpu:0')
            info = f"Using single GPU on {system}"
            
        else:
            # No GPUs: CPU training
            strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
            info = f"Using CPU-only training on {system} (no GPUs detected)"
    
    if verbose:
        print("=" * 60)
        print("Distribution Strategy")
        print("=" * 60)
        print(f"Platform: {system} ({platform.machine()})")
        print(f"GPUs detected: {num_gpus}")
        if num_gpus > 0:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        print(f"Strategy: {info}")
        print("=" * 60 + "\n")
    
    return strategy, info


def setup_gpu_memory_growth():
    """Enable GPU memory growth to avoid allocating all GPU memory at once.
    
    This is especially important on systems with limited GPU memory or when
    running multiple models. Should be called before creating any TensorFlow
    models.
    
    Note: On Apple Silicon with Metal, this may not have an effect as Metal
    manages memory differently.
    
    Example:
        >>> from multitask.utils.distributed import setup_gpu_memory_growth
        >>> setup_gpu_memory_growth()
        >>> # Now create models...
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled GPU memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"  Could not enable GPU memory growth: {e}")
            print("   (This is normal if GPUs were already initialized)")


def check_distributed_compatibility() -> dict:
    """Check platform compatibility with distributed training features.
    
    Returns:
        Dict with compatibility information and recommendations
        
    Example:
        >>> info = check_distributed_compatibility()
        >>> if not info['multi_gpu_supported']:
        ...     print(info['recommendation'])
    """
    system = platform.system()
    is_apple_silicon = system == 'Darwin' and platform.machine() == 'arm64'
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)
    
    info = {
        'platform': system,
        'architecture': platform.machine(),
        'is_apple_silicon': is_apple_silicon,
        'num_gpus': num_gpus,
        'multi_gpu_supported': False,
        'single_gpu_supported': False,
        'cpu_only': False,
        'recommendation': '',
        'notes': []
    }
    
    if is_apple_silicon:
        info['notes'].append("Apple Silicon detected (Metal backend)")
        if num_gpus > 0:
            info['single_gpu_supported'] = True
            info['recommendation'] = "Use single GPU training with Metal backend"
            info['notes'].append("MirroredStrategy not recommended on Apple Silicon")
            info['notes'].append("Metal backend may have compatibility issues with some TF ops")
        else:
            info['cpu_only'] = True
            info['recommendation'] = "Metal GPU not detected, will use CPU training"
            
    elif num_gpus >= 2:
        info['multi_gpu_supported'] = True
        info['single_gpu_supported'] = True
        info['recommendation'] = f"Can use MirroredStrategy with {num_gpus} GPUs"
        
    elif num_gpus == 1:
        info['single_gpu_supported'] = True
        info['recommendation'] = "Use single GPU training"
        
    else:
        info['cpu_only'] = True
        info['recommendation'] = "No GPUs detected, will use CPU training"
        info['notes'].append("Consider using a GPU for faster training")
    
    return info


def print_device_info():
    """Print detailed information about available compute devices.
    
    Useful for debugging platform-specific issues.
    """
    print("\n" + "=" * 60)
    print("TensorFlow Device Information")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python version: {platform.python_version()}")
    
    # Check for GPU support
    print(f"\nBuilt with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # List all devices
    print("\nPhysical devices:")
    for device_type in ['CPU', 'GPU']:
        devices = tf.config.list_physical_devices(device_type)
        print(f"  {device_type}: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"    {i}: {device}")
    
    # Check compatibility
    print("\nDistributed training compatibility:")
    compat = check_distributed_compatibility()
    print(f"  Multi-GPU supported: {compat['multi_gpu_supported']}")
    print(f"  Single GPU supported: {compat['single_gpu_supported']}")
    print(f"  Recommendation: {compat['recommendation']}")
    if compat['notes']:
        print(f"  Notes:")
        for note in compat['notes']:
            print(f"    - {note}")
    
    print("=" * 60 + "\n")
