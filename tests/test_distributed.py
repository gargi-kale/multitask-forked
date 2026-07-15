"""Unit tests for distributed training utilities.

These tests verify the logic of platform detection and strategy selection
without requiring TensorFlow/GPUs to be available.
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import platform

_DISTRIBUTED_PY = Path(__file__).resolve().parent.parent / "multitask" / "utils" / "distributed.py"


class TestDistributedUtils(unittest.TestCase):
    """Test distributed training utility functions."""
    
    def setUp(self):
        """Setup mocks for TensorFlow."""
        # Preserve the real TF module (may already be imported at process start) so
        # tearDown can restore it rather than deleting it. Deleting and re-importing
        # TF after mock teardown causes Metal's C-level plugin to crash in the same
        # process on Apple Silicon.
        self._real_tf = sys.modules.get('tensorflow')

        # Mock tensorflow module
        self.tf_mock = MagicMock()
        self.tf_mock.__version__ = '2.15.0'
        self.tf_mock.test.is_built_with_cuda.return_value = False
        sys.modules['tensorflow'] = self.tf_mock
        
        # Import after mocking - import module directly to avoid package dependencies
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "distributed",
            str(_DISTRIBUTED_PY),
        )
        self.distributed = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.distributed)
        
    def tearDown(self):
        """Cleanup mocks."""
        if self._real_tf is not None:
            sys.modules['tensorflow'] = self._real_tf
        elif 'tensorflow' in sys.modules:
            del sys.modules['tensorflow']
        if 'multitask.utils.distributed' in sys.modules:
            del sys.modules['multitask.utils.distributed']
    
    @patch('platform.system')
    @patch('platform.machine')
    def test_check_compatibility_apple_silicon(self, mock_machine, mock_system):
        """Test compatibility check on Apple Silicon."""
        mock_system.return_value = 'Darwin'
        mock_machine.return_value = 'arm64'
        
        # Mock 1 GPU (Metal)
        gpu_mock = Mock()
        gpu_mock.name = '/physical_device:GPU:0'
        self.tf_mock.config.list_physical_devices.return_value = [gpu_mock]
        
        info = self.distributed.check_distributed_compatibility()
        
        self.assertEqual(info['platform'], 'Darwin')
        self.assertEqual(info['architecture'], 'arm64')
        self.assertTrue(info['is_apple_silicon'])
        self.assertEqual(info['num_gpus'], 1)
        self.assertFalse(info['multi_gpu_supported'])
        self.assertTrue(info['single_gpu_supported'])
        self.assertFalse(info['cpu_only'])
        self.assertIn('Metal', info['recommendation'])
        
    @patch('platform.system')
    @patch('platform.machine')
    def test_check_compatibility_linux_multi_gpu(self, mock_machine, mock_system):
        """Test compatibility check on Linux with multiple GPUs."""
        mock_system.return_value = 'Linux'
        mock_machine.return_value = 'x86_64'
        
        # Mock 2 GPUs
        gpu_mocks = [Mock(), Mock()]
        for i, gpu in enumerate(gpu_mocks):
            gpu.name = f'/physical_device:GPU:{i}'
        self.tf_mock.config.list_physical_devices.return_value = gpu_mocks
        
        info = self.distributed.check_distributed_compatibility()
        
        self.assertEqual(info['platform'], 'Linux')
        self.assertFalse(info['is_apple_silicon'])
        self.assertEqual(info['num_gpus'], 2)
        self.assertTrue(info['multi_gpu_supported'])
        self.assertTrue(info['single_gpu_supported'])
        self.assertFalse(info['cpu_only'])
        self.assertIn('MirroredStrategy', info['recommendation'])
        
    @patch('platform.system')
    @patch('platform.machine')
    def test_check_compatibility_cpu_only(self, mock_machine, mock_system):
        """Test compatibility check with no GPUs."""
        mock_system.return_value = 'Windows'
        mock_machine.return_value = 'AMD64'
        
        # Mock 0 GPUs
        self.tf_mock.config.list_physical_devices.return_value = []
        
        info = self.distributed.check_distributed_compatibility()
        
        self.assertEqual(info['platform'], 'Windows')
        self.assertEqual(info['num_gpus'], 0)
        self.assertFalse(info['multi_gpu_supported'])
        self.assertFalse(info['single_gpu_supported'])
        self.assertTrue(info['cpu_only'])
        self.assertIn('CPU', info['recommendation'])
        
    @patch('platform.system')
    @patch('platform.machine')
    def test_get_strategy_auto_apple_silicon(self, mock_machine, mock_system):
        """Test auto strategy selection on Apple Silicon."""
        mock_system.return_value = 'Darwin'
        mock_machine.return_value = 'arm64'
        
        # Mock 1 GPU
        gpu_mock = Mock()
        gpu_mock.name = '/physical_device:GPU:0'
        self.tf_mock.config.list_physical_devices.return_value = [gpu_mock]
        
        # Mock strategy
        strategy_mock = Mock()
        self.tf_mock.distribute.OneDeviceStrategy.return_value = strategy_mock
        
        strategy, info = self.distributed.get_distribution_strategy('auto', verbose=False)
        
        self.assertEqual(strategy, strategy_mock)
        self.assertIn('Metal', info)
        self.assertIn('Apple Silicon', info)
        self.tf_mock.distribute.OneDeviceStrategy.assert_called_once_with('/gpu:0')
        
    @patch('platform.system')
    def test_get_strategy_force_cpu(self, mock_system):
        """Test forcing CPU strategy."""
        mock_system.return_value = 'Linux'
        
        # Mock strategy
        strategy_mock = Mock()
        self.tf_mock.distribute.OneDeviceStrategy.return_value = strategy_mock
        
        strategy, info = self.distributed.get_distribution_strategy('cpu', verbose=False)
        
        self.assertEqual(strategy, strategy_mock)
        self.assertIn('CPU', info)
        self.tf_mock.distribute.OneDeviceStrategy.assert_called_once_with('/cpu:0')
        
    @patch('platform.system')
    @patch('platform.machine')
    def test_get_strategy_mirrored_with_multiple_gpus(self, mock_machine, mock_system):
        """Test MirroredStrategy with multiple GPUs."""
        mock_system.return_value = 'Linux'
        mock_machine.return_value = 'x86_64'
        
        # Mock 3 GPUs
        gpu_mocks = [Mock(), Mock(), Mock()]
        self.tf_mock.config.list_physical_devices.return_value = gpu_mocks
        
        # Mock strategy
        strategy_mock = Mock()
        self.tf_mock.distribute.MirroredStrategy.return_value = strategy_mock
        
        strategy, info = self.distributed.get_distribution_strategy('mirrored', verbose=False)
        
        self.assertEqual(strategy, strategy_mock)
        self.assertIn('MirroredStrategy', info)
        self.assertIn('3 GPUs', info)
        self.tf_mock.distribute.MirroredStrategy.assert_called_once()
        
    @patch('platform.system')
    def test_get_strategy_mirrored_fallback(self, mock_system):
        """Test MirroredStrategy fallback when insufficient GPUs."""
        mock_system.return_value = 'Linux'
        
        # Mock 0 GPUs
        self.tf_mock.config.list_physical_devices.return_value = []
        
        # Mock strategy
        strategy_mock = Mock()
        self.tf_mock.distribute.OneDeviceStrategy.return_value = strategy_mock
        
        strategy, info = self.distributed.get_distribution_strategy('mirrored', verbose=False)
        
        self.assertEqual(strategy, strategy_mock)
        self.assertIn('CPU', info)
        self.assertIn('no GPUs available', info)
        self.tf_mock.distribute.OneDeviceStrategy.assert_called_once_with('/cpu:0')
        
    def test_setup_gpu_memory_growth_with_gpus(self):
        """Test GPU memory growth setup."""
        # Mock 2 GPUs
        gpu_mocks = [Mock(), Mock()]
        self.tf_mock.config.list_physical_devices.return_value = gpu_mocks
        self.tf_mock.config.experimental.set_memory_growth = Mock()
        
        # Should not raise error
        self.distributed.setup_gpu_memory_growth()
        
        # Should be called for each GPU
        self.assertEqual(
            self.tf_mock.config.experimental.set_memory_growth.call_count,
            2
        )
        
    def test_setup_gpu_memory_growth_no_gpus(self):
        """Test GPU memory growth setup with no GPUs."""
        # Mock 0 GPUs
        self.tf_mock.config.list_physical_devices.return_value = []
        
        # Should not raise error (just does nothing)
        self.distributed.setup_gpu_memory_growth()
        
    def test_setup_gpu_memory_growth_runtime_error(self):
        """Test GPU memory growth setup handles runtime error gracefully."""
        # Mock 1 GPU
        gpu_mock = Mock()
        self.tf_mock.config.list_physical_devices.return_value = [gpu_mock]
        self.tf_mock.config.experimental.set_memory_growth.side_effect = RuntimeError("Already initialized")
        
        # Should not raise error (catches and prints warning)
        self.distributed.setup_gpu_memory_growth()


class TestDistributedIntegration(unittest.TestCase):
    """Integration tests for distributed training.
    
    These tests verify that the utilities work correctly with actual
    TensorFlow if it's installed, otherwise they're skipped.
    """
    
    def setUp(self):
        """Check if TensorFlow is available."""
        try:
            import tensorflow as tf
            self.tf = tf
            self.tf_available = True
        except ImportError:
            self.tf_available = False
            self.skipTest("TensorFlow not installed")
    
    def test_import_distributed_utils(self):
        """Test that distributed utils can be imported."""
        from multitask.utils.distributed import (
            get_distribution_strategy,
            setup_gpu_memory_growth,
            check_distributed_compatibility,
            print_device_info
        )
        
        # Should not raise
        self.assertTrue(callable(get_distribution_strategy))
        self.assertTrue(callable(setup_gpu_memory_growth))
        self.assertTrue(callable(check_distributed_compatibility))
        self.assertTrue(callable(print_device_info))
    
    def test_check_compatibility_real(self):
        """Test compatibility check with real TensorFlow."""
        from multitask.utils.distributed import check_distributed_compatibility
        
        info = check_distributed_compatibility()
        
        # Should return valid info
        self.assertIn('platform', info)
        self.assertIn('num_gpus', info)
        self.assertIn('recommendation', info)
        self.assertIsInstance(info['num_gpus'], int)
        self.assertGreaterEqual(info['num_gpus'], 0)
    
    def test_get_strategy_auto_real(self):
        """Test auto strategy selection with real TensorFlow."""
        from multitask.utils.distributed import get_distribution_strategy
        
        strategy, info = get_distribution_strategy('auto', verbose=False)
        
        # Should return valid strategy and info
        self.assertIsNotNone(strategy)
        self.assertIsInstance(info, str)
        self.assertGreater(len(info), 0)
    
    def test_get_strategy_cpu_real(self):
        """Test CPU strategy with real TensorFlow."""
        from multitask.utils.distributed import get_distribution_strategy
        
        strategy, info = get_distribution_strategy('cpu', verbose=False)
        
        self.assertIn("OneDeviceStrategy", type(strategy).__name__)
        self.assertIn('CPU', info)


if __name__ == '__main__':
    unittest.main()
