"""
Device management utilities for multi-GPU and CPU environments.

This module handles device detection, setup, and optimization for
different hardware configurations.
"""

import torch
import torch.nn as nn
import platform
import psutil
import os
import sys
from typing import Tuple, Dict, Optional, List


def setup_device(force_cpu: bool = False, max_gpus: Optional[int] = None, 
                 verbose: bool = True) -> Tuple[torch.device, int]:
    """
    Setup and configure the compute device(s) for training.
    
    Args:
        force_cpu: Force CPU usage even if GPU is available
        max_gpus: Maximum number of GPUs to use (None = use all)
        verbose: Print device information
        
    Returns:
        Tuple of (device, gpu_count)
    """
    if force_cpu:
        device = torch.device("cpu")
        gpu_count = 0
        if verbose:
            print("  Forced CPU usage")
        return device, gpu_count
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        
        # Limit GPU count if specified
        if max_gpus is not None:
            gpu_count = min(gpu_count, max_gpus)
        
        # Use primary GPU
        device = torch.device("cuda:0")
        
        if verbose:
            print(f" CUDA available with {torch.cuda.device_count()} GPU(s)")
            if max_gpus is not None and max_gpus < torch.cuda.device_count():
                print(f"   Limited to {gpu_count} GPU(s) as requested")
            
            for i in range(min(gpu_count, torch.cuda.device_count())):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory = gpu_props.total_memory / (1024**3)
                compute_capability = f"{gpu_props.major}.{gpu_props.minor}"
                
                print(f"   GPU {i}: {gpu_name}")
                print(f"     Memory: {gpu_memory:.1f} GB")
                print(f"     Compute Capability: {compute_capability}")
            
            print(f" Using primary device: {device}")
            
    else:
        device = torch.device("cpu")
        gpu_count = 0
        if verbose:
            print("💻 CUDA not available, using CPU")
    
    return device, gpu_count


def get_system_info() -> Dict:
    """
    Get comprehensive system information.
    
    Returns:
        Dictionary containing system details
    """
    info = {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version.split()[0]
        },
        'cpu': {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'max_frequency': getattr(psutil.cpu_freq(), 'max', 'Unknown') if psutil.cpu_freq() else 'Unknown',
            'current_frequency': getattr(psutil.cpu_freq(), 'current', 'Unknown') if psutil.cpu_freq() else 'Unknown'
        },
        'memory': {
            'total_gb': psutil.virtual_memory().total / (1024**3),
            'available_gb': psutil.virtual_memory().available / (1024**3),
            'used_percent': psutil.virtual_memory().percent
        },
        'pytorch': {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None
        }
    }
    
    # Add GPU information if available
    if torch.cuda.is_available():
        info['gpu'] = {
            'count': torch.cuda.device_count(),
            'devices': []
        }
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info = {
                'index': i,
                'name': props.name,
                'memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multiprocessor_count
            }
            info['gpu']['devices'].append(gpu_info)
    
    return info


def print_system_info(info: Optional[Dict] = None):
    """Print formatted system information."""
    if info is None:
        info = get_system_info()
    
    print("💻 System Information:")
    print(f"   OS: {info['platform']['system']} {info['platform']['release']}")
    print(f"   Python: {info['platform']['python_version']}")
    print(f"   CPU: {info['cpu']['physical_cores']} cores ({info['cpu']['logical_cores']} logical)")
    print(f"   RAM: {info['memory']['total_gb']:.1f} GB total, {info['memory']['available_gb']:.1f} GB available")
    print(f"   PyTorch: {info['pytorch']['version']}")
    
    if info['pytorch']['cuda_available']:
        print(f"   CUDA: {info['pytorch']['cuda_version']}")
        print(f"   cuDNN: {info['pytorch']['cudnn_version']}")
        print(f"   GPUs: {info['gpu']['count']}")
        
        for gpu in info['gpu']['devices']:
            print(f"     GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
    else:
        print("   CUDA: Not available")


def optimize_for_device(model: nn.Module, device: torch.device, gpu_count: int) -> nn.Module:
    """
    Optimize model for the target device configuration.
    
    Args:
        model: PyTorch model to optimize
        device: Target device
        gpu_count: Number of GPUs available
        
    Returns:
        Optimized model
    """
    # Move model to device
    model = model.to(device)
    
    # Enable DataParallel for multi-GPU
    if gpu_count > 1 and device.type == 'cuda':
        print(f"🔄 Enabling DataParallel across {gpu_count} GPUs")
        model = nn.DataParallel(model)
        
        # Optimize for multi-GPU
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
    
    return model


def get_memory_info() -> Dict:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory information
    """
    info = {
        'system_memory': {
            'total_gb': psutil.virtual_memory().total / (1024**3),
            'used_gb': psutil.virtual_memory().used / (1024**3),
            'available_gb': psutil.virtual_memory().available / (1024**3),
            'percent_used': psutil.virtual_memory().percent
        }
    }
    
    if torch.cuda.is_available():
        info['gpu_memory'] = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            
            info['gpu_memory'][f'gpu_{i}'] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - reserved,
                'percent_used': (reserved / total) * 100 if total > 0 else 0
            }
    
    return info


def clear_gpu_cache():
    """Clear GPU memory cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_deterministic_behavior(seed: int = 42):
    """
    Set deterministic behavior for reproducible results.
    
    Args:
        seed: Random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    
    # For older PyTorch versions
    if hasattr(torch.backends.cudnn, 'deterministic'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimal_batch_size(base_batch_size: int, gpu_count: int, memory_factor: float = 1.0) -> int:
    """
    Calculate optimal batch size based on hardware configuration.
    
    Args:
        base_batch_size: Base batch size for single GPU/CPU
        gpu_count: Number of GPUs available
        memory_factor: Factor to adjust for available memory (1.0 = no adjustment)
        
    Returns:
        Optimal batch size
    """
    if gpu_count <= 1:
        return max(1, int(base_batch_size * memory_factor))
    
    # Scale with GPU count, but consider memory limitations
    optimal_size = base_batch_size * gpu_count * memory_factor
    
    # Ensure batch size is at least 1 and reasonable
    return max(1, int(optimal_size))


def monitor_resources(device: torch.device) -> Dict:
    """
    Monitor current resource usage.
    
    Args:
        device: Device being used
        
    Returns:
        Resource usage information
    """
    info = {
        'timestamp': torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None,
        'cpu_percent': psutil.cpu_percent(interval=None),
        'memory_percent': psutil.virtual_memory().percent
    }
    
    if device.type == 'cuda' and torch.cuda.is_available():
        # GPU utilization (approximate)
        try:
            info['gpu_memory_allocated'] = torch.cuda.memory_allocated(device.index) / (1024**3)
            info['gpu_memory_reserved'] = torch.cuda.memory_reserved(device.index) / (1024**3)
        except:
            info['gpu_memory_allocated'] = 0
            info['gpu_memory_reserved'] = 0
    
    return info


def suggest_optimization_settings(system_info: Dict) -> Dict:
    """
    Suggest optimization settings based on system configuration.
    
    Args:
        system_info: System information dictionary
        
    Returns:
        Suggested settings
    """
    suggestions = {
        'batch_size_multiplier': 1.0,
        'num_workers': 0,
        'pin_memory': False,
        'mixed_precision': False,
        'gradient_accumulation': 1
    }
    
    # Adjust based on available memory
    memory_gb = system_info['memory']['total_gb']
    if memory_gb >= 32:
        suggestions['batch_size_multiplier'] = 2.0
        suggestions['num_workers'] = min(4, system_info['cpu']['physical_cores'])
    elif memory_gb >= 16:
        suggestions['batch_size_multiplier'] = 1.5
        suggestions['num_workers'] = min(2, system_info['cpu']['physical_cores'])
    elif memory_gb < 8:
        suggestions['batch_size_multiplier'] = 0.5
        suggestions['gradient_accumulation'] = 2
    
    # GPU-specific optimizations
    if system_info['pytorch']['cuda_available']:
        suggestions['pin_memory'] = True
        
        # Check for modern GPU capabilities
        for gpu in system_info.get('gpu', {}).get('devices', []):
            if gpu['memory_gb'] >= 8:
                suggestions['mixed_precision'] = True
                break
    
    return suggestions


if __name__ == "__main__":
    # Test device utilities
    print("Testing device utilities...")
    
    # Setup device
    device, gpu_count = setup_device()
    
    # Get and print system info
    system_info = get_system_info()
    print_system_info(system_info)
    
    # Get memory info
    memory_info = get_memory_info()
    print(f"\nMemory Usage:")
    print(f"  System: {memory_info['system_memory']['used_gb']:.1f} / {memory_info['system_memory']['total_gb']:.1f} GB")
    
    if 'gpu_memory' in memory_info:
        for gpu_id, gpu_mem in memory_info['gpu_memory'].items():
            print(f"  {gpu_id.upper()}: {gpu_mem['allocated_gb']:.1f} / {gpu_mem['total_gb']:.1f} GB")
    
    # Optimization suggestions
    suggestions = suggest_optimization_settings(system_info)
    print(f"\nOptimization Suggestions:")
    for key, value in suggestions.items():
        print(f"  {key}: {value}")
    
    # Test optimal batch size calculation
    optimal_batch = get_optimal_batch_size(4, gpu_count)
    print(f"\nOptimal batch size: {optimal_batch} (base: 4, GPUs: {gpu_count})")
    
    print("\nDevice utilities test completed!")