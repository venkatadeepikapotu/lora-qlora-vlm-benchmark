"""
Utility functions for LoRA/QLoRA VLM Benchmark.

This module contains device management, metrics, and other utilities.
"""

from .device import (
    setup_device,
    get_system_info,
    print_system_info,
    optimize_for_device,
    get_memory_info,
    clear_gpu_cache,
    set_deterministic_behavior,
    get_optimal_batch_size,
    monitor_resources,
    suggest_optimization_settings
)

from .metrics import (
    save_results,
    print_results,
    create_comparison_table_text,
    create_efficiency_analysis_text,
    calculate_efficiency_score,
    save_training_curves_data,
    generate_summary_report
)

__all__ = [
    # Device utilities
    'setup_device',
    'get_system_info', 
    'print_system_info',
    'optimize_for_device',
    'get_memory_info',
    'clear_gpu_cache',
    'set_deterministic_behavior',
    'get_optimal_batch_size',
    'monitor_resources',
    'suggest_optimization_settings',
    
    # Metrics utilities
    'save_results',
    'print_results',
    'create_comparison_table_text',
    'create_efficiency_analysis_text', 
    'calculate_efficiency_score',
    'save_training_curves_data',
    'generate_summary_report'
]