"""
LoRA/QLoRA VLM Benchmark Package

A comprehensive benchmark comparing LoRA, QLoRA, and regular fine-tuning
for Vision-Language Models.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@university.edu"

from .models import SimpleVLM, LoRAVLM, QLoRAVLM
from .data import SmallVLDataset, create_dataloader
from .training import BenchmarkTrainer
from .utils import setup_device, get_system_info

__all__ = [
    'SimpleVLM',
    'LoRAVLM', 
    'QLoRAVLM',
    'SmallVLDataset',
    'create_dataloader',
    'BenchmarkTrainer',
    'setup_device',
    'get_system_info'
]