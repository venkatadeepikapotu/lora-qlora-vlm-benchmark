# src/data/__init__.py
"""
Dataset implementations for LoRA/QLoRA VLM Benchmark.

This module contains the real vision-language dataset used for benchmarking
with actual images downloaded from public sources.
"""

from .dataset import (
    TinyRealVLDataset,
    ExtendedTinyDataset,
    create_tiny_dataloader,
    create_custom_dataset
)

__all__ = [
    'TinyRealVLDataset',
    'ExtendedTinyDataset', 
    'create_tiny_dataloader',
    'create_custom_dataset'
]