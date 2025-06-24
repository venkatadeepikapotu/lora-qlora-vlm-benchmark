#!/usr/bin/env python3
"""
Main benchmark script for LoRA vs QLoRA vs Regular Fine-tuning comparison.

Usage:
    python scripts/run_benchmark.py [options]

Examples:
    # Basic run
    python scripts/run_benchmark.py
    
    # Custom configuration
    python scripts/run_benchmark.py --epochs 5 --batch-size 8 --lora-rank 8
    
    # Save models and custom output
    python scripts/run_benchmark.py --save-models --output-dir results/experiment_1
    
    # GPU-specific run
    CUDA_VISIBLE_DEVICES=0,1 python scripts/run_benchmark.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.device import setup_device, get_system_info
from data.dataset import TinyRealVLDataset, create_tiny_dataloader
from models.base_vlm import SimpleVLM
from models.lora_vlm import LoRAVLM
from models.qlora_vlm import QLoRAVLM
from training.trainer import BenchmarkTrainer
from utils.metrics import save_results, print_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA vs QLoRA vs Regular Fine-tuning Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Base batch size (will be scaled for multi-GPU)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    
    # Model parameters
    parser.add_argument('--lora-rank', type=int, default=4,
                        help='LoRA rank parameter')
    parser.add_argument('--lora-alpha', type=float, default=1.0,
                        help='LoRA alpha parameter')
    parser.add_argument('--embed-dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension')
    
    # Dataset parameters
    parser.add_argument('--dataset-repeats', type=int, default=20,
                        help='Number of times to repeat the 3-image dataset')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Image size for resizing (smaller = faster for CPU)')
    
    # Output and logging
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for this experiment')
    
    # Hardware
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU usage even if GPU available')
    parser.add_argument('--max-gpus', type=int, default=None,
                        help='Maximum number of GPUs to use')
    
    return parser.parse_args()


def setup_experiment(args):
    """Setup experiment directories and logging."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment subdirectory if name provided
    if args.experiment_name:
        exp_dir = output_dir / args.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        output_dir = exp_dir
    
    # Create timestamp-based subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def create_models(args, vocab_size, image_size):
    """Create all model variants for benchmarking."""
    print(f"\n Creating models with vocab_size={vocab_size}, image_size={image_size}")
    
    # Calculate actual input dimension based on image size
    input_dim = 3 * image_size * image_size
    
    # Create base model with adaptive input dimension
    base_model = SimpleVLM(
        vocab_size=vocab_size,
        image_dim=input_dim,
        embed_dim=args.embed_dim
    )
    
    models = {
        'Regular': SimpleVLM(
            vocab_size=vocab_size,
            image_dim=input_dim,
            embed_dim=args.embed_dim
        ),
        'LoRA': LoRAVLM(
            base_model=base_model,
            rank=args.lora_rank,
            alpha=args.lora_alpha
        ),
        'QLoRA': QLoRAVLM(
            base_model=base_model,
            rank=args.lora_rank,
            alpha=args.lora_alpha
        )
    }
    
    print(" Models created successfully")
    return models


def main():
    """Main benchmark execution."""
    print(" LoRA vs QLoRA vs Regular Fine-tuning Benchmark")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup device and system info
    device, gpu_count = setup_device(force_cpu=args.force_cpu, max_gpus=args.max_gpus)
    system_info = get_system_info()
    
    if not args.quiet:
        print(f"\n System Information:")
        print(f"   Device: {device}")
        print(f"   GPUs available: {gpu_count}")
        print(f"   Python version: {sys.version.split()[0]}")
    
    # Setup experiment directory
    run_dir = setup_experiment(args)
    print(f"\n Results will be saved to: {run_dir}")
    
    # Save experiment configuration
    config = {
        'args': vars(args),
        'system_info': system_info,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create dataset and dataloader
    print(f"\n Setting up dataset...")
    
    # Use command line image size, with CPU optimization
    image_size = args.image_size
    if args.force_cpu and args.image_size > 96:
        image_size = 96  # Force smaller size for CPU efficiency
        print(f"   Using smaller image size {image_size} for CPU optimization")
    
    # Scale batch size for multi-GPU (same as original logic)
    effective_batch_size = args.batch_size * max(1, gpu_count)
    
    dataloader, dataset = create_tiny_dataloader(
        image_size=image_size,
        batch_size=effective_batch_size,
        repeat_factor=args.dataset_repeats
    )
    
    vocab_size = len(dataset.vocab)
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Training samples: {len(dataloader.dataset)}")
    print(f"   Batch size: {dataloader.batch_size}")
    print(f"   Image size: {image_size}x{image_size}")
    print(f"   Input features: {3 * image_size * image_size}")
    
    # Create models
    models = create_models(args, vocab_size, image_size)
    
    # Initialize trainer
    trainer = BenchmarkTrainer(
        device=device,
        gpu_count=gpu_count,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        verbose=not args.quiet
    )
    
    # Run benchmarks
    print(f"\n Starting benchmark with {args.epochs} epochs...")
    start_time = time.time()
    
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"BENCHMARKING: {model_name}")
        print(f"{'=' * 60}")
        
        try:
            results = trainer.benchmark_model(
                model=model,
                model_name=model_name,
                dataloader=dataloader,
                num_epochs=args.epochs
            )
            
            all_results[model_name] = results
            
            # Save individual model if requested
            if args.save_models:
                model_path = run_dir / f"{model_name.lower()}_model.pth"
                trainer.save_model(model, model_path)
                print(f"    Model saved to: {model_path}")
                
        except Exception as e:
            print(f" Error benchmarking {model_name}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Print and save results
    if all_results:
        print_results(all_results, total_benchmark_time=total_time)
        save_results(all_results, run_dir, config)
        
        print(f"\n Benchmark completed successfully!")
        print(f" Results saved to: {run_dir}")
        print(f"  Total time: {total_time:.1f} seconds")
        
        # Print key files
        print(f"\n Key output files:")
        print(f"    benchmark_results.json - Raw metrics")
        print(f"    comparison_table.txt - Human-readable comparison")
        print(f"    efficiency_analysis.txt - Detailed analysis")
        print(f"    config.json - Experiment configuration")
        
    else:
        print(" No successful benchmarks completed")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Benchmark failed with error: {e}")
        raise