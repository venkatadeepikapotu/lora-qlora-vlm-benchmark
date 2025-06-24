"""
Training and benchmarking utilities for LoRA/QLoRA VLM comparison.

This module contains the core training logic and benchmarking functionality
for comparing different fine-tuning approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import psutil
import os

from utils.device import get_memory_info, clear_gpu_cache, optimize_for_device


@dataclass
class BenchmarkResults:
    """Store comprehensive benchmark results."""
    model_type: str
    total_params: int
    trainable_params: int
    trainable_percentage: float
    memory_usage_mb: float
    gpu_memory_mb: float
    avg_iteration_time: float
    total_training_time: float
    final_loss: float
    convergence_epoch: int
    peak_memory_mb: float
    training_losses: List[float]
    learning_curve: List[float]


class BenchmarkTrainer:
    """
    Comprehensive trainer for benchmarking LoRA, QLoRA, and regular fine-tuning.
    """
    
    def __init__(self, device: torch.device, gpu_count: int = 0, 
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 verbose: bool = True, log_interval: int = 10):
        """
        Initialize the benchmark trainer.
        
        Args:
            device: Device to train on
            gpu_count: Number of available GPUs
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            verbose: Whether to print detailed progress
            log_interval: How often to log progress (in batches)
        """
        self.device = device
        self.gpu_count = gpu_count
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.log_interval = log_interval
        
    def count_parameters(self, model: nn.Module) -> Tuple[int, int, float]:
        """
        Count model parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Tuple of (total_params, trainable_params, trainable_percentage)
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0
        
        return total_params, trainable_params, trainable_percentage
    
    def get_current_memory_usage(self) -> Tuple[float, float]:
        """
        Get current memory usage.
        
        Returns:
            Tuple of (system_memory_mb, gpu_memory_mb)
        """
        # System memory
        process = psutil.Process(os.getpid())
        system_memory = process.memory_info().rss / (1024 * 1024)
        
        # GPU memory
        gpu_memory = 0
        if self.device.type == 'cuda' and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(self.device.index) / (1024 * 1024)
        
        return system_memory, gpu_memory
    
    def setup_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Setup optimizer for the model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Configured optimizer
        """
        # Get trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        if not trainable_params:
            raise ValueError("No trainable parameters found in model")
        
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def compute_contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss for vision-language learning.
        
        Args:
            logits: Similarity logits of shape (batch_size, batch_size)
            
        Returns:
            Contrastive loss value
        """
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        
        # Image-to-text and text-to-image losses
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # Average the losses
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                   optimizer: torch.optim.Optimizer, epoch: int) -> Tuple[List[float], List[float]]:
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            dataloader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            Tuple of (iteration_times, losses)
        """
        model.train()
        iteration_times = []
        losses = []
        
        total_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            start_time = time.time()
            
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            captions = batch['caption'].to(self.device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images, captions)
            
            # Compute loss
            loss = self.compute_contrastive_loss(logits)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Record metrics
            iteration_time = time.time() - start_time
            iteration_times.append(iteration_time)
            losses.append(loss.item())
            
            # Logging
            if self.verbose and batch_idx % self.log_interval == 0:
                current_memory, gpu_memory = self.get_current_memory_usage()
                print(f"    Batch {batch_idx:3d}/{total_batches}: "
                      f"Loss={loss.item():.4f}, "
                      f"Time={iteration_time:.3f}s, "
                      f"Mem={current_memory:.0f}MB")
                
                # Show sample caption occasionally
                if batch_idx % (self.log_interval * 2) == 0 and len(batch['raw_caption']) > 0:
                    print(f"      Sample: '{batch['raw_caption'][0]}'")
            
            # Clear GPU cache periodically
            if self.device.type == 'cuda' and batch_idx % 20 == 0:
                clear_gpu_cache()
        
        return iteration_times, losses
    
    def evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            model: Model to evaluate
            dataloader: Validation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device, non_blocking=True)
                captions = batch['caption'].to(self.device, non_blocking=True)
                
                logits = model(images, captions)
                loss = self.compute_contrastive_loss(logits)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return {
            'avg_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def detect_convergence(self, losses: List[float], patience: int = 10, 
                          min_delta: float = 1e-4) -> bool:
        """
        Detect if training has converged.
        
        Args:
            losses: List of recent losses
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to consider as improvement
            
        Returns:
            True if converged, False otherwise
        """
        if len(losses) < patience:
            return False
        
        recent_losses = losses[-patience:]
        
        # Check if loss has stopped decreasing
        for i in range(1, len(recent_losses)):
            if recent_losses[0] - recent_losses[i] > min_delta:
                return False
        
        return True
    
    def benchmark_model(self, model: nn.Module, model_name: str, 
                       dataloader: DataLoader, num_epochs: int = 5) -> BenchmarkResults:
        """
        Comprehensive benchmark of a model.
        
        Args:
            model: Model to benchmark
            model_name: Name of the model for logging
            dataloader: Training data loader
            num_epochs: Number of epochs to train
            
        Returns:
            BenchmarkResults object with comprehensive metrics
        """
        if self.verbose:
            print(f"\n Starting benchmark for {model_name}")
            print(f"   Epochs: {num_epochs}")
            print(f"   Batches per epoch: {len(dataloader)}")
        
        # Setup model and optimizer
        model = optimize_for_device(model, self.device, self.gpu_count)
        optimizer = self.setup_optimizer(model)
        
        # Get parameter counts
        total_params, trainable_params, trainable_percentage = self.count_parameters(model)
        
        if self.verbose:
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Trainable percentage: {trainable_percentage:.2f}%")
        
        # Initialize tracking variables
        initial_memory, initial_gpu_memory = self.get_current_memory_usage()
        peak_memory = initial_memory
        
        all_iteration_times = []
        all_losses = []
        epoch_losses = []
        convergence_epoch = num_epochs
        
        # Training loop
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            if self.verbose:
                print(f"\n   Epoch {epoch + 1}/{num_epochs}")
            
            epoch_start_time = time.time()
            
            # Train epoch
            iteration_times, losses = self.train_epoch(model, dataloader, optimizer, epoch)
            
            # Update tracking
            all_iteration_times.extend(iteration_times)
            all_losses.extend(losses)
            
            # Epoch statistics
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = np.mean(losses)
            epoch_losses.append(avg_epoch_loss)
            
            # Memory tracking
            current_memory, gpu_memory = self.get_current_memory_usage()
            peak_memory = max(peak_memory, current_memory)
            
            if self.verbose:
                print(f"     Epoch time: {epoch_time:.1f}s")
                print(f"     Average loss: {avg_epoch_loss:.4f}")
                print(f"     Memory usage: {current_memory:.0f}MB (peak: {peak_memory:.0f}MB)")
                if gpu_memory > 0:
                    print(f"     GPU memory: {gpu_memory:.0f}MB")
            
            # Check for convergence
            if self.detect_convergence(epoch_losses):
                convergence_epoch = epoch + 1
                if self.verbose:
                    print(f"      Converged at epoch {convergence_epoch}")
                break
        
        total_training_time = time.time() - training_start_time
        
        # Final memory measurement
        final_memory, final_gpu_memory = self.get_current_memory_usage()
        memory_usage = final_memory - initial_memory
        gpu_memory_usage = final_gpu_memory - initial_gpu_memory
        
        # Calculate final metrics
        avg_iteration_time = np.mean(all_iteration_times) if all_iteration_times else 0
        final_loss = np.mean(all_losses[-10:]) if len(all_losses) >= 10 else np.mean(all_losses)
        
        # Create results object
        results = BenchmarkResults(
            model_type=model_name,
            total_params=total_params,
            trainable_params=trainable_params,
            trainable_percentage=trainable_percentage,
            memory_usage_mb=memory_usage,
            gpu_memory_mb=gpu_memory_usage,
            avg_iteration_time=avg_iteration_time,
            total_training_time=total_training_time,
            final_loss=final_loss,
            convergence_epoch=convergence_epoch,
            peak_memory_mb=peak_memory,
            training_losses=all_losses,
            learning_curve=epoch_losses
        )
        
        if self.verbose:
            print(f"\n {model_name} benchmark completed:")
            print(f"   Training time: {total_training_time:.1f}s")
            print(f"   Average iteration time: {avg_iteration_time:.4f}s")
            print(f"   Final loss: {final_loss:.4f}")
            print(f"   Memory usage: {memory_usage:.1f}MB")
            print(f"   Peak memory: {peak_memory:.1f}MB")
            if gpu_memory_usage > 0:
                print(f"   GPU memory usage: {gpu_memory_usage:.1f}MB")
        
        return results
    
    def save_model(self, model: nn.Module, path: str):
        """Save model checkpoint."""
        # Handle DataParallel models
        model_to_save = model.module if hasattr(model, 'module') else model
        
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'model_class': model_to_save.__class__.__name__,
        }
        
        torch.save(checkpoint, path)
        
        if self.verbose:
            print(f"Model saved to {path}")


def create_trainer(device: torch.device, gpu_count: int = 0, **kwargs) -> BenchmarkTrainer:
    """
    Factory function to create a benchmark trainer.
    
    Args:
        device: Device to train on
        gpu_count: Number of available GPUs
        **kwargs: Additional trainer parameters
        
    Returns:
        Configured BenchmarkTrainer
    """
    return BenchmarkTrainer(device=device, gpu_count=gpu_count, **kwargs)


if __name__ == "__main__":
    # Test the trainer
    print("Testing BenchmarkTrainer...")
    
    # This would require the full setup, so just test initialization
    device = torch.device("cpu")
    trainer = BenchmarkTrainer(device=device, gpu_count=0, verbose=True)
    
    print(f"Trainer initialized for device: {device}")
    print("Trainer test completed!")