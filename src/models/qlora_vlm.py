"""
QLoRA (Quantized LoRA) implementation for Vision-Language Models.

This module implements QLoRA, which combines LoRA with quantization
for even more memory-efficient fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .base_vlm import SimpleVLM


class QLoRALayer(nn.Module):
    """
    QLoRA (Quantized Low-Rank Adaptation) layer.
    
    Combines LoRA with 4-bit quantization for maximum memory efficiency.
    This is a simplified implementation for demonstration.
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 4, 
                 alpha: float = 1.0, dropout: float = 0.0, quantization_bits: int = 4):
        """
        Initialize QLoRA layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: Rank of the decomposition
            alpha: Scaling factor for LoRA weights
            dropout: Dropout probability
            quantization_bits: Number of bits for quantization (4 for QLoRA)
        """
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.quantization_bits = quantization_bits
        
        # LoRA matrices (will be quantized during forward pass)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.1)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Quantization parameters (buffers, not parameters)
        self.register_buffer('scale_A', torch.ones(1))
        self.register_buffer('scale_B', torch.ones(1))
        self.register_buffer('zero_point_A', torch.zeros(1))
        self.register_buffer('zero_point_B', torch.zeros(1))
        
        # Quantization range for 4-bit: [-7, 7]
        self.quant_min = -(2 ** (quantization_bits - 1)) + 1
        self.quant_max = 2 ** (quantization_bits - 1) - 1
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def quantize_tensor(self, tensor, scale, zero_point):
        """
        Quantize a tensor to specified bit-width.
        
        Args:
            tensor: Input tensor to quantize
            scale: Quantization scale factor
            zero_point: Quantization zero point
            
        Returns:
            Quantized tensor (still in float format for computation)
        """
        # Quantize: round((tensor / scale) + zero_point)
        quantized = torch.round(tensor / (scale + 1e-8) + zero_point)
        
        # Clamp to valid range
        quantized = torch.clamp(quantized, self.quant_min, self.quant_max)
        
        return quantized
    
    def dequantize_tensor(self, quantized_tensor, scale, zero_point):
        """
        Dequantize a tensor back to full precision.
        
        Args:
            quantized_tensor: Quantized tensor
            scale: Quantization scale factor
            zero_point: Quantization zero point
            
        Returns:
            Dequantized tensor
        """
        return (quantized_tensor - zero_point) * scale
    
    def update_quantization_params(self):
        """
        Update quantization parameters based on current weight statistics.
        This simulates dynamic quantization.
        """
        # Update scale and zero point for A matrix
        a_min, a_max = self.lora_A.min(), self.lora_A.max()
        self.scale_A = (a_max - a_min) / (self.quant_max - self.quant_min)
        self.zero_point_A = self.quant_min - a_min / (self.scale_A + 1e-8)
        
        # Update scale and zero point for B matrix
        b_min, b_max = self.lora_B.min(), self.lora_B.max()
        self.scale_B = (b_max - b_min) / (self.quant_max - self.quant_min)
        self.zero_point_B = self.quant_min - b_min / (self.scale_B + 1e-8)
    
    def forward(self, x):
        """
        Forward pass with quantized LoRA computation.
        
        Args:
            x: Input tensor
            
        Returns:
            QLoRA output
        """
        # Update quantization parameters
        self.update_quantization_params()
        
        # Quantize the LoRA matrices
        quantized_A = self.quantize_tensor(self.lora_A, self.scale_A, self.zero_point_A)
        quantized_B = self.quantize_tensor(self.lora_B, self.scale_B, self.zero_point_B)
        
        # Dequantize for computation (in practice, this would use optimized kernels)
        dequant_A = self.dequantize_tensor(quantized_A, self.scale_A, self.zero_point_A)
        dequant_B = self.dequantize_tensor(quantized_B, self.scale_B, self.zero_point_B)
        
        # Compute LoRA output: x @ A^T @ B^T * scaling
        result = F.linear(x, dequant_B @ dequant_A) * self.scaling
        
        return self.dropout(result)
    
    def get_memory_savings(self):
        """
        Estimate memory savings from quantization.
        
        Returns:
            Dictionary with memory information
        """
        # Original 32-bit parameters
        original_bits = 32
        total_params = self.lora_A.numel() + self.lora_B.numel()
        
        # Quantized parameters
        quantized_bits = self.quantization_bits
        
        original_memory = total_params * original_bits / 8  # bytes
        quantized_memory = total_params * quantized_bits / 8  # bytes
        
        savings_ratio = 1 - (quantized_memory / original_memory)
        
        return {
            'original_memory_bytes': original_memory,
            'quantized_memory_bytes': quantized_memory,
            'memory_savings_ratio': savings_ratio,
            'compression_ratio': original_bits / quantized_bits
        }
    
    def extra_repr(self):
        """String representation of QLoRA layer."""
        return (f'rank={self.rank}, alpha={self.alpha}, '
                f'quantization_bits={self.quantization_bits}, '
                f'scaling={self.scaling:.3f}')


class QLoRAVLM(nn.Module):
    """
    Vision-Language Model with QLoRA adaptation.
    
    Combines LoRA with quantization for maximum parameter and memory efficiency.
    """
    
    def __init__(self, base_model: SimpleVLM, rank: int = 4, alpha: float = 1.0,
                 dropout: float = 0.0, quantization_bits: int = 4, 
                 target_modules: Optional[List[str]] = None):
        """
        Initialize QLoRA-adapted VLM.
        
        Args:
            base_model: Base VLM model to adapt
            rank: LoRA rank parameter
            alpha: LoRA alpha scaling parameter
            dropout: Dropout for LoRA layers
            quantization_bits: Bits for quantization (4 for QLoRA)
            target_modules: Module names to apply QLoRA to
        """
        super().__init__()
        
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.quantization_bits = quantization_bits
        
        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add QLoRA layers
        self.vision_qlora = nn.ModuleList()
        self.text_qlora = nn.ModuleList()
        
        # Apply QLoRA to vision encoder
        self._add_qlora_to_sequential(
            self.base_model.vision_encoder.layers,
            self.vision_qlora,
            target_modules
        )
        
        # Apply QLoRA to text encoder
        self._add_qlora_to_sequential(
            self.base_model.text_encoder.layers,
            self.text_qlora,
            target_modules
        )
        
        # Keep temperature trainable
        self.base_model.temperature.requires_grad = True
        
        print(f"QLoRA applied with rank={rank}, alpha={alpha}, {quantization_bits}-bit quantization")
        print(f"Vision QLoRA layers: {len([l for l in self.vision_qlora if l is not None])}")
        print(f"Text QLoRA layers: {len([l for l in self.text_qlora if l is not None])}")
        
    def _add_qlora_to_sequential(self, sequential: nn.Sequential, qlora_list: nn.ModuleList,
                                target_modules: Optional[List[str]]):
        """Add QLoRA layers corresponding to a sequential module."""
        for i, layer in enumerate(sequential):
            if isinstance(layer, nn.Linear):
                if target_modules is None or f"{i}" in target_modules or "linear" in target_modules:
                    qlora_layer = QLoRALayer(
                        layer.in_features,
                        layer.out_features,
                        self.rank,
                        self.alpha,
                        self.dropout,
                        self.quantization_bits
                    )
                    qlora_list.append(qlora_layer)
                else:
                    qlora_list.append(None)
            else:
                qlora_list.append(None)
    
    def _forward_with_qlora(self, x, base_layers: nn.Sequential, qlora_layers: nn.ModuleList):
        """Forward pass through base layers with QLoRA additions."""
        for base_layer, qlora_layer in zip(base_layers, qlora_layers):
            if isinstance(base_layer, nn.Linear) and qlora_layer is not None:
                # Apply base layer + QLoRA adaptation
                base_output = base_layer(x)
                qlora_output = qlora_layer(x)
                x = base_output + qlora_output
            else:
                # Just apply base layer
                x = base_layer(x)
        return x
    
    def forward(self, images, texts):
        """
        Forward pass with QLoRA adaptations.
        
        Args:
            images: Image tensor of shape (batch_size, 3, 224, 224)
            texts: Text tensor of shape (batch_size, seq_len)
            
        Returns:
            Similarity logits of shape (batch_size, batch_size)
        """
        # Vision encoding with QLoRA
        vision_x = images.view(images.size(0), -1)
        vision_x = self._forward_with_qlora(
            vision_x,
            self.base_model.vision_encoder.layers,
            self.vision_qlora
        )
        
        # Text encoding with QLoRA
        text_x = self.base_model.text_encoder.embedding(texts).mean(dim=1)
        text_x = self._forward_with_qlora(
            text_x,
            self.base_model.text_encoder.layers,
            self.text_qlora
        )
        
        # Normalize and compute similarity
        vision_features = F.normalize(vision_x, dim=-1)
        text_features = F.normalize(text_x, dim=-1)
        
        logits = torch.matmul(vision_features, text_features.T) * self.base_model.temperature.exp()
        return logits
    
    def get_qlora_parameters(self):
        """Get all QLoRA parameters for optimization."""
        qlora_params = []
        
        for qlora_layer in self.vision_qlora:
            if qlora_layer is not None:
                qlora_params.extend(qlora_layer.parameters())
                
        for qlora_layer in self.text_qlora:
            if qlora_layer is not None:
                qlora_params.extend(qlora_layer.parameters())
        
        # Include temperature parameter
        qlora_params.append(self.base_model.temperature)
        
        return qlora_params
    
    def get_memory_savings_info(self):
        """Get detailed memory savings information."""
        total_savings = {
            'original_memory_bytes': 0,
            'quantized_memory_bytes': 0,
            'total_qlora_params': 0,
            'compression_ratio': 0
        }
        
        layer_count = 0
        
        for qlora_layer in self.vision_qlora:
            if qlora_layer is not None:
                savings = qlora_layer.get_memory_savings()
                total_savings['original_memory_bytes'] += savings['original_memory_bytes']
                total_savings['quantized_memory_bytes'] += savings['quantized_memory_bytes']
                total_savings['total_qlora_params'] += qlora_layer.lora_A.numel() + qlora_layer.lora_B.numel()
                layer_count += 1
        
        for qlora_layer in self.text_qlora:
            if qlora_layer is not None:
                savings = qlora_layer.get_memory_savings()
                total_savings['original_memory_bytes'] += savings['original_memory_bytes']
                total_savings['quantized_memory_bytes'] += savings['quantized_memory_bytes']
                total_savings['total_qlora_params'] += qlora_layer.lora_A.numel() + qlora_layer.lora_B.numel()
                layer_count += 1
        
        if total_savings['original_memory_bytes'] > 0:
            total_savings['memory_savings_ratio'] = 1 - (
                total_savings['quantized_memory_bytes'] / total_savings['original_memory_bytes']
            )
            total_savings['compression_ratio'] = 32 / self.quantization_bits
        
        total_savings['num_qlora_layers'] = layer_count
        
        return total_savings
    
    def print_memory_savings(self):
        """Print detailed memory savings information."""
        savings = self.get_memory_savings_info()
        
        print(f"\nQLoRA Memory Savings Analysis:")
        print(f"  Number of QLoRA layers: {savings['num_qlora_layers']}")
        print(f"  Total QLoRA parameters: {savings['total_qlora_params']:,}")
        print(f"  Original memory (32-bit): {savings['original_memory_bytes']:.1f} bytes")
        print(f"  Quantized memory ({self.quantization_bits}-bit): {savings['quantized_memory_bytes']:.1f} bytes")
        print(f"  Memory savings: {savings.get('memory_savings_ratio', 0)*100:.1f}%")
        print(f"  Compression ratio: {savings.get('compression_ratio', 1):.1f}x")
    
    def save_qlora_weights(self, path: str):
        """Save QLoRA weights and quantization parameters."""
        qlora_state = {
            'vision_qlora': self.vision_qlora.state_dict(),
            'text_qlora': self.text_qlora.state_dict(),
            'temperature': self.base_model.temperature.data,
            'config': {
                'rank': self.rank,
                'alpha': self.alpha,
                'dropout': self.dropout,
                'quantization_bits': self.quantization_bits
            }
        }
        torch.save(qlora_state, path)
    
    def load_qlora_weights(self, path: str):
        """Load QLoRA weights."""
        qlora_state = torch.load(path, map_location='cpu')
        self.vision_qlora.load_state_dict(qlora_state['vision_qlora'])
        self.text_qlora.load_state_dict(qlora_state['text_qlora'])
        self.base_model.temperature.data = qlora_state['temperature']
    
    def print_trainable_parameters(self):
        """Print information about trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        return total_params, trainable_params


def create_qlora_model(base_model: SimpleVLM, **kwargs):
    """
    Factory function to create a QLoRA-adapted VLM.
    
    Args:
        base_model: Base VLM model to adapt
        **kwargs: QLoRA configuration parameters
        
    Returns:
        QLoRA-adapted model
    """
    return QLoRAVLM(base_model, **kwargs)


if __name__ == "__main__":
    # Test QLoRA implementation
    from .base_vlm import SimpleVLM
    
    vocab_size = 1000
    batch_size = 4
    
    # Create base model
    base_model = SimpleVLM(vocab_size=vocab_size)
    
    # Create QLoRA model
    qlora_model = QLoRAVLM(base_model, rank=4, alpha=1.0, quantization_bits=4)
    
    # Test forward pass
    images = torch.randn(batch_size, 3, 224, 224)
    texts = torch.randint(0, vocab_size, (batch_size, 10))
    
    logits = qlora_model(images, texts)
    
    print(f"QLoRA model test successful!")
    print(f"Output shape: {logits.shape}")
    
    qlora_model.print_trainable_parameters()
    qlora_model.print_memory_savings()