"""
LoRA (Low-Rank Adaptation) implementation for Vision-Language Models.

This module implements LoRA adaptation layers that enable parameter-efficient
fine-tuning by adding low-rank decomposition matrices to existing linear layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .base_vlm import SimpleVLM


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer.
    
    Implements the LoRA technique by decomposing weight updates into
    low-rank matrices A and B, such that ΔW = B @ A.
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension  
            rank: Rank of the decomposition (lower = more parameter efficient)
            alpha: Scaling factor for LoRA weights
            dropout: Dropout probability for LoRA path
        """
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        # A matrix: random Gaussian initialization
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.1)
        
        # B matrix: zero initialization (ensures ΔW = 0 at start)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        """
        Forward pass for LoRA layer.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            LoRA adaptation output of shape (..., out_features)
        """
        # LoRA forward: x @ A^T @ B^T * scaling
        result = F.linear(x, self.lora_B @ self.lora_A) * self.scaling
        return self.dropout(result)
    
    def extra_repr(self):
        """String representation of LoRA layer."""
        return f'rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}'


class LoRAVLM(nn.Module):
    """
    Vision-Language Model with LoRA adaptation.
    
    This class wraps a base VLM and adds LoRA adaptation layers to
    selected linear layers for parameter-efficient fine-tuning.
    """
    
    def __init__(self, base_model: SimpleVLM, rank: int = 4, alpha: float = 1.0, 
                 dropout: float = 0.0, target_modules: Optional[List[str]] = None):
        """
        Initialize LoRA-adapted VLM.
        
        Args:
            base_model: Base VLM model to adapt
            rank: LoRA rank parameter
            alpha: LoRA alpha scaling parameter
            dropout: Dropout for LoRA layers
            target_modules: List of module names to apply LoRA to (None = all linear)
        """
        super().__init__()
        
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add LoRA layers
        self.vision_lora = nn.ModuleList()
        self.text_lora = nn.ModuleList()
        
        # Apply LoRA to vision encoder
        self._add_lora_to_sequential(
            self.base_model.vision_encoder.layers,
            self.vision_lora,
            target_modules
        )
        
        # Apply LoRA to text encoder  
        self._add_lora_to_sequential(
            self.base_model.text_encoder.layers,
            self.text_lora,
            target_modules
        )
        
        # Keep temperature trainable
        self.base_model.temperature.requires_grad = True
        
        print(f"LoRA applied with rank={rank}, alpha={alpha}")
        print(f"Vision LoRA layers: {len([l for l in self.vision_lora if l is not None])}")
        print(f"Text LoRA layers: {len([l for l in self.text_lora if l is not None])}")
        
    def _add_lora_to_sequential(self, sequential: nn.Sequential, lora_list: nn.ModuleList, 
                               target_modules: Optional[List[str]]):
        """Add LoRA layers corresponding to a sequential module."""
        for i, layer in enumerate(sequential):
            if isinstance(layer, nn.Linear):
                # Check if this layer should get LoRA
                if target_modules is None or f"{i}" in target_modules or "linear" in target_modules:
                    lora_layer = LoRALayer(
                        layer.in_features,
                        layer.out_features,
                        self.rank,
                        self.alpha,
                        self.dropout
                    )
                    lora_list.append(lora_layer)
                else:
                    lora_list.append(None)
            else:
                lora_list.append(None)
    
    def _forward_with_lora(self, x, base_layers: nn.Sequential, lora_layers: nn.ModuleList):
        """Forward pass through base layers with LoRA additions."""
        for base_layer, lora_layer in zip(base_layers, lora_layers):
            if isinstance(base_layer, nn.Linear) and lora_layer is not None:
                # Apply base layer + LoRA adaptation
                base_output = base_layer(x)
                lora_output = lora_layer(x)
                x = base_output + lora_output
            else:
                # Just apply base layer
                x = base_layer(x)
        return x
    
    def forward(self, images, texts):
        """
        Forward pass with LoRA adaptations.
        
        Args:
            images: Image tensor of shape (batch_size, 3, 224, 224)
            texts: Text tensor of shape (batch_size, seq_len)
            
        Returns:
            Similarity logits of shape (batch_size, batch_size)
        """
        # Vision encoding with LoRA
        vision_x = images.view(images.size(0), -1)
        vision_x = self._forward_with_lora(
            vision_x, 
            self.base_model.vision_encoder.layers,
            self.vision_lora
        )
        
        # Text encoding with LoRA
        text_x = self.base_model.text_encoder.embedding(texts).mean(dim=1)
        text_x = self._forward_with_lora(
            text_x,
            self.base_model.text_encoder.layers, 
            self.text_lora
        )
        
        # Normalize and compute similarity (same as base model)
        vision_features = F.normalize(vision_x, dim=-1)
        text_features = F.normalize(text_x, dim=-1)
        
        logits = torch.matmul(vision_features, text_features.T) * self.base_model.temperature.exp()
        return logits
    
    def get_lora_parameters(self):
        """Get all LoRA parameters for optimization."""
        lora_params = []
        
        for lora_layer in self.vision_lora:
            if lora_layer is not None:
                lora_params.extend(lora_layer.parameters())
                
        for lora_layer in self.text_lora:
            if lora_layer is not None:
                lora_params.extend(lora_layer.parameters())
        
        # Include temperature parameter
        lora_params.append(self.base_model.temperature)
        
        return lora_params
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights (not the full model)."""
        lora_state = {
            'vision_lora': self.vision_lora.state_dict(),
            'text_lora': self.text_lora.state_dict(),
            'temperature': self.base_model.temperature.data,
            'config': {
                'rank': self.rank,
                'alpha': self.alpha,
                'dropout': self.dropout
            }
        }
        torch.save(lora_state, path)
        
    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        lora_state = torch.load(path, map_location='cpu')
        self.vision_lora.load_state_dict(lora_state['vision_lora'])
        self.text_lora.load_state_dict(lora_state['text_lora'])
        self.base_model.temperature.data = lora_state['temperature']
        
    def merge_lora_weights(self):
        """
        Merge LoRA weights into base model weights.
        WARNING: This modifies the base model and cannot be undone easily.
        """
        print("Merging LoRA weights into base model...")
        
        # Merge vision encoder
        self._merge_lora_into_sequential(
            self.base_model.vision_encoder.layers,
            self.vision_lora
        )
        
        # Merge text encoder
        self._merge_lora_into_sequential(
            self.base_model.text_encoder.layers,
            self.text_lora
        )
        
        print("LoRA weights merged successfully")
    
    def _merge_lora_into_sequential(self, sequential: nn.Sequential, lora_layers: nn.ModuleList):
        """Merge LoRA weights into sequential layers."""
        for base_layer, lora_layer in zip(sequential, lora_layers):
            if isinstance(base_layer, nn.Linear) and lora_layer is not None:
                # Compute LoRA weight: B @ A * scaling
                lora_weight = lora_layer.lora_B @ lora_layer.lora_A * lora_layer.scaling
                
                # Add to base weight
                base_layer.weight.data += lora_weight
    
    def print_trainable_parameters(self):
        """Print information about trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        return total_params, trainable_params


def create_lora_model(base_model: SimpleVLM, **kwargs):
    """
    Factory function to create a LoRA-adapted VLM.
    
    Args:
        base_model: Base VLM model to adapt
        **kwargs: LoRA configuration parameters
        
    Returns:
        LoRA-adapted model
    """
    return LoRAVLM(base_model, **kwargs)


if __name__ == "__main__":
    # Test LoRA implementation
    from .base_vlm import SimpleVLM
    
    vocab_size = 1000
    batch_size = 4
    
    # Create base model
    base_model = SimpleVLM(vocab_size=vocab_size)
    
    # Create LoRA model
    lora_model = LoRAVLM(base_model, rank=4, alpha=1.0)
    
    # Test forward pass
    images = torch.randn(batch_size, 3, 224, 224)
    texts = torch.randint(0, vocab_size, (batch_size, 10))
    
    logits = lora_model(images, texts)
    
    print(f"LoRA model test successful!")
    print(f"Output shape: {logits.shape}")
    
    lora_model.print_trainable_parameters()