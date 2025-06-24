"""
Base Vision-Language Model implementation.

This module contains the core VLM architecture used as the foundation
for LoRA and QLoRA adaptations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleVisionEncoder(nn.Module):
    """Simple vision encoder for processing image inputs."""
    
    def __init__(self, input_dim: int = 3*224*224, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass for vision encoder.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Vision features of shape (batch_size, output_dim)
        """
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 3*224*224)
        return self.layers(x)


class SimpleTextEncoder(nn.Module):
    """Simple text encoder for processing text inputs."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass for text encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices
            
        Returns:
            Text features of shape (batch_size, output_dim)
        """
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.mean(dim=1)      # Simple average pooling: (batch_size, embed_dim)
        return self.layers(x)


class SimpleVLM(nn.Module):
    """
    Simple Vision-Language Model for contrastive learning.
    
    This model learns to align vision and text representations in a shared
    embedding space using contrastive learning.
    """
    
    def __init__(self, vocab_size: int, image_dim: int = 3*224*224, embed_dim: int = 256, 
                 vision_hidden: int = 512, text_hidden: int = 256):
        super().__init__()
        
        # Encoders
        self.vision_encoder = SimpleVisionEncoder(
            input_dim=image_dim,
            hidden_dim=vision_hidden,
            output_dim=embed_dim
        )
        
        self.text_encoder = SimpleTextEncoder(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=text_hidden,
            output_dim=embed_dim
        )
        
        # Learnable temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, images, texts):
        """
        Forward pass for VLM.
        
        Args:
            images: Image tensor of shape (batch_size, 3, 224, 224)
            texts: Text tensor of shape (batch_size, seq_len) containing token indices
            
        Returns:
            Similarity logits of shape (batch_size, batch_size)
        """
        # Get features from both encoders
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(texts)
        
        # Normalize features for cosine similarity
        vision_features = F.normalize(vision_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix with learnable temperature
        logits = torch.matmul(vision_features, text_features.T) * self.temperature.exp()
        
        return logits
    
    def get_vision_features(self, images):
        """Extract normalized vision features."""
        features = self.vision_encoder(images)
        return F.normalize(features, dim=-1)
    
    def get_text_features(self, texts):
        """Extract normalized text features."""
        features = self.text_encoder(texts)
        return F.normalize(features, dim=-1)
    
    def compute_similarity(self, images, texts):
        """Compute similarity between images and texts."""
        vision_features = self.get_vision_features(images)
        text_features = self.get_text_features(texts)
        
        return torch.matmul(vision_features, text_features.T)


def create_vlm_model(vocab_size: int, **kwargs):
    """
    Factory function to create a VLM model.
    
    Args:
        vocab_size: Size of the text vocabulary
        **kwargs: Additional model parameters
        
    Returns:
        Initialized SimpleVLM model
    """
    return SimpleVLM(vocab_size=vocab_size, **kwargs)


def count_parameters(model):
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Test the model
    vocab_size = 1000
    batch_size = 4
    
    # Create model
    model = SimpleVLM(vocab_size=vocab_size)
    
    # Create dummy data
    images = torch.randn(batch_size, 3, 224, 224)
    texts = torch.randint(0, vocab_size, (batch_size, 10))
    
    # Forward pass
    logits = model(images, texts)
    
    print(f"Model created successfully!")
    print(f"Input shapes: images {images.shape}, texts {texts.shape}")
    print(f"Output shape: {logits.shape}")
    
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")