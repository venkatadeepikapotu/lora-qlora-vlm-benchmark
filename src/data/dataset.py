import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
from typing import Dict, List
import requests
from io import BytesIO

class TinyRealVLDataset(Dataset):
    """
    Tiny real vision-language dataset with actual images
    Perfect for CPU training with minimal memory usage
    """
    
    def __init__(self, max_length: int = 12, image_size: int = 128, download_images: bool = True):
        """
        Initialize with real image URLs or local files
        
        Args:
            max_length: Max sequence length for captions
            image_size: Resize images to this size (smaller = faster for CPU)
            download_images: Whether to download sample images
        """
        self.max_length = max_length
        self.image_size = image_size
        
        # 5 real image-caption pairs with public domain/free images
        self.data = [
            {
                "caption": "a cute golden retriever sitting in grass",
                "image_url": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400",
                "image_path": "dog.jpg",
                "image_id": 0
            },
            {
                "caption": "snow covered mountain peaks under blue sky",
                "image_url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
                "image_path": "mountains.jpg",
                "image_id": 1
            },
            {
                "caption": "red car parked on street",
                "image_url": "https://images.unsplash.com/photo-1619767886558-efdc259cde1a?w=400", 
                "image_path": "car.jpg",
                "image_id": 2
            },
            {
                "caption": "orange cat lying on floor",
                "image_url": "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400",
                "image_path": "cat.jpg", 
                "image_id": 3
            },
            {
                "caption": "tropical beach with palm trees and blue water",
                "image_url": "https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=400",
                "image_path": "beach.jpg",
                "image_id": 4
            }
        ]
        
        # Create images directory
        os.makedirs("tiny_dataset_images", exist_ok=True)
        
        # Download images if requested
        if download_images:
            self._download_sample_images()
        
        # Build vocabulary from captions
        self.vocab = self._build_vocabulary()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        # Image preprocessing (lightweight for CPU)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Small size for CPU
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        print(f" TinyRealVLDataset created:")
        print(f"    {len(self.data)} real image-caption pairs")
        print(f"    Vocabulary size: {self.vocab_size}")
        print(f"    Image size: {image_size}x{image_size}")
        print(f"    Optimized for CPU training")
        
    def _download_sample_images(self):
        """Download sample images from URLs"""
        print(" Downloading sample images...")
        
        for item in self.data:
            image_path = os.path.join("tiny_dataset_images", item["image_path"])
            
            if os.path.exists(image_path):
                print(f"    {item['image_path']} already exists")
                continue
                
            try:
                print(f"     Downloading {item['image_path']}...")
                response = requests.get(item["image_url"], timeout=10)
                response.raise_for_status()
                
                # Save image
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                    
                print(f"    {item['image_path']} downloaded successfully")
                
            except Exception as e:
                print(f"    Failed to download {item['image_path']}: {e}")
                print(f"      You can manually download from: {item['image_url']}")
    
    def _build_vocabulary(self):
        """Build vocabulary from captions"""
        all_words = set()
        for item in self.data:
            words = item['caption'].lower().split()
            all_words.update(words)
        
        vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + sorted(list(all_words))
        return vocab
    
    def _tokenize_caption(self, caption: str):
        """Convert caption to token indices"""
        words = caption.lower().split()
        tokens = [self.word_to_idx['<START>']]
        
        for word in words:
            token_id = self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
            tokens.append(token_id)
        
        tokens.append(self.word_to_idx['<END>'])
        
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens.extend([self.word_to_idx['<PAD>']] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length-1] + [self.word_to_idx['<END>']]
            
        return torch.tensor(tokens, dtype=torch.long)
    
    def _load_real_image(self, image_path: str):
        """Load and preprocess real image"""
        full_path = os.path.join("tiny_dataset_images", image_path)
        
        try:
            # Load image
            image = Image.open(full_path).convert('RGB')
            # Apply transforms
            image_tensor = self.transform(image)
            return image_tensor
            
        except Exception as e:
            print(f" Error loading image {image_path}: {e}")
            # Fallback to small random tensor if image fails
            return torch.randn(3, self.image_size, self.image_size)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load real image
        image = self._load_real_image(item['image_path'])
        
        # Tokenize caption
        caption_tokens = self._tokenize_caption(item['caption'])
        
        return {
            'image': image,
            'caption': caption_tokens,
            'raw_caption': item['caption'],
            'image_id': item['image_id']
        }

# Extended dataset for more training samples
class ExtendedTinyDataset(Dataset):
    """Extend tiny dataset with data augmentation for more samples"""
    
    def __init__(self, base_dataset, repeat_factor: int = 20, add_noise: bool = False):
        self.base_dataset = base_dataset
        self.repeat_factor = repeat_factor
        self.add_noise = add_noise
        
        # Additional light augmentations for variety
        self.augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f" Extended dataset: {len(base_dataset)} → {len(base_dataset) * repeat_factor} samples")
    
    def __len__(self):
        return len(self.base_dataset) * self.repeat_factor
    
    def __getitem__(self, idx):
        # Map to base dataset
        base_idx = idx % len(self.base_dataset)
        sample = self.base_dataset[base_idx]
        
        # Add slight variation every few repetitions
        if idx // len(self.base_dataset) > 0 and torch.random.manual_seed(idx):
            # Denormalize, augment, renormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            # Denormalize
            denorm_image = sample['image'] * std + mean
            denorm_image = torch.clamp(denorm_image, 0, 1)
            
            # Apply light augmentation
            try:
                augmented_image = self.augment_transform(denorm_image)
                sample['image'] = augmented_image
            except:
                pass  # Use original if augmentation fails
        
        return sample

def create_tiny_dataloader(image_size: int = 128, batch_size: int = 2, repeat_factor: int = 15):
    """
    Create DataLoader optimized for CPU training
    
    Args:
        image_size: Smaller = faster for CPU (64, 96, 128 recommended)
        batch_size: Small batch for CPU (1-4 recommended)
        repeat_factor: Repeat dataset for more training iterations
    """
    print(" Creating tiny real dataset for CPU training...")
    
    # Create base dataset
    dataset = TinyRealVLDataset(image_size=image_size, download_images=True)
    
    # Extend for more training samples  
    extended_dataset = ExtendedTinyDataset(dataset, repeat_factor=repeat_factor)
    
    # Create DataLoader (no pin_memory for CPU)
    dataloader = DataLoader(
        extended_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 for CPU training
        pin_memory=False  # False for CPU
    )
    
    print(f" DataLoader ready:")
    print(f"    Batch size: {batch_size}")
    print(f"    Total samples: {len(extended_dataset)}")
    print(f"    Batches per epoch: {len(dataloader)}")
    print(f"    CPU optimized")
    
    return dataloader, dataset

# Alternative: Use your own images
def create_custom_dataset(image_folder: str, captions: List[str]):
    """
    Create dataset from your own images
    
    Args:
        image_folder: Path to folder with your images
        captions: List of captions matching your images
    """
    
    class CustomDataset(TinyRealVLDataset):
        def __init__(self, image_folder, captions, image_size=128):
            self.image_folder = image_folder
            self.image_size = image_size
            
            # Create data from your images
            image_files = [f for f in os.listdir(image_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            self.data = []
            for i, (img_file, caption) in enumerate(zip(image_files, captions)):
                self.data.append({
                    "caption": caption,
                    "image_path": img_file,
                    "image_id": i
                })
            
            # Build vocabulary and transforms
            self.vocab = self._build_vocabulary()
            self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.vocab_size = len(self.vocab)
            
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
        def _load_real_image(self, image_path):
            full_path = os.path.join(self.image_folder, image_path)
            try:
                image = Image.open(full_path).convert('RGB')
                return self.transform(image)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                return torch.randn(3, self.image_size, self.image_size)
    
    return CustomDataset(image_folder, captions)

if __name__ == "__main__":
    # Test the real dataset with 5 images
    print(" Testing Tiny Real Dataset with 5 images...")
    
    dataloader, dataset = create_tiny_dataloader(
        image_size=96,  # Small for CPU
        batch_size=2,   # Small batch
        repeat_factor=10  # 5 images → 50 training samples
    )
    
    # Test loading a batch
    print("\n Testing batch loading...")
    batch = next(iter(dataloader))
    
    print(f" Batch loaded successfully:")
    print(f"   Images shape: {batch['image'].shape}")
    print(f"   Captions shape: {batch['caption'].shape}")
    print(f"   Sample caption: '{batch['raw_caption'][0]}'")
    
    print("\n Ready for CPU training with real images!")