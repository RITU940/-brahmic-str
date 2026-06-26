"""
Bengali Scene Text PyTorch Dataset
- Loads images and ground truth labels
- Applies augmentation for training
- Handles CTC-compatible label encoding
"""
import os
import json
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset


class BengaliSceneTextDataset(Dataset):
    """PyTorch Dataset for Bengali scene text recognition."""
    
    def __init__(self, pairs, char2idx, img_height=32, img_width=128,
                 augment=False, max_label_len=25):
        self.pairs = pairs
        self.char2idx = char2idx
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        self.max_label_len = max_label_len
    
    def __len__(self):
        return len(self.pairs)
    
    def encode_label(self, text):
        """Encode text to list of indices using char2idx."""
        encoded = []
        for ch in text:
            if ch in self.char2idx:
                encoded.append(self.char2idx[ch])
        return encoded
    
    def preprocess_image(self, img):
        """Resize image maintaining aspect ratio, pad to fixed size."""
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Calculate aspect-ratio-preserving resize
        w, h = img.size
        target_h = self.img_height
        scale = target_h / h
        target_w = min(int(w * scale), self.img_width)
        
        img = img.resize((target_w, target_h), Image.LANCZOS)
        
        # Pad to fixed width
        padded = Image.new('L', (self.img_width, self.img_height), 255)
        padded.paste(img, (0, 0))
        
        return padded
    
    def apply_augmentation(self, img):
        """Apply random augmentations for training."""
        # Random rotation (±3 degrees)
        if random.random() < 0.3:
            angle = random.uniform(-3, 3)
            img = img.rotate(angle, fillcolor=255, expand=False)
        
        # Random contrast adjustment
        if random.random() < 0.4:
            factor = random.uniform(0.7, 1.5)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        
        # Random brightness adjustment
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.3)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        
        # Random blur
        if random.random() < 0.2:
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # Random sharpening
        if random.random() < 0.2:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(random.uniform(1.5, 3.0))
        
        # Random noise (by adding slight perturbation)
        if random.random() < 0.2:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 5, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        
        # Random erosion/dilation effect
        if random.random() < 0.15:
            if random.random() < 0.5:
                img = img.filter(ImageFilter.MinFilter(3))  # erosion
            else:
                img = img.filter(ImageFilter.MaxFilter(3))  # dilation
        
        # Random horizontal stretch/compress
        if random.random() < 0.2:
            w, h = img.size
            stretch = random.uniform(0.85, 1.15)
            new_w = max(int(w * stretch), 10)
            img = img.resize((new_w, h), Image.LANCZOS)
            # Re-crop/pad to original width
            if new_w > w:
                img = img.crop((0, 0, w, h))
            else:
                padded = Image.new('L', (w, h), 255)
                padded.paste(img, (0, 0))
                img = padded
        
        return img
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img_path = pair['image']
        gt_text = pair['gt']
        
        # Load image
        try:
            img = Image.open(img_path)
        except Exception as e:
            # Return a blank image on error
            img = Image.new('L', (self.img_width, self.img_height), 255)
            gt_text = ''
        
        # Preprocess
        img = self.preprocess_image(img)
        
        # Augment (training only)
        if self.augment:
            img = self.apply_augmentation(img)
        
        # To tensor: normalize to [0, 1], then standardize
        img_array = np.array(img, dtype=np.float32) / 255.0
        # Invert: make text white (high) on black (low) background
        img_array = 1.0 - img_array
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0)  # (1, H, W)
        
        # Encode label
        label = self.encode_label(gt_text)
        label_tensor = torch.IntTensor(label)
        label_length = len(label)
        
        return img_tensor, label_tensor, label_length, gt_text


def collate_fn(batch):
    """Custom collate for variable-length labels in CTC."""
    images, labels, label_lengths, texts = zip(*batch)
    
    # Stack images (all same size)
    images = torch.stack(images, 0)
    
    # Concatenate labels (CTC expects flat labels)
    labels = torch.cat(labels, 0)
    label_lengths = torch.IntTensor(label_lengths)
    
    return images, labels, label_lengths, texts


def load_dataset_splits(json_path, base_dir='.'):
    """Load dataset splits from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pairs = data['pairs']
    char2idx = data['vocabulary']['char2idx']
    
    # Fix paths if needed
    for p in pairs:
        if not os.path.isabs(p['image']):
            p['image'] = os.path.join(base_dir, p['image'])
    
    splits = data['splits']
    train_pairs = [pairs[i] for i in splits['train']]
    val_pairs = [pairs[i] for i in splits['val']]
    test_pairs = [pairs[i] for i in splits['test']]
    
    return train_pairs, val_pairs, test_pairs, char2idx, data
