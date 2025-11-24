import os
import fnmatch
import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    Multi-class version: Handles nested class folders
    256*256 RGB images
    as input:
    modulo: [0, 1] float, as float32
    modulo_edge: [0, 1] float, as float32
    as target:
    fold_number_edge: binary, as float32
    """
    def __init__(self, data_dir='data', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Collect all samples from all class folders
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        for class_name in class_dirs:
            class_path = os.path.join(data_dir, class_name)
            modulo_dir = os.path.join(class_path, 'modulo')
            modulo_edge_dir = os.path.join(class_path, 'modulo_edge_dir')
            fold_number_edge_dir = os.path.join(class_path, 'fold_number_edge')
            
            # Check if directories exist
            if not all(os.path.exists(d) for d in [modulo_dir, modulo_edge_dir, fold_number_edge_dir]):
                print(f"Warning: Skipping class {class_name} - missing directories")
                continue
            
            # Get all .npy files in modulo directory
            names = fnmatch.filter(os.listdir(modulo_dir), '*.npy')
            
            # Add to samples list
            for name in names:
                self.samples.append({
                    'class_name': class_name,
                    'name': name,
                    'modulo_path': os.path.join(modulo_dir, name),
                    'modulo_edge_path': os.path.join(modulo_edge_dir, name),
                    'fold_number_edge_path': os.path.join(fold_number_edge_dir, name)
                })
        
        print(f"Loaded {len(self.samples)} samples from {len(class_dirs)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # Load data (H, W, C)
        modulo = np.load(sample['modulo_path'])
        modulo_edge = np.load(sample['modulo_edge_path'])
        fold_number_edge = np.load(sample['fold_number_edge_path'])
        
        name = sample['name'].split('.')[0]
        class_name = sample['class_name']
        
        assert modulo.ndim == 3  # for RGB image
        
        # Convert to (C, H, W)
        modulo = torch.tensor(np.transpose(modulo / np.max(modulo), (2, 0, 1)), dtype=torch.float32)
        modulo_edge = torch.tensor(np.transpose(modulo_edge, (2, 0, 1)), dtype=torch.float32)
        fold_number_edge = torch.tensor(np.transpose(fold_number_edge, (2, 0, 1)), dtype=torch.float32)
        
        if self.transform:
            modulo = self.transform(modulo)
            modulo_edge = self.transform(modulo_edge)
            fold_number_edge = self.transform(fold_number_edge)
        
        return {
            'modulo': modulo,
            'modulo_edge': modulo_edge,
            'fold_number_edge': fold_number_edge,
            'name': name,
            'class_name': class_name
        }


class InferDataset(Dataset):
    """
    Multi-class inference dataset
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Collect all samples from all class folders
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        for class_name in class_dirs:
            class_path = os.path.join(data_dir, class_name)
            modulo_dir = os.path.join(class_path, 'modulo')
            
            if not os.path.exists(modulo_dir):
                print(f"Warning: Skipping class {class_name} - modulo directory not found")
                continue
            
            names = fnmatch.filter(os.listdir(modulo_dir), '*.npy')
            
            for name in names:
                self.samples.append({
                    'class_name': class_name,
                    'name': name,
                    'modulo_path': os.path.join(modulo_dir, name)
                })
        
        print(f"Loaded {len(self.samples)} inference samples from {len(class_dirs)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # Load data (H, W, C)
        modulo = np.load(sample['modulo_path'])
        
        name = sample['name'].split('.')[0]
        class_name = sample['class_name']
        
        assert modulo.ndim == 3  # for RGB image
        
        # Convert to (C, H, W)
        modulo = torch.tensor(np.transpose(modulo, (2, 0, 1)), dtype=torch.float32)
        
        if self.transform:
            modulo = self.transform(modulo)
        
        return {
            'modulo': modulo,
            'name': name,
            'class_name': class_name
        }