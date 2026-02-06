import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset


class GaussianNormalizer:
    """Zero mean, unit variance normalization."""
    
    def __init__(self, x, eps=1e-5):
        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean
    
    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


def load_data(data_path, batch_size=32, num_workers=4):
    """Load .npz dataset. Returns DataLoader and shapes."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")
        
    data = np.load(data_path)
    
    a = torch.from_numpy(data['a']).float()
    f = torch.from_numpy(data['f']).float()
    u = torch.from_numpy(data['u']).float()
    
    if a.ndim == 3: a = a.unsqueeze(1)
    if f.ndim == 3: f = f.unsqueeze(1)
    if u.ndim == 3: u = u.unsqueeze(1)
    
    dataset = TensorDataset(a, f, u)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return loader, (a.shape, f.shape, u.shape)


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
