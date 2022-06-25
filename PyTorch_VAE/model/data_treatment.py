import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from PyTorch_VAE.model.settings import *


def load_data(batch_size):
    # データのロード
    transform = transforms.Compose([transforms.ToTensor(), ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader
