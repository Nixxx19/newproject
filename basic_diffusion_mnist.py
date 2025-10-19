import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 5
batch_size = 128
T = 300  # number of diffusion steps
lr = 1e-4

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data = datasets.MNIST(root='./data', download=True, transform=transform)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
