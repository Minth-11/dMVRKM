import torch
from torchvision import datasets, transforms

dataset1 = datasets.MNIST('../data', train=True,  download=True)
dataset2 = datasets.MNIST('../data', train=False, download=True)
