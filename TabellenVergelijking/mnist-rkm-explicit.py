import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as T

# ref:
# https://github.com/sonnyachten/dMVRKM/tree/main/srcs/model

# setup dataset
trainData = datasets.MNIST('../data', train=True,  download=True, transform=T.ToTensor())
testTotData = datasets.MNIST('../data', train=False, download=True, transform=T.ToTensor())
ttl = len(testTotData.targets)
testData, valData = torch.utils.data.random_split(testTotData, [int(ttl/2),int(ttl/2)])
loaderNWorkers = 16
loaderBatchSize = 1
loaders = {
    'train' : torch.utils.data.DataLoader(trainData,
                                          batch_size=loaderBatchSize,
                                          shuffle=True,
                                          num_workers=loaderNWorkers,
                                          ),
    'test'  : torch.utils.data.DataLoader(testData,
                                          batch_size=loaderBatchSize,
                                          shuffle=True,
                                          num_workers=loaderNWorkers),
    'val'   : torch.utils.data.DataLoader(valData,
                                          batch_size=loaderBatchSize, # use all for validation
                                          shuffle=True,
                                          num_workers=loaderNWorkers),
    }

# setup model
nViews = 2
Cs = []

class explicitFreatureMapMNIST(nn.Module):
    def __init__(self,nViews):
        super(explicitFreatureMapMNIST, self).__init__()
        self.nViews = nViews
        # parameters for feature maps of images, taken from https://github.com/MrPandey01/Stiefel_Restricted_Kernel_Machine/blob/main/code/stiefel_rkm_model.py, Net1
        nChannels = 1 # input size phi - 1 for black and white?
        capacity = 64 # hiervoor "self.args.capacity"
        self.phi = nn.Sequential(
            nn.Conv2d(nChannels, self.args.capacity),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(capacity, capacity * 2),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(capacity * 2, capacity * 4),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Flatten(),
            nn.Linear(capacity * 4 * cnn_kwargs[2] ** 2, self.args.x_fdim1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.args.x_fdim1, self.args.x_fdim2),
        )
        # U matrix


    def forward(self,x)
        # setup C matrices -> use feature map per nView

        # center C
        # unsupervised loss
        # supervised loss
        # add
