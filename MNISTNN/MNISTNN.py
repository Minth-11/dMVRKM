import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.autograd import Variable
import copy



def main():
    train_tot = datasets.MNIST('../data', train=True,  download=True, transform=T.ToTensor())
    train_data, val_data = torch.utils.data.random_split(train_tot, [50000, 10000])
    test_data = datasets.MNIST('../data', train=False, download=True, transform=T.ToTensor())
    loaders = {
    'train' : torch.utils.data.DataLoader(train_data,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=1,
                                          ),
    'test'  : torch.utils.data.DataLoader(test_data,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=1),
    'val'   : torch.utils.data.DataLoader(val_data,
                                          batch_size=10000, # use all for validation
                                          shuffle=True,
                                          num_workers=1),
    }

    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    nEpochs = 100
    patience = 15
    train(nEpochs, model, loaders, criterion, optimizer,patience)
    test(model,loaders)
    torch.save(model,"mnist.pth")
    model = torch.load("mnist.pth")
    test(model,loaders)
    model = torch.load("mnist.pth")
    test(model,loaders)
    print(model)

def train(nEpochs, model, loaders, loss_func, optimizer, patience):
    model.train()
    total_step = len(loaders['train'])
    bestLoss = float('inf')
    best_model_weights = None
    patCtr = patience
    loopObj = tqdm(range(nEpochs))
    for epoch in loopObj:
        model.train()
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)[0]
            loss = loss_func(output, b_y)
            # clear gradients for this training step
            optimizer.zero_grad()
            # backpropagation, compute gradients
            loss.backward()                # apply gradients
            optimizer.step()

        def barInfo():
            desc = "v:{:.3f} b:".format(valLoss.item()) + "{:.3f}".format(bestLoss) + " p:" + str(patCtr)
            loopObj.set_description(desc)

        # validation
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(loaders['val']):
                b_x = Variable(images)   # batch x
                b_y = Variable(labels)   # batch y
                output = model(b_x)[0]
                valLoss = loss_func(output, b_y)
                barInfo()
        if valLoss < bestLoss:
            bestLoss = valLoss
            best_model_weights = copy.deepcopy(model.state_dict())
            patCtr = patience
            barInfo()
        else:
            patCtr -= 1
            if patCtr == 0:
                barInfo()
                print("Patience ran out")
                break
    # use best model on val
    model.load_state_dict(best_model_weights)

def test(model,loaders):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    print('Test Accuracy of the model on the 10000 test images: %.4f' % accuracy)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization

if __name__ == '__main__':
    main()
