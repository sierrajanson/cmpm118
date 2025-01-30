"""
About: ♡ CNN Framework ♡
Authors: Ruthwika, Shubhi, Sierra (and chatGPT...)
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import nn

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# hyperparameters (constants) to mess around with
EPOCHS = 3

# load data
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) # change this to tonic (for NMNIST)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) # need to import both train and test


# define convolutional layers
# in_channels --> initially # of colors (1 for grayscale, 3 for RGB)
# out_channels --> how many features derived from conv layer, to be passed as input to next layer
# kernel (filter) size (amplifier square multiplied w/ OG image)--> 3 is standard
conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1) 
conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
fc = torch.nn.Linear(32 * 7 * 7, 10)

# optimizer
optimizer = torch.optim.Adam(
    list(conv1.parameters()) + list(conv2.parameters()) + list(fc.parameters()), 
    lr=0.001
    )

# forward function
def forward(x):
    x = F.relu(conv1(x))
    x = F.max_pool2d(x, 2, 2)
    # add leaky snn here
    x = F.relu(conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(x.size(0), -1) # dynamically flattens sizes (alledgedly)
    x = fc(x) # fully connected
    # add leaky snn here (i think :/)
    return x

# training loop
# need to add the testing accuracy portion + spk_rec
for epoch in range(EPOCHS):
    for images, labels in trainloader:
        optimizer.zero_grad()
        x = forward(images) # forward pass
        loss = F.cross_entropy(x, labels)
        loss.backward()
        optimizer.step()
        print(loss.item())
    print(f"♡ epoch {epoch+1}/{EPOCHS}, loss: {loss.item():.4f} ♡")
    break # remove later
