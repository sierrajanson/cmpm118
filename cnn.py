"""
About: ♡ CNN Framework ♡
Authors: Ruthwika, Shubhi, Sierra (and chatGPT...)
"""
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn

import snntorch as snn
import tonic 
import snntorch.functional as F
import tonic.transforms as transforms
from snntorch import surrogate

# hyperparameters (constants) to mess around with
EPOCHS = 3
BETA = 0.99

# load data
transform = transforms.ToTensor()
trainset = tonic.datasets.NMNIST(root='./data', train=True, download=True, transform=transform) # change this to tonic (for NMNIST)
testset = tonic.datasets.NMNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) # need to import both train and test
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# define convolutional layers
# in_channels --> initially # of colors (1 for grayscale, 3 for RGB)
# out_channels --> how many features derived from conv layer, to be passed as input to next layer
# kernel (filter) size (amplifier square multiplied w/ OG image)--> 3 is standard
conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1) 
conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
fc = torch.nn.Linear(32 * 7 * 7, 10)

# optimizer
optimizer = torch.optim.Adam(list(conv1.parameters()) + list(conv2.parameters()) + list(fc.parameters()),  lr=0.001)
gradient = surrogate.fast_sigmoid(slope=25)

# forward function
num_inputs = 784
num_hidden = 1000
num_outputs = 10
beta = 0.99

# fc1 = nn.Linear(num_inputs, num_hidden)
# lif1 = snn.Leaky(beta=beta)
# fc2 = nn.Linear(num_hidden, num_outputs)
# lif2 = snn.Leaky(beta=beta)

# mem1 = lif1.init_leaky()
# mem2 = lif2.init_leaky()

# mem2_rec = []
# spk1_rec = []
# spk2_rec = []

net = nn.Sequential(conv1, nn.MaxPool2d(2), snn.Leaky(beta=beta, spike_grad=gradient), conv2, nn.MaxPool2d(2), snn.Leaky(beta=beta, spike_grad=gradient), nn.Flatten(), fc1, snn.Leaky(x,beta=beta, spike_grad=gradient, output=True))

def forward(net, x):
    spk_rec = []
    utils.reset(net)
    for step in range(x.size(0)):
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)
# def forward(x):
#     x = F.relu(conv1(x))
#     x = F.max_pool2d(x, 2, 2) # add leaky snn here
#     x = snn.Leaky(beta=beta, spike_grad=gradient)
#     x = F.relu(conv2(x))
#     x = F.max_pool2d(x, 2, 2)
#     x = snn.Leaky(x,beta=beta, spike_grad=gradient)
#     x = F.Flatten() # dynamically flattens sizes (alledgedly)
#     x = fc(x) # fully connected
#     x = snn.Leaky(x,beta=beta, spike_grad=gradient, output=True) # add leaky snn here (i think :/)
#     return x

# training loop
# need to add the testing accuracy portion + spk_rec
for epoch in range(EPOCHS):
    for images, labels in trainloader:
        optimizer.zero_grad()
        x = forward(images) # forward pass
        # loss = F.cross_entropy(x, labels)
        loss = F.mse_count_loss(x, labels)
        loss.backward()
        optimizer.step()
        print(loss.item())
    print(f"♡ epoch {epoch+1}/{EPOCHS}, loss: {loss.item():.4f} ♡")
    break # remove later
