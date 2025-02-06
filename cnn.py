"""
About: ♡ CNN Framework ♡
Authors: Ruthwika, Shubhi, Sierra (and chatGPT...)
"""
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn

import snntorch as snn
from snntorch import utils
import tonic 
import snntorch.functional as F
import tonic.transforms as transforms
from snntorch import surrogate

# hyperparameters (constants) to mess around with
EPOCHS = 3
BETA = 0.99

# load data
# Define the sensor size for NMNIST
sensor_size = tonic.datasets.NMNIST.sensor_size

# Create a transform pipeline
transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=sensor_size, time_window=10000)
])

# Load the datasets
trainset = tonic.datasets.NMNIST(save_to='./data', train=True, transform=transform)
testset = tonic.datasets.NMNIST(save_to='./data', train=False, transform=transform)

# Create data loaders
batch_size = 64
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=True))
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=True))

# define convolutional layers
# in_channels --> initially # of colors (1 for grayscale, 3 for RGB)
# out_channels --> how many features derived from conv layer, to be passed as input to next layer
# kernel (filter) size (amplifier square multiplied w/ OG image)--> 3 is standard
conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
fc = nn.Linear(32 * 7 * 7, 10)

# optimizer
optimizer = torch.optim.Adam(list(conv1.parameters()) + list(conv2.parameters()) + list(fc.parameters()),  lr=0.001)
gradient = surrogate.fast_sigmoid(slope=25)

# forward function
num_inputs = 784
num_hidden = 1000
num_outputs = 10
beta = 0.99

# define SNN layers
snn1 = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid(slope=25))
snn2 = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid(slope=25))
snn3 = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid(slope=25), output=True)

net = nn.Sequential(
    conv1, nn.MaxPool2d(2), snn1,
    conv2, nn.MaxPool2d(2), snn2,
    nn.Flatten(), fc, snn3
)

def forward(net, x):
    spk_rec = []
    utils.reset(net)  
    for step in range(x.size(0)): 
        spk_out, mem_out = net(x[step])
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
        x = forward(net, images) # forward pass
        # loss = F.cross_entropy(x, labels)
        loss = F.mse_count_loss(x, labels)
        loss.backward()
        optimizer.step()
        print(loss.item())
    print(f"♡ epoch {epoch+1}/{EPOCHS}, loss: {loss.item():.4f} ♡")
    break # remove later
