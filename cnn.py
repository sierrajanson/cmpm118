import torch
import torch.optim as optim
import torchvision.transforms as torch_transforms
from torch import nn

import snntorch as snn
from snntorch import utils
import tonic 
import snntorch.functional as F
import tonic.transforms as transforms
from snntorch import surrogate
from snntorch import utils

# hyperparameters
EPOCHS = 3
BETA = 0.5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
ITERS = 50

# Define the sensor size and transforms
sensor_size = tonic.datasets.NMNIST.sensor_size
transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=sensor_size, time_window=1000), 
    transforms.SpatialJitter(jitter=0.1)
])
test_transform = tonic.transforms.Compose([torch.from_numpy, torch_transforms.RandomRotation([-10,10])])

# Load datasets
trainset = tonic.datasets.NMNIST(save_to='./data', train=True, transform=transform)
testset = tonic.datasets.NMNIST(save_to='./data', train=False, transform=transform)

# Create data loaders
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    collate_fn=tonic.collation.PadTensors(batch_first=False)
)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    collate_fn=tonic.collation.PadTensors(batch_first=False)
)

sample, label = trainset[0]
print(label)
print(sample.shape)

net = nn.Sequential(nn.Conv2d(2, 12, 5),
                    snn.Leaky(beta=BETA, spike_grad=surrogate.atan(), init_hidden=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(12, 32, 5),
                    snn.Leaky(beta=BETA, spike_grad=surrogate.atan(), init_hidden=True),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(32*5*5, 10),
                    snn.Leaky(beta=BETA, spike_grad=surrogate.atan(), init_hidden=True, output=True)
                    )

# Initialize network and optimizer
# net = Net()
def forward(net, data):  
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(data.size(0)):  # data.size(0) = number of time steps
      spk_out, mem_out = net(data[step])
      spk_rec.append(spk_out)
  
  return torch.stack(spk_rec)

optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_fn = F.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

loss_hist = []
acc_hist = []

# training loop
for epoch in range(EPOCHS):
    for i, (data, targets) in enumerate(iter(trainloader)):

        net.train()
        spk_rec = forward(net, data)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())
 
        print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
    
        acc = F.accuracy_rate(spk_rec, targets) 
        acc_hist.append(acc)
        print(f"Accuracy: {acc * 100:.2f}%\n")

        # This will end training after 50 iterations by default
        if i == ITERS:
          break