import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
import snntorch as snn
from snntorch import utils
import tonic
import tonic.transforms as transforms
from snntorch import surrogate

EPOCHS = 3
BETA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 0.001

sensor_size = tonic.datasets.NMNIST.sensor_size
transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=sensor_size, time_window=10000)
])

trainset = tonic.datasets.NMNIST(save_to='./data', train=True, transform=transform)
testset = tonic.datasets.NMNIST(save_to='./data', train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=True))
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=True))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 12, kernel_size=5)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=5)
        self.fc = nn.Linear(32 * 5 * 5, 10)
        self.snn1 = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.snn2 = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.snn3 = snn.Leaky(beta=BETA, spike_grad=surrogate.fast_sigmoid(slope=25), output=True)

    def forward(self, x):
        self.reset_states()
        x = nn.functional.max_pool2d(self.conv1(x), 2)
        spk1, _ = self.snn1(x)
        x = nn.functional.max_pool2d(self.conv2(spk1), 2)
        spk2, _ = self.snn2(x)
        x = self.fc(torch.flatten(spk2, 1))
        spk3, _ = self.snn3(x)
        return spk3

    def reset_states(self):
        self.snn1.reset(), self.snn2.reset(), self.snn3.reset()

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_fn = snn.loss.ce_count_loss(correct_rate=0.8, incorrect_rate=0.2)

def train_step(data, labels):
    net.reset_states()
    optimizer.zero_grad()
    spk_rec = [net(data[step]) for step in range(data.size(0))]
    spk_rec = torch.stack(spk_rec)
    loss = loss_fn(spk_rec, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(EPOCHS):
    total_loss, batch_count = 0, 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if data.size(0) == BATCH_SIZE:
            loss = train_step(data, labels)
            total_loss += loss
            batch_count += 1
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss:.4f}")
    print(f"♡ Epoch {epoch+1}/{EPOCHS}, Average Loss: {total_loss/batch_count:.4f} ♡")