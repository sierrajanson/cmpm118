import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import spikegen
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Dataset (MNIST)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Model Definition
class SpikingConvNet(nn.Module):
    def __init__(self):
        super(SpikingConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=0.9)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.lif2 = snn.Leaky(beta=0.9)
        self.fc1 = nn.Linear(16 * 14 * 14, 10)#64 * 7 * 7, 10)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        # mem2 = self.lif2.init_leaky()
        spk_out = 0

        for step in range(x.size(0)):
            cur_x = x[step]  # Shape: [batch_size, channels, height, width]
            cur_x = F.max_pool2d(F.relu(self.conv1(cur_x)), 2)
            spk1, mem1 = self.lif1(cur_x, mem1)
            
            cur_x = F.max_pool2d(F.relu(self.conv2(spk1)), 2)
            spk2, mem2 = self.lif2(cur_x, mem2)
            
            cur_x = spk2.view(spk2.size(0), -1)
            # cur_x = spk1.view(spk1.size(0), -1)
            spk_out += self.fc1(cur_x)

        return spk_out / x.size(0)

# Model, Loss, and Optimizer
model = SpikingConvNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import snntorch.functional as SF
# Use snntorch's cross-entropy loss based on spike counts
loss_fn = SF.mse_count_loss()

# Training Loop
num_epochs = 1
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Generate spike trains
        spike_train = spikegen.rate(images, num_steps=10, gain=5)  # Poisson spike trains
        
        # Forward pass
        outputs = model(spike_train)
        
        # Compute snntorch's cross-entropy count loss
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
# Save the model
torch.save(model.state_dict(), 'spiking_convnet_snntorch_loss.pth')
