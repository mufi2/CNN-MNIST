import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# CNN model
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Hyperameters
in_channels = 1
num_classes = 10
learning_rate = 3e-4
batch_size = 64
num_epoch = 15

# Loading data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initializing Network
model = CNN(in_channels= in_channels, num_classes= num_classes)

## Loss and criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network
epoch_losses = []
for epoch in range(num_epoch):
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        scores = model(data)
        loss = criterion(scores, target)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    
plt.plot(range(1, num_epoch + 1), epoch_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss vs Epoch for Number of Epoch:{num_epoch}")
plt.savefig(f"Results/result_{num_epoch}.png")
plt.show()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set (Number of Epoch {num_epoch}): {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set (Number of Epoch {num_epoch}): {check_accuracy(test_loader, model)*100:.2f}")
         


torch.save(model.state_dict(), "Model/cnn_model.pkl")
print("Model saved as 'Results/cnn_model.pkl'")


