import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.MNIST(root = './data', train=True, download = True, transform=transform)
testset = torchvision.datasets.MNIST(root= './data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCNN()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 5

for epoch in range(epochs):
    running_loss = 0.001
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch: {epoch}/{epochs}, Loss = {running_loss/len(trainloader):.4f}")

# Test
model.eval()  # doesn't do anything here
correct = 0 
total = 0 
with torch.no_grad():
     for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct/total}%")


