# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create fully connected network

class NN(nn.Module):
    def __init__(self, input_size, num_classes): # input = 28*28
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, num_classes)
    
    def forward(self, input):
        output_from_layer1 = F.relu(self.fc1(input))
        output_from_layer_2 = self.fc2(output_from_layer1)
        return output_from_layer_2
    
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load data
train_dataset = datasets.MNIST(root = '/Users/anantvirsingh/MyData/pytorch_practice/dataset', train = True, transform = transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_dataset = datasets.MNIST(root = '/Users/anantvirsingh/MyData/pytorch_practice/dataset', train = False, transform = transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# Initialize the model
net = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = learning_rate)


# Train Network

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Reshape data to be 784 contiguous neurons 
        reshaped_data = data.reshape(data.shape[0], -1)
        
        # Forward pass
        scores = net(reshaped_data)

        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()

# check accuracy on train and test

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            scores = net(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)}/{float(num_samples)}*100')
    model.train()


check_accuracy(train_loader, net)
check_accuracy(test_loader, net)



