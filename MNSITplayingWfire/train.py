import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
# Define your training code here
# Train the model and save it using torch.save


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from model import Network  # Import your custom network (assuming it's in network.py)

# Load and preprocess the MNIST dataset for training and testing
training = datasets.MNIST("", train=True, download=True, 
                          transform=transforms.Compose([transforms.ToTensor()]))
testing = datasets.MNIST("", train=False, download=True, 
                          transform=transforms.Compose([transforms.ToTensor()]))

train_set = torch.utils.data.DataLoader(training, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(testing, batch_size=10, shuffle=True)

# Initialize your custom network
network = Network()

# Define the optimizer
learn_rate = optim.Adam(network.parameters(), lr=0.001)

# Define the number of training epochs
epochs = 4

# Training loop
for epoch in range(epochs):
    for data in train_set:
        image, output = data
        network.zero_grad()
        result = network(image.view(-1, 784))
        loss = F.nll_loss(result, output)
        loss.backward()
        learn_rate.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the network
network.eval()

correct = 0
total = 0

with torch.no_grad():
    for data in test_set:
        image, output = data
        result = network(image.view(-1, 784))
        for index, tensor_value in enumerate(result):
            total += 1
            if torch.argmax(tensor_value) == output[index]:
                correct += 1
                
accuracy = correct / total
print(f"Accuracy: {accuracy}")

# Save the trained model
torch.save(network.state_dict(), 'network.pth')
print("Model saved.")
