import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)




training = datasets.FashionMNIST("", train=True, download=True, 
                          transform = transforms.Compose([transforms.ToTensor()]))

testing = datasets.FashionMNIST("",
                                train=False, 
                                download=True, 
                                transform = transforms.Compose([transforms.ToTensor()])
                                )

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
train_set = torch.utils.data.DataLoader(training_data, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True)

network = NeuralNetwork()

learn_rate = optim.Adam(network.parameters(), lr=0.001)
epochs = 4

for i in range(epochs):
    for data in train_set:
        image, output = data
        network.zero_grad()
        result = network(image.view(-1,784))
        loss = F.nll_loss(result,output)
        loss.backward()
        learn_rate.step()
    print(loss)


# Test the network
network.eval()

correct = 0
total = 0

with torch.no_grad():
    for data in test_set:
        image, output = data
        result = network(image.view(-1,784))
        for index, tensor_value in enumerate(result):
            total += 1
            if torch.argmax(tensor_value) == output[index]:
                correct += 1
                
accuracy = correct / total
print(f"Accuracy: {accuracy}")


# Look image processing
from PIL import Image
import numpy as np
import PIL.ImageOps   

img = Image.open("firstTest.png")
img = img.resize((28,28))
img = img.convert("L")
img = PIL.ImageOps.invert(img)

plt.imshow(img)

img = np.array(img)
img = img / 255
image = torch.from_numpy(img)
image = image.float()

result = network.forward(image.view(-1,28*28))
print(torch.argmax(output))







