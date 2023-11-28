import torch
import torch.nn as nn


import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # image 28 X 28 = 784
        self.input_layer = nn.Linear(784, 64)
        self.hidden1 = nn.Linear(64, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 10)
    def forward(self, data):
        data = F.relu(self.input_layer(data))
        data = F.relu(self.hidden1(data))
        data = F.relu(self.hidden2(data))
        data = self.output(data)
        
        return F.log_softmax(data, dim=1)
        


# You can also define functions related to the model in this file
def load_model(model_path):
    model = Network()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


