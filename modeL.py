import torch
import torch.nn as nn

class Neuralnet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.ll1 = nn.Linear(input_size, hidden_size) 
        self.ll2 = nn.Linear(hidden_size, hidden_size) 
        self.ll3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.ll1(x)
        out = self.relu(out)
        out = self.ll2(out)
        out = self.relu(out)
        out = self.ll3(out)
    
        return out
