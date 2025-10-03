import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dim
        self.num_classes = output_dim
        self.flatten = nn.Flatten()  
        self.fc = nn.Linear(self.input_dimension, self.num_classes, bias=bias)

    def forward(self, x):
        x = self.flatten(x)  
        return self.fc(x)

