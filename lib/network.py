import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Network(nn.Module):
    """
    Build a physics informed neural network (PINN) model for the steady Navier-Stokes equations.

    Attributes: 
        activations: custom activation functions.
    """

    def __init__(self, num_inputs=2, layers=[3, 20, 20, 20, 20, 20, 20, 20, 20, 2], activation='swish', num_outputs=2):
        super(Network, self).__init__()

        #input layer
        self.fc1 = nn.Linear(5000, 3)
        self.fc2 = nn.Linear(3, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, 20)
        self.fc6 = nn.Linear(20, 20)
        self.fc7 = nn.Linear(20, 20)
        self.fc8 = nn.Linear(20, 20)
        self.fc9 = nn.Linear(20, 20)
        self.fc10 = nn.Linear(20, 2)
        #hidden layers

    def forward(self, X):
        X = nn.Tanh
        
