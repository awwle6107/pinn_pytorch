import numpy as np
import torch as torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from lib.network import Network

if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model
    for the cavity flow governed by the steady Navier-Stokes equation.
    """
    # number of training samples
    num_train_samples = 10000
    # number of test samples
    num_test_samples = 100

    # inlet flow velocity
    u0 = 1
    # density
    rho = 1
    # viscosity

    model = Network()
    print(model)