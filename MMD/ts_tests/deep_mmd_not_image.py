"""
The methods here are taken from Liu et al
https://github.com/fengliu90/DK-for-TST/blob/master/Baselines_Blob.py
"""
from sklearn.utils import check_random_state
from argparse import Namespace
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
import pickle
from tqdm.auto import tqdm
from utils_HD import MatConvert, MMDu, TST_MMD_u
from mmdvar import IncomMMDVar, ComMMDVar, h1_mean_var_gram

# Setup seeds
np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True
is_cuda = True

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False


def deep_mmd_not_image(sample_p, sample_q, use_1sample_U, complete, n_epochs=1000):
    assert sample_p.shape[1] == sample_q.shape[1]
    sample_p = np.array(sample_p, dtype='float32')
    sample_q = np.array(sample_q, dtype='float32')
    
    # Setup for all experiments
    alpha = 0.05 # test threshold
    x_in = sample_p.shape[1] # number of neurons in the input layer, i.e., dimension of data
    H = 50 # number of neurons in the hidden layer
    x_out = 50 # number of neurons in the output layer
    learning_rate = 0.0005 # learning rate for MMD-D on Blob
    N_epoch = n_epochs # number of training epochs

    # prepare datasets
    sample_p = torch.from_numpy(sample_p)
    sample_q = torch.from_numpy(sample_q)
    
    # split data 50/50
    x_train, x_test = sample_p[:len(sample_p)//2], sample_p[len(sample_p)//2:]
    y_train, y_test = sample_q[:len(sample_q) // 2], sample_q[len(sample_q) // 2:]

    # Initialize parameters
    model_u = ModelLatentF(x_in, H, x_out)
    if cuda:
        model_u.cuda()
    epsilonOPT = MatConvert(np.random.rand(1) * (10 ** (-10)), device, dtype)
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.sqrt(np.random.rand(1) * 0.3), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.sqrt(np.random.rand(1) * 0.002), device, dtype)
    sigma0OPT.requires_grad = True

    # Setup optimizer for training deep kernel
    optimizer_u = torch.optim.Adam(list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT], lr=learning_rate) #

    # Train deep kernel to maximize test power
    S = torch.cat([x_train.cpu(), y_train.cpu()], 0).to(device)
    # S = MatConvert(S, device, dtype)
    N1 = len(x_train)
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for t in tqdm(range(N_epoch)):
        # Compute epsilon, sigma and sigma_0
        ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
        sigma = sigmaOPT ** 2
        sigma0_u = sigma0OPT ** 2
        # Compute output of the deep network
        modelu_output = model_u(S)
        # Compute J (STAT_u)
        TEMP = MMDu(modelu_output, N1, S, sigma, sigma0_u, ep, use_1sample_U=use_1sample_U, complete=complete)
        mmd_value_temp = -1 * TEMP[0]
        mmd_std_temp = torch.sqrt(TEMP[1]+10**(-6))
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        # Initialize optimizer and Compute gradient
        optimizer_u.zero_grad()
        STAT_u.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer_u.step()

    # Compute test power of deep kernel based MMD
    S = torch.cat([x_test.cpu(), y_test.cpu()], 0).to(device)
    # S = MatConvert(S, device, dtype)
    N1 = len(x_test)
    N_per = 500
    alpha = 0.05
    # MMD-D
    dec, pvalue, _ = TST_MMD_u(model_u(S), N_per, N1, S, sigma, sigma0_u, ep, alpha, device, dtype, use_1sample_U=use_1sample_U, complete=complete)
    return dec


class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant