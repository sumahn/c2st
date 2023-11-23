"""
The methods here are taken from Liu et al:
https://github.com/fengliu90/DK-for-TST/blob/master/Deep_Baselines_CIFAR10.py
"""
from argparse import Namespace
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
import torchvision 
import scipy.stats as stats
from tqdm.auto import tqdm
from utils_HD import get_item, MatConvert, Pdist2, MMDu, TST_MMD_u
from mmdvar import IncomMMDVar, ComMMDVar, h1_mean_var_gram

torch.backends.cudnn.deterministic = True
is_cuda = True

dtype = torch.float
device = torch.device("cuda:0")
cuda = True if torch.cuda.is_available() else False

def deep_mmd_image(sample_p, sample_q, use_1sample_U, complete, n_epochs=1000):
    assert sample_p.shape[1] == sample_q.shape[1]
    
    # Setup seeds
    np.random.seed(819)
    torch.manual_seed(819)
    torch.cuda.manual_seed(819)
    
    # prepare datasets
    sample_p = np.array(sample_p, dtype='float32')
    sample_q = np.array(sample_q, dtype='float32')
    sample_p = torch.from_numpy(sample_p)
    sample_q = torch.from_numpy(sample_q)
    #sample_p = sample_p / 255
    #sample_q = sample_q / 255
    #sample_p = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sample_p)
    #sample_q = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sample_q)
        
    # Parameters
    opt = Namespace()
    opt.n_epochs = n_epochs
    opt.batch_size = 100
    opt.img_size = sample_p.shape[-1]
    opt.orig_img_size = sample_p.shape[-1]
    opt.channels = sample_p.shape[1]
    opt.lr = 0.0002
    opt.n = sample_p.shape[0]
    N_per = 100 # permutation times
    alpha = 0.05 # test threshold

    # split data 50/50
    x_train, x_test = sample_p[:opt.n // 2], sample_p[opt.n // 2:]
    y_train, y_test = sample_q[:opt.n // 2], sample_q[opt.n // 2:]
    
    # Loss function
    adversarial_loss = torch.nn.CrossEntropyLoss()

    # Define the deep network for MMD-D
    class Featurizer(nn.Module):
        def __init__(self):
            super(Featurizer, self).__init__()

            def discriminator_block(in_filters, out_filters, bn=True):
                block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)] #0.25
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block

            self.model = nn.Sequential(
                *discriminator_block(opt.channels, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
            )

            # The height and width of downsampled image
            ds_size = opt.img_size // 2 ** 4
            self.adv_layer = nn.Sequential(
                nn.Linear(128 * ds_size ** 2, 300))

        def forward(self, img):
            out = self.model(img)
            out = out.view(out.shape[0], -1)
            feature = self.adv_layer(out)

            return feature

    featurizer = Featurizer()
    # Initialize parameters
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    sigma0OPT.requires_grad = True
    if cuda:
        featurizer.cuda()
        adversarial_loss.cuda()

    # Initialize optimizers
    optimizer_F = torch.optim.Adam(list(featurizer.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=opt.lr)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # Dataloader
    dataloader_x_train = torch.utils.data.DataLoader(
        x_train,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # -----------------------------------------------------
    #  Training deep networks for MMD-D (called featurizer)
    # -----------------------------------------------------
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for epoch in tqdm(range(opt.n_epochs)):
        for _, x_train_batch in enumerate(dataloader_x_train):
            ind = np.random.choice(y_train.shape[0], x_train_batch.shape[0], replace=False)
            y_train_batch = y_train[ind]

            x_train_batch = Variable(x_train_batch.type(Tensor))
            y_train_batch = Variable(y_train_batch.type(Tensor))
            X = torch.cat([x_train_batch, y_train_batch], 0)

            # ------------------------------
            #  Train deep network for MMD-D
            # ------------------------------
            # Initialize optimizer
            optimizer_F.zero_grad()
            # Compute output of deep network
            modelu_output = featurizer(X)
            # Compute epsilon, sigma and sigma_0
            ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            sigma0_u = sigma0OPT ** 2
            # Compute Compute J (STAT_u)
            TEMP = MMDu(modelu_output, x_train_batch.shape[0], X.reshape(X.shape[0], -1), sigma, sigma0_u, ep, use_1sample_U=use_1sample_U, complete=complete)
            mmd_value_temp = -1 * (TEMP[0])
            mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
            STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
            # Compute gradient
            STAT_u.backward()
            # Update weights using gradient descent
            optimizer_F.step()

    # Run two-sample test on the test set
    S = torch.cat([x_test.cpu(), y_test.cpu()], 0).to(device)  
    Sv = S.view(x_test.shape[0] + y_test.shape[0], -1)
    h, threshold, mmd_value = TST_MMD_u(
        featurizer(S), 
        N_per, 
        x_test.shape[0], 
        Sv, 
        sigma, 
        sigma0_u, 
        ep, 
        alpha, 
        device, 
        dtype,
        use_1sample_U=use_1sample_U,
        complete=complete
    )
    return h