from seed import generate_seed 
import itertools 
import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import MyModel 
from tests import TST_C2ST 
import argparse

# Parameter settings 
parser = argparse.ArgumentParser() 
parser.add_argument('--exp', type=str, 
                    help='Explanation for experiment')
parser.add_argument('--rep', type=int, default=500,
                    help = 'Number of repetitions')
parser.add_argument('--cifar10_path', type=str, default='./data', 
                    help = 'Path for CIFAR-10')
parser.add_argument('--cifar10_1_path', type=str, 
                    help = 'Path for CIFAR-10.1')
parser.add_argument('--device', help="CUDA number")
args = parser.parse_args()

# Import data 
cifar10_transform_list = [
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
]
cifar10_transform = transforms.Compose(cifar10_transform_list)
cifar10_dataset = datasets.CIFAR10(args.cifar10_path, train=False, download=False, 
                                   transform=cifar10_transform)
cifar10_dataloader = DataLoader(cifar10_dataset, batch_size = len(cifar10_dataset))

# Obtain CIFAR10 images 
for i, (imgs, labels) in enumerate(cifar10_dataloader):
    data_all = imgs 
    label_all = labels 
Ind_all = np.arange(len(data_all))

# Obtain CIFAR10.1 images 
cifar101 = np.load(args.cifar10_1_path + 'cifar10.1_v4_data.npy')
cifar101_T = np.transpose(cifar101, [0, 3, 1, 2])
ind_M = np.random.choice(len(cifar101_T), len(cifar101_T), replace=False)
data_T = cifar101_T[ind_M] 
trans = transforms.ToPILImage() 
data_trans = torch.zeros([len(data_T), 3, 32, 32]) 
data_T_tensor = torch.from_numpy(data_T) 

for i in range(len(data_T)):
    d0 = trans(data_T_tensor[i]) 
    data_trans[i] = cifar10_transform(d0) 
Ind_v4_all = np.arange(len(data_T)) 

# Setup parameters 
sample_size_m = 2000 
sample_sizes_n = [2000, 4000, 6000, 8000, 10000] 
v_num = len(sample_sizes_n) 
n_epochs = [500, 1000] 
e_num = len(n_epochs) 
learning_rates = [0.0002, 0.0005, 0.001] 
lr_num = len(learning_rates) 
alpha = 0.05
dtype = torch.float32

index_vals = [] 
results = [] 

for e, r, v in itertools.product(range(e_num), range(lr_num), range(v_num)):
    n_epoch = n_epochs[e]
    n = sample_sizes_n[v] 
    m = sample_size_m
    learning_rate = learning_rates[r]
    test_output_list = [] 
    
    # Sampling 
    rs = np.random.RandomState(1203 + e + r + v) 
    idx_X = rs.choice(len(data_all), size = n, replace=False) 
    idx_Y = rs.choice(len(data_T), size=m, replace=False) 
    s1 = torch.stack([data_all[i] for i in idx_X]).to(args.device) 
    s2 = torch.stack([data_trans[i] for i in idx_Y]).to(args.device) 
    
    for i in range(args.rep):
        model = MyModel(in_channels=3, img_size=32, device=args.device, dtype=dtype) 
        seed = generate_seed(e, 2, 3, r, v, i) 
        test_output_list.append(
            TST_C2ST(
                s1,
                s2,
                model,
                alpha,
                learning_rate,
                n_epoch,
                seed, 
                loss_fn = model.smooth_objective,
                device=args.device
            )
        )
    power = np.mean(test_output_list) 
    index_val = (
        args.exp, 
        args.rep,
        m, 
        n, 
        n_epoch, 
        learning_rate
    )
    index_vals.append(index_val) 
    results.append(power) 
    print(f"N_epoch: {n_epoch}, Learning rate: {learning_rate}, n: {n}, m: {m}, power: {power}")
    
index_names = (
    "experiment",
    "repetitions", 
    "m",
    "n",
    "n_epoch",
    "learning_rate"
)
index = pd.MultiIndex.from_tuples(index_vals, names=index_names)
results_df = pd.Series(results, index=index).to_frame("power") 
results_df.reset_index().to_csv(f"user/raw/results_{args.exp}.csv")
results_df.to_pickle(f"user/raw/results_{args.exp}.pkl")
