# Standard Libraries
import argparse
from random import sample
import numpy as np
from tqdm.auto import tqdm

# PyTorch and Torchvision
import torch
import torch.backends.cudnn as cudnn
from torchvision import models, datasets, transforms
from torchvision.transforms import transforms

# Custom Modules
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator
from model.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from exceptions.exceptions import InvalidDatasetSelection
from PIL import Image


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description="SimCLR")
parser.add_argument("-cifar10", metavar="DIR", default="/data4/oldrain123/C2ST/data/cifar_data/cifar10",
                    help="path to cifar10")
parser.add_argument("-cifar10_1", metavar="DIR", default="/data4/oldrain123/C2ST/data/cifar_data/cifar10.1_v4_data.npy",
                    help="path to cifar10_1")
# parser.add_argument("-dataset-name", default="cifar10",
#                     help="dataset name", choices=['cifar10', 'stl10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) + 
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total batch size')
parser.add_argument('--lr', '--learning_rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int, 
                    help='seed for initialization training.')
parser.add_argument('--disable_cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int, 
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float, 
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='GPU index.')

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported."
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True 
        cudnn.benchmark = True 
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset_test = datasets.CIFAR10(root=args.cifar10, download=True,train=False,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True, num_workers=1)
    
    # Obtain CIFAR10 images
    for i, (imgs, Labels) in enumerate(dataloader_test):
        data_all = imgs
        label_all = Labels
    Ind_all = np.arange(len(data_all))

    # Obtain CIFAR10.1 images
    data_new = np.load(args.cifar10_1)
    
    data_T = np.transpose(data_new, [0,3,1,2])
    ind_M = np.random.choice(len(data_T), len(data_T), replace=False)
    data_T = data_T[ind_M]
    TT = transforms.Compose([
        # ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(32),2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans = transforms.ToPILImage()
    data_trans = torch.zeros([len(data_T),3,32,32])
    data_T_tensor = torch.from_numpy(data_T)
    
    for i in range(len(data_T)):
        d0 = trans(data_T_tensor[i])
        data_trans[i] = TT(d0)
    Ind_v4_all = np.arange(len(data_T))
    
    contrastive_gen = ContrastiveLearningViewGenerator(ContrastiveLearningDataset.get_simclr_pipeline_transform(64), 2)

    # Convert the tensor to a PIL Image
    trans = transforms.ToPILImage()

    # Initialize a tensor to store the transformed images.
    data_trans_transformed = torch.zeros([len(data_trans), 2, 3, 64, 64])
    data_all_transformed = torch.zeros([len(data_all), 2, 3, 64, 64])

    for i in tqdm(range(len(data_trans))):
        # Convert tensor to PIL image, and un-normalize it for conversion to work
        pil_image1 = trans((data_trans[i] * 0.5) + 0.5)  # un-normalize

        # Apply the ContrastiveLearningViewGenerator
        transformed_views1 = contrastive_gen(pil_image1)
        
        # Convert the transformed PIL images back to tensors and normalize them
        transformed_tensors1 = [(tensor - 0.5) / 0.5 for tensor in transformed_views1]  # normalize
        
        # Store them in new tensor
        data_trans_transformed[i] = torch.stack(transformed_tensors1)
        
    for i in tqdm(range(len(data_all))):
        # Convert tensor to PIL image, and un-normalize it for conversion to work
        pil_image2 = trans((data_all[i] * 0.5) + 0.5)

        # Apply the ContrastiveLearningViewGenerator
        transformed_views2 = contrastive_gen(pil_image2)
        
        # Convert the transformed PIL images back to tensors and normalize them
        transformed_tensors2 = [(tensor - 0.5) / 0.5 for tensor in transformed_views2]
        
        # Store them in new tensor
        data_all_transformed[i] = torch.stack(transformed_tensors2)
    
    # Collect CIFAR10 images
    Ind_tr = np.random.choice(len(data_all), len(data_T), replace=False)
    cifar10_data = []
    labels0 = np.zeros(len(data_all_transformed))
    for i in Ind_tr:
        cifar10_data.append(data_all_transformed[i])
        # cifar10_data.append(data_all_transformed[i])

    # Collect CIFAR10.1 images
    Ind_tr_v4 = np.random.choice(len(data_T), len(data_T), replace=False)
    New_CIFAR_tr = data_trans_transformed[Ind_tr_v4]
    new_cifar10_data = []
    labels1 = np.ones(len(New_CIFAR_tr))
    for i in Ind_tr_v4:
        new_cifar10_data.append(New_CIFAR_tr[i])
        # new_cifar10_data.append(New_CIFAR_tr[i])

    train_dataset = torch.concatenate((data_all_transformed[Ind_tr], New_CIFAR_tr[Ind_tr_v4]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)

if __name__ == "__main__":
    main()
