from torchvision.transforms import transforms 
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
 
class CustomCIFAR10_1(Dataset):
    def __init__(self, data_path, transform=None):
        self.images = np.load(data_path, allow_pickle=True)
        print("Loaded data type:", type(self.images))
        print("Loaded data shape:", self.images.shape)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample_image = self.images[idx]

        # Convert to PIL Image if needed
        if sample_image.shape == (3, 32, 32):  # If it's already a tensor shape
            sample_image = torch.from_numpy(sample_image).float()
        else:  # Otherwise, assume it's an image and convert it to a tensor
            sample_image = Image.fromarray(np.uint8(sample_image))
            if self.transform:
                sample_image = self.transform(sample_image)

        return sample_image


    
class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        
    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a list of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply([color_jitter], p=0.8),
                                                transforms.RandomGrayscale(p=0.2),
                                                GaussianBlur(kernel_size=int(0.1*size)),
                                                transforms.ToTensor()])
        return data_transforms
    
    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10' : lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                                transform = ContrastiveLearningViewGenerator(
                                                                    self.get_simclr_pipeline_transform(32),
                                                                    n_views),
                                                                download=True),
                        'stl10': lambda: datasets.STL10(self.root_finder, split='unlabeled',
                                                        transform = ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(96),
                                                            n_views),
                                                        download=True),
                        'cifar10_1': lambda: CustomCIFAR10_1('/data4/oldrain123/C2ST/data/cifar_data/cifar10.1_v4_data.npy',
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views))
                        }
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()