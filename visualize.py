import random 
import torch
import matplotlib.pyplot as plt 

def tensor_to_image(tensor):
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to('cpu', torch.uint8).numpy() 
    
    return ndarr 


def visualize_dataset(dataset, nrow=12, classes=None, nchw=False):
    
    from torchvision.utils import make_grid
    
    if isinstance(dataset, torch.utils.data.Dataset):
        images = torch.tensor(dataset.data)
        labels = torch.tensor(dataset.targets)
    else:
        images, labels = dataset 
    
    samples = []
    img_half_width = images.shape[2] // 2 
    
    if classes:
        for y, cls in enumerate(classes):
            tx = -4 
            ty = (img_half_width * 2 + 2) * y + (img_half_width + 2)
            plt.text(tx, ty, cls, ha='right')
            inds = (labels == y).nonzero().view(-1)
            ind = inds[torch.randperm(inds.shape[0])][:nrow]
            samples.append(images[ind])
        samples = torch.cat(samples, dim=0)
    else:
        nrow_sq = nrow * nrow 
        ind = torch.randperm(images.shape[0])[:nrow_sq]
        samples = images[ind]
    if not nchw: # make_grid gets NCHW 
        samples = samples.permute(0, 3, 1, 2)
    img = make_grid(samples, nrow=nrow)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
            