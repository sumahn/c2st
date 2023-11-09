"""
Classifiers
"""

import numpy as np
from collections import OrderedDict

import torch 
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 


from torch.optim.lr_scheduler import ReduceLROnPlateau

class PlainBlock(nn.Module):
    """Define simple neural network"""
    
    expansion: int = 1
    
    def __init__(self, Cin, Cout, downsample=False):
        super().__init__() 
        
        if downsample:
            stride = 2
        else:
            stride = 1 
        
        self.model = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm2d(Cin)),
            ('relu1', nn.ReLU()),
            ('conv1', nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(Cout)),
            ('relu2', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=Cout, out_channels=Cout, kernel_size=3, stride=1, padding=1, bias=False))
        ]))
    
    def forward(self, x):
        return self.model(x)
    

class ResidualBlock(nn.Module):
    
    expansion: int = 1
    
    def __init__(self, Cin, Cout, downsample=False):
        
        super().__init__()
        
        if downsample:
            stride = 2
        else:
            stride = 1
            
        self.block = PlainBlock(Cin, Cout, downsample)
        
        if (Cout == Cin) and (downsample == False):
            self.shortcut = nn.Identity() 
        elif Cout != Cin:
            self.shortcut = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=1, stride=stride, bias=False)
        
    def forward(self, x):
        return self.block(x) + self.shortcut(x)
    

class ResNetStage(nn.Module):
    
    def __init__(self, Cin, Cout, num_blocks, downsample=False, block=ResidualBlock):
        
        super().__init__()
        blocks = [block(Cin, Cout, downsample)]
        for _ in range(num_blocks-1):
            blocks.append(block(Cout * block.expansion, Cout))
        self.net = nn.Sequential(*blocks) 
        
    def forward(self, x):
        return self.net(x)
    
    
class ResNetStem(nn.Module):
    
    def __init__(self, Cin=3, Cout=16):
        
        super().__init__()
        
        self.net = nn.Conv2d(Cin, Cout, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        return self.net(x)
    

class ResNet(nn.Module):
    
    def __init__(self, stage_args, in_channels=3, block=ResidualBlock, num_classes=2):
        
        """
        inputs: 
        - stage_args: A tuple of (C, num_blocks, downsample), where 
            - C: Number of channels
            - num_blocks: Number of blocks 
            - downsample: Add downsampling (a conv with stride=2) if True
        - in_channels: Number of input channels for stem 
        - block: Class of the building block 
        - num_classes: Number of scores to produce from the final linear layer. 
        """
        super().__init__() 
        
        layers = [] 
        Cin = stage_args[0][0]
        for i, (Cout, num_blocks, downsample) in enumerate(stage_args):
            if i == 0:
                layers.append(ResNetStage(Cin, Cout, num_blocks, downsample=downsample, block=block))
            else:
                layers.append(ResNetStage(Cin * block.expansion, Cout, num_blocks, downsample, block=block))
            Cin = Cout
            
        self.cnn = nn.Sequential(
            ResNetStem(Cin=in_channels, Cout=stage_args[0][0]),
            *layers,
            nn.BatchNorm2d(Cin * block.expansion),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.fc = nn.Linear(stage_args[-1][0] * block.expansion, num_classes)
        
    def forward(self, x):
        x = self.cnn(x) 
        x = torch.flatten(x, 1) 
        scores = self.fc(x) 
        return scores
    
    
class ResidualBottleneck(nn.Module):
    
    expansion: int = 4 
    
    def __init__(self, Cin, Cout, downsample=False):
        
        super().__init__() 
        
        if downsample:
            stride=2
        else:
            stride=1 
            
        self.block = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm2d(Cin)), 
            ('relu1', nn.ReLU()),
            ('conv1', nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=1, stride=stride, padding=0, bias=False)),
            ('bn2', nn.BatchNorm2d(Cout)),
            ('relu2', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=Cout, out_channels=Cout, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(Cout)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(in_channels=Cout, out_channels=Cout * 4, kernel_size=1, stride=1,padding=0, bias=False))
        ]))
        
        if (Cout * 4 == Cin) and (downsample == False):
            self.shortcut = nn.Identity() 
        elif Cout * 4 != Cin:
            self.shortcut = nn.Conv2d(in_channels=Cin, out_channels=Cout * 4, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        return self.block(x) + self.shortcut(x)
    
networks = {
    'plain34': {
        'block': PlainBlock,
        'stage_args': [
            (16, 3, False),
            (32, 4, True),
            (64, 6, True),
            (128, 3, True),
        ]
    },
    'resnet34': {
        'block': ResidualBlock,
        'stage_args': [
            (16, 3, False),
            (32, 4, True),
            (64, 6, True),
            (128, 3, True),
        ]
    },
    'resnet50': {
        'block': ResidualBottleneck,
        'stage_args': [
            (16, 3, False),
            (32, 4, True),
            (64, 6, True),
            (128, 3, True),
        ],
    },
}



def get_resnet(arch_name):
    return ResNet(**networks[arch_name])
    
class MyModel(nn.Module):
    def __init__(self, in_channels, img_size, device, dtype):
        super(MyModel, self).__init__()
        self.channels = in_channels
        self.img_size = img_size
        self.device = device 
        self.dtype = dtype 
        # self.k = 8
        self.k = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        # # 1. CNN version
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.0)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            # *discriminator_block(256, 512),
        )
        
        # The height and width of downsampled image
        ds_size = self.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(64 * ds_size ** 2, 300),
            nn.ReLU(),
            nn.Linear(300, 2),
            nn.Softmax(dim=1))
        
        
        # 2. ResNet version
        # self.model = get_resnet('plain34')
        
    def forward(self, img):
        
        # 1. CNN version
        out1 = self.model(img)
        out2 = out1.view(out1.shape[0], -1)
        validity = self.adv_layer(out2)
        # print(validity)
        # 2. ResNet version 
        # validity = self.model(img)
        return validity
    
    def smooth_objective(self, outputs, labels):  
        n0 = torch.sum(1-labels).float() 
        n1 = torch.sum(labels).float() 
        
        # k is a hyperparameter to be tuned
        def f(x):
            return 0.5 * (torch.tanh(self.k * x) + 1)

        tau0 = torch.sum((1-labels) * (outputs[:, 1] > outputs[:, 0])) /  n0
        tau1 = torch.sum((labels) * (outputs[:, 0] > outputs[:, 1])) / n1
        tau0_soft = torch.sum((1-labels) * f(x = outputs[:, 1] - outputs[:, 0])) / n0
        tau1_soft = torch.sum(labels * f(x = outputs[:, 0] - outputs[:, 1])) / n1

        CE_term =  1 - tau0_soft - tau1_soft
        Var_term = torch.sqrt((tau0_soft*(1-tau0_soft)/n0) + (tau1_soft*(1-tau1_soft)/n1))
        return - CE_term / Var_term
        
    def fit(self, trainloader, lr, n_epochs, loss_fn, valloader=None):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-5)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=0.00001, verbose=False)
        
        best_val_loss = float('inf') 
        epochs_without_improvement = 0
        
        # training phase
        for epoch in range(n_epochs):
            self.train()
            train_loss = 0
            same = 0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs) 
                preds = outputs.argmax(dim=1)
                correct = (preds == labels).sum().item()
                same += correct
                loss = loss_fn(outputs, labels) 
                loss.backward()
                optimizer.step() 
                train_loss += loss.item()
                
        return self 
        
    def compute_objective(self, dataloader):
        self.eval()  # Set the model to evaluation mode

        tau0_sum = 0.0
        tau1_sum = 0.0
        n0 = 0.0
        n1 = 0.0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                preds = outputs.argmax(dim=1)
                preds_0 = torch.sum(preds == 0).item() 
                preds_1 = torch.sum(preds == 1).item() 
                
                # print(f"0 preds: {preds_0}, 1 preds: {preds_1}")
                correct = (preds == labels.to(self.device)).sum().item()
                
                # print(f"Correct Predictions: {correct}/{len(labels)}")
                tau0_sum += torch.sum((1-labels) * (outputs[:, 1] > outputs[:, 0]).float()).item()  # Predicted as 1, true label 0
                tau1_sum += torch.sum(labels * (outputs[:, 0] > outputs[:, 1]).float()).item()       # Predicted as 0, true label 1
                
                n0 += torch.sum(1 - labels).item()
                n1 += torch.sum(labels).item()

                loss = self.smooth_objective(outputs, labels)
                
        tau0 = tau0_sum / n0 
        tau1 = tau1_sum / n1

        obj = (1 - tau0 - tau1) / np.sqrt((tau0 * (1 - tau0) / n0) + (tau1 * (1 - tau1) / n1))

        # print(f"The power statistic: {obj.item()}")
        return loss.item(), obj.item(), tau0, tau1, correct

class NeuralNet(torch.nn.Module):
    """define deep neural networks"""
    def __init__(self, x_in, H, x_out, device, dtype):
        super().__init__()
        self._x_in = x_in
        self._H = H
        self._x_out = x_out
        self.device = device
        self.dtype = dtype
        self.W = nn.Linear(self._x_out, 2)
        self.dropout = nn.Dropout(0.2)  # Add dropout layer with dropout rate 0.5
        self.k = 8 
        
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H),
            torch.nn.BatchNorm1d(H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 2*H),
            torch.nn.BatchNorm1d(2*H),
            torch.nn.ReLU(),
            torch.nn.Linear(2*H, 4*H),
            torch.nn.BatchNorm1d(4*H),
            torch.nn.ReLU(),
            torch.nn.Linear(4*H, 2*H),
            torch.nn.BatchNorm1d(2*H),
            torch.nn.ReLU(),
            torch.nn.Linear(2*H, H),
            torch.nn.BatchNorm1d(H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, x_out),
        )

    
    def forward(self, input):
        fealant = self.latent(input.to(self.dtype))
        fealant = self.W(fealant)
        return fealant.to(self.dtype)
    
    def smooth_objective(self, outputs, labels):    
        n0 = torch.sum(1-labels).float() 
        n1 = torch.sum(labels).float() 
        def f(x):
            return 0.5 * (torch.tanh(self.k * x) + 1)

        preds = outputs.argmax(dim=1)

        tau0 = torch.sum((1-labels) * (outputs[:, 1] > outputs[:, 0])) /  n0
        tau1 = torch.sum((labels) * (outputs[:, 0] > outputs[:, 1])) / n1
        tau0_soft = torch.sum((1-labels) * f(x = outputs[:, 1] - outputs[:, 0])) / n0
        tau1_soft = torch.sum(labels * f(x = outputs[:, 0] - outputs[:, 1])) / n1
                        
        # CE_term = loss_fn(outputs, labels.long())

        CE_term =  1 - tau0_soft - tau1_soft
        var_term = 1 * torch.sqrt((tau0_soft*(1-tau0_soft)/n0) + (tau1_soft*(1-tau1_soft)/n1))
        # print(f"tau0: {np.round(tau0_soft.item(), 4)}, tau1: {np.round(tau1_soft.item(), 4)}")
        return - CE_term / var_term
        
    def fit(self, trainloader, lr, n_epochs, loss_fn, valloader=None):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=0.00001, verbose=False)
        for epoch in range(n_epochs):
            self.train()
            train_loss = 0
            same = 0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = self(inputs) 
                preds = outputs.argmax(dim=1)
                preds_0 = torch.sum(preds == 0).item() 
                preds_1 = torch.sum(preds == 1).item() 
                
                # print(f"0 preds: {preds_0}, 1 preds: {preds_1}")
                correct = (preds == labels.to(self.device)).sum().item()
                
                # print(f"Batch {i+1}, Correct Predictions: {correct}/{len(labels)}")
                same += correct
                loss = loss_fn(outputs, labels) 
                loss.backward()
                optimizer.step() 
                train_loss += loss.item()
        
            # validation phase 
            if valloader :
                self.eval() 
                valid_obj = 0.0 
                
                with torch.no_grad():
                    val_loss, valid_obj, tau0, tau1 = self.compute_objective(valloader)
                    scheduler.step(-valid_obj)
                
        return self 
    
        
    def compute_objective(self, dataloader):
        self.eval()  # Set the model to evaluation mode

        tau0_sum = 0.0
        tau1_sum = 0.0
        n0 = 0.0
        n1 = 0.0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                preds = outputs.argmax(dim=1)
                preds_0 = torch.sum(preds == 0).item() 
                preds_1 = torch.sum(preds == 1).item() 
                
                # print(f"0 preds: {preds_0}, 1 preds: {preds_1}")
                correct = (preds == labels.to(self.device)).sum().item()
                
                # print(f"Correct Predictions: {correct}/{len(labels)}")
                tau0_sum += torch.sum((1-labels) * (outputs[:, 1] > outputs[:, 0]).float()).item()  # Predicted as 1, true label 0
                tau1_sum += torch.sum(labels * (outputs[:, 0] > outputs[:, 1]).float()).item()       # Predicted as 0, true label 1
                
                n0 += torch.sum(1 - labels).item()
                n1 += torch.sum(labels).item()

                loss = self.smooth_objective(outputs, labels)
                
        tau0 = tau0_sum / n0 
        tau1 = tau1_sum / n1

        obj = (1 - tau0 - tau1) / np.sqrt((tau0 * (1 - tau0) / n0) + (tau1 * (1 - tau1) / n1))

        # print(f"The power statistic: {obj.item()}")
        return loss.item(), obj.item(), tau0, tau1, correct
    


class MNISTResNet(nn.Module):
    def __init__(self, base_model, in_channels, img_size, device, dtype):
        super().__init__()
        self.channels = in_channels
        self.img_size = img_size
        self.device = device 
        self.dtype = dtype 
        # Adjust the input layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Use all layers except the first (conv1) and the last (fc)
        self.features = nn.Sequential(*list(base_model.children())[1:-1])
        # Adjust the final layer
        self.fc = nn.Linear(512, 2)  # Assuming you use ResNet18 or 34. For ResNet50/101/152, use 2048 instead of 512.

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def smooth_objective(self, outputs, labels):  
        lamb = 1e-8  
        n0 = torch.sum(1-labels).float() 
        n1 = torch.sum(labels).float() 
        
        # k is a hyperparameter to be tuned
        def f(x, k=8):
            return 0.5 * (torch.tanh(k * x) + 1)

        preds = outputs.argmax(dim=1)

        tau0 = torch.sum((1-labels) * (outputs[:, 1] > outputs[:, 0])) /  n0
        tau1 = torch.sum((labels) * (outputs[:, 0] > outputs[:, 1])) / n1
        # tau0_soft = torch.sum((1-labels) * outputs[:, 1]) /  n0
        # tau1_soft = torch.sum((labels) * outputs[:, 0]) / n1
        tau0_soft = torch.sum((1-labels) * f(x = outputs[:, 1] - outputs[:, 0])) / n0
        tau1_soft = torch.sum(labels * f(x = outputs[:, 0] - outputs[:, 1])) / n1

        CE_term =  1 - tau0_soft - tau1_soft
        Var_term = 1 * torch.sqrt((tau0_soft*(1-tau0_soft)/n0) + (tau1_soft*(1-tau1_soft)/n1))
        # print(f"tau0: {np.round(tau0_soft.item(), 4)}, tau1: {np.round(tau1_soft.item(), 4)}")
        max_sample_size = torch.max(n0, n1)
        return - CE_term / ((Var_term) * max_sample_size)
        # return - CE_term / Var_term
        
    def fit(self, trainloader, lr, n_epochs, loss_fn, valloader=None):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-5)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=0.00001, verbose=False)
        
        best_val_loss = float('inf') 
        epochs_without_improvement = 0
        
        # training phase
        for epoch in range(n_epochs):
            self.train()
            train_loss = 0
            same = 0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = self(inputs) 
                preds = outputs.argmax(dim=1)
                correct = (preds == labels.to(self.device)).sum().item()
                same += correct
                loss = loss_fn(outputs, labels) 
                loss.backward()
                optimizer.step() 
                train_loss += loss.item()
            
            # validation phase 
            if valloader :
                self.eval() 
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in valloader: 
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self(inputs) 
                        loss = loss_fn(outputs, labels)
                        val_loss += loss
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss 
                    epochs_without_improvement = 0 
                else:
                    epochs_without_improvement += 1
                
                if (epoch+1) % 100 == 0:
                    print(f'Epoch: {epoch+1}, Training Stats: {-train_loss:.4f}, Training Accuracy: {same / len(trainloader.dataset):.4f}, Validation Stats: {-val_loss:.4f}')
            
            else:
                scheduler.step(train_loss)
                if (epoch +1) % 100 == 0:
                    print(f'Epoch: {epoch+1}, Training Stats: {-train_loss:.4f}, Training Accuracy: {same / len(trainloader.dataset):.4f}')

        return self 
    
    def predict_proba(self, x):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self(x) 
            return outputs.cpu().numpy()
        
    def compute_objective(self, dataloader):
        self.eval()  # Set the model to evaluation mode

        tau0_sum = 0.0
        tau1_sum = 0.0
        n0 = 0.0
        n1 = 0.0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                preds = outputs.argmax(dim=1)
                preds_0 = torch.sum(preds == 0).item() 
                preds_1 = torch.sum(preds == 1).item() 
                
                # print(f"0 preds: {preds_0}, 1 preds: {preds_1}")
                correct = (preds == labels.to(self.device)).sum().item()
                
                # print(f"Correct Predictions: {correct}/{len(labels)}")
                tau0_sum += torch.sum((1-labels) * (outputs[:, 1] > outputs[:, 0]).float()).item()  # Predicted as 1, true label 0
                tau1_sum += torch.sum(labels * (outputs[:, 0] > outputs[:, 1]).float()).item()       # Predicted as 0, true label 1
                
                n0 += torch.sum(1 - labels).item()
                n1 += torch.sum(labels).item()

                loss = self.smooth_objective(outputs, labels)
                
        tau0 = tau0_sum / n0 
        tau1 = tau1_sum / n1

        obj = (1 - tau0 - tau1) / np.sqrt((tau0 * (1 - tau0) / n0) + (tau1 * (1 - tau1) / n1))

        # print(f"The power statistic: {obj.item()}")
        return loss.item(), obj.item(), tau0, tau1, correct


