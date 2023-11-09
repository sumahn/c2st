import logging # a package that provides a way to configure and generate log messages
import os 
import sys 
import torch 
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast 
from torch.utils.tensorboard import SummaryWriter 
from tqdm.auto import tqdm 
from utils_simclr import *

torch.manual_seed(0) 

class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args'] 
        self.model = kwargs['model'].to(self.args.device) 
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter() 
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device) 
        
    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        # Feature Normalization
        features = F.normalize(features, dim=1)
        # 위의 feature 기반으로 similarity matrix 계산
        # -> 이 simliarty matrix를 MMD로 바꾼다?
        # https://arxiv.org/pdf/2208.00789.pdf 이것 참고해보자 
        similarity_matrix = torch.matmul(features, features.T)
                
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # Select and combine multiple positives 
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        
        # Select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) 
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        
        logits = logits/self.args.temperature
        
        return logits,labels 
    
    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        # Save config file 
        save_config_file(self.writer.log_dir, self.args)
        
        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.") 
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        
        for epoch in range(self.args.epochs):
            for batch in tqdm(train_loader):
                images0 = batch[:, 0, :, :]
                images1 = batch[:, 1, :, :]
                images = torch.cat([images0, images1], dim=0)
                images = images.to(self.args.device)
                # Mixed-precision training
                # Performance improvement by leveraging GPU more effectively
                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels) 
                
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                n_iter += 1
            
            # Warmup for the first 10 epochs 
            if epoch >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
        
        logging.info("Training has finished.")
        # Save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch':self.args.epochs,
            'arch':self.args.arch,
            'state_dict':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        
                    