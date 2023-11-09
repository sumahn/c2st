import os 
import shutil  # package for file operations
import torch 
import yaml 

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # 모델 save
    torch.save(state, filename)
    if is_best: # 가장 결과가 좋은 모델은 best_model로 copy해서 저장
        shutil.copyfile(filename, 'best_model.pth.tar')

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)
            
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the top k predictions
    for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk) 
        batch_size = target.size(0) 
        
        _, pred = output.topk(maxk, 1, True, True) 
        pred = pred.t() 
        correct = pred.eq(target.view(1, -1).expand_as(pred)) 
        
        res = [] 
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) 
            res.append(correct_k.mul_(100.0 / batch_size)) 
        return res
    