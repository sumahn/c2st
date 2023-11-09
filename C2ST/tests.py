import numpy as np 
import torch
from util import MMDu_var
from scipy.stats import norm

# MMD-D
def deep_mmd_permutation(X, Y, params, model, n_perm, device):
    mmd = MMDu_var(X, Y, params, model, device)[0]
    perm_stat = torch.zeros(n_perm)
    count = 0 
    N = X.shape[0]
    
    for i in range(n_perm):
        idx = torch.randperm(N)
        perm_Y = Y[idx, :]
        perm_mmd = MMDu_var(X, perm_Y, params, model, device)[0]
        
        if perm_mmd >= mmd:
            count += 1
        else:
            count += 0 
        
        # Compute p-value 
        p_value = (count + 1) / (n_perm + 1)
        
        return p_value, mmd
    

def TST_MMD(X, Y, params, model, n_perm, alpha, device):
    tmp = MMDu_var(X, Y, params, model, device)[0]
    p_value, mmd = deep_mmd_permutation(X, Y, params, model, n_perm, device)
    
    if p_value > alpha: 
        h = 0           # Do not reject H0 
    else:
        h = 1           # Reject H0 
    
    return h, tmp, p_value



# C2ST 
def TST_C2ST(X, Y, model, alpha, learning_rate, n_epochs, seed, loss_fn, device):
    
    from sklearn.model_selection import train_test_split
    from split_data import split_data
    
    labels = (torch.cat((torch.zeros(len(X), 1), torch.ones(len(Y), 1)), 0)).squeeze(1).to(device).long()
    labels_np = labels.to('cpu').numpy() 
    
    # Merge two dataset
    dataset = torch.cat([X, Y], dim=0).to('cpu')
    
    n = len(dataset)
    # print(f"Total sample size : {n}")
    # Split to training and test dataset
    
    # Imbalanced training set / Imbalanced validation and test dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels_np, test_size = int(0.5 * n), random_state=seed)
    # X_te, X_val, y_te, y_val = train_test_split(X_test, y_test, test_size=int(0.2 * len(X_test)), random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=int(0.2 * len(X_train)), random_state=seed)
    
    
    # Imbalanced training set / Balanced validation and test dataset
    # X_train, X_val, X_test, y_train, y_val, y_test = split_data(dataset, labels_np, 0.5, 0.1)
        
    # Dataloader 
    train_dataset = torch.utils.data.TensorDataset(X_train.to(device), torch.tensor(y_train).to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = len(train_dataset), shuffle=True)
    # val_dataset = torch.utils.data.TensorDataset(X_val.to(device), torch.tensor(y_val).to(device)) 
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = len(val_dataset), shuffle=True) 
    test_dataset = torch.utils.data.TensorDataset(X_test.to(device), torch.tensor(y_test).to(device)) 
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    # Train model(classifier)
    model.fit(train_dataloader, lr = learning_rate, n_epochs = n_epochs, loss_fn = loss_fn, valloader=None)
    
    # Calculate test statistics
    _, stats, tau0, tau1, correct = model.compute_objective(test_dataloader)

    # Decision 
    if stats >= 1 - norm.ppf(alpha):
        h = 1
    else:
        h = 0 
    
    print(f"stats: {np.round(stats, 4)}, tau0: {np.round(tau0, 4)}, tau1: {np.round(tau1, 4)}, accuracy: {correct / len(test_dataset)}")
    return h


def c2st_tst(X_tr, Y_tr, X_te, Y_te,  model, alpha, learning_rate, n_epochs, seed, loss_fn, device):
    
    labels_tr = (torch.cat((torch.zeros(len(X_tr), 1), torch.ones(len(Y_tr), 1)), 0)).squeeze(1).to(device).long()
    labels_te = (torch.cat((torch.zeros(len(X_te), 1), torch.ones(len(Y_te), 1)), 0)).squeeze(1).to(device).long()
    
    # Merge two dataset
    tr_dataset = torch.cat([X_tr, Y_tr], dim=0)
    te_dataset = torch.cat([X_te, Y_te], dim=0)
           
    # Dataloader 
    train_dataset = torch.utils.data.TensorDataset(tr_dataset, labels_tr)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = len(train_dataset), shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(te_dataset, labels_te) 
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    # Train model(classifier)
    model.fit(train_dataloader, lr = learning_rate, n_epochs = n_epochs, loss_fn = loss_fn, valloader=None)
    
    # Calculate test statistics
    _, stats, tau0, tau1, correct = model.compute_objective(test_dataloader)

    # Decision 
    if stats >= 1 - norm.ppf(alpha):
        h = 1
    else:
        h = 0 
    
    print(f"stats: {np.round(stats, 4)}, tau0: {np.round(tau0, 4)}, tau1: {np.round(tau1, 4)}, accuracy: {correct / len(test_dataset)}")
    return h

    