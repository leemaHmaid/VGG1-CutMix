import numpy as np
import torch
from torch.utils.data import  DataLoader
from CustomImageDataset import train_dataset


def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (targets, shuffled_targets, lam)
    
    return data, targets

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutMixCollator:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        data, targets = zip(*batch)
        data = torch.stack(data)
        targets = torch.tensor(targets)
        data, targets = cutmix(data, targets, self.alpha)
        
        return data, targets
    


train_loader_cutmix = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4,collate_fn=CutMixCollator(alpha=1.0))
    
