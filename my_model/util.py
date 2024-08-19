import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter

path_all = './results/'

def setup_seed(seed):
    np.random.seed(seed) #
    random.seed(seed)  # python random module
    os.environ['PYTHONHASHSEED'] = str(seed) #
    torch.manual_seed(seed) #
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) #
        torch.cuda.manual_seed_all(seed) # 
        torch.backends.cudnn.benchmark = False # 
        torch.backends.cudnn.deterministic = True 
                                                    
            
        
     
def count_labels(data_loader):
    labels = []
    for _, label in data_loader:
        labels.extend(label.tolist())
    return Counter(labels)



def compute_mean_std(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1) 
    return round(mean, 4), round(std, 4)








