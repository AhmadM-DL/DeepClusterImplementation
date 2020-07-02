"""
Created on Tuesday April 24 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""
import torch 
import numpy as np
import random 
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)