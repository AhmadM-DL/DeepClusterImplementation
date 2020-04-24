"""
Created on Tuesday April 20 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch
import numpy as np

class DeepClusteringDataset(Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, original_dataset, transform=None):
        self.original_imgs = original_dataset.imgs
        self.original_dataset = original_dataset
        self.imgs = self.original_imgs
        self.transform = original_dataset.transform
            
    def __len__(self):
        return self.original_dataset.__len__()
    
    def __getitem__(self, index):
        return self.original_dataset.__getitem__(index)

    def get_targets(self):
        return [target for (path,target) in self.original_dataset.imgs]
    
    def set_pseudolabels(self, pseudolabels):
        for i, pseudolabel in enumerate(pseudolabels):
            self.imgs[i] = (self.imgs[i][0], torch.tensor(pseudolabel, dtype=torch.long))
    
    def get_pseudolabels(self):
        return [pseudolabel.item() for (path, pseudolabel) in self.imgs]
    
    def unset_pseudolabels(self):
        self.imgs= self.original_imgs

    def group_indices_by_labels(self):
        n_labels = len(np.unique([ label for (_, label) in self.imgs]))
        grouped_indices = [[] for i in range(n_labels)]
        for i, (path, label) in enumerate(self.imgs):
            grouped_indices[label].append(i)
        return grouped_indices
