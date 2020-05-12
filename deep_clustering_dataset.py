"""
Created on Tuesday April 20 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets import VisionDataset
import torch
import numpy as np

class DeepClusteringDataset(Dataset):
    """ A Datset Decorator that adds changing labels to pseudolabels
    functionality.
    Args:
        original_dataset (list): Pytorch Dataset
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

        if isinstance(self.original_dataset, ImageFolder):
            self.imgs = self.original_dataset.imgs.copy()

        elif isinstance(self.original_dataset, VisionDataset):
            self.data = original_dataset.data
            self.targets = original_dataset.targets
        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")

        self.transform = original_dataset.transform
            
    def __len__(self):
        return self.original_dataset.__len__()
    
    def __getitem__(self, index):
        return self.original_dataset.__getitem__(index)

    def get_targets(self):
        if isinstance(self.original_dataset, ImageFolder):
            return [target for (path ,target) in self.original_dataset.imgs]

        elif isinstance(self.original_dataset, VisionDataset):
            return self.targets

        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")
    
    def set_pseudolabels(self, pseudolabels):

        if isinstance(self.original_dataset, ImageFolder):
            for i, pseudolabel in enumerate(pseudolabels):
                self.imgs[i] = (self.imgs[i][0], torch.tensor(pseudolabel, dtype=torch.long))

        elif isinstance(self.original_dataset, VisionDataset):
             self.targets = pseudolabels

        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")
    
    def get_pseudolabels(self):
        if isinstance(self.original_dataset, ImageFolder):
            return [pseudolabel.item() for (path, pseudolabel) in self.imgs]

        elif isinstance(self.original_dataset, VisionDataset):
             return self.targets

        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")
        
    def unset_pseudolabels(self):
        if isinstance(self.original_dataset, ImageFolder):
            self.imgs= self.original_dataset.imgs

        elif isinstance(self.original_dataset, VisionDataset):
             self.targets = self.original_dataset.targets

        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")
                

    def group_indices_by_labels(self):
        if isinstance(self.original_dataset, ImageFolder):
            n_labels = len(np.unique([ label for (_, label) in self.imgs]))
            grouped_indices = [[] for i in range(n_labels)]
            for i, (path, label) in enumerate(self.imgs):
                grouped_indices[label].append(i)
            return grouped_indices

        elif isinstance(self.original_dataset, VisionDataset):
            n_labels = len(np.unique(self.targets))
            grouped_indices = [[] for i in range(n_labels)]
            for i, label in enumerate(self.targets):
                grouped_indices[label].append(i)
            return grouped_indices

        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")
             

