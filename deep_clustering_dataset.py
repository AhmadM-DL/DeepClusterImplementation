"""
Created on Tuesday April 20 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets import VisionDataset
import torch
import copy
import numpy as np
import os

class DeepClusteringDataset(Dataset):
    """ A Datset **Decorator** that adds changing labels to pseudolabels
    functionality.
    Args:
        original_dataset (list): Pytorch Dataset
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, original_dataset, transform=None):
        self.dataset = copy.deepcopy(original_dataset)
        self.original_dataset = original_dataset

        if isinstance(self.original_dataset, ImageFolder):
            self.imgs = self.dataset.imgs

        elif isinstance(self.original_dataset, VisionDataset):
            self.data = self.dataset.data
            if hasattr(self.dataset, "targets"):
                self.targets = self.dataset.targets
            elif hasattr(self.dataset, "labels"):
                self.targets = self.dataset.labels
            else: 
                raise Exception("The entered dataset is not supported - no labels/targets variables")
        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")

        if transform:
            self.dataset.transform = transform
        else:
            self.dataset.transform = original_dataset.transform

        self.transform = self.dataset.transform
        self.instance_wise_weights= None
    
    def set_transform(self, transform):
        self.dataset.transform = transform
        self.transform = self.dataset.transform

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, index):
        if self.instance_wise_weights:
            return self.dataset.__getitem__(index)+ (self.instance_wise_weights[index],)
        else:
            return self.dataset.__getitem__(index)

    def get_targets(self):
        if isinstance(self.original_dataset, ImageFolder):
            return [target for (path ,target) in self.original_dataset.imgs]

        elif isinstance(self.original_dataset, VisionDataset):
            
            if hasattr(self.original_dataset, "targets"):
                return self.original_dataset.targets
            
            elif hasattr(self.original_dataset, "labels"):
                return self.original_dataset.labels
            
            else: 
                raise Exception("The entered dataset is not supported - no labels/targets variables")
        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")
    
    def set_pseudolabels(self, pseudolabels):

        if isinstance(self.dataset, ImageFolder):
            for i, pseudolabel in enumerate(pseudolabels):
                self.imgs[i] = (self.imgs[i][0], torch.tensor(pseudolabel, dtype=torch.long))

        elif isinstance(self.dataset, VisionDataset):

            if hasattr(self.original_dataset, "targets"):
                self.dataset.targets = torch.tensor(pseudolabels, dtype=torch.long)
                self.targets = self.dataset.targets
            
            elif hasattr(self.original_dataset, "labels"):
                self.dataset.labels = torch.tensor(pseudolabels, dtype=torch.long)
                self.targets = self.dataset.labels            
            else: 
                raise Exception("The entered dataset is not supported - no labels/targets variables")
        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")
    
    # TODO - remove if unused
    def set_instance_wise_weights(self, weights):
        self.instance_wise_weights = weights
        return

    def unset_instance_wise_weights(self):
        self.instance_wise_weights = None
        return

    def save_pseudolabels(self, path , tag):

        if isinstance(self.dataset, ImageFolder):
            data_to_save = [ (index,pseudolabel.item()) for (index, (path, pseudolabel)) in enumerate(self.imgs)]
            np.save(os.path.join(path, tag), data_to_save )

        elif isinstance(self.dataset, VisionDataset):
            
            data_to_save = [ (index,pseudolabel) for (index, pseudolabel) in enumerate(self.targets)]
            np.save(os.path.join(path, tag), data_to_save )

        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")


    def get_pseudolabels(self):
        if isinstance(self.dataset, ImageFolder):
            return [pseudolabel.item() for (path, pseudolabel) in self.imgs]

        elif isinstance(self.dataset, VisionDataset):
             return self.targets

        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")

    def unset_pseudolabels(self):
        if isinstance(self.dataset, ImageFolder):
            self.imgs= self.original_dataset.imgs

        elif isinstance(self.dataset, VisionDataset):
             self.targets = self.original_dataset.targets

        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")
                

    def group_indices_by_labels(self):
        if isinstance(self.dataset, ImageFolder):
            n_labels = len(np.unique([ label for (_, label) in self.imgs]))
            grouped_indices = [[] for i in range(n_labels)]
            for i, (path, label) in enumerate(self.imgs):
                grouped_indices[label].append(i)
            return grouped_indices

        elif isinstance(self.dataset, VisionDataset):
            n_labels = len(np.unique(self.targets))
            grouped_indices = [[] for i in range(n_labels)]
            for i, label in enumerate(self.targets):
                grouped_indices[label].append(i)
            return grouped_indices

        else:
            raise Exception("The passed original dataset is of unsupported dataset instance")
             

