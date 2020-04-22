"""
Created on Tuesday April 20 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

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
        self.imgs = self.original_imgs
        self.transform = original_dataset.transform
    
    def set_pseudolabels(self, pseudolabels):
        for i, pseudolabel in enumerate(pseudolabels):
            self.imgs[i][1] = pseudolabel
    
    def unset_pseudolabels(self):
        self.imgs= self.original_imgs