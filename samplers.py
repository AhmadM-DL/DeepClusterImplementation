"""
Created on Tuesday April 23 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""
import numpy as np
from deep_clustering_dataset import DeepClusteringDataset
import torch

class UnifAverageLabelSampler(torch.utils.data.Sampler):

    def __init__(self, dataset: DeepClusteringDataset, dataset_multiplier=1, shuffle= True):
        self.dataset = dataset
        self.dataset_multiplier = dataset_multiplier
        self.shuffle = shuffle
        self.indexes = self._generate_indexes_epoch()

    # ToDo : re_implement size constraints
    def _generate_indexes_epoch(self):

        grouped_indices = self.dataset.group_indices_by_labels()

        # nmb_non_empty_clusters = 0
        # for i in range(len(self.images_lists)):
        #     if len(self.images_lists[i]) != 0:
        #         nmb_non_empty_clusters += 1
                
        target_sizes = [len(target_group) for target_group in grouped_indices]
        avg_target_size = int(np.average(target_sizes)) + 1
        n = int(self.dataset_multiplier * avg_target_size * len(grouped_indices))

        res = np.zeros(n)

        for i, target_group in enumerate(grouped_indices):
            indexes = np.random.choice(
                target_group,
                avg_target_size,
                replace=(len(target_group) <= avg_target_size)
            )
            res[i * avg_target_size: (i + 1) * avg_target_size] = indexes

        if self.shuffle:
            np.random.shuffle(res)

        # res = list(res.astype('int'))
        # if len(res) >= self.N:
        #     return res[:self.N]
        # res += res[: (self.N - len(res))]

        return res.astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)
