"""
Created on Tuesday April 23 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""
import numpy as np

class UnifAverageLabelSampler(Sampler):

    def __init__(self, dataset, dataset_multiplier=1):
        self.imgs = dataset.imgs
        self.dataset_multiplier = dataset_multiplier
        self.indexes = self._generate_indexes_epoch()

    def _group_indices_by_target(self):
        n_targets = np.unique([ target for (_,target) in self.imgs])
        grouped_indices = [[] for i in range(n_targets)]
        for i, (path,target) in enumerate(self.imgs):
            grouped_indices[target].append(i)
        return grouped_indices

    def _generate_indexes_epoch(self):
        grouped_indices = self._group_indices_by_target()

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

        np.random.shuffle(res)
        return res.astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)
