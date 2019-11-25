# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 13:26:39 2019

@author: amm90
"""

import time

import numpy as np
import utils
import torch.utils.data as data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class labels_reassigned_dataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, original_dataset, image_indexes, pseudolabels, transform=None):

        self.imgs = self.make_dataset(image_indexes, pseudolabels, original_dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):

        images = []
        for idx in image_indexes:
            path = dataset.imgs[idx][0]
            images.append((path, pseudolabels[idx]))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel = self.imgs[index]
        img = utils.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)


def clustered_data_indices_to_list(clustered_data_indices):
    pseudolabels = []
    image_indexes = []

    for cluster, images in enumerate(clustered_data_indices):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    indexes = np.argsort(image_indexes)

    pseudolabels = np.asarray(pseudolabels)[indexes]

    return [np.sort(image_indexes), pseudolabels]


class neural_features_kmeans_with_preprocessing():

    def __init__(self, data, n_clusters, pca=0, verbose=0, **kwargs):

        self.data = data
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.verbose = verbose
        self.pca = pca

        self.clustered_data_indices = [[] for i in range(n_clusters)]

        self.preprocessed_data = None
        self.inertia = None
        self.assignments = None

    def cluster(self):

        end = time.time()

        # Preprocess features
        self.preprocessed_data = self.__preprocess_neural_features(data=self.data, pca=self.pca, verbose=self.verbose,
                                                                   random_State= self.kwargs.get("random_state", None) )

        if self.verbose:
            print('Preprocessing Features (PCA, Whitening, L2_normalization) Time: {0:.0f} s'.format(time.time() - end))

        kmeans_object = KMeans(self.n_clusters, max_iter=self.kwargs.get("max_iter", 20),
                               n_init=self.kwargs.get("n_init", 1),
                               verbose=1, random_state=self.kwargs.get("random_state", None))

        kmeans_object.fit_predict(self.preprocessed_data)

        if self.verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        self.inertia = kmeans_object.inertia_
        self.assignments = kmeans_object.labels_

        if self.verbose:
            print('k-means loss evolution (inertia): {0}'.format(self.inertia))

        for i in range(len(self.data)):
            self.clustered_data_indices[self.assignments[i]].append(i)

    def __preprocess_neural_features(self, data, pca=256, verbose=0, random_State=None):

        """Preprocess an array of features.
        Args:
            data (np.array N * ndim): features to preprocess
            pca (int): dim of output
        Returns:
            np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
        """
        _, ndim = data.shape
        data = data.astype('float32')

        # Apply PCA-whitening with sklearn pca
        if(pca):
            if(verbose):print("Applying PCA with %d components on features"%(pca))
            mat = PCA(n_components=pca, whiten=True, random_state= random_State)
            mat.fit(data)
            data = mat.transform(data)

        # L2 normalization
        if(verbose):print("Computing L2 norm of features")
        row_sums = np.linalg.norm(data, axis=1)
        data = data / row_sums[:, np.newaxis]

        return data
