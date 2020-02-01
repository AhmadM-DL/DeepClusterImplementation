# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 13:26:39 2019

@author: amm90
"""

import time
import numpy as np
import utils
import os
import torch.utils.data as data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy
from PIL import Image


def split_dataset(original_dataset, split_ratios: list, random_state=0):
    subsets = []
    if not np.sum(split_ratios) == 1:
        raise Exception("Error the passed split_ratios doesn't sum to 1")
    all_indices = range(0,len(original_dataset))
    random_splitter = np.random.RandomState(random_state)
    for ratio in split_ratios:
        subset_indices = random_splitter.choice(all_indices, int(len(original_dataset)*ratio))
        subsets.append(MySubset(original_dataset, subset_indices))
    return subsets


class MySubset(data.Dataset):

    def __init__(self, original_dataset: data.Dataset, indices: list) -> None:
        self.imgs = [original_dataset.imgs[i] for i in indices]
        self.transform = original_dataset.transform

    def __getitem__(self, index: int):
        path, pseudolabel = self.imgs[index]
        img = utils.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self) -> int:
        return len(self.imgs)


class LabelsReassignedDataset(data.Dataset):
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
        for i, idx in enumerate(image_indexes):
            path = dataset.imgs[idx][0]
            images.append((path, pseudolabels[i]))
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


def clustered_data_indices_to_list(clustered_data_indices, reindex=False):
    pseudolabels = []
    image_indexes = []

    for cluster, images in enumerate(clustered_data_indices):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    indexes = np.argsort(image_indexes)
    pseudolabels = np.asarray(pseudolabels)[indexes]
    image_indexes = np.sort(image_indexes)

    if (reindex):
        reindexed_clustered_data_indices = [[] for i in range(len(clustered_data_indices))]
        for i in range(len(image_indexes)):
            reindexed_clustered_data_indices[pseudolabels[i]].append(i)

    if (not reindex):
        return [image_indexes, pseudolabels]
    else:
        return [image_indexes, pseudolabels], reindexed_clustered_data_indices


class Neural_Features_Clustering_With_Preprocessing():

    def __init__(self, data, pca=0, verbose=0, **kwargs):

        self.data = data
        self.kwargs = kwargs
        self.verbose = verbose
        self.pca = pca

        self.clustered_data_indices = None  # [[] for i in range(n_clusters)]

        self.preprocessed_data = None
        self.assignments = None
        self.koutputs = {}

    def cluster(self, algorithm="kmeans", **kwargs):

        end = time.time()

        # Preprocess features
        self.preprocessed_data, self.koutputs["pca"] = self.__preprocess_neural_features(data=self.data, pca=self.pca,
                                                                                         verbose=self.verbose,
                                                                                         random_State=self.kwargs.get(
                                                                                             "random_state", None),
                                                                                         return_pca_object=True)

        if self.verbose:
            print('Preprocessing Features (PCA, Whitening, L2_normalization) Time: {0:.0f} s'.format(time.time() - end))

        if (algorithm == "kmeans"):
            clustering_object = KMeans(kwargs.get("n_clusters"), max_iter=self.kwargs.get("max_iter", 20),
                                       n_init=self.kwargs.get("n_init", 1),
                                       verbose=1, random_state=self.kwargs.get("random_state", None))

            clustering_object.fit_predict(self.preprocessed_data)
            self.koutputs["inertia"] = clustering_object.inertia_

            if self.verbose: print('k-means time: {0:.0f} s'.format(time.time() - end))
            if self.verbose: print('k-means loss evolution (inertia): {0}'.format(self.koutputs["inertia"]))

        elif (algorithm == "hdbscan"):
            clustering_object = hdbscan.HDBSCAN(min_cluster_size=kwargs.get("min_cluster_size", 100),
                                                min_samples=kwargs.get("min_samples", 100),
                                                cluster_selection_method=kwargs.get("cluster_selection_method", "eom"),
                                                metric=kwargs.get("metric", "euclidean"),
                                                memory=kwargs.get("memory", None)
                                                )
            clustering_object.fit_predict(self.preprocessed_data)

            self.koutputs["probabilities"] = clustering_object.probabilities_
            self.koutputs["condensed_tree"] = clustering_object.condensed_tree_

            if self.verbose: print('hdbscan time: {0:.0f} s'.format(time.time() - end))

        self.assignments = clustering_object.labels_
        number_of_clusters = clustering_object.labels_.max() + 1

        self.clustered_data_indices = [[] for i in range(number_of_clusters)]

        for i in range(len(self.data)):
            self.clustered_data_indices[self.assignments[i]].append(i)

    def __preprocess_neural_features(self, data, pca=256, verbose=0, random_State=None, return_pca_object=False):

        """Preprocess an array of features.
        Args:
            data (np.array N * ndim): features to preprocess
            pca (int): dim of output
        Returns:
            np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
        """
        _, ndim = data.shape
        data = data.astype('float32')
        mat = None

        # Apply PCA-whitening with sklearn pca
        if (pca):
            if (verbose): print("Applying PCA with %d components on features" % (pca))
            mat = PCA(n_components=pca, whiten=True, random_state=random_State)
            mat.fit(data)
            data = mat.transform(data)

        # L2 normalization
        if (verbose): print("Computing L2 norm of features")
        row_sums = np.linalg.norm(data, axis=1)
        data = data / row_sums[:, np.newaxis]

        if (return_pca_object):
            return data, mat
        else:
            return data


def cross_2_models_clustering_output(model_1_clusters, model_2_clusters, take_top=None):
    results = []
    sizes = []

    model_1_clusters_as_sets = np.array([set(x) for x in model_1_clusters])
    model_2_clusters_as_sets = np.array([set(x) for x in model_2_clusters])

    for c1 in model_1_clusters_as_sets:
        for c2 in model_2_clusters_as_sets:
            cr = c1 & c2
            results.append(cr)
            sizes.append(len(cr))

    sorted_indices_by_size = np.argsort(sizes)[::-1]  ## descending order
    results = np.array(results)[sorted_indices_by_size]
    results = [list(x) for x in results]

    if (take_top):
        return results[:take_top]
    else:
        return results[:len(model_1_clusters)]


class ClusteringTracker(object):

    def __init__(self):
        self.clustering_log = []

    def update(self, epoch, clustered_data_indices):
        self.clustering_log.append( (epoch, clustered_data_indices) )

    def size_new_data_btw_epochs(self):

        new_data_sizes = []

        for i in range(1, len(self.clustering_log)):
            prev_clusters = self.clustering_log[i - 1][1]
            curr_clusters = self.clustering_log[i][1]

            flat_prev_clusters = set([item for cluster in prev_clusters for item in cluster])
            flat_curr_clusters = set([item for cluster in curr_clusters for item in cluster])

            new_data_indices = flat_curr_clusters - flat_prev_clusters

            new_data_sizes.append(len(new_data_indices))

        return new_data_sizes

    def epochs_avg_entropy(self, ground_truth):
        avg_entropies = []

        for (_, clusters) in self.clustering_log:
            entropies = []
            for cluster in clusters:
                images_original_classes = [ground_truth[image_index] for image_index in cluster]
                _, counts = np.unique(images_original_classes, return_counts=True)
                entropies.append(entropy(counts))
            avg_entropies.append(np.average(entropies))

        return avg_entropies

    def cluster_evolution(self, start_epoch, target_cluster_index):

        results = []

        start_epoch_index = [ epoch for epoch,_ in self.clustering_log].index(start_epoch)
        target_cluster = set(self.clustering_log[start_epoch_index][1][target_cluster_index])

        for i in range(start_epoch_index + 1, len(self.clustering_log)):
            for k, cluster in enumerate(self.clustering_log[i][1]):
                intersection = list(set(cluster) & target_cluster)
                if (len(intersection) > 0):
                    results.append((i, k, intersection))

        return results

    def save_clustering_log(self, path, override=False):
        if not os.path.exists(path):
            # The file don't exist; save
            np.save(path, self.clustering_log)
            print("Clustering Log saved at: %s" % path)
            return
        if os.path.exists(path):
            if override:
                np.save(path, self.clustering_log)
                print("Clustering Log saved at: %s" % path)
                return
            else:
                print("Error the file already exists, rerun with parameter override=True to override.")
                return

    def load_clustering_log(self, path):
        if not os.path.isfile(path):
            # The file dosen't exist
            print("The provided path %s doesn't exist" % path)
        else:
            self.clustering_log = np.load(path, allow_pickle=True).tolist()
            print("Loaded Clustering Log from : %s" % path)

    def plot_cluster_evolution(self, cluster_evolution, final_epoch, weight_in_percent=True):

        cluster_evolution = [("C", k, len(indices)) for (i, k, indices) in cluster_evolution if i == final_epoch]

        size_target_cluster = np.sum([w for (_, _, w) in cluster_evolution])

        G = nx.DiGraph()
        G.add_weighted_edges_from(cluster_evolution)

        pos = nx.spring_layout(G)

        weights = nx.get_edge_attributes(G, 'weight')

        if (weight_in_percent):
            weights = {key: int(value / size_target_cluster * 100) for (key, value) in weights.items()}

        nx.draw(G, pos, edge_color='black',
                width=1, linewidths=1, node_size=500,
                node_color='pink', alpha=0.9,
                labels={node: node for node in G.nodes()})

        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_color='red')
        plt.axis('off')
        plt.show()
        return

def old_plot_clustering_log(clustering_log_path, trainset, plots_output_path=None, **kwargs):
    clustering_tracker = ClusteringTracker()
    filename = os.path.split(clustering_log_path)[1].split(".")[0]
    clustering_tracker.load_clustering_log(clustering_log_path)
    clustering_tracker.clustering_log= [ (epoch, clusters) for epoch,clusters in enumerate(clustering_tracker.clustering_log)]

    # Plot clusters avg. entropy vs. epochs
    fig = plt.figure(kwargs.get("figsize",(8,8)))
    plt.plot(clustering_tracker.epochs_avg_entropy(ground_truth=[t for (_,t )in trainset.imgs]))
    plt.title("Intersected Clusters Entropy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Avg. Entropy")

    if plots_output_path:
        plt.savefig(plots_output_path+"/"+filename+"_entropy.png")

    # Plot crossed clusters size vs epochs
    fig = plt.figure(kwargs.get("figsize",(8,8)))
    l_sizes= []
    for _,clusters_log in clustering_tracker.clustering_log:
        sizes = []
        for cluster in clusters_log:
            sizes.append(len(cluster))
        l_sizes.append(np.sum(sizes))
    plt.plot(l_sizes)
    plt.title("Intersected Clusters Size")
    plt.xlabel("Epoch")
    plt.ylabel("Size")

    if plots_output_path:
        plt.savefig(plots_output_path+"/"+filename+"_cluster_size.png")

    # Plot size of new images arising from crossed clusters vs epochs
    fig = plt.figure(kwargs.get("figsize",(8,8)))
    plt.plot( clustering_tracker.size_new_data_btw_epochs() )
    plt.title("The size of new images arising from crossing clusters vs epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Size")

    if plots_output_path:
        plt.savefig(plots_output_path+"/"+filename+"new_imgs_size.png")
