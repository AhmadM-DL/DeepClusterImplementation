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
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy
from PIL import Image


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
        for i,idx in enumerate(image_indexes):
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

    if(reindex):
        reindexed_clustered_data_indices = [ [] for i in range( len( clustered_data_indices ) ) ]
        for i in range(len(image_indexes)):
            reindexed_clustered_data_indices[ pseudolabels[i] ].append(i)

    if(not reindex):
        return [image_indexes, pseudolabels]
    else:
        return [image_indexes, pseudolabels], reindexed_clustered_data_indices


class NeuralFeaturesKmeansWithPreprocessing():

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
        self.pca_object= None

    def cluster(self):

        end = time.time()

        # Preprocess features
        self.preprocessed_data, self.pca_object = self.__preprocess_neural_features(data=self.data, pca=self.pca, verbose=self.verbose,
                                                                   random_State= self.kwargs.get("random_state", None),
                                                                   return_pca_object=True)

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

        if(return_pca_object):
            return data, pca
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

    sorted_indices_by_size = np.argsort(sizes)[::-1] ## descending order
    results = np.array(results)[sorted_indices_by_size]
    results = [list(x) for x in results]

    if(take_top):
        return results[:take_top]
    else:
        return results[:len(model_1_clusters)]


class ClusteringTracker(object):

    def __init__(self):
        self.clustering_log = []
        self.merged_clustering_log = []
        self.epochs= []

    def update(self, epoch, clustered_data_indices):
        self.clustering_log.append(clustered_data_indices)
        self.epochs.append(epoch)

    def size_new_data_btw_epochs(self):

        new_data_sizes = []
        
        for i in range( 1,  len(self.clustering_log) ):

            prev_clusters = self.clustering_log[i-1]
            curr_clusters = self.clustering_log[i]

            flat_prev_clusters = set([item for cluster in prev_clusters for item in cluster])
            flat_curr_clusters = set([item for cluster in curr_clusters for item in cluster])

            new_data_indices = flat_curr_clusters - flat_prev_clusters

            new_data_sizes.append(len(new_data_indices))

        return new_data_sizes

    def epochs_avg_entropy(self, ground_truth):
        avg_entropies = []

        for i,clusters in enumerate(self.clustering_log):
            entropies = []
            for j,cluster in enumerate(self.clustering_log[i]):
                images_original_classes = [ground_truth[image_index] for image_index in cluster]
                values, counts = np.unique(images_original_classes, return_counts=True)
                entropies.append(entropy(counts))
            avg_entropies.append( np.average(entropies) )

        return avg_entropies

    def cluster_evolution(self, start_epoch, target_cluster_index):

        results = []

        epoch_index = self.epochs.index(start_epoch)
        target_cluster = set(self.clustering_log[epoch_index][target_cluster_index])

        for i in range(epoch_index + 1, len(self.clustering_log)):
            for k, cluster in enumerate(self.clustering_log[i]):
                intersection = list(set(cluster) & target_cluster)
                if (len(intersection) > 0):
                    results.append((i, k, intersection))

        return results

    def plot_cluster_evolution(self, cluster_evolution, final_epoch, weight_in_percent=True):

        cluster_evolution = [("C", k, len(indices)) for (i, k, indices) in cluster_evolution if i == final_epoch]

        size_target_cluster = np.sum([w for (_,_,w) in cluster_evolution])

        G = nx.DiGraph()
        G.add_weighted_edges_from(cluster_evolution)

        pos = nx.spring_layout(G)

        weights = nx.get_edge_attributes(G, 'weight')

        if(weight_in_percent):
            weights = {key: int(value / size_target_cluster*100) for (key, value) in weights.items()}

        nx.draw(G, pos, edge_color='black',
                width=1, linewidths=1, node_size=500,
                node_color='pink', alpha=0.9,
                labels={node: node for node in G.nodes()})

        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_color='red')
        plt.axis('off')
        plt.show()
        return

    def plot_cluster_images(self, epoch, cluster_index, images_paths, percent_of_images_to_plot=100):

        cluster_images_indices = [ image_index for image_index in self.clustering_log[epoch][cluster_index] ]
        number_images_to_plot = len(cluster_images_indices)*percent_of_images_to_plot//100
        images_to_plot_indices = np.random.choice(cluster_images_indices,number_images_to_plot, replace=False)
        self.plot_set_of_images(images_to_plot_indices, images_paths)


    def plot_set_of_images(self, images_indices, images_paths, figsize=(20,20)):

        N = len(images_indices)
        fig = plt.figure(figsize=figsize)
        images_to_plot_paths = np.array(images_paths)[images_indices]

        for i, image_path in enumerate(images_to_plot_paths):
            plt.subplot(N // 10 + 1, 10, i + 1)
            im = Image.open(image_path)
            plt.axis("off")
            plt.imshow(im)

    def plot_clusters_histograms(self, epoch, ground_truth):
        fig = plt.figure(figsize=(20, 30))

        for (n, cluster) in enumerate(self.clustering_log[epoch]):

            images_original_classes = [ground_truth[image_index] for image_index in cluster]
            plt.subplot(len(self.clustering_log[epoch]) // 10 + 1, 10, n + 1)
            plt.xticks(rotation='horizontal')
            plt.yticks([])

            values, counts = np.unique(images_original_classes, return_counts=True)
            cluster_entropy =  entropy(counts)
            #max_count_target = values[np.argmax(counts)]

            plt.ylabel("C %d E %f " % (n, cluster_entropy))
            plt.hist(np.array(images_original_classes).astype(str))

    def merge_clustering_log(self, ground_truth, merging_entropy_threshold=0):
        self.merged_clustering_log = []
        for (n, epoch_clustering_log) in enumerate(self.clustering_log):

            clusters_to_merge_indices = [[] for i in range(len(np.unique(ground_truth)))]

            for (k,cluster) in enumerate(epoch_clustering_log):
                images_original_classes = [ground_truth[image_index] for image_index in cluster]
                values, counts = np.unique(images_original_classes, return_counts=True)
                cluster_entropy = entropy(counts)
                max_count_target = values[np.argmax(counts)]

                if(cluster_entropy<=merging_entropy_threshold):
                    clusters_to_merge_indices[max_count_target].extend([k])

            clusters_to_merge_indices = set(np.concatenate(clusters_to_merge_indices))
            clusters_to_persist_indices = set(range(len(epoch_clustering_log))) - clusters_to_merge_indices

            epoch_merged_clustering_log = [ epoch_clustering_log[i] for i in clusters_to_persist_indices]

            for group_to_cluster in clusters_to_merge_indices:
                new_cluster = []
                for cluster_index in group_to_cluster:
                    new_cluster.extend(epoch_clustering_log[cluster_index])
                epoch_merged_clustering_log.append(new_cluster)










