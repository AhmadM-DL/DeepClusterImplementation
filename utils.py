"""
Created on Tuesday April 24 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""

from deep_clustering_net import DeepClusteringNet
from deep_clustering_dataset import DeepClusteringDataset
from preprocessing import l2_normalization, sklearn_pca_whitening
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.metrics import normalized_mutual_info_score as NMI


from torch.utils.tensorboard import SummaryWriter
from scipy.stats import entropy
import matplotlib.pyplot as plt

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

def qualify_space(model: DeepClusteringNet, dataset: DeepClusteringDataset,
                 k_list: list, writer:SummaryWriter,
                 verbose=True, random_state=0,**kwargs):

    # clustering algorithm:
    clustering_algorithm = kwargs.get("clustering_algorithm", "kmeans")

    # full feedforward
    features = model.full_feed_forward(
        dataloader=torch.utils.data.DataLoader(dataset,
                                                batch_size=256,
                                                shuffle=False,
                                                pin_memory=True), verbose=True)
    # pre-processing pca-whitening
    if kwargs.get("pca_components", None) == None:
        pass
    else:
        if verbose:
            print(" - Features PCA + Whitening")
        features = sklearn_pca_whitening(features, n_components=kwargs.get("pca_components"), random_state=random_state)
    
    # pre-processing l2-normalization
    if verbose:
        print(" - Features L2 Normalization")
    features = l2_normalization(features)

    # cluster
    if verbose:
        print(" - Clustering")

    if clustering_algorithm=="kmeans":
        CMs = [ KMeans(n_clusters = k, random_state=random_state) for k in k_list ]
    elif clustering_algorithm== "gmm":
        CMs = [ GaussianMixture(n_components = k, random_state=random_state) for k in k_list ]
    else:
        raise Exception("Error an unsupported clustering algorithm was provided")

    k_assignments = [ CM.fit_predict(features) for CM in CMs ]
    k_grouped_assignments_indices = [ group_by_index(assignments) for assignments in k_assignments]
    k_entropies = [[] for i in range(len(k_list))]

    for i, grouped_assignments_indices in enumerate(k_grouped_assignments_indices):
        grouped_original_labels = [[dataset.get_targets()[index] for index in group] for group in grouped_assignments_indices]
        grouped_counts = [np.unique(group, return_counts=True)[1] for group in grouped_original_labels]
        entropies = [entropy(group) for group in grouped_counts]
        k_entropies[i] = entropies
    
    k_avg_entropies = [np.average(entropies) for entropies in k_entropies ]
    k_min_entropies = [np.min(entropies) for entropies in k_entropies ]
    k_max_entropies = [np.max(entropies) for entropies in k_entropies ]


    for i,k in enumerate(k_list):
        writer.add_histogram(clustering_algorithm+"/Space Quality k=%d"%k, np.array(k_entropies[i]), global_step=0)
    
    avg_entropies_vs_k = plt.figure()
    plt.plot(k_list, k_avg_entropies,"-*", label="Avg")
    plt.plot(k_list, k_min_entropies,"-+", label="Min")
    plt.plot(k_list, k_max_entropies,"-o", label="Max")
    plt.title("Space Quality")
    plt.xlabel("Number of clusters")
    plt.ylabel(" Cluster Entropy")
    plt.legend()

    writer.add_figure(clustering_algorithm+"/Space Quality Entropy", avg_entropies_vs_k, global_step=0)

    k_nmis = [ NMI(assignments, dataset.get_targets()) for assignments in k_assignments ]

    nmis_vs_k = plt.figure()

    plt.plot(k_list, k_nmis)
    plt.title("Space Quality")
    plt.xlabel("Number of clusters")
    plt.ylabel("Predicted/GroundTruth NMI")
    plt.legend()

    writer.add_figure(clustering_algorithm+"/Space Quality NMI", nmis_vs_k, global_step=0)

def group_by_index(labels):
    n_labels = len(np.unique(labels))
    grouped_indices = [[] for i in range(n_labels)]
    for i, label in enumerate(labels):
        grouped_indices[label].append(i)
    return grouped_indices





