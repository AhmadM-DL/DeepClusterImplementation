"""
Created on Tuesday April 15 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""
from deep_clustering_net import DeepClusteringNet
from deep_clustering_dataset import DeepClusteringDataset
from clustering import sklearn_kmeans
from preprocessing import l2_normalization, kmeans_pca_whitening
import torch


def deep_cluster(model: DeepClusteringNet, dataset: DeepClusteringDataset,  n_clusters, loss_fn, optimizer, n_cycles, device, random_state=0, verbose=0, **kwargs):
    """   
    """

    for i in n_cycles:

        # remove top layer 
        if model.top_layer:
            model.top_layer == None

        # remove top_layer parameters from optimizer
        if len(optimizer.param_groups)>1:
            del optimizer.param_groups[1]

        # full feedforward
        features = model.full_feed_forward(
            dataloader=dataloader, device=device, verbose=verbose)

        # pre-processing pca-whitening
        features = kmeans_pca_whitening(features, n_components=kwargs.get(
            "n_components", 256), random_state=random_state)

        # pre-processing l2-normalization
        features = l2_normalization(features)

        # cluster
        assignments = sklearn_kmeans(
            features, n_clusters=n_clusters, random_state=random_state, verbose=verbose)

        # re assign labels
        dataset.set_pseudolabels(assignments)

        # add top layer
        model.add_top_layer(n_clusters)

        # add top layer parameters to optimizer
        lr = optimizer.defaults["lr"]
        weight_decay = optimizer.defaults["weight_decay"]
        optimizer.add_param_group({"params": model.top_layer.parameters(),
                                   "lr": lr,
                                   "weight_decay": weight_decay})
        
        # train network
        loss = model.train(dataloader=dataloader, optimizer=optimizer, loss_fn= loss_fn, verbose=verbose)
