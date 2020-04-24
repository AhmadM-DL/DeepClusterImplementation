"""
Created on Tuesday April 15 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""
from deep_clustering_net import DeepClusteringNet
from deep_clustering_dataset import DeepClusteringDataset
from clustering import sklearn_kmeans
from samplers import UnifAverageLabelSampler
from preprocessing import l2_normalization, kmeans_pca_whitening
import torch
from torch.utils.tensorboard import SummaryWriter



def deep_cluster(model: DeepClusteringNet, dataset: DeepClusteringDataset, n_clusters, loss_fn, optimizer, n_cycles,
                 random_state=0, verbose=0, writer:SummaryWriter=None,**kwargs):
    """ 
    The main method in this repo. it implements the DeepCluster pipeline
    introduced by caron et. al. in "Deep Clustering for Unsupervised Learning of Visual Features"  
    params:
        model(DeepClusterNet): A nn.module that implements DeepClusteringNet Class
        dataset(DeepClusteringDataset): A torch.utils.dataset that implements DeepClusteringDataset
        n_clusters(int): The number of clusters to use in DeepCluster Clustering part
        loss_fn(torch.nn): Pytorch Loss Criterion i.e. CrossEntropyLoss
        optimizer(torch.optim.Optimizer): Pytorch Optimizer i.e. torch.optim.SGD
        n_cycles(int): The number of DeepCluster cycles to be performed each cycle includes a clustering
                       step and a network training step
        random_state(int): Random State argument for reproducing results
        verbose(int): verbose level
        kwargs: Other relevent arguments that lesser used. i.e. n_components for PCA before clustering, ...
    """

    for i in range(n_cycles):

        # remove top layer
        if model.top_layer:
            model.top_layer = None

        # remove top_layer parameters from optimizer
        if len(optimizer.param_groups) > 1:
            del optimizer.param_groups[1]

        # full feedforward
        features = model.full_feed_forward(
            dataloader=torch.utils.data.DataLoader(dataset,
                                                   batch_size=kwargs.get("loading_batch_size", 256), pin_memory=True), verbose=verbose)

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

        # initialize uniform sample
        sampler = UnifAverageLabelSampler(dataset,
                                          dataset_multiplier=kwargs.get("dataset_multiplier", 1))

        # initialize training data loader
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=kwargs.get("training_batch_size", 256),
            sampler=sampler,
            pin_memory=True,
        )

        # add top layer
        model.add_top_layer(n_clusters)

        # add top layer parameters to optimizer
        lr = optimizer.defaults["lr"]
        weight_decay = optimizer.defaults["weight_decay"]
        optimizer.add_param_group({"params": model.top_layer.parameters(),
                                   "lr": lr,
                                   "weight_decay": weight_decay})

        # train network
        loss = model.deep_cluster_train(dataloader=train_dataloader,
                           optimizer=optimizer, loss_fn=loss_fn, verbose=verbose, writer=writer)
