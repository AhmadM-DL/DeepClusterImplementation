"""
Created on Tuesday April 15 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""

import torch
from torch.utils.tensorboard import SummaryWriter

from deep_clustering_net import DeepClusteringNet
from deep_clustering_dataset import DeepClusteringDataset
from preprocessing import l2_normalization, sklearn_pca_whitening
from samplers import UnifAverageLabelSampler

from clustering import sklearn_kmeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from scipy.stats import entropy
from sklearn.manifold import TSNE

import random
import numpy as np
import os

def cross_clusters(clustersA, clustersB):
    return

def deep_cluster(modelA: DeepClusteringNet,
                 modelB: DeepClusteringNet,
                 dataset: DeepClusteringDataset,
                 n_clusters, loss_fn, optimizerA, optimizerB, n_cycles,
                 loading_transform=None, training_transform=None,
                 random_state=0, verbose=0, writer: SummaryWriter = None,
                 checkpoint=None,
                 **kwargs):
    """ 
    TODO
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
        checkpoint: A path to model checkpoint to resume training from / save model to
        kwargs: Other relevent arguments that lesser used 
                - "pca_components" for PCA before clustering default= None
                - "loading_batch_size" default=256
                - "training_batch_size" default=256
                - "taining_shuffle" default= False
                - "loading_shuffle" default= False
                - "dataset_multiplier" default=1
                - "n_epochs" default=1
                - "embeddings_sample_size" used for writer to write embeddings default 500
                - "embeddings_checkpoint" the percent of cycles to be performed between written embeddings default 20
                - ...
    """
    if writer:
        dummy_input_A = torch.rand((1,) + modelA.input_size)
        dummy_input_B = torch.rand((1,) + modelB.input_size)

        writer.add_graph(modelA, dummy_input_A.to(modelA.device))
        writer.add_graph(modelB, dummy_input_B.to(modelB.device))

    # validations
    if not modelA.device == modelB.device:
        raise Exception("Co Deep Clustering Error: Models should exist on same device")
    
    start_cycle = 0

    loss_fn.to(modelA.device)

    if checkpoint:
        # if checkpoint exist load model from
        if os.path.isdir(checkpoint):
            start_cycle = modelA.load_model_parameters(
                checkpoint+"/modelA.pth", optimizer=optimizerA)
            start_cycle = modelB.load_model_parameters(
                checkpoint+"/modelB.pth", optimizer=optimizerB)

    for cycle in range(start_cycle, n_cycles):

        if verbose:
            print("Cycle %d:" % (cycle))

        # remove top layer
        if modelA.top_layer:
            modelA.top_layer = None
            if verbose:
                print(" - Remove A Top Layer ")

        if modelB.top_layer:
            modelB.top_layer = None
            if verbose:
                print(" - Remove B Top Layer")
        
        # remove top_layer parameters from optimizer
        if len(optimizerA.param_groups) > 1:
            del optimizerA.param_groups[1]
            if verbose:
                print(" - Remove A Top_layer Params from Optimizer")
        
        if len(optimizerA.param_groups) > 1:
            del optimizerA.param_groups[1]
            if verbose:
                print(" - Remove B Top_layer Params from Optimizer")

        if checkpoint:
            # save model
            modelA.save_model_parameters(
                checkpoint+"/modelA.pth", optimizer=optimizerA, epoch=cycle)
            modelB.save_model_parameters(
                checkpoint+"/modelB.pth", optimizer=optimizerB, epoch=cycle)

        # Set Loading Transform else consider the dataset transform
        if loading_transform:
            dataset.set_transform(loading_transform)

        # full feedforward
        featuresA = modelA.full_feed_forward(
            dataloader=torch.utils.data.DataLoader(dataset,
                                                   batch_size=kwargs.get(
                                                       "loading_batch_size", 256),
                                                   shuffle=kwargs.get(
                                                       "loading_shuffle", False),
                                                   pin_memory=True), verbose=verbose)
        featuresB = modelA.full_feed_forward(
            dataloader=torch.utils.data.DataLoader(dataset,
                                                   batch_size=kwargs.get(
                                                       "loading_batch_size", 256),
                                                   shuffle=kwargs.get(
                                                       "loading_shuffle", False),
                                                   pin_memory=True), verbose=verbose)

        # if writer and we completed a 20% of all cycles: add embeddings
        if writer and cycle % (int(n_cycles*(kwargs.get("embeddings_checkpoint", 20)/100))) == 0:

            embeddings_sample_size = kwargs.get("embeddings_sample_size", 500)
            to_embed_A = featuresA[0:embeddings_sample_size]
            to_embed_B = featuresB[0:embeddings_sample_size]

            images_labels = [dataset.__getitem__(
                i) for i in range(0, embeddings_sample_size)]

            images = torch.stack([tuple[0] for tuple in images_labels])
            labels = torch.tensor([tuple[1] for tuple in images_labels])

            writer.add_embedding(mat=to_embed_A, metadata=labels,
                                 label_img=images, global_step=cycle)
            writer.add_embedding(mat=to_embed_B, metadata=labels,
                                 label_img=images, global_step=cycle)

        # pre-processing pca-whitening
        if kwargs.get("pca_components", None) == None:
            pass
        else:
            if verbose:
                print(" - Features A PCA + Whitening")
            featuresA = sklearn_pca_whitening(featuresA, n_components=kwargs.get(
                "pca_components"), random_state=random_state)
            if verbose:
                print(" - Features PCA + Whitening")
            featuresB = sklearn_pca_whitening(featuresB, n_components=kwargs.get(
                "pca_components"), random_state=random_state)

        # pre-processing l2-normalization
        if verbose:
            print(" - Features A L2 Normalization")
        featuresA = l2_normalization(featuresA)
        if verbose:
            print(" - Features B L2 Normalization")
        featuresB = l2_normalization(featuresB)

        # cluster
        if verbose:
            print(" - Clustering A")
        assignmentsA = sklearn_kmeans(
            featuresA, n_clusters=n_clusters, random_state=random_state, verbose=verbose-1)
        if verbose:
            print(" - Clustering B")
        assignmentsB = sklearn_kmeans(
            featuresB, n_clusters=n_clusters, random_state=random_state, verbose=verbose-1)

        # cross intersections
        clustersA = group_indices_by_labels(assignmentsA)
        clustersB = group_indices_by_labels(assignmentsB)

        # cross clusters
        crossed_clusters = cross_clusters(clustersA, clustersB)
        # take top sized n_clusters clusters:
        crossed_clusters = crossed_clusters[:n_clusters]

        instance_wise_weights = [kwargs.get("weak_instance_weight",0.6)]*dataset.__len__()
        for cluster in cross_clusters:
            for index in cluster:
                instance_wise_weights[index] = kwargs.get("strong_instance_weight",1.2)


        if writer:
            # write NMI between consecutive pseudolabels
            if cycle > 0:
                writer.add_scalar(
                    "NMI/pt_vs_pt-1/A", NMI(assignmentsA, dataset.get_pseudolabels()), cycle)
                writer.add_scalar(
                    "NMI/pt_vs_pt-1/B", NMI(assignmentsB, dataset.get_pseudolabels()), cycle)

            # write NMI between lables and pseudolabels
            writer.add_scalar("NMI/pt_vs_labels/A",
                              NMI(assignmentsA, dataset.get_targets()), cycle)
            writer.add_scalar("NMI/pt_vs_labels/B",
                              NMI(assignmentsB, dataset.get_targets()), cycle)

        if writer:
           # write original labels entropy distribution in pseudoclasses
            pseudoclasses_labels_A = [[dataset.get_targets()[index] for index in pseudoclass] for pseudoclass in clustersA]
            pseudoclasses_labels_counts = [np.unique(pseudoclass_labels, return_counts=True)[1] for pseudoclass_labels in pseudoclasses_labels_A]
            entropies = [entropy(pseudoclass_labels_counts)for pseudoclass_labels_counts in pseudoclasses_labels_counts]
            writer.add_histogram("pseudoclasses_entropies_A",np.array(entropies), cycle)

            pseudoclasses_labels_B = [[dataset.get_targets()[index] for index in pseudoclass] for pseudoclass in clustersB]
            pseudoclasses_labels_counts = [np.unique(pseudoclass_labels, return_counts=True)[1] for pseudoclass_labels in pseudoclasses_labels_B]
            entropies = [entropy(pseudoclass_labels_counts)for pseudoclass_labels_counts in pseudoclasses_labels_counts]
            writer.add_histogram("pseudoclasses_entropies_B",np.array(entropies), cycle)

        # re assign labels
        if verbose:
            print(" - Reassign pseudo_labels")
        if random.random()>=0.5:
            dataset.set_pseudolabels(assignmentsA)
        else:
            dataset.set_pseudolabels(assignmentsB)

        # initialize uniform sample
        sampler = UnifAverageLabelSampler(dataset,
                                          dataset_multiplier=kwargs.get(
                                              "dataset_multiplier", 1),
                                          shuffle=kwargs.get(
                                              "training_shuffle", True),
                                          )

        # set training transform else consider dataset transform
        if training_transform:
            dataset.set_transform(training_transform)

        # initialize training data loader
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=kwargs.get("training_batch_size", 256),
            sampler=sampler,
            pin_memory=True,
        )

        # add top layer
        if verbose:
            print(" - Add A Top Layer")
        modelA.add_top_layer(n_clusters)
        if verbose:
            print(" - Add B Top Layer")
        modelB.add_top_layer(n_clusters)

        # add top layer parameters to optimizer
        if verbose:
            print(" - Add A Top Layer Params to Optimizer")
        lr = optimizerA.defaults["lr"]
        weight_decay = optimizerA.defaults["weight_decay"]
        optimizerA.add_param_group({"params": modelA.top_layer.parameters(),
                                   "lr": lr,
                                   "weight_decay": weight_decay})
        if verbose:
            print(" - Add B Top Layer Params to Optimizer")
        lr = optimizerB.defaults["lr"]
        weight_decay = optimizerB.defaults["weight_decay"]
        optimizerB.add_param_group({"params": modelB.top_layer.parameters(),
                                   "lr": lr,
                                   "weight_decay": weight_decay})

        # train network
        n_epochs = kwargs.get("n_epochs", 1)
        for epoch in range(n_epochs):
            lossA = modelA.deep_cluster_train_with_weights(dataloader=train_dataloader,
                                              optimizer=optimizerA,
                                              epoch=cycle*n_epochs+epoch,
                                              loss_fn=loss_fn,
                                              instance_wise_weights=instance_wise_weights,
                                              verbose=verbose,
                                              writer=writer)
            
            lossB = modelB.deep_cluster_train_with_weights(dataloader=train_dataloader,
                                              optimizer=optimizerB,
                                              epoch=cycle*n_epochs+epoch, 
                                              loss_fn=loss_fn,
                                              instance_wise_weights=instance_wise_weights,
                                              verbose=verbose, 
                                              writer=writer)

def group_indices_by_labels(labels):
    """ Group the indices of a list of labels by labels.
        [l1,l2,l1,l1,l3,l2,l2] => [[0,2,3],[1,5,6],[4]]  

    Args:
        labels (list(int)): A list of labels

    Returns:
        list of lists: grouped indices
    """
    n_labels = len(np.unique(labels))
    grouped_indices = [[] for i in range(n_labels)]
    for i, label in enumerate(labels):
        grouped_indices[label].append(i)
    return grouped_indices

def cross_clusters(clustersA, clustersB):
    """ A method to find the intersection between two sets of clusters.
        [ [A1], ..., [AN] ] X [ [B1], ..., [BN] ] => [ [A1&B1], ..., [A1&BN], ..., [AN&BN] ]

    Args:
        clustersA (list of lists): The first set of clusters
        clustersB (list of lists): The second set of clusters

    Returns:
        list of lists: The list of crossed clusters
    """
    crossed_clusters = []
    sizes = []

    clustersA_as_sets = np.array([set(x) for x in clustersA])
    clustersB_as_sets = np.array([set(x) for x in clustersB])

    for c1 in clustersA_as_sets:
        for c2 in clustersB_as_sets:
            cr = c1 & c2
            crossed_clusters.append(cr)
            sizes.append(len(cr))

    sorted_indices_by_size = np.argsort(sizes)[::-1]  ## descending order
    crossed_clusters = np.array(crossed_clusters)[sorted_indices_by_size]
    crossed_clusters = [list(x) for x in crossed_clusters]

    return crossed_clusters

