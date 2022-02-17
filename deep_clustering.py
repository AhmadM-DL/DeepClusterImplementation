"""
Created on Tuesday April 15 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""

import torch
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from deep_clustering_net import DeepClusteringNet
from deep_clustering_dataset import DeepClusteringDataset
from preprocessing import l2_normalization, sklearn_pca_whitening
from samplers import UnifAverageLabelSampler

from clustering import sklearn_kmeans, faiss_kmeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from scipy.stats import entropy
from sklearn.manifold import TSNE

import numpy as np
import os


def deep_cluster(model: DeepClusteringNet, dataset: DeepClusteringDataset, n_clusters, loss_fn, optimizer, n_cycles,
                 loading_transform=None, training_transform=None,
                 random_state=0, verbose=0, writer: SummaryWriter = None,
                 checkpoints=None, resume=None, in_loop_transform=False,
                 **kwargs):
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
        checkpoint: A path to checkpoint dir to resume training from / save model to
        kwargs: Other relevent arguments that lesser used 
                - checkpoints_interval default 10
                - "pca_components" for PCA before clustering default= None
                - "loading_batch_size" default=256
                - "training_batch_size" default=256
                - "taining_shuffle" default= False
                - "loading_shuffle" default= False
                - "dataset_multiplier" default=1
                - "n_epochs" default=1
                - "embeddings_sample_size" used for writer to write embeddings default 500
                - "embeddings_checkpoint" the percent of cycles to be performed between written embeddings default 20
                - "halt_clustering"
                - "kmeans_max_iter"
                - "use_faiss" used to use facebook faiss clustering rather than kmeans
                - ..
    """
    if writer:
        # TODO dummy input should be based on dataset
        dummy_input = torch.rand((1,) + model.input_size)
        # I am not really sure why I have to add an input for add_graph
        # also move dummy input to models device
        #writer.add_graph(model, dummy_input.to(model.device))

    start_cycle = 0

    loss_fn.to(model.device)

    if checkpoints:
        if not os.path.isdir(checkpoints):
            os.makedirs(checkpoints)

    if resume:
        # if resume exist load model from
        if os.path.isfile(resume):
            start_cycle = model.load_model_parameters(
                resume, optimizer=optimizer)

    for cycle in range(start_cycle, n_cycles):

        if cycle == n_cycles -1 :
            break

        if verbose:
            print("Cycle %d:" % (cycle))

        # remove top layer
        if model.top_layer:
            model.top_layer = None
            if verbose:
                print(" - Remove Top Layer")

        # remove top_layer parameters from optimizer
        if len(optimizer.param_groups) > 1:
            del optimizer.param_groups[1]
            # The following is due a bug in PyTorch implementation
            state_ids= [id(k) for k,v in optimizer.state.items()]
            param_ids = [ id(p) for p in optimizer.param_groups[0]["params"]]
            remove_ids = [ param_id for param_id in state_ids if param_id not in param_ids]
            remove_keys = [ list(optimizer.state.keys())[state_ids.index(param_id)] for param_id in remove_ids]
            for key in remove_keys:
                del optimizer.state[key] 
            
            if verbose:
                print(" - Remove Top_layer Params from Optimizer")

        intermediate_checkpoint = kwargs.get("checkpoints_interval", 0)
        if checkpoints :
            # save last model
            model.save_model_parameters(
                os.path.join(checkpoints, "last_model.pth"), optimizer=optimizer, epoch=cycle)

            if intermediate_checkpoint:
                if cycle % intermediate_checkpoint == 0:
                    # save intermediate model
                    model.save_model_parameters(
                        os.path.join(checkpoints, "model_%d.pth" % cycle), optimizer=optimizer, epoch=cycle)

        halt_clustering = kwargs.get("halt_clustering", None)

        if  halt_clustering and cycle >= halt_clustering:
            pass
        else:
            
            # Set Loading Transform else consider the dataset transform
            if loading_transform:
                if in_loop_transform:
                    dataset.in_loop_transform = loading_transform
                else:
                    dataset.set_transform(loading_transform)

            # full feedforward
            features = model.full_feed_forward(
                dataloader=torch.utils.data.DataLoader(dataset,
                                                    batch_size=kwargs.get(
                                                        "loading_batch_size", 256),
                                                    shuffle=kwargs.get(
                                                        "loading_shuffle", False),
                                                    pin_memory=True), verbose=verbose,
                                                    transform_inside_loop= in_loop_transform)

        # if writer and we completed a 20% of all cycles: add embeddings
        # if writer and cycle % (int(n_cycles*(kwargs.get("embeddings_checkpoint", 20)/100))) == 0:

        #     embeddings_sample_size = kwargs.get("embeddings_sample_size", 500)
        #     to_embed = features[0:embeddings_sample_size]

        #     images_labels = [dataset.original_dataset.__getitem__(i) for i in range(0, embeddings_sample_size)]
        #     images = torch.stack([ transforms.ToTensor()(tuple[0]) for tuple in images_labels])
        #     labels = torch.tensor([tuple[1] for tuple in images_labels])

        #     writer.add_embedding(mat=to_embed, metadata=labels,
        #                          label_img=images, global_step=cycle)

        if halt_clustering and cycle>=halt_clustering:
            assignments= None
        else:
            # pre-processing pca-whitening
            if kwargs.get("pca_components", None) == None:
                pass
            else:
                if verbose:
                    print(" - Features PCA + Whitening")
                features = sklearn_pca_whitening(features, n_components=kwargs.get(
                    "pca_components"), random_state=random_state)

            # pre-processing l2-normalization
            if verbose:
                print(" - Features L2 Normalization")
            features = l2_normalization(features)

            # cluster
            if verbose:
                print(" - Clustering")
            
            # Change random state at each k-means so that the randomly picked
            # initialization centroids do not correspond to the same feature ids
            # from an epoch to another.
            rnd_state = kwargs.get("kmeans_rnd_state", None)
            max_iter  = kwargs.get("kmeans_max_iter", 20)
            if not rnd_state:
                rnd_state = np.random.randint(1234)
            use_faiss = kwargs.get("use_faiss", None)
            if use_faiss:
                assignments = faiss_kmeans(
                    features, n_clusters=n_clusters,
                    random_state=rnd_state,
                    verbose=verbose-1,
                    fit_partial=kwargs.get("partial_fit", None))
            else:
                assignments = sklearn_kmeans(
                    features, n_clusters=n_clusters,
                    random_state=rnd_state,
                    verbose=verbose-1,
                    max_iter = max_iter,
                    fit_partial=kwargs.get("partial_fit", None))

        if writer:
            if assignments!=None:
                # write NMI between consecutive pseudolabels
                if cycle > 0:
                    writer.add_scalar(
                        "NMI/pt_vs_pt-1", NMI(assignments, dataset.get_pseudolabels()), cycle)
                # write NMI between lables and pseudolabels
                writer.add_scalar("NMI/pt_vs_labels",
                                NMI(assignments, dataset.get_targets()), cycle)

        # re assign labels
        if halt_clustering and cycle>=halt_clustering:
            pass
        else:
            if verbose:
                print(" - Reassign pseudo_labels")

            dataset.set_pseudolabels(assignments)
            #dataset.save_pseudolabels(writer.get_logdir()+"/clusters", cycle)

        # set training transform else consider dataset transform
        if training_transform:
            if in_loop_transform:
                dataset.in_loop_transform = training_transform
            else:
                dataset.set_transform(training_transform)

        if writer:
           # write original labels entropy distribution in pseudoclasses
            pseudoclasses = dataset.group_indices_by_labels()
            pseudoclasses_labels = [[dataset.get_targets(
            )[index] for index in pseudoclass] for pseudoclass in pseudoclasses]
            pseudoclasses_labels_counts = [np.unique(pseudoclass_labels, return_counts=True)[
                1] for pseudoclass_labels in pseudoclasses_labels]
            entropies = [entropy(pseudoclass_labels_counts)
                         for pseudoclass_labels_counts in pseudoclasses_labels_counts]
            writer.add_histogram("pseudoclasses_entropies",
                                 np.array(entropies), cycle)

        # initialize uniform sample
        sampler = UnifAverageLabelSampler(dataset,
                                          dataset_multiplier=kwargs.get(
                                              "dataset_multiplier", 1),
                                          shuffle=kwargs.get(
                                              "training_shuffle", True),
                                          )

        # initialize training data loader
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=kwargs.get("training_batch_size", 256),
            sampler=sampler,
            pin_memory=True,
        )

        # add top layer
        if verbose:
            print(" - Add Top Layer")
        model.add_top_layer(n_clusters)

        if verbose:
            print(" - Add Top Layer Params to Optimizer")
        # add top layer parameters to optimizer
        lr = optimizer.defaults["lr"]
        weight_decay = optimizer.defaults["weight_decay"]
        optimizer.add_param_group({"params": model.top_layer.parameters(),
                                   "lr": lr,
                                   "weight_decay": weight_decay})

        # train network
        n_epochs = kwargs.get("n_epochs", 1)
        for epoch in range(n_epochs):
            loss = model.deep_cluster_train(dataloader=train_dataloader,
                                            optimizer=optimizer, epoch=cycle*n_epochs+epoch, 
                                            loss_fn=loss_fn, verbose=verbose, writer=writer,
                                            transform_inside_loop=in_loop_transform)
