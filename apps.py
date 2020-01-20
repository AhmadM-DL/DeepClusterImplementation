import models
import utils
import deep_cluster
import os, json
import torch
import re, time
import numpy as np
from torch import nn


def dual_deep_cluster(model_1, model_2, n_epochs, output_directory,
                      training_transformation, trainset,
                      learning_rate, momentum, weight_decay, n_clusters,
                      trainloader, trainloader_batch_size=256,
                      training_batch_size=256,
                      epochs_per_checkpoint=20, random_state=0, pca=0,
                      size_per_pseudolabel="average", network_iterations=1,
                      device_name="cuda:0", clustering_tech="kmeans", run_from_checkpoint=False,
                      verbose=0):

    utils.create_directory(output_directory, verbose)
    if epochs_per_checkpoint:
        checkpoint_dir = utils.create_directory(output_directory + "/checkpoints", verbose)

    cfg_file = output_directory + "/cfg.json"
    if verbose:
        print("Saving Configuration: %s" % cfg_file)
    cfg = {"n_epochs": n_epochs,
           "output_directory": output_directory,
           "learning_rate": learning_rate,
           "momentum": momentum,
           "weight_decay": weight_decay,
           "n_clusters": n_clusters,
           "trainloader_batch_size": trainloader_batch_size,
           "training_batch_size": training_batch_size,
           "epochs_per_checkpoint": epochs_per_checkpoint,
           "random_state": random_state,
           "pca": pca,
           "size_per_pseudolabel": size_per_pseudolabel,
           "network_iterations": network_iterations,
           "device_name": device_name,
           "clustering_tech": clustering_tech,
           "run_from_checkpoint": run_from_checkpoint
           }
    json.dump(cfg, open(cfg_file, "w"))

    device = torch.device(device_name)

    if verbose:
        print("Connected to device %s" % device_name)

    optimizer_1 = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model_1.parameters()),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    optimizer_2 = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model_2.parameters()),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    loss_criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    nmi_meter_1 = utils.NMIMeter()
    nmi_meter_2 = utils.NMIMeter()

    clustering_tracker = deep_cluster.ClusteringTracker()

    if run_from_checkpoint:

        # Get checkpoint files
        files = [(f, int(re.findall("\d+", f)[0]), int(re.findall("\d+", f)[1])) for f in os.listdir(checkpoint_dir) if
                 re.match("model_\d+_checkpoint_\d+.pth", f)]

        # Get latest model 1 file
        model_1_files = [(f, chkpt_number) for (f, model_id, chkpt_number) in files if model_id == 1]
        model_1_chkpts = [chkpt_number for (_, chkpt_number) in model_1_files]
        model_1_latest_chkpt = model_1_files[np.argmax(model_1_chkpts)][0]

        # Load model 1
        last_epoch_1 = models.load_from_checkpoint(model_1, optimizer_1, checkpoint_dir + "/" + model_1_latest_chkpt)

        # Get latest model 2 file
        model_2_files = [(f, chkpt_number) for (f, model_id, chkpt_number) in files if model_id == 2]
        model_2_chkpts = [chkpt_number for (_, chkpt_number) in model_2_files]
        model_2_latest_chkpt = model_2_files[np.argmax(model_2_chkpts)][0]

        # Load model 2
        last_epoch_2 = models.load_from_checkpoint(model_2, optimizer_2, checkpoint_dir + "/" + model_2_latest_chkpt)

        if not last_epoch_1 == last_epoch_2:
            raise Exception("Error in loading from checkpoint: the 2 models doesn't have the same latest epoch")
        else:
            start_epoch = last_epoch_1 + 1

        # Load previous nmi values
        nmi_meter_1.load_from_csv(output_directory + "/" + "model_1_nmi.csv")
        nmi_meter_2.load_from_csv(output_directory + "/" + "model_2_nmi.csv")

        # Trim NMI to last_epoch
        for (index, epoch, _) in enumerate(nmi_meter_1.nmi_array):
            if epoch > last_epoch_1:
                nmi_meter_1.nmi_array.pop(index)

        for (index, epoch, _) in enumerate(nmi_meter_2.nmi_array):
            if epoch > last_epoch_1:
                nmi_meter_2.nmi_array.pop(index)

        # Load previous clustering log
        clustering_tracker.load_clustering_log(output_directory + "/" + "clustering_log.npy")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, n_epochs):

        if verbose:
            print("Epoch %d:" % epoch)

        model_1_output_size = model_1.get_model_output_size()
        model_2_output_size = model_2.get_model_output_size()

        # save checkpoint
        if epochs_per_checkpoint > 0 and epoch != 0 and epoch % epochs_per_checkpoint == 0:
            models.save_checkpoint(model=model_1, optimizer=optimizer_1, epoch=epoch,
                                   path=checkpoint_dir + "/model_1_checkpoint_%d.pth" % epoch)

            models.save_checkpoint(model=model_2, optimizer=optimizer_2, epoch=epoch,
                                   path=checkpoint_dir + "/model_2_checkpoint_%d.pth" % epoch)
            if verbose:
                print(" Saved models checkpoints at %s/model_n_checkpoint_$d.pth" % (checkpoint_dir, epoch))

        if verbose:
            print("Computing network output of the training set")

        inputs_1, features_1, targets_1 = models.compute_network_output(trainloader,
                                                                        model_1, device,
                                                                        batch_size=trainloader_batch_size,
                                                                        data_size=len(trainloader.dataset),
                                                                        verbose=verbose,
                                                                        return_inputs=True, return_targets=True)

        inputs_2, features_2, targets_2 = models.compute_network_output(trainloader,
                                                                        model_2, device,
                                                                        batch_size=trainloader_batch_size,
                                                                        data_size=len(trainloader.dataset),
                                                                        verbose=verbose,
                                                                        return_inputs=True, return_targets=True)
        if verbose:
            print("Clustering Features:")  # Todo make Neural_Features_Clustering_With_Preprocessing 2 functions

        deepcluster_1 = deep_cluster.Neural_Features_Clustering_With_Preprocessing(data=features_1, verbose=verbose,
                                                                                   random_state=random_state, pca=pca)

        deepcluster_2 = deep_cluster.Neural_Features_Clustering_With_Preprocessing(data=features_2, verbose=verbose,
                                                                                   random_state=random_state, pca=pca)

        deepcluster_1.cluster(algorithm=clustering_tech, n_clusters=n_clusters)
        deepcluster_2.cluster(algorithm=clustering_tech, n_clusters=n_clusters)

        if verbose:
            print("Crossing clustering results")
        crossed_clusters = deep_cluster.cross_2_models_clustering_output(deepcluster_1.clustered_data_indices,
                                                                         deepcluster_2.clustered_data_indices)

        clustering_tracker.update(epoch, crossed_clusters)

        images_pseudolabels, rearranged_crossed_clusters = deep_cluster.clustered_data_indices_to_list(crossed_clusters,
                                                                                                       reindex=True)

        deep_cluster_trainset = deep_cluster.LabelsReassignedDataset(original_dataset=trainset,
                                                                     image_indexes=images_pseudolabels[0],
                                                                     pseudolabels=images_pseudolabels[1],
                                                                     transform=training_transformation)

        sampler = utils.UnifLabelSampler(images_lists=rearranged_crossed_clusters,
                                         dataset_multiplier=network_iterations,
                                         size_per_pseudolabel=size_per_pseudolabel)

        train_dataloader = torch.utils.data.DataLoader(
            deep_cluster_trainset,
            batch_size=training_batch_size,
            num_workers=4,
            sampler=sampler,
            pin_memory=True,
        )

        if verbose:
            print("Adding top layer with output sizes:\nmodel_1 %d\nmodel_2 %d" % (
                model_1_output_size, model_2_output_size))
        models.add_top_layer(model_1, [('L', model_1_output_size, len(rearranged_crossed_clusters))], device=device)
        models.add_top_layer(model_2, [('L', model_2_output_size, len(rearranged_crossed_clusters))], device=device)

        tl_optimizer_1 = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model_1.top_layer.parameters()),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        tl_optimizer_2 = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model_2.top_layer.parameters()),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        if verbose:
            print("Training Models: ")

        loss_1 = models.deep_cluster_train(dataloader=train_dataloader, model=model_1,
                                           loss_criterion=loss_criterion,
                                           net_optimizer=optimizer_1,
                                           annxed_layers_optimizer=tl_optimizer_1,
                                           epoch=epoch, device=device)

        loss_2 = models.deep_cluster_train(dataloader=train_dataloader, model=model_2,
                                           loss_criterion=loss_criterion,
                                           net_optimizer=optimizer_2,
                                           annxed_layers_optimizer=tl_optimizer_2,
                                           epoch=epoch, device=device)
        if verbose:
            print("Getting trained model output on pseudo-training_set to measure performance")

        outputs_1, targets_1 = models.compute_network_output(dataloader=trainloader,
                                                             model=model_1, device=device,
                                                             batch_size=trainloader_batch_size,
                                                             data_size=len(trainloader.dataset),
                                                             verbose=True,
                                                             return_targets=True)

        outputs_2, targets_2 = models.compute_network_output(dataloader=trainloader,
                                                             model=model_2, device=device,
                                                             batch_size=trainloader_batch_size,
                                                             data_size=len(trainloader.dataset),
                                                             verbose=True,
                                                             return_targets=True)

        nmi_meter_1.update(epoch, ground_truth=targets_1, predictions=np.argmax(outputs_1, 1))
        nmi_meter_2.update(epoch, ground_truth=targets_2, predictions=np.argmax(outputs_2, 1))

        clustering_tracker.save_clustering_log(path=output_directory + "/" + "clustering_log.npy", override=True)
        nmi_meter_1.store_as_csv(path=output_directory + "/" + "model_1_nmi.csv", override=True)
        nmi_meter_2.store_as_csv(path=output_directory + "/" + "model_2_nmi.csv", override=True)

        if verbose:
            print("Removing added top layers")

        models.remove_top_layer(model_1)
        models.remove_top_layer(model_2)

    # Save final model
    models.save_model_parameter(model=model_1, path=output_directory + "/final_model_1.pth", override=True)

    models.save_model_parameter(model=model_2, path=output_directory + "/final_model_2.pth", override=True)
    if verbose:
        print(" Saved final models at %s/final_model_.pth" % output_directory)


def mono_deep_cluster(model, n_epochs, output_directory,
                      training_transformation, trainset,
                      learning_rate, momentum, weight_decay, n_clusters,
                      trainloader, trainloader_batch_size=256,
                      training_batch_size=256, epochs_per_checkpoint=20,
                      random_state=0, pca=0, size_per_pseudolabel="average",
                      network_iterations=1, device_name="cuda:0", clustering_tech="kmeans",
                      run_from_checkpoint=False, verbose=0):

    utils.create_directory(output_directory, verbose)
    if epochs_per_checkpoint:
        checkpoint_dir = utils.create_directory(output_directory + "/checkpoints", verbose)

    cfg_file = output_directory + "/cfg.json"
    if verbose:
        print("Saving Configuration: %s" % cfg_file)
    cfg = {"n_epochs": n_epochs,
           "output_directory": output_directory,
           "learning_rate": learning_rate,
           "momentum": momentum,
           "weight_decay": weight_decay,
           "n_clusters": n_clusters,
           "trainloader_batch_size": trainloader_batch_size,
           "training_batch_size": training_batch_size,
           "epochs_per_checkpoint": epochs_per_checkpoint,
           "random_state": random_state,
           "pca": pca,
           "size_per_pseudolabel": size_per_pseudolabel,
           "network_iterations": network_iterations,
           "device_name": device_name,
           "clustering_tech": clustering_tech,
           "run_from_checkpoint": run_from_checkpoint
           }
    json.dump(cfg, open(cfg_file, "w"))

    device = torch.device(device_name)
    if verbose:
        print("Connected to device %s" % device_name)

    utils.create_directory(output_directory, verbose)
    if epochs_per_checkpoint:
        checkpoint_dir = utils.create_directory(output_directory + "/checkpoints", verbose)

    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    loss_criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    nmi_meter = utils.NMIMeter()
    clustering_tracker = deep_cluster.ClusteringTracker()

    if run_from_checkpoint:

        # Get checkpoint files
        model_files = [(f, int(re.findall("\d+", f)[0])) for f in os.listdir(checkpoint_dir) if
                       re.match("model_checkpoint_\d+.pth", f)]

        # Get latest model 1 file
        model_chkpts = [chkpt_number for (_, chkpt_number) in model_files]
        model_latest_chkpt = model_files[np.argmax(model_chkpts)][0]

        # Load model
        last_epoch = models.load_from_checkpoint(model, optimizer, checkpoint_dir + "/" + model_latest_chkpt)
        start_epoch = last_epoch + 1

        # Load previous nmi values
        nmi_meter.load_from_csv(output_directory + "/" + "model_nmi.csv")

        # Trim NMI to last_epoch
        for (index, epoch, _) in enumerate(nmi_meter.nmi_array):
            if epoch > last_epoch:
                nmi_meter.nmi_array.pop(index)

        # Load previous clustering log
        clustering_tracker.load_clustering_log(output_directory + "/" + "clustering_log.npy")

    else:

        start_epoch = 0

    for epoch in range(start_epoch, n_epochs):

        if verbose:
            print("Epoch %d:" % epoch)

        model_output_size = model.get_model_output_size()

        # save checkpoint
        if epochs_per_checkpoint > 0 and epoch != 0 and epoch % epochs_per_checkpoint == 0:
            models.save_checkpoint(model=model, optimizer=optimizer, epoch=epoch,
                                   path=checkpoint_dir + "/model_checkpoint_%d.pth" % epoch)
            if verbose:
                print(" Saved a checkpoint at %s/model_checkpoint_$d.pth" % (checkpoint_dir, epoch))

        if verbose:
            print("Computing network output of the training set")

        inputs, features, targets = models.compute_network_output(trainloader,
                                                                  model, device,
                                                                  batch_size=trainloader_batch_size,
                                                                  data_size=len(trainloader.dataset),
                                                                  verbose=verbose,
                                                                  return_inputs=True, return_targets=True)

        if verbose:
            print("Clustering Features:")  # Todo make Neural_Features_Clustering_With_Preprocessing 2 functions

        deepcluster = deep_cluster.Neural_Features_Clustering_With_Preprocessing(data=features, verbose=verbose,
                                                                                 random_state=random_state, pca=pca)

        deepcluster.cluster(algorithm=clustering_tech, n_clusters=n_clusters)

        clustering_tracker.update(epoch, deepcluster.clustered_data_indices)

        images_pseudolabels, rearranged_clusters = deep_cluster.clustered_data_indices_to_list(
            deepcluster.clustered_data_indices,
            reindex=True)

        deep_cluster_trainset = deep_cluster.LabelsReassignedDataset(original_dataset=trainset,
                                                                     image_indexes=images_pseudolabels[0],
                                                                     pseudolabels=images_pseudolabels[1],
                                                                     transform=training_transformation)

        sampler = utils.UnifLabelSampler(images_lists=rearranged_clusters,
                                         dataset_multiplier=network_iterations,
                                         size_per_pseudolabel=size_per_pseudolabel)

        train_dataloader = torch.utils.data.DataLoader(
            deep_cluster_trainset,
            batch_size=training_batch_size,
            num_workers=4,
            sampler=sampler,
            pin_memory=True,
        )

        if verbose:
            print("Adding top layer with output size: %d" % model_output_size)

        models.add_top_layer(model, [('L', model_output_size, len(rearranged_clusters))], device=device)

        tl_optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model.top_layer.parameters()),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        if verbose:
            print("Training Model: ")

        loss = models.deep_cluster_train(dataloader=train_dataloader, model=model,
                                         loss_criterion=loss_criterion,
                                         net_optimizer=optimizer,
                                         annxed_layers_optimizer=tl_optimizer,
                                         epoch=epoch, device=device)

        if verbose:
            print("Getting trained model output on pseudo-training_set to measure performance")

        outputs, targets = models.compute_network_output(dataloader=trainloader,
                                                         model=model, device=device,
                                                         batch_size=trainloader_batch_size,
                                                         data_size=len(trainloader.dataset),
                                                         verbose=True,
                                                         return_targets=True)

        nmi_meter.update(epoch, ground_truth=targets, predictions=np.argmax(outputs, 1))

        clustering_tracker.save_clustering_log(path=output_directory + "/" + "clustering_log.npy", override=True)
        nmi_meter.store_as_csv(path=output_directory + "/" + "model_nmi.csv", override=True)

        if verbose:
            print("Removing added top layer")

        models.remove_top_layer(model)

    # Save final model
    models.save_model_parameter(model=model, path=output_directory + "/final_model.pth", override=True)
    if verbose:
        print(" Saved final model at %s/final_model.pth" % output_directory)


def multinomial_regressor_train(model, model_path, dataloader,
                                learning_rate, momentum, weight_decay, n_epochs,
                                number_of_classes, device, output_directory, verbose=0):

    models.load_model_parameter(model, model_path)

    model_output_size = model.get_output_size()
    models.add_top_layer(model, [('L', model_output_size, number_of_classes)], device=device)

    models.freeze_module(model.features)
    models.freeze_module(model.classifier)

    loss_criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.top_layer.parameters()),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # switch to train mode
    model.train()

    for epoch in range(n_epochs):
        models.normal_train(model, dataloader, loss_criterion, optimizer, epoch, device, verbose)



