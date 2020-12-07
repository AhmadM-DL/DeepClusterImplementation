import torch
import json
import logging
import argparse
import datetime

import sys
import importlib
import os

sys.path.append("C:\\Users\\PC\\Desktop\\Projects\\DeepClusterImplementation")

from torchvision import transforms
from torchvision.datasets import SVHN
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop
from deep_clustering_models import AlexNet_Micro
from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering import deep_cluster

from evaluation.linear_probe import eval_linear
from utils import set_seed

def run(device, batch_norm, lr, wd, momentum, n_cycles,
        n_clusters, pca, training_batch_size, sobel, training_shuffle,
        random_state, dataset_path):

    logging.info("Set Seed")
    set_seed(random_state)
    
    logging.info("Loading Dataset")
    svhn = SVHN(dataset_path, download=True)

    device = torch.device(device)

    logging.info("Build Model")
    model = AlexNet_Micro(sobel=sobel, batch_normalization=batch_norm, device=device)

    logging.info("Build Optimizer")
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=wd,
    )

    logging.info("Defining Loss Function")
    loss_function = torch.nn.CrossEntropyLoss()

    logging.info("Decorating Dataset")
    dataset = DeepClusteringDataset(svhn)

    logging.info("Defining Transformations")
    normalize = transforms.Normalize(mean = (0.437, 0.443, 0.472),
                                     std = (0.198, 0.201, 0.197))

    training_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
        normalize])

    loading_transform = transforms.Compose([
        transforms.Resize(42),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize])

    logging.info("Defining Writer")
    writer_file = "svhn_"+"batchnorm(%d)_" % batch_norm + \
        "lr(%f)_" % lr+"momentum(%f)_" % momentum+"wdecay(%f)_" % wd + \
        "n_clusters(%d)_" % n_clusters + \
        "n_cycles(%d)_" % n_cycles+"rnd(%d)_" % random_state + \
        "t_batch_size(%d)_" % training_batch_size + \
        "shuffle(%d)_" % training_shuffle
    if pca:
        writer_file=writer_file+"pca(%d)_"%pca
    else:
        writer_file=writer_file+"pca(None)_"

    writer = SummaryWriter('runs/'+writer_file)

    if os.path.isfile(writer_file+"/checkpoints/last_model.pth"):
        resume = "checkpoints/"+writer_file+"/last_model.pth"
        logging.INFO("##########\nResuming from: %s\n##########"%resume)
    else: 
        resume = None
        logging.INFO("##########\nRun: %s\n##########"%writer_file)
    
    deep_cluster(model=model,
                 dataset=dataset,
                 n_clusters=n_clusters,
                 loss_fn=loss_function,
                 optimizer=optimizer,
                 n_cycles=n_cycles,
                 loading_transform=loading_transform,
                 training_transform=training_transform,
                 random_state=random_state,
                 verbose=1,
                 training_batch_size=training_batch_size,
                 training_shuffle=training_shuffle,
                 pca_components=pca,
                 checkpoints= "checkpoints/"+writer_file,
                 resume=resume,
                 writer=writer)
    
    svhn_test = SVHN(dataset_path, train=False, download=True)

    traindataset = svhn
    validdataset = svhn_test

    transformations_val = [transforms.Resize(42),
                                transforms.CenterCrop(32),
                                transforms.ToTensor(),
                                normalize]

    transformations_train = [transforms.Resize(42),
                                transforms.CenterCrop(32),
                                #transforms.RandomCrop(32),
                                #transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize]

    svhn.transform = transforms.Compose(transformations_train)
    svhn.transform = transforms.Compose(transformations_val)

    logging.INFO("Evaluation")
    eval_linear(model=model,
                n_epochs=20,
                traindataset=traindataset,
                validdataset=validdataset,
                target_layer="relu_5",
                n_labels=10,
                features_size= 512,
                avg_pool= None,
                random_state=random_state,
                writer= writer,
                verbose=1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    parser.add_argument('--hyperparam', default="./hyper.json", type=str, help='Path to hyperparam json file')
    parser.add_argument('--dataset', default="../datasets", type=str, help="Path to datasets")
    parser.add_argument('--device', default="cpu", type=str, help="Device to use")
    parser.add_argument('--seed', default=666, type=int, help="Random Seed")
    parser.add_argument('--chkp', default=None, type=str, help="Checkpoint")
    args = parser.parse_args()

    logging.basicConfig(filename='app.log', filemode='a',
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    logging.INFO("##########\n%s##########"%(datetime.now()))
    hparams = json.load(open(args.hyperparam, "r"))
    device = torch.device(args.device)
    if args.chkp:
        executed_runs = int(open(args.chkp, "r").read())
        logging.INFO("Running from checkpoint: run(%d)"%executed_runs)
    else:
        executed_runs = 0 
    counter=1
    for lr in hparams['lr']:
        for wd in hparams['wd']:
            for momentum in hparams['momentum']:
                for batch_norm in hparams['batch_norm']:
                    for n_cycles in hparams["n_cycles"]:
                        for n_clusters in hparams["n_clusters"]:
                            for pca in hparams["pca"]:
                                for training_batch_size in hparams["training_batch_size"]:
                                    for training_shuffle in hparams["training_shuffle"]:
                                        for sobel in hparams["sobel"]:
                                            if counter <= executed_runs:
                                                continue
                                            run(device, batch_norm, lr, wd, momentum, n_cycles, n_clusters,
                                            pca, training_batch_size, training_shuffle, sobel, random_state=args.seed, dataset_path=args.dataset)
                                            counter+=1
                                            open(args.chpk, "w").write(str(counter))
