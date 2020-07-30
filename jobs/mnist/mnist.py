import torch
import json
import logging
import argparse

import sys
import importlib

sys.path.append("C:\\Users\\PC\\Desktop\\Projects\\DeepClusterImplementation")


from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop
from deep_clustering import deep_cluster
from deep_clustering_models import LeNet
from deep_clustering_dataset import DeepClusteringDataset

from evaluation.linear_probe import eval_linear


def run(device, batch_norm, lr, wd, momentum, n_cycles,
        n_clusters, pca, training_batch_size, training_shuffle,
        random_state, ):

    logging.info("Loading Dataset")
    mnist = MNIST("./datasets/", download=True)

    device = torch.device(device)

    logging.info("Build Model")
    model = LeNet(batch_normalization=batch_norm, device=device)

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
    dataset = DeepClusteringDataset(mnist)

    logging.info("Defining Transformations")
    normalize = transforms.Normalize(mean=(0.1307,),
                                     std=(0.3081,))

    training_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
        normalize])

    loading_transform = transforms.Compose([
        transforms.Resize(44),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize])

    logging.info("Defining Writer")
    writer_file = "mnist_"+"batchnorm(%d)_" % batch_norm + \
        "lr(%f)_" % lr+"momentum(%f)_" % momentum+"wdecay(%f)_" % wd + \
        "n_clusters(%d)_" % n_clusters + \
        "n_cycles(%d)_" % n_cycles+"rnd(%d)_" % random_state + \
        "t_batch_size(%d)_" % training_batch_size + \
        "shuffle(%d)_" % training_shuffle
    if pca:
        writer_file=writer_file+"pca(%d)_"%pca
    else:
        writer_file=writer_file+"pca(None)_"

    writer = SummaryWriter(
        'runs/'+writer_file)

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
                 writer=writer)
    
    mnist_test = MNIST("../datasets/", train=False, download=True)

    traindataset = mnist
    validdataset = mnist_test

    transformations_val = [transforms.Resize(44),
                                transforms.CenterCrop(32),
                                transforms.ToTensor(),
                                normalize]

    transformations_train = [transforms.Resize(44),
                                transforms.CenterCrop(32),
                                #transforms.RandomCrop(32),
                                #transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize]

    mnist.transform = transforms.Compose(transformations_train)
    mnist_test.transform = transforms.Compose(transformations_val)

    eval_linear(model=model,
                n_epochs=20,
                traindataset=traindataset,
                validdataset=validdataset,
                target_layer="conv_2",
                n_labels=10,
                features_size=1600,
                avg_pool= None,
                random_state=random_state,
                writer= writer,
                verbose=1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    parser.add_argument('--hyperparam', default="C:\\Users\\PC\\Desktop\\Projects\\DeepClusterImplementation\\jobs\\mnist\\hyper.json", type=str, help='Path to hyperparam json file')
    args = parser.parse_args()

    logging.basicConfig(filename='app.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')
    hparams = json.load(open(args.hyperparam, "r"))
    device = torch.device("cpu")

    for lr in hparams['lr']:
        for wd in hparams['wd']:
            for momentum in hparams['momentum']:
                for batch_norm in hparams['batch_norm']:
                    for n_cycles in hparams["n_cycles"]:
                        for n_clusters in hparams["n_clusters"]:
                            for pca in hparams["pca"]:
                                for training_batch_size in hparams["training_batch_size"]:
                                    for training_shuffle in hparams["training_shuffle"]:
                                        run(device, batch_norm, lr, wd, momentum, n_cycles, n_clusters,
                                        pca, training_batch_size, training_shuffle, random_state=0)
