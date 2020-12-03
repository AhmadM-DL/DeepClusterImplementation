
import logging
import argparse

import sys
import importlib

sys.path.append("C:\\Users\\PC\\Desktop\\Projects\\DeepClusterImplementation")

import torch

from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering_models import AlexNet_Small
from deep_clustering import deep_cluster
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--device', default='cpu', type=str,
                        help='Device type(default: cpu)')
    parser.add_argument('--learning_rate','--lr', default=0.01, type=float,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay','--wd', default=0.00001, type=float,
                        help='weight decay pow (default: 0.00001)')
    parser.add_argument('--momentum', default=0.9,
                        type=float, help='momentum (default: 0.9)')
    parser.add_argument('--random_state', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--n_cycles', type=int, default=50,
                        help='number of total cycles to run (default: 50)')
    parser.add_argument('--n_clusters', '--k', type=int, default=10,
                        help='number of cluster for k-means (default: 10)')

    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--batch_norm', action='store_true',
                        help='Batch Normalization')

    return parser.parse_args()


def Cifar10Grid():

    logging.info("Loading Dataset")
    cifar = CIFAR10("./datasets/", download=True)

    device = torch.device(args.device)

    logging.info("Build Model")
    model = AlexNet_Small(
        sobel=args.sobel, batch_normalization=args.batch_norm, device=device)

    logging.info("Build Optimizer")
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    logging.info("Defining Loss Function")
    loss_function = torch.nn.CrossEntropyLoss()

    logging.info("Decorating Dataset")
    dataset = DeepClusteringDataset(cifar)

    logging.info("Defining Transformations")
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.247, 0.243, 0.261])
    training_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    loading_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    logging.info("Defining Writer")
    writer_file = "cifar10_"+"sobel(%d)_" %args.sobel+"batchnorm(%d)_"%args.batch_norm + \
        "lr(%f)_" %args.learning_rate+"momentum(%f)_" %args.momentum+"wdecay(%f)_"%args.weight_decay+"n_clusters(%d)_"%args.n_clusters+\
        "n_cycles(%d)_"%args.n_cycles+"rnd(%d)_"%args.random_state

    writer = SummaryWriter(
        'runs/'+writer_file)

    deep_cluster(model=model,
                 dataset=dataset,
                 n_clusters=args.n_clusters,
                 loss_fn=loss_function,
                 optimizer=optimizer,
                 n_cycles=args.n_cycles,
                 loading_transform=loading_transform,
                 training_transform=training_transform,
                 random_state=args.random_state,
                 verbose=1,
                 writer=writer)

if __name__ == '__main__':
    logging.basicConfig(filename='app.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')
    args = parse_args()
    main(args)
