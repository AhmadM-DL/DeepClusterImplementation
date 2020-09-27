
import logging
import torch
import time
import numpy as np

from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering_models import AlexNet_Small
from deep_clustering import deep_cluster
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision import transforms
from linear_probe import eval_linear
from utils import set_seed, qualify_space

hparams= {
    'lr': 0.05,
    'momentum': 0.9,
    "weight_decay": 0.00001,
    "n_clusters": 2000,
    "n_cycles": 150,
    "random_state":0,
    "pca":256,
    "batch_norm":True,
    "sobel":True,
    "batch_size":256,
    "checkpoints_interval":10,
}

def main(args):

    max_iters = [20, 50, 100, 300]

    for max_iter in max_iters:

        set_seed(42)

        #logging.info("Loading Dataset")
        cifar = CIFAR10(args.dataset)
        cifar_test = CIFAR10(args.dataset, train=False)

        device = torch.device("cuda:0")

        #logging.info("Build Model")
        model = AlexNet_Small(
            sobel=hparams["sobel"], batch_normalization=hparams["batch_norm"], device=device)

        #logging.info("Build Optimizer")
        optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=hparams["lr"],
            momentum=hparams["momentum"],
            weight_decay=hparams["weight_decay"],
        )

        #logging.info("Defining Loss Function")
        loss_function = torch.nn.CrossEntropyLoss()

        #logging.info("Decorating Dataset")
        dataset = DeepClusteringDataset(cifar)

        #logging.info("Defining Transformations")
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

        #logging.info("Defining Writer")
        
        writer_file = "cifar10_alexnet_small"+"batchnorm(%d)_" % hparams["batch_norm"] + \
        "lr(%f)_"% hparams["lr"]+"momentum(%f)_" % hparams["momentum"]+"wdecay(%f)_" % hparams["weight_decay"]  + \
        "n_clusters(%d)_" % hparams["n_clusters"] + \
        "pca(%d)_" % hparams["pca"] + "sobel(%d)"%hparams["sobel"] +\
        "n_cycles(%d)_" % hparams["n_cycles"]+"rnd(%d)_" % hparams["random_state"] +\
        "t_batch_size(%d)_"% hparams["batch_size"]
        
        writer = SummaryWriter(
            'runs/'+writer_file)

        dataset.set_transform(loading_transform)

        end = time.time()
        deep_cluster(model=model,
                    dataset=dataset,
                    n_clusters=hparams["n_clusters"],
                    loss_fn=loss_function,
                    optimizer=optimizer,
                    n_cycles=hparams["n_cycles"],
                    loading_transform=loading_transform,
                    training_transform=training_transform,
                    random_state=hparams["random_state"],
                    pca_components=hparams["pca"],
                    kmeans_max_iter = max_iter,
                    verbose=1,
                    writer=writer
                    )
                    
        cifar_test.transform = loading_transform
        cifar.transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(256),
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize])
        
        eval_linear(
        model= model,
        n_epochs= 30,
        traindataset= cifar,
        validdataset= cifar_test,
        target_layer= "relu_5",
        n_labels= 10,
        features_size= 9216,
        avg_pool= {"kernel_size":2, "stride":2, "padding":0},
        writer= writer
        )
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    parser.add_argument('--dataset', default="./datasets", type=str, help="Path to datasets")
    args = parser.parse_args()
    main(args)
