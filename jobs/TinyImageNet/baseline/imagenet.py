
import logging
import torch
import numpy as np

from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering_models import AlexNet_Small
from deep_clustering import deep_cluster
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from linear_probe import eval_linear
from utils import set_seed, qualify_space
import os

hparams= {
    'lr': 0.05,
    'momentum': 0.9,
    "weight_decay": 0.00001,
    "n_clusters": 2000,
    "n_cycles": 100,
    "random_state":0,
    "pca":256,
    "batch_norm":True,
    "sobel":True,
    "batch_size":256,
    "checkpoints_interval":10,
}

def main():

    seeds = [39, 6, 3, 59, 11, 29, 0, 99, 28, 60]
    fixed_seed = 41

    #logging.info("Loading Dataset")
    tiny_imagenet = datasets.ImageFolder(os.path.join("./datasets/", "train"))
    tiny_imagenet_test = datasets.ImageFolder(os.path.join("./datasets/", "test"))

    device = torch.device("cuda:0")

    #logging.info("Defining Transformations")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
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

    for seed in seeds:

        set_seed(seed)

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
        dataset = DeepClusteringDataset(tiny_imagenet)

        #logging.info("Defining Writer")
        
        writer_file = "tiny_imagenet_alexnet_small"+"batchnorm(%d)_" % hparams["batch_norm"] + \
        "lr(%f)_"% hparams["lr"]+"momentum(%f)_" % hparams["momentum"]+"wdecay(%f)_" % hparams["weight_decay"]  + \
        "n_clusters(%d)_" % hparams["n_clusters"] + \
        "pca(%d)_" % hparams["pca"] + "sobel(%d)"%hparams["sobel"] +\
        "n_cycles(%d)_" % hparams["n_cycles"]+"rnd(%d)_" % hparams["random_state"] +\
        "t_batch_size(%d)_"% hparams["batch_size"] +\
        "seed(%d)_"%seed
        
        writer = SummaryWriter(
            'runs/'+writer_file)

        set_seed(fixed_seed)

        dataset.set_transform(loading_transform)
        space_quality = qualify_space(model, dataset, [ 2000 ], writer, clustering_algorithm="kmeans")
        np.save(file=writer.log_dir+"/space_quality",arr=space_quality)

        deep_cluster(model=model,
                    dataset=dataset,
                    n_clusters=hparams["n_clusters"],
                    loss_fn=loss_function,
                    optimizer=optimizer,
                    n_cycles=hparams["n_cycles"],
                    loading_transform=loading_transform,
                    training_transform=training_transform,
                    random_state=hparams["random_state"],
                    verbose=1,
                    writer=writer)

        tiny_imagenet_test.transform = loading_transform
        tiny_imagenet.transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(256),
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize])
        
        eval_linear(
        model= model,
        n_epochs= 30,
        traindataset= tiny_imagenet,
        validdataset= tiny_imagenet_test,
        target_layer= "relu_5",
        n_labels= 200,
        features_size= 9216,
        avg_pool= {"kernel_size":2, "stride":2, "padding":0},
        writer= writer
        )
    


if __name__ == '__main__':

    #logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    main()
