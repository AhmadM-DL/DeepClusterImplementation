# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import sys
import importlib

sys.path.append("../")


# %%
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop

from deep_clustering import deep_cluster
from deep_clustering_models import AlexNet_CIFAR
from deep_clustering_dataset import DeepClusteringDataset


# %%
cifar = CIFAR10("../datasets/", download=True)


# %%
device = torch.device("cpu")


# %%
model = AlexNet_CIFAR(sobel=True, batch_normalization=True, device=device)


# %%
optimizer = torch.optim.SGD(
    filter(lambda x: x.requires_grad, model.parameters() ) ,
    lr=0.01,
    momentum=0.9,
    weight_decay=0.00005,
)


# %%
loss_function = torch.nn.CrossEntropyLoss()


# %%
dataset = DeepClusteringDataset(cifar)


# %%
normalize = transforms.Normalize(mean= [0.4914, 0.4822, 0.4465],
                                 std= [0.247, 0.243, 0.261])
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


# %%
writer = SummaryWriter('runs/cifar10_sobel_batchnorm_lr0.01_moment0.9_decay10^-5_ncluster10_ncycles50_rnd0')


# %%
deep_cluster(model= model, 
dataset = dataset,
n_clusters=10,
loss_fn = loss_function,
optimizer= optimizer,
n_cycles= 50,
loading_transform= loading_transform,
training_transform= training_transform,
random_state= 0,
verbose=1,
writer=writer)


# %%
