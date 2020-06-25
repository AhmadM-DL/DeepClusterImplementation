# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import sys
import importlib

sys.path.append("../")


# %%
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop

from deep_clustering import deep_cluster
from deep_clustering_models import LeNet_MNIST
from deep_clustering_dataset import DeepClusteringDataset


# %%
mnist = MNIST("../datasets/", download=True)


# %%
device = torch.device("cpu")


# %%
model = LeNet_MNIST(batch_normalization=True, device=device)


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
dataset = DeepClusteringDataset(mnist)


# %%
normalize = transforms.Normalize(mean= (0.1307,) ,
                                 std= (0.3081,))

training_transform = transforms.Compose([
                            transforms.RandomResizedCrop(32),
                            transforms.ToTensor(),
                            normalize])

loading_transform = transforms.Compose([
                            transforms.Resize(44),
                            transforms.CenterCrop(32),
                            transforms.ToTensor(),
                            normalize])


# %%
writer = SummaryWriter('runs/mnist_batchnorm_lr0.01_moment0.9_decay10^-5_ncluster10_ncycles50_rnd0')


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
