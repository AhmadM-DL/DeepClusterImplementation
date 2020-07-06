# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import sys
import importlib

sys.path.append("../")
sys.path.append("../evaluation")

# %%
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop
from torch.utils.data import DataLoader

from deep_clustering import deep_cluster
from deep_clustering_models import LeNet
from deep_clustering_dataset import DeepClusteringDataset
from linear_probe import LinearProbe, eval_linear

# %%
mnist = MNIST("../datasets/", download=True)
dataset = DeepClusteringDataset(mnist)
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
device = torch.device("cpu")
model = LeNet(batch_normalization=True, device=device)
optimizer = torch.optim.SGD(
    filter(lambda x: x.requires_grad, model.parameters() ) ,
    lr=0.01,
    momentum=0.9,
    weight_decay=0.00005,
)
loss_function = torch.nn.CrossEntropyLoss()

# %%
writer = SummaryWriter('runs/mnist')

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
# Linear Probe
mnist_test = MNIST("../datasets/", train=False, download=True)

traindataset = mnist
validdataset = mnist_test

transformations_val = [transforms.Resize(44),
                            transforms.CenterCrop(32),
                            transforms.ToTensor(),
                            normalize]

transformations_train = [transforms.Resize(44),
                            transforms.CenterCrop(44),
                            transforms.RandomCrop(32),
                            #transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize]

mnist.transform = transforms.Compose(transformations_train)
mnist_test.transform = transforms.Compose(transformations_val)


eval_linear(model=model, n_epochs= 5, traindataset=traindataset,
            validdataset= validdataset, target_layer="conv_2", n_labels=10,
            features_size= 1600, avg_pool= None, random_state=0, writer= writer,
            verbose= True)