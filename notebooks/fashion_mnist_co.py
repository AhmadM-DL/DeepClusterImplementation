# %%
import sys
sys.path.append("../")
sys.path.append("../evaluation/")

# %%
from torchvision.datasets import FashionMNIST
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

#%%
from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering_models import LeNet
from co_deep_clustering import deep_cluster
from linear_probe import eval_linear

# %%
fashion_mnist = FashionMNIST("../datasets", download=True)
dataset = DeepClusteringDataset(fashion_mnist)

normalize = transforms.Normalize(mean=(0.2860,),std=(0.3205,))

loading_transform = transforms.Compose([
    transforms.Resize(44),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    normalize])

training_transform = transforms.Compose([
    transforms.Resize(44),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

# %%
# fashion_mnist.transform = transforms.Compose([transforms.ToTensor()])
# loader = DataLoader(fashion_mnist,
#                          batch_size=10,
#                          num_workers=0,
#                          shuffle=False)

# mean = 0.
# std = 0.
# for images, _ in loader:
#     batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
#     images = images.view(batch_samples, images.size(1), -1)
#     mean += images.mean(2).sum(0)
#     std += images.std(2).sum(0)

# mean /= len(loader.dataset) # 0.2860
# std /= len(loader.dataset) # 0.3205

# %%
device= torch.device("cpu")
modelA = LeNet(batch_normalization=True, device=device)
modelB = LeNet(batch_normalization=True, device=device)

loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

optimizerA = torch.optim.SGD(
    filter(lambda x: x.requires_grad, modelA.parameters()),
    lr = 0.01,
    momentum= 0.9,
    weight_decay= 0.00001,
)
optimizerB = torch.optim.SGD(
    filter(lambda x: x.requires_grad, modelB.parameters()),
    lr = 0.01,
    momentum= 0.9,
    weight_decay= 0.00001,
)

# %%
writer = SummaryWriter(log_dir="runs/fashion_mnist_co_strong1weak0.01")

# %%

deep_cluster(
    modelA= modelA,
    modelB= modelB,
    dataset= dataset,
    n_clusters= 10,
    loss_fn= loss_fn,
    optimizerA= optimizerA,
    optimizerB= optimizerB,
    n_cycles= 20,
    strong_instance_weight= 1,
    weak_instance_weight= 0.01,
    loading_transform= loading_transform,
    training_transform= training_transform,
    random_state=0,
    verbose=True,
    writer= writer
)

# %%

fashion_mnist_test = FashionMNIST("../datasets", train=False, download=True)

fashion_mnist.transform = training_transform
fashion_mnist_test.transform = loading_transform

eval_linear(
    model= model,
    n_epochs= 20,
    traindataset= fashion_mnist,
    validdataset= fashion_mnist_test,
    target_layer= "conv_2",
    n_labels= 10,
    features_size= 1600,
    avg_pool= None,
    random_state=0,
    writer= writer,
    verbose=True
)

# %%
