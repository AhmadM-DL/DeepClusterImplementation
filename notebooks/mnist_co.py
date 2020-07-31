# %%
import sys
sys.path.append("../")
sys.path.append("../evaluation/")

# %%
from torchvision.datasets import MNIST
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

#%%
from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering_models import LeNet
from co_deep_clustering import deep_cluster
from linear_probe import eval_linear
from utils import qualify_space, set_seed

# %%
hparams= {
    'lr': 0.01,
    'momentum': 0.9,
    "weight_decay": 0.00001,
    "n_clusters": 10,
    "n_cycles": 20,
    "strong_instance_weight":1,
    "weak_instance_weight":0.1,
    "random_state":0,
    "batch_norm":True,
}

# %%
mnist = MNIST("../datasets", download=True)
dataset = DeepClusteringDataset(mnist)

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
modelA = LeNet(batch_normalization=hparams["batch_norm"], device=device)
modelB = LeNet(batch_normalization=hparams["batch_norm"], device=device)

loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

optimizerA = torch.optim.SGD(
    filter(lambda x: x.requires_grad, modelA.parameters()),
    lr = hparams["lr"],
    momentum= hparams["momentum"],
    weight_decay= hparams["weight_decay"],
)
optimizerB = torch.optim.SGD(
    filter(lambda x: x.requires_grad, modelB.parameters()),
    lr = hparams["lr"],
    momentum= hparams["momentum"],
    weight_decay= hparams["weight_decay"],
)

# %%
writer = SummaryWriter(log_dir="runs/mnist_co_strong1weak0.1_3")

# %%
# dataset.set_transform(loading_transform)
# qualify_space(modelA, dataset, [5,10,12,20,50], writer, clustering_algorithm="gmm")

# %%
# write hyper parameters
#writer.add_hparams(hparams)

# %%

deep_cluster(
    modelA= modelA,
    modelB= modelB,
    dataset= dataset,
    n_clusters= hparams["n_clusters"],
    loss_fn= loss_fn,
    optimizerA= optimizerA,
    optimizerB= optimizerB,
    n_cycles= hparams["n_cycles"],
    strong_instance_weight= hparams["strong_instance_weight"],
    weak_instance_weight= hparams["weak_instance_weight"],
    loading_transform= loading_transform,
    training_transform= training_transform,
    random_state=hparams,
    verbose=True,
    writer= writer
)

# %%
writer.flush()
writer.close()

# %%

mnist_test = MNIST("../datasets", train=False, download=True)

mnist.transform = training_transform
mnist_test.transform = loading_transform

eval_linear(
    model= modelA,
    n_epochs= 20,
    traindataset= mnist,
    validdataset= mnist_test,
    target_layer= "conv_2",
    n_labels= 10,
    features_size= 1600,
    avg_pool= None,
    random_state=0,
    writer= writer,
    verbose=True
)

# %%
