# %%
import sys
sys.path.append("../")
sys.path.append("../evaluation")

# %%
import torch
from torchvision.datasets import KMNIST
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# %%
from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering_models import LeNet
from deep_clustering import deep_cluster
from linear_probe import eval_linear

# %%
kanji_mnist = KMNIST("../datasets", download= True)

dataset = DeepClusteringDataset(kanji_mnist)

normalize = transforms.Normalize(mean=(0.1918,),std=(0.3385,))
training_transform = transforms.Compose([
    transforms.Resize(44),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    normalize,
])

loading_transform = transforms.Compose([
    transforms.Resize(44),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    normalize,
])

# %%
# kanji_mnist.transform = transforms.Compose([transforms.ToTensor()])
# loader = DataLoader(kanji_mnist,
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

# mean /= len(loader.dataset) # 0.1918
# std /= len(loader.dataset) # 0.3385

# %%
device = torch.device("cpu")
model = LeNet(batch_normalization=True, device=device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    filter(lambda x: x.requires_grad, model.parameters()),
    lr = 0.01,
    momentum= 0.9,
    weight_decay= 0.00001,
)

# %%
name= "kanji_mnist"
writer = SummaryWriter(log_dir="runs/"+name)

# %%

deep_cluster(
    model= model,
    dataset= dataset,
    n_clusters= 10,
    loss_fn= loss_fn,
    optimizer= optimizer,
    n_cycles= 20,
    loading_transform= loading_transform,
    training_transform= training_transform,
    random_state=0,
    verbose=True,
    checkpoints= "checkpoints/"+name,
    writer= writer
)


# %%
