# %%
import sys
sys.path.append("../")
sys.path.append("../evaluation")

# %%
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# %%
from deep_clustering_models import AlexNet_Small
from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering import deep_cluster
 


#%%

svhn = SVHN("../datasets", download= True)

dataset = DeepClusteringDataset(svhn)

normalize = transforms.Normalize(mean= [0.4377, 0.4438, 0.4728],
                                 std= [0.1201, 0.1231, 0.1052])

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
# svhn.transform = transforms.Compose([transforms.ToTensor()])
# loader = DataLoader(svhn,
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

# mean /= len(loader.dataset) # 0.4377, 0.4438, 0.4728
# std /= len(loader.dataset) # 0.1201, 0.1231, 0.1052

# %%
