# %%
import sys
sys.path.append("../")
sys.path.append("../evaluation")

# %%
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
from torchvision import transforms


#%%

svhn = SVHN("../datasets", download= True)

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
