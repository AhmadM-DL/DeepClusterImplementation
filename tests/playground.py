#%%
import sys
sys.path.append("../.")

#%%
from deep_clustering_models import AlexNet_Micro
import torch
#%%
model = AlexNet_Micro(True, True, torch.device("cpu"))

#%%
s = model.extract_features(torch.rand((1,3,32,32)), "relu_5", flatten=False)
s.shape
# %%
