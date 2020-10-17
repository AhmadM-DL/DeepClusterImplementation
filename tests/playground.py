#%%
import sys
sys.path.append("../.")

#%%
from deep_clustering_net import DeepClusteringNet
import torch
from layers_stacker import stack_convolutional_layers, stack_linear_layers

#%%
sobel = True
concat_sobel = True
input_ = torch.rand((100,3,224,224))
n_input_channels = 2 + int(not sobel) if not concat_sobel else 5
device = torch.device("cpu")
batch_normalization= True

# %%
alexnet_features_cfg = [
            {
            "type": "convolution",
            "out_channels":96,
            "kernel_size":11,
            "stride":4,
            "padding":2,
            "activation":"ReLU",
            },

            {
            "type":"max_pool",
            "kernel_size":3,
            "stride":2,
            },

            {
            "type": "convolution",
            "out_channels":256,
            "kernel_size":5,
            "stride":1,
            "padding":2,
            "activation":"ReLU",
            },

            {
            "type":"max_pool",
            "kernel_size":3,
            "stride":2,
            },

            {
            "type": "convolution",
            "out_channels":384,
            "kernel_size":3,
            "stride":1,
            "padding":1,
            "activation":"ReLU",
            },

            {
            "type": "convolution",
            "out_channels":384,
            "kernel_size":3,
            "stride":1,
            "padding":1,
            "activation":"ReLU",
            },

            {
            "type": "convolution",
            "out_channels":256,
            "kernel_size":3,
            "stride":1,
            "padding":1,
            "activation":"ReLU",
            },

            {
            "type":"max_pool",
            "kernel_size":3,
            "stride":2,
            }         
            ]

#%%
classifier_cfg = [
                    {"type":"drop_out",
                    "drop_ratio": 0.5},

                    {"type":"linear",
                    "out_features":4096,
                    "activation":"ReLU"},

                    {"type":"drop_out",
                    "drop_ratio": 0.5},

                    {"type":"linear",
                    "out_features":4096}
    ]

#%%
model = DeepClusteringNet(input_size=(3,224,224),
                          features= stack_convolutional_layers(input_channels= n_input_channels, cfg=alexnet_features_cfg, batch_normalization=batch_normalization),
                          classifier= stack_linear_layers(input_features= 256 * 6 * 6, cfg= classifier_cfg),
                          top_layer = None,
                          with_sobel=sobel,
                          concat_sobel=True,
                          device=device)

#%%
model(input_)
