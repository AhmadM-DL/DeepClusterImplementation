# -*- coding: utf-8 -*-
"""
Created on Tuesday April 15 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)

This module represents a mini-framework to stack sequentially neural network layers from
a Simple JSON Configuration File. I build it for convenience and to be useful as a usable
module for Deep Learning Reseach. I will add the layers it support as much as I use.
Currently it only support Linear, Dropout, and 2D-Convolutional layers with ReLU activation
and Batch Normalization.

I had to choose between 2 design decisions:

    - give the user complete control on layers input (whether size or channels).

    - take only the initial input and dynamically link the output of one layer
      to the input of the next layer. 

I choose the second design choice as it felt more convineant for standard stacks.
I had to separate between stacking liner and convolutional layer because otherwise
and following my design I had to compute the flattened size of convolutions. This
require to know the input size which I prefer to be dynamic for convolutions. 

"""
import torch
from collections import OrderedDict

example_convolutional_cfg = [
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

example_linear_cfg = [{"type":"drop_out",
                       "drop_ratio": 0.5},

                      {"type":"linear",
                       "out_features":4096,
                       "activation":"ReLU"},

                      {"type":"drop_out",
                       "drop_ratio": 0.5},

                      {"type":"linear",
                      "out_features":4096,
                      "activation":"ReLU"}

                       ]

activations = ["ReLU", "Sigmoid"]

def stack_convolutional_layers(input_channels, cfg, with_modules_names=True, batch_normalization=False):
    """
    This method aims to stack convolutional layers on each other given a convolutional layers cfg with max pooling
    and batch normalization
    """
    layers = []
    in_channels = input_channels
    for layer_cfg in cfg:

        layer_type = layer_cfg.get("type", None)

        if not layer_type: 
            raise Exception("stack_convolutional_layers", "A layer cfg is missing 'type' attribute")

        if layer_type == "convolution":
            parsed_layers = parse_convolution(in_channels, layer_cfg)
            in_channels = layer_cfg["out_channels"]
            if batch_normalization:
                layers.extend([parsed_layers[0],
                              torch.nn.BatchNorm2d(in_channels),
                              parsed_layers[1]])
            else:
                layers.extend(parsed_layers)
            
        elif layer_type == "drop_out":
            layers.extend([parse_drop_out(layer_cfg)])

        elif layer_type == "max_pool":
            layers.extend([parse_max_pool(in_channels, layer_cfg)])
        
        else:
            raise Exception("stack_convolutional_layers", "Cfg object contains an unknown layer %s"%layer_type)
        
    if with_modules_names: 
        return torch.nn.Sequential(name_layers(layers))
    else:
        return torch.nn.Sequential(*layers)

def test_stack_convolutional_layers():
    named_layers = stack_convolutional_layers(input_channels=3, cfg=example_convolutional_cfg, batch_normalization=False)
    # get number conv modules
    n_conv = len([ name for (name,module) in list(named_layers.named_modules()) if "conv" in name])
    assert n_conv == 5
    assert len(list(named_layers.named_children())) == len(example_convolutional_cfg) + n_conv
    assert named_layers.relu_2
    assert named_layers.conv_4
    assert named_layers.max_pool_3
    named_layers = stack_convolutional_layers(input_channels=3, cfg=example_convolutional_cfg, batch_normalization=True)
    assert len(list(named_layers.named_children())) == len(example_convolutional_cfg) + 2*n_conv
    assert named_layers.batch_norm_3
    return 

def stack_linear_layers(input_features, cfg, with_modules_names=True):
    """
    This method aims to stack linear layers on each other given a linear layers cfg with drop out
    """
    layers = []
    in_features = input_features
    for layer_cfg in cfg:
        
        layer_type = layer_cfg.get("type", None)

        if not layer_type: 
            raise Exception("stack_linear_layers", "A layer cfg is missing 'type' attribute")

        if layer_type == "linear":
            layers.extend(parse_linear(in_features, layer_cfg))
            in_features = layer_cfg["out_features"]

        elif layer_type == "drop_out":
            layers.extend([parse_drop_out(layer_cfg)])
        
        else:
            raise Exception("stack_linear_layers", "Cfg object contains an unknown layer %s"%layer_type)
    
    if with_modules_names: 
        return torch.nn.Sequential(name_layers(layers))
    else:
        return torch.nn.Sequential(*layers)

def test_stack_linear_layers():
    named_layers = stack_linear_layers(input_features=256*6*6, cfg=example_linear_cfg)
    # get number conv modules
    n_lin = len([ name for (name,module) in list(named_layers.named_modules()) if "linear" in name])
    assert n_lin == 2
    assert len(list(named_layers.named_children())) == len(example_convolutional_cfg) + n_lin
    assert named_layers.relu_2
    assert named_layers.drop_out_1
    assert named_layers.linear_2
    return

def name_layers(layers_list):
    """
    A method that takes a list of nn.modules and returns them as an ordered Dict.
    """
    layers_counts= {}
    named_layers= []
    for layer in layers_list:
        if isinstance(layer, torch.nn.Conv2d):
            current_layer_count = layers_counts.get("convolution2d",1)
            named_layers.append(("conv_%d"%current_layer_count, layer))
            layers_counts["convolution2d"] = current_layer_count+1
        elif isinstance(layer, torch.nn.Linear):
            current_layer_count = layers_counts.get("linear",1)
            named_layers.append(("linear_%d"%current_layer_count, layer))
            layers_counts["linear"] = current_layer_count+1
        elif isinstance(layer, torch.nn.MaxPool2d):
            current_layer_count = layers_counts.get("max_pool",1)
            named_layers.append(("max_pool_%d"%current_layer_count, layer))
            layers_counts["max_pool"] = current_layer_count+1 
        elif isinstance(layer, torch.nn.Dropout):
            current_layer_count = layers_counts.get("drop_out",1)
            named_layers.append(("drop_out_%d"%current_layer_count, layer))
            layers_counts["drop_out"] = current_layer_count+1
        elif isinstance(layer, torch.nn.BatchNorm2d):
            current_layer_count = layers_counts.get("batch_norm",1)
            named_layers.append(("batch_norm_%d"%current_layer_count, layer))
            layers_counts["batch_norm"] = current_layer_count+1        
        elif isinstance(layer, torch.nn.ReLU):
            current_layer_count = layers_counts.get("relu",1)
            named_layers.append(("relu_%d"%current_layer_count, layer))
            layers_counts["relu"] = current_layer_count+1
        else:
            raise Exception("name_layers", "Paramters layers_list contain an unsupported layer type %s"%(type(layer)))
    return OrderedDict(named_layers)

def test_name_layers():
    layers = [ torch.nn.Conv2d(3,16,11,2),
               torch.nn.MaxPool2d(2, 3),
               torch.nn.Conv2d(3,16,11,2),
               torch.nn.Dropout(0.3),
               torch.nn.Linear(100,20) ]
    named_layers= name_layers(layers)

    assert named_layers.get("conv_1",0)!=0
    assert named_layers.get("max_pool_1",0)!=0
    assert named_layers.get("conv_2",0)!=0
    assert named_layers.get("drop_out_1",0)!=0
    assert named_layers.get("linear_1",0)!=0 

def parse_convolution(in_channels, cfg):
    """
    A method to read a convolution cfg and return a torch.nn.conv2d layer 
    """
    if cfg.get("out_channels",None)==None:
        raise Exception("Layers_Generator_CFG_Error", "Convolution should include out_channels attribute")
    
    if cfg.get("kernel_size",None)==None:
        raise Exception("Layers_Generator_CFG_Error", "Convolution should include kernel_size attribute")
    
    if cfg.get("stride",None)==None:
        raise Exception("Layers_Generator_CFG_Error", "Convolution should include stride attribute")
    
    if cfg.get("padding",None)==None:
        raise Exception("Layers_Generator_CFG_Error", "Convolution should include pad attribute")
    
    if cfg.get("activation",None)==None:
        raise Exception("Layers_Generator_CFG_Error", "Convolution should include activation attribute")

    if cfg["activation"] == "ReLU":
        activation_layer = torch.nn.ReLU()
    elif cfg["activation"] == "Sigmoid":
        activation_layer == torch.nn.Sigmoid()
    else:
        raise Exception("Layers_Generator_CFG_Error", "Activation %s is not supproted."%(cfg["activation"]))


    return [torch.nn.Conv2d(in_channels= in_channels,
                            out_channels= cfg["out_channels"],
                            kernel_size= cfg["kernel_size"],
                            stride= cfg["stride"],
                            padding= cfg["padding"]
                            ),
            activation_layer]

def parse_max_pool(in_channels, cfg):
    """
    A method to read a 2d max_pool cfg and return a torch.nn.MaxPool2d layer 
    """
    if cfg.get("kernel_size", None) == None:
        raise Exception("Layers_Generator_CFG_Error", "Max Pool should include kernel_size attribute")
    if cfg.get("stride", None) == None:
        raise Exception("Layers_Generator_CFG_Error", "Max Pool should include stride attribute")

    return torch.nn.MaxPool2d(kernel_size=cfg["kernel_size"], stride=cfg["stride"])

def parse_drop_out(cfg):
    """
    A method to read a drop_out cfg and return a torch.nn.Dropout layer 
    """
    if cfg.get("drop_ratio", None) == None:
        raise Exception("Layers_Generator_CFG_Error", "Drop out should include drop_ratio attribute")

    return torch.nn.Dropout(p=cfg["drop_ratio"])

def parse_linear(in_features, cfg):
    """
    A method to read a linear cfg and return a torch.nn.linear layer 
    """
    if cfg.get("out_features", None) == None:
        raise Exception("Layers_Generator_CFG_Error", "Linear should include out_channels attribute")
    if cfg.get("activation", None) == None:
        raise Exception("Layers_Generator_CFG_Error", "Linear should include activation attribute")

    if cfg["activation"] == "ReLU":
        activation_layer = torch.nn.ReLU()
    elif cfg["activation"] == "Sigmoid":
        activation_layer = torch.nn.Sigmoid()
    else:
        raise Exception("Layers_Generator_CFG_Error", "Activation %s is not supproted."%(cfg["activation"]))

    return [torch.nn.Linear(in_features= in_features,
                            out_features=cfg["out_features"]),
                activation_layer]

