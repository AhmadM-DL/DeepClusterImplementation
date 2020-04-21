from layers_stacker import *
import unittest
import torch

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
                      }

                       ]
class LayersStackerTests(unittest.TestCase):
    def test_parse_linear(self):
        linear_layer1 = parse_linear(in_features=100, cfg={"out_features":20, "activation":"ReLU"})
        linear_layer2 = parse_linear(in_features=100, cfg={"out_features":20, "activation":"Sigmoid"})
        assert linear_layer1[0].weight.shape == linear_layer2[0].weight.shape
        assert isinstance(linear_layer1[0], torch.nn.Linear)
        assert isinstance(linear_layer1[1], torch.nn.ReLU)
        assert isinstance(linear_layer2[1], torch.nn.Sigmoid)
        with self.assertRaises(Exception):
            linear_layer3 = parse_linear(in_features=100, cfg={"out_feature":20, "activation":"Sigmoid"})
        linear_layer4 = parse_linear(in_features=100, cfg={"out_features":20})
        assert isinstance(linear_layer4, torch.nn.Linear)
        

    def test_parse_drop_out(self):
        drp_layer = parse_drop_out(cfg={"drop_ratio":0.4})
        assert isinstance(drp_layer, torch.nn.Dropout)
        assert drp_layer.p == 0.4
        with self.assertRaises(Exception):
            drp_layer = parse_drop_out(cfg={"out_feature":20})
        

    def test_parse_max_pool(self):
        mxp_layer = parse_max_pool(100, cfg={"stride":2, "kernel_size":3})
        assert isinstance(mxp_layer, torch.nn.MaxPool2d)
        assert mxp_layer.stride == 2
        assert mxp_layer.kernel_size == 3
        with self.assertRaises(Exception):
            mxp_layer = parse_max_pool(100, cfg={"strides":2, "kernel_size":3})


    def test_parse_convolutional(self):
        conv_layer = parse_convolution(in_channels=3, cfg={"out_channels":96,
                                                            "stride": 2,
                                                            "padding":1,
                                                            "kernel_size":11,
                                                            "activation":"ReLU"})

        assert isinstance(conv_layer[0], torch.nn.Conv2d)
        assert conv_layer[0].out_channels == 96
        assert conv_layer[0].stride == (2,2)
        assert conv_layer[0].padding != 1
        assert conv_layer[0].kernel_size == (11, 11)
        assert isinstance(conv_layer[1], torch.nn.ReLU)
        with self.assertRaises(Exception):
            conv_layer = parse_convolution(in_channels=3, cfg={"out_channels":96,
                                                                "stride": 2,
                                                                "kernel_size":11,
                                                                "activation":"ReLU"})
    def test_name_layers(self):
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

    def test_stack_linear_layers(self):
        named_layers = stack_linear_layers(input_features=256*6*6, cfg=example_linear_cfg)
        # get number conv modules
        n_lin = len([ name for (name,module) in list(named_layers.named_modules()) if "linear" in name])
        assert n_lin == 2
        assert len(list(named_layers.named_children())) == len(example_linear_cfg) + n_lin - 1
        assert named_layers.drop_out_1
        assert named_layers.linear_2
        

    def test_stack_convolutional_layers(self):
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
         

if __name__ == '__main__':
    unittest.main()
