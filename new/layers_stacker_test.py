from layers_stacker import *
import unittest
import torch

class layer_generator_tests(unittest.TestCase):
    def test_parse_linear(self):
        linear_layer1 = parse_linear(in_features=100, cfg={"out_features":20, "activation":"ReLU"})
        linear_layer2 = parse_linear(in_features=100, cfg={"out_features":20, "activation":"Sigmoid"})
        assert linear_layer1[0].weight.shape == linear_layer2[0].weight.shape
        assert isinstance(linear_layer1[0], torch.nn.Linear)
        assert isinstance(linear_layer1[1], torch.nn.ReLU)
        assert isinstance(linear_layer2[1], torch.nn.Sigmoid)
        with self.assertRaises(Exception):
            linear_layer3 = parse_linear(in_features=100, cfg={"out_feature":20, "activation":"Sigmoid"})
        

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
if __name__ == '__main__':
    unittest.main()
