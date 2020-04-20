import unittest
from deep_learning_unittest import *
from deep_clustering_models import *


class DeepClusteringModelsTests(unittest.TestCase):
    def test_alexnet_imagenet(self):
        model = AlexNet_ImageNet(sobel=True, batch_normalization=True)

        assert model.top_layer == None
        assert model.sobel

        expected_features_layers = ['conv_1', 'batch_norm_1', 'relu_1',
                                    'max_pool_1', 'conv_2', 'batch_norm_2',
                                    'relu_2', 'max_pool_2', 'conv_3',
                                    'batch_norm_3', 'relu_3', 'conv_4',
                                    'batch_norm_4', 'relu_4', 'conv_5',
                                    'batch_norm_5', 'relu_5', 'max_pool_3']

        features_layers = [child[0]
                           for child in model.features.named_children()]
        assert expected_features_layers == features_layers

        expected_classifier_layers = [
            'drop_out_1', 'linear_1', 'relu_1', 'drop_out_2', 'linear_2']
        classifier_layers = [child[0]
                             for child in model.classifier.named_children()]
        assert expected_classifier_layers == classifier_layers

        _train_step(model,
                    loss_fn= torch.nn)

if __name__ == "__main__":
    unittest.main()
