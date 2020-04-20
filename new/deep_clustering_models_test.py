import unittest
from deep_learning_unittest import *
from deep_clustering_models import *


class DeepClusteringModelsTests(unittest.TestCase):
    def test_alexnet_imagenet_static(self):
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

    def test_alexnet_imagenet_dynamic(self):
        top_layer_output_size = 1000 
        model = AlexNet_ImageNet(sobel=True, batch_normalization=True)
        model.top_layer = torch.nn.Linear(model.output_size((1,2,244,244))[0], top_layer_output_size)
        do_train_step(model, 
                    loss_fn= torch.nn.CrossEntropyLoss(),
                    optim= torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),lr=0.01),
                    batch= (torch.rand(10,3,244,244), torch.rand(10,top_layer_output_size)),
                    device= torch.device("cpu"))

if __name__ == "__main__":
    unittest.main()
