import sys
sys.path.append("../")

import unittest
import torch
from deep_learning_unittest import do_train_step, do_forward_step, test_param_change, RandomDataset
from deep_clustering_models import *


class DeepClusteringModelsTests(unittest.TestCase):

    def test_alexnet_cifar_static(self):
        model = AlexNet_CIFAR(sobel=True, batch_normalization=True, device=torch.device("cpu"))
        return

    def test_alexnet_imagenet_static(self):

        model = AlexNet_ImageNet(
            sobel=True, batch_normalization=True, device=torch.device("cpu"))

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

        device = torch.device("cpu")
        model = AlexNet_ImageNet(
            sobel=True, batch_normalization=True, device=device)
        output_size = 1000
        batch_size = 10
        model.add_top_layer(output_size=output_size)

        dummy_batch = (torch.rand(batch_size, 3, 244, 244), torch.empty(
            batch_size, dtype=torch.long).random_(output_size))

        # Run a training cycle
        do_train_step(model,
                      loss_fn=torch.nn.CrossEntropyLoss(),
                      optim=torch.optim.SGD(
                          filter(lambda x: x.requires_grad, model.parameters()), lr=0.01),
                      batch=dummy_batch,
                      device=device)

        # Run a fead forward step
        do_forward_step(model, batch=dummy_batch, device=device)

        # Run a training cycle and assure that paramters are being updated
        test_param_change(vars_change=True, model=model,
                          loss_fn=torch.nn.CrossEntropyLoss(),
                          optim=torch.optim.SGD(
                              filter(lambda x: x.requires_grad, model.parameters()), lr=0.01),
                          batch=dummy_batch,
                          device=device)

    def test_full_feed_forward(self):

        device = torch.device("cpu")

        model = AlexNet_ImageNet(
            sobel=True, batch_normalization=True, device=device)
        output_size = 1000
        batch_size = 10
        model.add_top_layer(output_size=output_size)

        dataset = RandomDataset((3, 244, 244), 100, output_size)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size)

        model.full_feed_forward(dataloader=dataloader)

    def test_deep_cluster_train(self):
        device = torch.device("cpu")
        model = AlexNet_ImageNet(
            sobel=True, batch_normalization=True, device=device)

        optimizer = torch.optim.SGD(filter(
            lambda x: x.requires_grad, model.parameters()), lr=0.01, weight_decay=0.00001)
        
        loss_fn = torch.nn.CrossEntropyLoss()

        output_size = 1000
        batch_size = 10
        model.add_top_layer(output_size=output_size)

        dataset = RandomDataset((3, 244, 244), 100, output_size)
        dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size)

        model.deep_cluster_train(dataloader=dataloader, epoch=0, optimizer=optimizer, loss_fn=loss_fn)

if __name__ == "__main__":
    unittest.main()
