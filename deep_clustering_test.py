"""
Created on Tuesday April 14 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""

import unittest
from deep_clustering import deep_cluster
from deep_clustering_models import AlexNet_ImageNet
from deep_learning_unittest import RandomVisionDataset
from deep_clustering_dataset import DeepClusteringDataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch


class DeepClusteringTests(unittest.TestCase):

    def test_deep_cluster_1(self):
        device = torch.device("cpu")
        model = AlexNet_ImageNet(sobel=True, batch_normalization=True, device=device)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        tra = [transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
               normalize]

        dataset = RandomVisionDataset(
            (3, 244, 244), data_length=80, n_classes=5)
        dataset = DeepClusteringDataset(
            dataset, transform=transforms.Compose(tra))

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=0.01,
            weight_decay=10**-5,
        )
        deep_cluster(model=model, dataset=dataset, n_clusters=5, loss_fn=loss_fn,
                     optimizer=optimizer, n_cycles=2, random_state=0, verbose=1,
                     loading_batch_size=8, training_batch_size=8, n_components=20)
    
    def test_deep_cluster_with_writer(self):
        device = torch.device("cpu")
        model = AlexNet_ImageNet(sobel=True, batch_normalization=True, device=device)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        tra = [transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
               normalize]

        dataset = RandomVisionDataset(
            (3, 244, 244), data_length=80, n_classes=5)
        dataset = DeepClusteringDataset(
            dataset, transform=transforms.Compose(tra))

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=0.01,
            weight_decay=10**-5,
        )
        writer = SummaryWriter('runs/test_deep_cluster_with_writer')

        deep_cluster(model=model, dataset=dataset, n_clusters=5, loss_fn=loss_fn,
                     optimizer=optimizer, n_cycles=1, random_state=0, verbose=1,
                     loading_batch_size=8, training_batch_size=8, n_components=20, writer=writer)


if __name__ == "__main__":
    unittest.main()
