import sys
sys.path.append("../")

# import torch
# from deep_clustering_models import AlexNet_ImageNet
# from torchvision import transforms
# from deep_learning_unittest import RandomVisionDataset
# from deep_clustering_dataset import DeepClusteringDataset

#from linear_probe import LinearProbe
import unittest

class LinearProbeTests(unittest.TestCase):

    def train_linear_probe_test(self):
        # device = torch.device("cpu")
        # model = AlexNet_ImageNet(sobel=True, batch_normalization=True, device=device)

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # tra = [transforms.Resize(256),
        #        transforms.CenterCrop(224),
        #        transforms.ToTensor(),
        #        normalize]

        # dataset = RandomVisionDataset(
        #     (3, 244, 244), data_length=80, n_classes=5)
        # dataset = DeepClusteringDataset(
        #     dataset, transform=transforms.Compose(tra))

        # loss_fn = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(
        #     filter(lambda x: x.requires_grad, model.parameters()),
        #     lr=0.01,
        #     weight_decay=10**-5,
        # )

        print("test")
        return


if __name__ == '__main__':
    unittest.main()