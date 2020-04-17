from new.models.DeepClusteringModels import *
import unittest

class DeepClusteringModelsTests(unittest.TestCase):

    def test_alex_net(self):
        model = AlexNet(sobel=True, batch_normalization=True)


if __name__ == "__main__":
    unittest.main()


