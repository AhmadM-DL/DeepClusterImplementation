import unittest
from new.deep_clustering_models import AlexNet

class DeepClusteringModelsTests(unittest.TestCase):
        def test_alexnet(self):
                model = AlexNet(sobel=True, batch_normalization=True)
                assert True

if __name__ == "__main__":
    unittest.main()
