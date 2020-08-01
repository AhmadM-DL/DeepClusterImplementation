from torchvision.datasets import MNIST

MNIST("./datasets", download=True, train=True)
MNIST("./datasets", download=True, train=False)