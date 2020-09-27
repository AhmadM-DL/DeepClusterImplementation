from torchvision.datasets import FashionMNIST

FashionMNIST("./datasets", download=True, train=True)
FashionMNIST("./datasets", download=True, train=False)