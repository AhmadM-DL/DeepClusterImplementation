from torchvision.datasets import SVHN

SVHN("./datasets", download=True, train=True)
SVHN("./datasets", download=True, train=False)