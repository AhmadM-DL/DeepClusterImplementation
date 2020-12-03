from torchvision.datasets import SVHN

SVHN("./datasets", download=True, split="train")
SVHN("./datasets", download=True, split="test")
