from torchvision.datasets import CIFAR10

CIFAR10("./datasets", download=True, split="train")
CIFAR10("./datasets", download=True, split="test")
