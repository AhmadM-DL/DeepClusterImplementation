from torchvision.datasets import CIFAR100

CIFAR100("./datasets", download=True, split="train")
CIFAR100("./datasets", download=True, split="test")
