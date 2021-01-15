from torchvision.datasets import CIFAR10

CIFAR10("./datasets", download=True, train= True)
CIFAR10("./datasets", download=True, train= False)
