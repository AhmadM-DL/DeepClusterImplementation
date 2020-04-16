# -*- coding: utf-8 -*-
import torch

class SobelFilter(torch.nn.Module):
    def __init__(self):
        """
        In this constructor we initialize the Sobel Filter Layer which is composed 
        from two Conv2d layers. The first transforms RGB input to grayscale.
        The second apply sobel filter on grayscale input.
        """
        super(SobelFilter, self).__init__()

        self.grayscale = torch.nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        self.grayscale.weight.data.fill_(1.0 / 3.0)
        self.grayscale.bias.data.zero_()

        self.sobel_filter = torch.nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        self.sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        )
        self.sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        )
        self.sobel_filter.bias.data.zero_()

        for p in self.grayscale.parameters():
            p.requires_grad = False

        for p in self.sobel_filter.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.grayscale(x)
        x = self.sobel_filter(x)
        return x


