from new.custom_layers import *
import unittest
import torch
#import matplotlib.pyplot as plt
#import numpy as np

class CustomLayersTests(unittest.TestCase):
    def test_sobel_filter(self):
        sobel_filter = SobelFilter()
        img = torch.rand((1,3,100,100))
        #plt.imshow(np.moveaxis(img.numpy(), 1, 3)[0])

        img_grayscale = sobel_filter.grayscale(img)
        assert img_grayscale.size()[1] == 1
        #plt.imshow((np.moveaxis(img_grayscale.numpy(), 1, 3).squeeze()), cmap="gray")

        img_sobel = sobel_filter.sobel_filter(img_grayscale)
        assert img_sobel.size()[1] == 2
        #img_edge = img_sobel.numpy().squeeze()[0] + img_sobel.numpy().squeeze()[1] 
        #plt.imshow(img_edge, cmap="gray")

        img_sobel = sobel_filter(img)
        assert img_sobel.size()[1] == 2
        #plt.imshow(np.moveaxis(img_sobel.numpy(), 1, 3)[0])

if __name__ == '__main__':
    unittest.main()

