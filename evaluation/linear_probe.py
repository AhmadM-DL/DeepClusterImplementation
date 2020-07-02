"""

"""

import torch.nn as nn
import torch
from utils import set_seed
import numpy as np

import sys
sys.path.append("../")
from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering_net import DeepClusteringNet

def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class LinearProbe(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, model:DeepClusteringNet, num_labels, downsample, target_layer):
        """ The init function of the logistic Regression Module

        Args:
            num_labels (int): Number of labels that the model is supposed to be trained on
            downsample (tuple): A tuple contianing the downsample args [kernel_size, stride, padding, downsample output size]
        """
        super(LinearProbe, self).__init__()
        self.model= model
        self.avg_pool = nn.AvgPool2d(kernel_size= downsample[0], stride= downsample[1], padding=downsample[2])
        self.linear = nn.Linear(downsample[3], num_labels)
        self.device = model.device
        self.target_layer = target_layer

    def forward(self, x):
        x = self.model.extract_features(x, self.target_layer, flatten=False)
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x) 

    def train_(self, epoch, trainloader, optimizer, loss_fn, verbose=True):

        self.train()

        for i, (input_, target) in enumerate(trainloader):
            
            # adjust learning rate
            learning_rate_decay(optimizer, len(trainloader) * epoch + i, optimizer.lr)

            input_ = input_.to(self.device)

            target = target.to(self.device)
            output = self.model(input_)
            output = self(output).data.cpu().numpy()
            
            # compute output
            loss = loss_fn(output, target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss:.4f} \t'
                    'Prec@1 {acc1:.3f}\t'
                    'Prec@5 {acc5:.3f}'
                    .format(epoch, i, len(trainloader), loss=loss, acc1=acc1, acc5=acc5))
        
        return loss
    
    def validate(self, epoch, validloader, optimizer, loss_fn, verbose=True, tencrops=False):
        
        # switch to evaluate mode
        self.model.eval()

        softmax = nn.Softmax(dim=1).to(self.device)

        for i, (input_, target) in enumerate(validloader):

            if tencrops:
                batch_size, n_crops, channel, hight, width = input_.size()
                input_ = input_.view(-1, channel, hight, width)

            target = target.to(self.device)
            input_ = input_.to(self.device)

            output = self(input_)

            if tencrops:
                output_central = output.view(batch_size, n_crops, -1)[:, n_crops / 2 - 1, :]
                output = softmax(output)
                output = torch.squeeze(output.view(batch_size, n_crops, -1).mean(1))
            else:
                output_central = output

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            loss = loss_fn(output_central, target)

            if verbose and i % 100 == 0:
                print('Validation: [{0}/{1}]\t'
                    'Loss {loss:.4f} \t'
                    'Prec@1 {acc1:.3f}\t'
                    'Prec@5 {acc5.val:.3f}'
                    .format(i, len(validloader), loss=loss, acc1=acc1, acc5=acc5))

        return loss
        

def eval_linear(model: DeepClusteringNet, conv_layer, **kwargs):

    # set seed
    set_seed(kwargs.get("random_state",0))

    # Freeze features
    model.freeze_features()

    # 

