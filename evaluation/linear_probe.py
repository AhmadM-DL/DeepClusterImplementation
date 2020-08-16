"""
Created on 3-7-2020

"""

from deep_clustering_net import DeepClusteringNet
from deep_clustering_dataset import DeepClusteringDataset

import torch.nn as nn
import torch
from utils import set_seed
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# import sys
# sys.path.append("../")


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

class LogisticRegression(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, features_size, num_labels, avg_pool):
        """ The init function of the logistic Regression Module

        Args:
            num_labels (int): Number of labels that the model is supposed to be trained on
            downsample (tuple): A tuple contianing the downsample args [kernel_size, stride, padding, downsample output size]
        """
        super(LogisticRegression, self).__init__()
        if avg_pool:
            self.avg_pool = nn.AvgPool2d(
                kernel_size=avg_pool["kernel_size"], stride=avg_pool["stride"], padding=avg_pool["padding"])
        else:
            self.avg_pool = None
        self.linear = nn.Linear(features_size, num_labels)

    def forward(self, x):
        if self.avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)

def train(reglog, model:DeepClusteringNet, target_layer, epoch, trainloader, optimizer, loss_fn, device, verbose=True, lr_decay=False):

    model.eval()

    losses = []
    accuracies_1 = []
    accuracies_5 = []

    for i, (input_, target) in enumerate(trainloader):

        # adjust learning rate
        if lr_decay:
            learning_rate_decay(optimizer, len(trainloader)
                            * epoch + i, optimizer.defaults["lr"])

        input_ = input_.to(device)
        target = target.to(device)

        output = model.extract_features(input_, target_layer, flatten=False)
        output = reglog(output)

        # compute output
        loss = loss_fn(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.append(loss.item())
        accuracies_1.append(acc1.item())
        accuracies_5.append(acc5.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss:.4f} \t'
                    'ACC1 {acc1:.3f}\t'
                    'ACC5 {acc5:.3f}'
                    .format(epoch, i, len(trainloader), loss=loss.item(), acc1=acc1.item(), acc5=acc5.item()))

    return np.average(losses), np.average(accuracies_1), np.average(accuracies_5)

def validate(reglog, model:DeepClusteringNet, target_layer,validloader, loss_fn, device, verbose=True, tencrops=False):

    # switch to evaluate mode
    model.eval()

    losses = []
    accuracies_1 = []
    accuracies_5 = []

    softmax = nn.Softmax(dim=1).to(device)

    for i, (input_, target) in enumerate(validloader):

        if tencrops:
            batch_size, n_crops, channel, hight, width = input_.size()
            input_ = input_.view(-1, channel, hight, width)

        target = target.to(device)
        input_ = input_.to(device)

        output = reglog( model.extract_features(input_, target_layer, flatten=False) )

        if tencrops:
            output_central = output.view(
                batch_size, n_crops, -1)[:, n_crops / 2 - 1, :]
            output = softmax(output)
            output = torch.squeeze(output.view(
                batch_size, n_crops, -1).mean(1))
        else:
            output_central = output

        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        loss = loss_fn(output_central, target)

        losses.append(loss.item())
        accuracies_1.append(acc1[0])
        accuracies_5.append(acc5[0])

        if verbose and i % 100 == 0:
            print('Validation: [{0}/{1}]\t'
                    'Loss {loss:.4f} \t'
                    'ACC1 {acc1:.3f}\t'
                    'ACC5 {acc5:.3f}'
                    .format(i, len(validloader), loss=loss.item(), acc1=acc1.item(), acc5=acc5.item()))
    
    return np.average(losses), np.average(accuracies_1), np.average(accuracies_5)

def eval_linear(model: DeepClusteringNet, n_epochs, traindataset, validdataset,
                target_layer, n_labels, features_size, avg_pool=None,
                random_state=0, writer: SummaryWriter=None, verbose=True, **kwargs):

    # set random seed
    set_seed(random_state)

    # feeze model wights
    model.freeze_features()
    model.freeze_classifier()

    # define loaders
    traindataloader = DataLoader(
        dataset=traindataset, batch_size=kwargs.get("batch_size", 256), shuffle= kwargs.get("shuffle_train", True))
    
    if validdataset:
        validdataloader = DataLoader(
        dataset=validdataset, batch_size=int(kwargs.get("batch_size", 256)/2), shuffle= kwargs.get("shuffle_valid", False))

    # define loss_fn
    loss_fn = nn.CrossEntropyLoss().to(model.device)

    # define logistic regression on top of target layer
    reglog = LogisticRegression(features_size, n_labels, avg_pool)
    reglog.to(model.device)
    
    # define optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, reglog.parameters()),
        lr = kwargs.get("learning_rate", 0.01),
        momentum= kwargs.get("momentum", 0.9),
        weight_decay= kwargs.get("wight_decay", 10**(-4))
    )

    for epoch in range(0, n_epochs):
        t_loss, t_acc1, t_acc2 = train(reglog, model, target_layer, epoch, traindataloader, optimizer, loss_fn, model.device, verbose=verbose)
        if writer:
            writer.add_scalar("linear_probe_train/%s/loss"%target_layer, t_loss, global_step=epoch)
            writer.add_scalar("linear_probe_train/%s/acc1"%target_layer, t_acc1, global_step=epoch)
            writer.add_scalar("linear_probe_train/%s/acc2"%target_layer, t_acc2, global_step=epoch)

        if validdataset:
            v_loss, v_acc1, v_acc2 = validate(reglog, model, target_layer, validdataloader , loss_fn, model.device, verbose=verbose)
            if writer:
                writer.add_scalar("linear_probe_valid/%s/loss"%target_layer, v_loss, global_step=epoch)
                writer.add_scalar("linear_probe_valid/%s/acc1"%target_layer, v_acc1, global_step=epoch)
                writer.add_scalar("linear_probe_valid/%s/acc2"%target_layer, v_acc2, global_step=epoch)

    # unfreeze model wights
    model.unfreeze_classifier()
    model.unfreeze_features()

    return
