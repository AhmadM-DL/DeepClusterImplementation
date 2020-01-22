# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:44:21 2019

@author: amm90
"""
import torch.nn as nn
import torch
import math
import time
import utils
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import os


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


class NetBuilder(nn.Module):
    def __init__(self, features, classifier, top_layer, features_output, classifier_output, top_layer_output, apply_sobel):
        super(NetBuilder, self).__init__()
        self.sobel = None
        self.features = features
        self.classifier = classifier
        self.top_layer = top_layer
        self._initialize_weights()
        self.features_output = features_output
        self.classifier_output = classifier_output
        self.top_layer_output = top_layer_output

        if apply_sobel:
            self.sobel = make_sobel_layer()

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):

            if isinstance(m, nn.Conv2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_model_output_size(self):
        if self.top_layer:
            return self.top_layer_output
        if self.classifier:
            return self.classifier_output
        if self.features:
            return self.features_output
        raise Exception("The model doesn't have actual modules")


def make_convolutional_layers(cfg, input_n_channels, bn):
    # cfg:
    # (number of filters, kernel size, stride, pad)
    # ('M', kernel size, stride)

    layers = []
    in_channels = input_n_channels
    for v in cfg:
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1],
                               stride=v[2], padding=v[3])
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


def make_sobel_layer():
    grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
    grayscale.weight.data.fill_(1.0 / 3.0)
    grayscale.bias.data.zero_()

    sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)

    sobel_filter.weight.data[0, 0].copy_(
        torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    )
    sobel_filter.weight.data[1, 0].copy_(
        torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    )

    sobel_filter.bias.data.zero_()

    sobel = nn.Sequential(grayscale, sobel_filter)

    for p in sobel.parameters():
        p.requires_grad = False

    return sobel


def make_linear_layers(cfg):
    layers = []
    for v in cfg:
        if v[0] == 'L':
            layers += [nn.Linear(v[1], v[2])]
        if v[0] == 'R':
            layers += [nn.ReLU(inplace=v[1])]
            continue
        if v[0] == 'D':
            layers += [nn.Dropout(v[1])]
            continue
    return nn.Sequential(*layers)


def alexnet_cifar(sobel=False, bn=True, out=10):
    input_n_channels = 2 + int(not sobel)

    features_cfg = [(64, 3, 1, 2), ('M', 2, None), (192, 3, 1, 2),
                    ('M', 2, None), (384, 3, 1, 1), (256, 3, 1, 1),
                    (256, 3, 1, 1), ('M', 3, 2)]

    classifier_cfg = [('D', 0.5), ('L', 4096, 2048), ('R', True), ('D', 0.5),
                      ('L', 2048, 2048), ('R', True)]

    top_layer_cfg = [('L', 2048, out)]

    features = make_convolutional_layers(features_cfg, input_n_channels, bn=bn)
    classifier = make_linear_layers(classifier_cfg)
    top_layer = make_linear_layers(top_layer_cfg)[0]

    model = NetBuilder(features, classifier, top_layer,
                       features_output=4096, classifier_output=2048, top_layer_output=out,
                       apply_sobel=sobel)

    return model


def lenet_5(bn=False, out=10):
    features_cfg = [(6, 5, 1, 0), ('M', 2, None), (16, 5, 1, 0), ('M', 2, None)]

    classifier_cfg = [('L', 400, 120), ('R', True),
                      ('L', 120, 84), ('R', True)]

    top_layer_cfg = [('L', 84, out)]

    features = make_convolutional_layers(features_cfg, 1, bn=bn)
    classifier = make_linear_layers(classifier_cfg)
    top_layer = make_linear_layers(top_layer_cfg)[0]

    model = NetBuilder(features, classifier, top_layer,
                       features_output=400, classifier_output=84, top_layer_output=out,
                       apply_sobel=False)

    return model


def remove_top_layer(model):
    """
    :param model: The model than need its top layer to be removed
    :return: The model output size after removing top layer
    """
    new_output_size = model.top_layer.weight.size()[1]
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    return new_output_size


def add_top_layer(model, top_layer_cfg, device, weight_mean=0, weight_std=0.01):
    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=True).to(device))
    model.classifier = nn.Sequential(*mlp)

    model.top_layer = make_linear_layers(top_layer_cfg)[0]
    model.top_layer.weight.data.normal_(weight_mean, weight_std)
    model.top_layer.bias.data.zero_()

    model.top_layer.to(device)


def compute_network_output(dataloader, model, device, batch_size, data_size, verbose=False,
                           return_inputs=False, return_targets=False):
    if verbose:
        print('Computing Model Output')

    batch_time = utils.AverageMeter()
    end = time.time()

    model.eval()

    inputs = []
    targets = []

    for i, (input_tensor, target) in enumerate(dataloader):

        input = input_tensor.to(device)
        output = model(input).data.cpu().numpy()

        if i == 0:
            outputs = np.zeros((data_size, output.shape[1])).astype('float32')

        if i < len(dataloader) - 1:
            outputs[i * batch_size: (i + 1) * batch_size] = output.astype('float32')

        else:
            # special treatment for final batch
            outputs[i * batch_size:] = output.astype('float32')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        inputs.extend(input_tensor.data.numpy())
        targets.extend(target.data.numpy())

        if verbose and (i % 10) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))

    inputs = np.squeeze(np.array(inputs))
    targets = np.array(targets)

    if return_inputs and return_targets:
        return inputs, outputs, targets

    if not return_inputs and not return_targets:
        return outputs

    if return_inputs:
        return inputs, outputs

    if return_targets:
        return outputs, targets


def normal_train(model, dataloader, loss_criterion, optimizer, epoch, device, verbose=0):
    # switch to train mode
    model.train()

    batch_time = utils.AverageMeter()
    loss_meter = utils.AverageMeter()
    data_time = utils.AverageMeter()

    end = time.time()
    if verbose:
        print("Training")
    for i, (input_tensor, target_tensor) in enumerate(dataloader):

        data_time.update(time.time() - end)

        target = target_tensor.to(device)
        output = model(input_tensor.to(device))

        loss = loss_criterion(output, target)

        if loss.dim() == 0:
            print("Error This function expects a loss criterion that doesn't apply reduction\n")

        loss = loss.mean()

        loss_meter.update(loss.item(), input_tensor.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and (i % 10) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(dataloader), batch_time=batch_time,
                          data_time=data_time, loss=loss_meter))

    return loss_meter.avg


def normal_test(model, epoch, dataloader, device, loss_criterion=None, return_loss=False, verbose=0):

    correct_predictions = 0
    total_predictions = 0
    model.eval()
    losses = []

    if return_loss and not loss_criterion:
        raise Exception("Error please add a loss criterion")

    if verbose:
        print("Testing")
    for i, (input_tensor, target) in enumerate(dataloader):

        target = target.to(device)
        input_tensor = input_tensor.to(device)

        output = model(input_tensor)

        loss = loss_criterion(output, target)
        if loss.dim() == 0:
            raise Exception("Error this function expects a loss criterion that doesn't apply reduction\n")
        loss = loss.mean()
        losses.append(loss.item())

        predicted = torch.argmax(output, 1)

        total_predictions += target.size(0)

        correct_predictions += (predicted == target).sum().item()

        if verbose and (i % 10) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {3} (4})'
                  .format(epoch, i, len(dataloader), loss, np.mean(losses)))

    test_acc = (100 * correct_predictions / total_predictions)

    if not return_loss:
        return test_acc
    else:
        return test_acc, np.mean(losses)


def deep_cluster_train(dataloader, model, loss_criterion, net_optimizer, annxed_layers_optimizer, epoch, device,
                       return_inputs_outputs_targets_losses=False, verbose=True, checkpoint=0):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    data_time = utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    dataset_inputs = []
    dataset_outputs = []
    dataset_targets = []
    dataset_losses = []

    for i, (input_tensor, target_tensor) in enumerate(dataloader):

        data_time.update(time.time() - end)

        target = target_tensor.to(device)
        input = input_tensor.to(device)

        output = model(input)

        loss = loss_criterion(output, target)
        if (loss.dim() == 0):
            print("Error This function excpects a loss criterion that doesn't apply reduction\n")

        dataset_inputs.extend(input_tensor.data.numpy())
        dataset_losses.extend(loss.data.cpu().numpy())
        dataset_targets.extend(target_tensor.data.numpy())
        dataset_outputs.extend(output.cpu().data.numpy())

        loss = loss.mean()

        # record loss
        losses.update(loss.item(), input_tensor.size(0))

        net_optimizer.zero_grad()
        annxed_layers_optimizer.zero_grad()
        loss.backward()
        net_optimizer.step()
        annxed_layers_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and (i % 10) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(dataloader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    if (return_inputs_outputs_targets_losses):
        return losses.avg, dataset_inputs, dataset_outputs, dataset_targets, dataset_losses
    else:
        return losses.avg


def deep_cluster_test(dataloader, model, device):
    correct_predictions = 0
    total_predictions = 0
    model.eval()

    for input_tensor, target in dataloader:
        target = target.to(device)
        input_tensor = input_tensor.to(device)

        output = model(input_tensor)

        predicted = torch.argmax(output, 1)

        total_predictions += target.size(0)

        correct_predictions += (predicted == target).sum().item()

    test_acc = (100 * correct_predictions / total_predictions)

    return test_acc


def save_checkpoint(model, optimizer, epoch, path, architecture="unspecified", verbose=True):
    if verbose:
        print('Save checkpoint at: {0}'.format(path))

    torch.save({
        'epoch': epoch,
        'arch': architecture,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)


def load_from_checkpoint(model, optimizer, path):
    if os.path.isfile(path):
        print("    => loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        # remove top_layer parameters from checkpoint
        for key in checkpoint['state_dict'].copy():
            if 'top_layer' in key:
                del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("    => loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))
        return epoch
    else:
        print("    => no checkpoint found at '{}'".format(path))


def save_model_parameter(model, path, override=False):
    if not os.path.isfile(path):
        # The file don't exist; save
        torch.save(model.state_dict(), path)
        print("Model saved in directory: %s" % path)
        return
    if os.path.isfile(path):
        if override:
            torch.save(model.state_dict(), path)
            print("Model saved in directory: %s" % path)
            return
        else:
            print("Error the file already exists, rerun with parameter override=True to override.")
            return


def load_model_parameter(model, path):
    if not os.path.isfile(path):
        # The file dosen't exist
        print("The provided path %s doesn't exist" % path)
    else:
        model.load_state_dict(torch.load(path))
        print("Loaded model parameters from : %s" % path)
