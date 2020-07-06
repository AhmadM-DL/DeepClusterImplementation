# -*- coding: utf-8 -*-
"""
Created on Tuesday April 14 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)

This module implements a number of standard benchmarks architectures, however by utilizing
DeepClusteringNet class.
"""
import torch.nn as nn
import torch

import math
import time
import os

import numpy as np

from custom_layers import SobelFilter
from deep_clustering_net import DeepClusteringNet
from sklearn.metrics import normalized_mutual_info_score

from  layers_stacker import stack_convolutional_layers, stack_linear_layers

def AlexNet_CIFAR(sobel, batch_normalization, device):
    """Implementation of AlexNet for CIFAR dataset

    Arguments:
        sobel {Boolean} -- Add sobel filter prior to input or not 
        batch_normalization {Boolean} -- Add normalization after convolution or not
        device {torch.device} -- Pytorch device to send the model to (cpu/gpu)
    """
    n_input_channels = 2 + int(not sobel)
    
    alexnet_features_cfg = [
                {
                "type": "convolution",
                "out_channels":64,
                "kernel_size":11,
                "stride":4,
                "padding":2,
                "activation":"ReLU",
                },

                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                },

                {
                "type": "convolution",
                "out_channels":192,
                "kernel_size":5,
                "stride":1,
                "padding":2,
                "activation":"ReLU",
                },

                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                },

                {
                "type": "convolution",
                "out_channels":384,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },

                {
                "type": "convolution",
                "out_channels":256,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },

                {
                "type": "convolution",
                "out_channels":256,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },
                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                }         
                ]

    classifier_cfg = [
                      {"type":"drop_out",
                       "drop_ratio": 0.5},

                      {"type":"linear",
                       "out_features":2048,
                       "activation":"ReLU"},

                      {"type":"drop_out",
                       "drop_ratio": 0.5},

                      {"type":"linear",
                      "out_features":2048}
        ]

    model = DeepClusteringNet(input_size=(n_input_channels,224,224),
                              features= stack_convolutional_layers(input_channels= n_input_channels, cfg=alexnet_features_cfg, batch_normalization=batch_normalization),
                              classifier= stack_linear_layers(input_features= 256 * 6 * 6, cfg= classifier_cfg),
                              top_layer = None,
                              with_sobel=sobel,
                              device=device)
    return model

def AlexNet_ImageNet(sobel, batch_normalization, device):
    """Implementation of AlexNet for Imagenet dataset

    Arguments:
        sobel {Boolean} -- Add sobel filter prior to input or not 
        batch_normalization {Boolean} -- Add normalization after convolution or not
        device {torch.device} -- Pytorch device to send the model to (cpu/gpu)
    """
    n_input_channels = 2 + int(not sobel)
    
    alexnet_features_cfg = [
                {
                "type": "convolution",
                "out_channels":96,
                "kernel_size":11,
                "stride":4,
                "padding":2,
                "activation":"ReLU",
                },

                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                },

                {
                "type": "convolution",
                "out_channels":256,
                "kernel_size":5,
                "stride":1,
                "padding":2,
                "activation":"ReLU",
                },

                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                },

                {
                "type": "convolution",
                "out_channels":384,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },

                {
                "type": "convolution",
                "out_channels":384,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },

                {
                "type": "convolution",
                "out_channels":256,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },

                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                }         
                ]

    classifier_cfg = [
                      {"type":"drop_out",
                       "drop_ratio": 0.5},

                      {"type":"linear",
                       "out_features":4096,
                       "activation":"ReLU"},

                      {"type":"drop_out",
                       "drop_ratio": 0.5},

                      {"type":"linear",
                      "out_features":4096}
        ]

    model = DeepClusteringNet(input_size=(n_input_channels,224,224),
                              features= stack_convolutional_layers(input_channels= n_input_channels, cfg=alexnet_features_cfg, batch_normalization=batch_normalization),
                              classifier= stack_linear_layers(input_features= 256 * 6 * 6, cfg= classifier_cfg),
                              top_layer = None,
                              with_sobel=sobel,
                              device=device)
    return model

def LeNet(batch_normalization, device):
    """
    Implementation of LeNet
    """
    lenet_features_cfg = [{"type":"convolution",
                            "out_channels": 6,
                            "kernel_size": 5,
                            "padding": 0,
                            "stride":1,
                            "activation": "ReLU",
                            },
                            {"type":"max_pool",
                            "kernel_size": 2,
                            "stride": 2,
                            },
                            {"type":"convolution",
                            "out_channels": 16,
                            "kernel_size": 5,
                            "padding": 0,
                            "stride":1,
                            "activation": "ReLU",
                            },
                            {"type":"max_pool",
                            "kernel_size": 2,
                            "stride": 2,
                            }]

    classifier_cfg = [{"type":"linear", "out_features": 120, "activation":"ReLU"},
                      {"type":"linear", "out_features": 84 }]

    model = DeepClusteringNet(
                       input_size=(1,32,32),
                       features= stack_convolutional_layers(input_channels=1, cfg=lenet_features_cfg, batch_normalization=batch_normalization),
                       classifier= stack_linear_layers(input_features=16*5*5, cfg= classifier_cfg),
                       top_layer= None,
                       with_sobel=False,
                       device=device)

    return model


# def remove_top_layer(model):
#     """
#     :param model: The model than need its top layer to be removed
#     :return: The model output size after removing top layer
#     """
#     new_output_size = model.top_layer.weight.size()[1]
#     model.top_layer = None
#     model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
#     return new_output_size


# def add_top_layer(model, top_layer_cfg, device, weight_mean=0, weight_std=0.01):
#     mlp = list(model.classifier.children())
#     mlp.append(nn.ReLU(inplace=True).to(device))
#     model.classifier = nn.Sequential(*mlp)

#     model.top_layer = make_linear_layers(top_layer_cfg)[0]
#     model.top_layer.weight.data.normal_(weight_mean, weight_std)
#     model.top_layer.bias.data.zero_()

#     model.top_layer.to(device)


# def compute_network_output(dataloader, model, device, batch_size, data_size, verbose=False,
#                            return_inputs=False, return_targets=False):
#     if verbose:
#         print('Computing Model Output')

#     batch_time = utils.AverageMeter()
#     end = time.time()

#     model.eval()

#     inputs = []
#     targets = []

#     for i, (input_tensor, target) in enumerate(dataloader):

#         input_tensor = input_tensor.to(device)
#         output = model(input_tensor).data.cpu().numpy()

#         if i == 0:
#             outputs = np.zeros((data_size, output.shape[1])).astype('float32')

#         if i < len(dataloader) - 1:
#             outputs[i * batch_size: (i + 1) * batch_size] = output.astype('float32')

#         else:
#             # special treatment for final batch
#             outputs[i * batch_size:] = output.astype('float32')

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         inputs.extend(input_tensor.data.cpu().numpy())
#         targets.extend(target.data.numpy())

#         if verbose and (i % 10) == 0:
#             print('{0} / {1}\t'
#                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
#                   .format(i, len(dataloader), batch_time=batch_time))

#     inputs = np.squeeze(np.array(inputs))
#     targets = np.array(targets)

#     if return_inputs and return_targets:
#         return inputs, outputs, targets

#     if not return_inputs and not return_targets:
#         return outputs

#     if return_inputs:
#         return inputs, outputs

#     if return_targets:
#         return outputs, targets


# def normal_train(model, dataloader, loss_criterion, optimizer, epoch, device, verbose=0):
#     # switch to train mode
#     model.train()

#     batch_time = utils.AverageMeter()
#     loss_meter = utils.AverageMeter()
#     data_time = utils.AverageMeter()

#     end = time.time()
#     if verbose:
#         print("Training")
#     for i, (input_tensor, target_tensor) in enumerate(dataloader):

#         data_time.update(time.time() - end)

#         target = target_tensor.to(device)
#         output = model(input_tensor.to(device))

#         loss = loss_criterion(output, target)

#         if loss.dim() == 0:
#             print("Error This function expects a loss criterion that doesn't apply reduction\n")

#         loss = loss.mean()

#         loss_meter.update(loss.item(), input_tensor.size(0))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if verbose and (i % 10) == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss: {loss.val:.4f} ({loss.avg:.4f})'
#                   .format(epoch, i, len(dataloader), batch_time=batch_time,
#                           data_time=data_time, loss=loss_meter))

#     return loss_meter.avg


# def normal_test(model, epoch, dataloader, device, loss_criterion, verbose=0):
#     correct_predictions = 0
#     total_predictions = 0
#     model.eval()
#     losses = []

#     if verbose:
#         print("Testing")

#     for i, (input_tensor, target) in enumerate(dataloader):

#         target = target.to(device)
#         input_tensor = input_tensor.to(device)

#         output = model(input_tensor)

#         loss = loss_criterion(output, target)
#         if loss.dim() == 0:
#             raise Exception("Error this function expects a loss criterion that doesn't apply reduction\n")
#         loss = loss.mean()
#         losses.append(loss.item())

#         predicted = torch.argmax(output, 1)

#         total_predictions += target.size(0)

#         correct_predictions += (predicted == target).sum().item()

#         if verbose and (i % 10) == 0:
#             print('Epoch: [%d][%d/%d]\t'
#                   'Loss: %0.4f (%0.4f)'
#                   % (epoch, i, len(dataloader), loss, np.mean(losses)))

#     test_acc = (100 * correct_predictions / total_predictions)

#     return test_acc, np.mean(losses)


# def deep_cluster_train(dataloader, model, loss_criterion, net_optimizer, annxed_layers_optimizer, epoch, device,
#                        return_inputs_outputs_targets_losses=False, verbose=True, checkpoint=0):
#     """Training of the CNN.
#         Args:
#             loader (torch.utils.data.DataLoader): Data loader
#             model (nn.Module): CNN
#             crit (torch.nn): loss
#             opt (torch.optim.SGD): optimizer for every parameters with True
#                                    requires_grad in model except top layer
#             epoch (int)
#     """
#     batch_time = utils.AverageMeter()
#     losses = utils.AverageMeter()
#     data_time = utils.AverageMeter()

#     # switch to train mode
#     model.train()

#     end = time.time()

#     dataset_inputs = []
#     dataset_outputs = []
#     dataset_targets = []
#     dataset_losses = []

#     for i, (input_tensor, target_tensor) in enumerate(dataloader):

#         data_time.update(time.time() - end)

#         target = target_tensor.to(device)
#         input = input_tensor.to(device)

#         output = model(input)

#         loss = loss_criterion(output, target)
#         if (loss.dim() == 0):
#             print("Error This function excpects a loss criterion that doesn't apply reduction\n")

#         dataset_inputs.extend(input_tensor.data.numpy())
#         dataset_losses.extend(loss.data.cpu().numpy())
#         dataset_targets.extend(target_tensor.data.numpy())
#         dataset_outputs.extend(output.cpu().data.numpy())

#         loss = loss.mean()

#         # record loss
#         losses.update(loss.item(), input_tensor.size(0))

#         net_optimizer.zero_grad()
#         annxed_layers_optimizer.zero_grad()
#         loss.backward()
#         net_optimizer.step()
#         annxed_layers_optimizer.step()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if verbose and (i % 10) == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss: {loss.val:.4f} ({loss.avg:.4f})'
#                   .format(epoch, i, len(dataloader), batch_time=batch_time,
#                           data_time=data_time, loss=losses))

#     if (return_inputs_outputs_targets_losses):
#         return losses.avg, dataset_inputs, dataset_outputs, dataset_targets, dataset_losses
#     else:
#         return losses.avg


# def deep_cluster_test(dataloader, model, device):
#     correct_predictions = 0
#     total_predictions = 0
#     model.eval()

#     for input_tensor, target in dataloader:
#         target = target.to(device)
#         input_tensor = input_tensor.to(device)

#         output = model(input_tensor)

#         predicted = torch.argmax(output, 1)

#         total_predictions += target.size(0)

#         correct_predictions += (predicted == target).sum().item()

#     test_acc = (100 * correct_predictions / total_predictions)

#     return test_acc


# def save_checkpoint(model, optimizer, epoch, path, architecture="unspecified", verbose=True):
#     if verbose:
#         print('Save checkpoint at: {0}'.format(path))

#     torch.save({
#         'epoch': epoch,
#         'arch': architecture,
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict()
#     }, path)


# def load_from_checkpoint(model, optimizer, path):
#     if os.path.isfile(path):
#         print("    => loading checkpoint '{}'".format(path))
#         checkpoint = torch.load(path)
#         epoch = checkpoint['epoch']
#         # remove top_layer parameters from checkpoint
#         for key in checkpoint['state_dict'].copy():
#             if 'top_layer' in key:
#                 del checkpoint['state_dict'][key]
#             model.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             print("    => loaded checkpoint '{}' (epoch {})"
#                   .format(path, checkpoint['epoch']))
#         return epoch
#     else:
#         print("    => no checkpoint found at '{}'".format(path))


# def add_probe_layer(model, layer_input_size, layer_output_size, device, weight_mean=0, weight_std=0.01):
#     probe_layer = nn.Linear(layer_input_size, layer_output_size)
#     probe_layer.weight.data.normal_(weight_mean, weight_std)
#     probe_layer.bias.data.zero_()
#     probe_layer.to(device)

#     model.top_layer = probe_layer

#     return


# def save_model_parameter(model, path, override=False):
#     if not os.path.isfile(path):
#         # The file don't exist; save
#         torch.save(model.state_dict(), path)
#         print("Model saved in directory: %s" % path)
#         return
#     if os.path.isfile(path):
#         if override:
#             torch.save(model.state_dict(), path)
#             print("Model saved in directory: %s" % path)
#             return
#         else:
#             print("Error the file already exists, rerun with parameter override=True to override.")
#             return


# def load_model_parameter(model, path):
#     if not os.path.isfile(path):
#         # The file dosen't exist
#         print("The provided path %s doesn't exist" % path)
#     else:
#         model.load_state_dict(torch.load(path))
#         print("Loaded model parameters from : %s" % path)
