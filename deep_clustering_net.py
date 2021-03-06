# -*- coding: utf-8 -*-
"""
Created on Tuesday April 14 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""
import torch
import math
import time
import os
import numpy as np
from custom_layers import SobelFilter
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.tensorboard import SummaryWriter


class DeepClusteringNet(torch.nn.Module):

    def __init__(self, input_size, features, classifier, top_layer, device, with_sobel=False, concat_sobel=False):
        super().__init__()
        self.sobel = SobelFilter() if with_sobel else None
        self.features = features
        self.classifier = classifier
        self.top_layer = top_layer
        self.input_size = input_size
        self.concat_sobel = concat_sobel


        self._initialize_weights()
        self.device = device
        self.to(self.device)

        return

    def forward(self, x):
        if self.sobel:
            if self.concat_sobel:
                aug = self.sobel(x)
                x = torch.cat( (x, aug), 1)
            else:
                x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.top_layer:
            x = torch.nn.functional.relu(x)
            x = self.top_layer(x)
        return x
    
    def extract_features(self, x, target_layer, flatten=True):
        if self.sobel:
            x = self.sobel(x)
        
        for module_name, module in self.features.named_children():
            x = module(x)
            if module_name == target_layer:
                break
        if flatten:
            x = x.view(x.size(0), -1)
            
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def output_size(self, single_input_size):
        """
        A method that computes model output size by feeding forward
        a dummy single input. This method doesn't consider the batch size
        :param single_input_size: tuple
            a tuple that includes the input size in the form of (#Channels, Width, Height)
        :return: tuple
            a tuple that includes the output size
        """
        x = torch.rand(size=(1, *single_input_size), device=self.device)
        x = self.forward(x)
        return tuple(x.size()[1:])

    def add_top_layer(self, output_size):
        # get model output size
        model_output_size = self.output_size(self.input_size)[0]
        linear_layer = torch.nn.Linear(model_output_size, output_size)
        linear_layer.weight.data.normal_(0, 0.01)
        linear_layer.bias.data.zero_()
        self.top_layer = torch.nn.Sequential(torch.nn.ReLU(),linear_layer)
        self.top_layer.to(self.device)

    def remove_top_layer(self):
        self.top_layer == None

    def freeze_features(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_features(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def freeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = True

    def deep_cluster_train(self, dataloader, epoch, optimizer: torch.optim.Optimizer, loss_fn, verbose=False,
                           writer: SummaryWriter = None):

        if verbose:
            print('Training Model')

        self.train()
        end = time.time()

        for i, (input_, target) in enumerate(dataloader):

            input_ = input_.to(self.device)
            target = target.to(self.device)
            output = self(input_)

            loss = loss_fn(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if writer:

                writer.add_scalar("training_loss",
                                  scalar_value=loss.item(),
                                  global_step=epoch * len(dataloader) + i)

            if verbose and len(dataloader) >= 10 and (i % (len(dataloader)//10)) == 0:
                print('{0} / {1}\tTime: {2:.3f}'.format(i,
                                                        len(dataloader), time.time() - end))

            end = time.time()
    
    def deep_cluster_train_with_weights(self, dataloader,
                           epoch, optimizer: torch.optim.Optimizer,
                           loss_fn, 
                           instance_wise_weights:torch.tensor,
                           verbose=False,
                           writer: SummaryWriter = None,
                           writer_tag= None
                           ):

        if verbose:
            print('Training Model')

        self.train()
        end = time.time()

        if not writer_tag:
            writer_tag= ""
        else:
            writer_tag= "/"+writer_tag

        dataloader.dataset.set_instance_wise_weights(instance_wise_weights)

        for i, (input_, target, instance_wise_weight) in enumerate(dataloader):

            input_ = input_.to(self.device)
            target = target.to(self.device)
            instance_wise_weight = instance_wise_weight.to(self.device)
            instance_wise_weight = torch.as_tensor(instance_wise_weight, dtype=torch.float64)

            output = self(input_)

            loss = loss_fn(output, target)
            loss = torch.as_tensor(loss, dtype=torch.float64)
            if (loss.dim() == 0):
                raise Exception("Error This function expects a loss criterion that doesn't apply reduction\n")
            
            loss = loss * instance_wise_weight
            loss = loss.mean()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if writer:
                
                writer.add_scalar("training_loss"+writer_tag,
                                  scalar_value=loss.item(),
                                  global_step=epoch * len(dataloader) + i)

            if verbose and len(dataloader) >= 10 and (i % (len(dataloader)//10)) == 0:
                print('{0} / {1}\tTime: {2:.3f}'.format(i,
                                                        len(dataloader), time.time() - end))

            end = time.time()
        
        dataloader.dataset.unset_instance_wise_weights()


    def full_feed_forward(self, dataloader, verbose=False):

        if verbose:
            print('Computing Model Output')

        self.eval()
        end = time.time()

        for i, (input_, _) in enumerate(dataloader):

            batch_size = dataloader.batch_size
            input_ = input_.to(self.device)
            output = self(input_).data.cpu().numpy()

            if i == 0:
                outputs = np.zeros(
                    shape=(len(dataloader.dataset), output.shape[1]), dtype=np.float32)

            if i < len(dataloader) - 1:
                outputs[i * batch_size: (i + 1) *
                        batch_size] = output.astype('float32')
            else:
                # special treatment for final batch
                outputs[i * batch_size:] = output.astype('float32')

            if verbose and len(dataloader) >= 10 and (i % (len(dataloader)//10)) == 0:
                print('{0} / {1}\tTime: {2:.3f}'.format(i,
                                                        len(dataloader), time.time() - end))

            end = time.time()

        return outputs

    def load_model_parameters(self, model_parameters_path, optimizer=None):
        if os.path.isfile(model_parameters_path):
            print("=> loading checkpoint '{}'".format(model_parameters_path))
            checkpoint = torch.load(model_parameters_path)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})".format(
                model_parameters_path, checkpoint['epoch']))

            return start_epoch
        else:
            print("=> no checkpoint found at '{}'".format(model_parameters_path))
            raise Exception("No checkpoint found at %s" %(model_parameters_path))

    def save_model_parameters(self, model_parameters_path, epoch, optimizer=None):
        model_dict = {'epoch': epoch,
                      'state_dict': self.state_dict()}
        if optimizer:
            model_dict['optimizer'] = optimizer.state_dict()

        torch.save(model_dict, model_parameters_path)

        return


# def alexnet_cifar(sobel=False, bn=True, out=10):
#     input_n_channels = 2 + int(not sobel)

#     features_cfg = [(64, 3, 1, 2), ('M', 2, None), (192, 3, 1, 2),
#                     ('M', 2, None), (384, 3, 1, 1), (256, 3, 1, 1),
#                     (256, 3, 1, 1), ('M', 3, 2)]

#     classifier_cfg = [('D', 0.5), ('L', 4096, 2048), ('R', True), ('D', 0.5),
#                       ('L', 2048, 2048), ('R', True)]

#     top_layer_cfg = [('L', 2048, out)]

#     features = make_convolutional_layers(features_cfg, input_n_channels, bn=bn)
#     classifier = make_linear_layers(classifier_cfg)
#     top_layer = make_linear_layers(top_layer_cfg)[0]

#     model = NetBuilder(features, classifier, top_layer,
#                        features_output=4096, classifier_output=2048, top_layer_output=out,
#                        apply_sobel=sobel)

#     return model


# def lenet_5(bn=False, out=10):
#     features_cfg = [(6, 5, 1, 0), ('M', 2, None), (16, 5, 1, 0), ('M', 2, None)]

#     classifier_cfg = [('L', 400, 120), ('R', True),
#                       ('L', 120, 84), ('R', True)]

#     top_layer_cfg = [('L', 84, out)]

#     features = make_convolutional_layers(features_cfg, 1, bn=bn)
#     classifier = make_linear_layers(classifier_cfg)
#     top_layer = make_linear_layers(top_layer_cfg)[0]

#     model = NetBuilder(features, classifier, top_layer,
#                        features_output=400, classifier_output=84, top_layer_output=out,
#                        apply_sobel=False)

#     return model


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
