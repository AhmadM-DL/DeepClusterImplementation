"""
Created on Tuesday April 20 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)

I created this model to serve as a modular and re-usable Deep Learning
Test Suit it contains methods that tests differentaspects of Deep Learning
modules. For example one method is to test whether the parameter changed after
an optimization/training step or not.
"""

import torch
from torch.utils.data import Dataset

class RandomDataset(Dataset):

    def __init__(self, input_size, data_length, n_classes):
        self.len = data_length
        self.data = torch.randn(data_length, *input_size)
        self.targets = torch.empty(data_length, dtype=torch.long).random_(n_classes)        
        device = torch.device("cpu")

    def __getitem__(self, index):
        return (self.data[index], self.targets[index])

    def __len__(self):
        return self.len

def do_train_step(model, loss_fn, optim, batch, device):
    """Run a training step on model for a given batch of data
    Parameters of the model accumulate gradients and the optimizer performs
    a gradient update on the parameters
    Parameters
    ----------
    model : torch.nn.Module
      torch model, an instance of torch.nn.Module
    loss_fn : function
      a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
      an optimizer instance
    batch : list
      a 2 element list of inputs and labels, to be fed to the model
    """

    # put model in train mode
    model.train()
    model.to(device)

    #  run one forward + backward step
    # clear gradient
    optim.zero_grad()
    # inputs and targets
    inputs, targets = batch[0], batch[1]
    # move data to DEVICE
    inputs = inputs.to(device)
    targets = targets.to(device)
    # forward
    likelihood = model(inputs)
    # calc loss
    loss = loss_fn(likelihood, targets)
    # backward
    loss.backward()
    # optimization step
    optim.step()


def do_forward_step(model, batch, device):
    """Run a forward step of model for a given batch of data
    Parameters
    ----------
    model : torch.nn.Module
      torch model, an instance of torch.nn.Module
    batch : list
      a 2 element list of inputs and labels, to be fed to the model
    Returns
    -------
    torch.tensor
      output of model's forward function
    """

    # put model in eval mode
    model.eval()
    model.to(device)

    with torch.no_grad():
        # inputs and targets
        inputs = batch[0]
        # move data to DEVICE
        inputs = inputs.to(device)
        # forward
        return model(inputs)


def test_param_change(vars_change, model, loss_fn, optim, batch, device, params=None):
    """Check if given variables (params) change or not during training
    If parameters (params) aren't provided, check all parameters.
    Parameters
    ----------
    vars_change : bool
      a flag which controls the check for change or not change
    model : torch.nn.Module
      torch model, an instance of torch.nn.Module
    loss_fn : function
      a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
      an optimizer instance
    batch : list
      a 2 element list of inputs and labels, to be fed to the model
    params : list, optional
      list of parameters of form (name, variable)
    Raises
    ------
    VariablesChangeException
      if vars_change is True and params DO NOT change during training
      if vars_change is False and params DO change during training
    """

    if params is None:
        # get a list of params that are allowed to change
        params = [np for np in model.named_parameters() if np[1].requires_grad]

    # take a copy
    initial_params = [(name, p.clone()) for (name, p) in params]

    # run a training step
    do_train_step(model, loss_fn, optim, batch, device)

    # check if variables have changed
    for (_, p0), (name, p1) in zip(initial_params, params):
        if vars_change:
            assert not torch.equal(p0.to(device), p1.to(device))
        else:
            assert torch.equal(p0.to(device), p1.to(device))
            
