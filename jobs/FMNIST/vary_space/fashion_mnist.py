import torch
import json
import logging
import argparse
import os
import datetime
import sys
import importlib

sys.path.append("C:\\Users\\PC\\Desktop\\Projects\\DeepClusterImplementation")


from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop
from deep_clustering_models import LeNet
from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering import deep_cluster

from evaluation.linear_probe import eval_linear
from utils import set_seed

from datetime import datetime
import traceback


## CHANGE
DATASET = "FMNIST"
## CHANGE
MODEL = "LENET"

CHECKPOINTS = "checkpoints"
TENSORBOARD = "runs"


def run(device, batch_norm, lr, wd, momentum, n_cycles,
        n_clusters, pca, training_batch_size, training_shuffle,
        random_state, dataset_path, use_faiss, log_dir=None):
    logging.info("New Experiment ##########################################")
    logging.info("%s" % datetime.now())

    logging.info("Set Seed")
    set_seed(random_state)

    logging.info("Set Log Dir")
    if not log_dir:
        log_dir = "./"
    
    logging.info("Loading Dataset")
    ## CHANGE
    dataset_train = FashionMNIST(dataset_path, download=True)

    device = torch.device(device)

    logging.info("Build Model")
    model = LeNet(batch_normalization=batch_norm, device=device)

    logging.info("Build Optimizer")
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=wd,
    )

    logging.info("Defining Loss Function")
    loss_function = torch.nn.CrossEntropyLoss()

    logging.info("Decorating Dataset")
    dataset = DeepClusteringDataset(dataset_train)

    logging.info("Defining Transformations")
    ## CHANGE
    normalize = transforms.Normalize(mean=(0.2860,),
                                     std=(0.3205,))
    ## CHANGE
    main_transform = transforms.ToTensor()
    dataset.set_transform(main_transform)                            
    ## CHANGE
    in_loop_training_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        normalize])
    ## CHANGE
    in_loop_loading_transform = transforms.Compose([
        transforms.Resize(44),
        transforms.CenterCrop(32),
        normalize])

    logging.info("Defining Writer")
    writer_file = "{dataset}_{model}_batchnorm({bt})_lr({lr})_momentum({mom})_wdecay({wd})_n_clusters({nclusters})_n_cycles({ncycles})_rnd({seed})_t_batch_size({tbsize})_shuffle({shfl})_"
    writer_file = writer_file.format(dataset=DATASET, model=MODEL, bt=batch_norm, lr=lr, mom=momentum,
                                     wd=wd, nclusters=n_clusters, ncycles=n_cycles, seed=random_state,
                                     tbsize=training_batch_size, shfl=training_shuffle)
    if pca:
        writer_file = writer_file+"pca(%d)" % pca
    else:
        writer_file = writer_file+"pca(None)"

    writer = SummaryWriter(os.path.join(log_dir, TENSORBOARD, writer_file))

    if os.path.isfile(os.path.join(log_dir, CHECKPOINTS, writer_file, "last_model.pth")):
        resume = os.path.join(log_dir, CHECKPOINTS, writer_file, "last_model.pth")
        logging.info("Resuming from: %s" % resume)
    else:
        resume = None
        logging.info("Run: %s" % writer_file)

    deep_cluster(model=model,
                 dataset=dataset,
                 n_clusters=n_clusters,
                 loss_fn=loss_function,
                 optimizer=optimizer,
                 n_cycles=n_cycles,
                 loading_transform=in_loop_loading_transform,
                 training_transform=in_loop_training_transform,
                 random_state=random_state,
                 verbose=1,
                 training_batch_size=training_batch_size,
                 training_shuffle=training_shuffle,
                 pca_components=pca,
                 checkpoints=os.path.join(log_dir, CHECKPOINTS, writer_file),
                 use_faiss=use_faiss,
                 resume=resume,
                 in_loop_transform=True,
                 writer=writer)
    
    ## CHANGE
    dataset_test = FashionMNIST(dataset_path, train=False, download=True)

    traindataset = dataset_train
    validdataset = dataset_test

    ## CHANGE
    transformations_val = [transforms.Resize(44),
                           transforms.CenterCrop(32),
                           transforms.ToTensor(),
                           normalize]
    ## CHANGE
    transformations_train = [transforms.Resize(44),
                            transforms.CenterCrop(32),
                            #transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize]

    dataset_train.transform = transforms.Compose(transformations_train)
    dataset_test.transform = transforms.Compose(transformations_val)

    eval_linear(model=model,
                n_epochs=20,
                traindataset=traindataset,
                validdataset=validdataset,
                target_layer="conv_2",
                n_labels=10,
                features_size=1600,
                avg_pool= None,
                random_state=random_state,
                writer= writer,
                verbose=1)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--log', default="exp.log", type=str)
    parser.add_argument('--expcheck', default="exp.chkp", type=str)

    parser.add_argument('--hyperparam', default="./hyper.json",
                        type=str, help='Path to hyperparam json file')

    parser.add_argument('--dataset', default="../datasets",
                        type=str, help="Path to datasets")

    parser.add_argument('--device', default="cpu",
                        type=str, help="Device to use")

    parser.add_argument('--seed', default=666, type=int, help="Random Seed")

    parser.add_argument('--log_dir', default="./",
                        type=str, help="Logs directory")

    parser.add_argument("--use_faiss", action="store_true",
                        help="Use facebook FAISS for clustering")

    args = parser.parse_args()

    EXPERIMENT_CHECK = args.log
    LOGS = args.expcheck

    # create logs file if not available
    if not os.path.isfile(os.path.join(args.log_dir, LOGS)):
        f = open( os.path.join(args.log_dir, LOGS), 'a')
        f.close()

    logging.basicConfig(format='%(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(filename=os.path.join(args.log_dir, LOGS), mode="a"),
                            logging.StreamHandler(sys.stdout)
                        ]
    )

    logging.info("New Batch ####################################")
    logging.info("\n##########\n%s\n##########\n" % datetime.now())


    hparams = json.load(open(args.hyperparam, "r"))
    device = torch.device(args.device)

    if os.path.isfile(os.path.join(args.log_dir, EXPERIMENT_CHECK)):
        executed_runs = int(
            open(os.path.join(args.log_dir, EXPERIMENT_CHECK), "r").read())
        logging.info("Running from checkpoint: run(%d)" % executed_runs)
    else:
        executed_runs = 0
    counter=1

    for lr in hparams['lr']:
        for wd in hparams['wd']:
            for momentum in hparams['momentum']:
                for batch_norm in hparams['batch_norm']:
                    for n_cycles in hparams["n_cycles"]:
                        for n_clusters in hparams["n_clusters"]:
                            for pca in hparams["pca"]:
                                for training_batch_size in hparams["training_batch_size"]:
                                    for training_shuffle in hparams["training_shuffle"]:
                                        for seed in hparams["seed"]:
                                             #logging.info("Experiment %d"%counter)
                                            if counter <= executed_runs:
                                                counter += 1
                                                continue
                                            try:
                                                run(device, batch_norm, lr, wd, momentum, n_cycles, n_clusters,
                                                    pca, training_batch_size, training_shuffle,
                                                    random_state=seed, dataset_path=args.dataset,
                                                    use_faiss=args.use_faiss, log_dir=args.log_dir)
                                                counter += 1
                                                open(os.path.join(args.log_dir, EXPERIMENT_CHECK), "w").write(
                                                    str(counter))
                                            except Exception as e:
                                                logging.error(traceback.format_exception(*sys.exc_info()))
                                                logging.error(e)
                                                counter += 1
                                                open(os.path.join(args.log_dir, EXPERIMENT_CHECK), "w").write(
                                                    str(counter))
                                                continue
  