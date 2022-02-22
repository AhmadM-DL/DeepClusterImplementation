from evaluation.linear_probe import eval_linear
from deep_clustering import deep_cluster
from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering_models import AlexNet_Micro

import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop
from torchvision.datasets import SVHN
from torchvision import transforms

from utils import set_seed
from preprocessing import l2_normalization, sklearn_pca_whitening
from clustering import sklearn_kmeans, faiss_kmeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from scipy.stats import entropy

# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter

from utils import set_seed

import torch
import json,  logging, argparse
from datetime import datetime
import sys, traceback, importlib, os

#sys.path.append("C:\\Users\\PC\\Desktop\\Projects\\DeepClusterImplementation")

## CHANGE
DATASET = "SVHN"
## CHANGE
MODEL = "AlexNetMicro"

CHECKPOINTS = "checkpoints"
TENSORBOARD = "runs"



def run(device, batch_norm, lr, wd, momentum, n_cycles,
        n_clusters, pca, training_batch_size, sobel, training_shuffle,
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
    dataset_train = SVHN(dataset_path, download=True, split='train')

    device = torch.device(device)

    logging.info("Build Model")
    ## CHANGE
    model = AlexNet_Micro(
        sobel=sobel, batch_normalization=batch_norm, device=device)

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
    normalize = transforms.Normalize(mean = (0.437, 0.443, 0.472),
                                     std = (0.198, 0.201, 0.197))
    ## CHANGE
    main_transform = transforms.ToTensor()
    dataset.set_transform(main_transform)
    ## CHANGE
    in_loop_training_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        normalize])
    ## CHANGE
    in_loop_loading_transform = transforms.Compose([
        transforms.Resize(45),
        transforms.CenterCrop(32),
        normalize])

    logging.info("Defining Writer")
    writer_file = "{dataset}_{model}_batchnorm({bt})_lr({lr})_momentum({mom})_wdecay({wd})_n_clusters({nclusters})_n_cycles({ncycles})_rnd({seed})_t_batch_size({tbsize})_shuffle({shfl})_sobel({sobel})_"
    writer_file = writer_file.format(dataset=DATASET, model=MODEL, bt=batch_norm, lr=lr, mom=momentum,
                                     wd=wd, nclusters=n_clusters, ncycles=n_cycles, seed=random_state,
                                     tbsize=training_batch_size, shfl=training_shuffle, sobel=sobel)
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

    return deep_cluster(model=model,
                 dataset=dataset,
                 n_clusters=n_clusters,
                 loss_fn=loss_function,
                 optimizer=optimizer,
                 n_cycles=n_cycles,
                 loading_transform= in_loop_loading_transform,
                 training_transform= in_loop_training_transform,
                 random_state=random_state,
                 verbose=1,
                 training_batch_size=training_batch_size,
                 training_shuffle=training_shuffle,
                 pca_components=pca,
                 checkpoints=os.path.join(log_dir, CHECKPOINTS, writer_file),
                 use_faiss=use_faiss,
                 resume=resume,
                 in_loop_transform=True,
                 writer=writer,
                 only_clustering=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of DeepCluster')

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

    EXPERIMENT_CHECK = args.expcheck
    LOGS = args.log

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
    device = args.device

    nmis= []
    all_entropies = []
    all_noises = []
    seed_range = [1, 101]
    for seed in np.arange(seed_range[0], seed_range[1]):
        nmi, entropies, noises = run(device, hparams['batch_norm'], 0.01, 0.0001, 0.9, 1,
                                     hparams["n_clusters"], hparams["pca"],  hparams["training_batch_size"],
                                     hparams["sobel"], True, random_state=seed, dataset_path=args.dataset,
                                     use_faiss=args.use_faiss, log_dir=args.log_dir)
        nmis.append(nmi)
        all_entropies.append(list(entropies))
        all_noises.append(list(noises))

    fig = plt.figure()
    plt.hist(nmis)
    plt.xlabel("NMI values")
    plt.ylabel("Frequency")
    plt.title("%s NMI Distribution"%DATASET)
    writer.add_figure("NMI Dist", fig)    
    
    json.dump({"dataset": DATASET,
               "seed_range": seed_range,
               "hparams": hparams,
               "nmis": nmis,
               "entropies": all_entropies,
               "noises": all_noises
               }, open(os.path.join(args.log_dir, TENSORBOARD,"space_run.json"), "w"))          