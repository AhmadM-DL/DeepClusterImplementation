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



def run(device, batch_norm, n_clusters, pca, sobel, training_batch_size, random_state, dataset_path, use_faiss, log_dir=None):
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

    logging.info("Decorating Dataset")
    dataset = DeepClusteringDataset(dataset_train)

    ## CHANGE
    normalize = transforms.Normalize(mean = (0.437, 0.443, 0.472),
                                     std = (0.198, 0.201, 0.197))

    logging.info("Defining Transformations")
    loading_transform = transforms.Compose([
    transforms.Resize(44),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    normalize])

    logging.info("Remove Top Layer")
    if model.top_layer:
        model.top_layer = None

    if loading_transform:
        dataset.set_transform(loading_transform)

    logging.info(" Full Feedforward")
    features = model.full_feed_forward(
        dataloader=torch.utils.data.DataLoader(dataset,
                                            batch_size=training_batch_size,
                                            shuffle=None,
                                            pin_memory=True))

    logging.info("  Pre-processing pca/whitening/l2_normalization")
    if pca == None:
        pass
    else:
        features = sklearn_pca_whitening(features, n_components=pca, random_state=random_state)
    features = l2_normalization(features)

    logging.info(" Clustering")
    if use_faiss:
        assignments = faiss_kmeans(
            features, n_clusters=n_clusters,
            random_state=random_state,
            fit_partial=None)
    else:
        assignments = sklearn_kmeans(
            features, n_clusters=n_clusters,
            random_state=random_state,
            max_iter = 20,
            fit_partial=None)

    dataset.set_pseudolabels(assignments)

    nmi =  NMI(assignments, dataset.get_targets())
    
    pseudoclasses = dataset.group_indices_by_labels()
    pseudoclasses_labels = [[dataset.get_targets()[index] for index in pseudoclass] for pseudoclass in pseudoclasses]
    pseudoclasses_labels_counts = [np.unique(pseudoclass_labels, return_counts=True)[1] for pseudoclass_labels in pseudoclasses_labels]
    entropies = [entropy(pseudoclass_labels_counts) for pseudoclass_labels_counts in pseudoclasses_labels_counts]
    noises = [ 1 - np.max(pseudoclass_labels_counts)/np.sum(pseudoclass_labels_counts) for pseudoclass_labels_counts in pseudoclasses_labels_counts]

    return nmi, entropies, noises
    

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
    device = torch.device(args.device)

    nmis= []
    all_entropies = []
    all_noises = []
    seed_range = [1, 101]
    for seed in np.arange(seed_range[0], seed_range[1]):
        writer = SummaryWriter( os.path.join(args.log_dir, TENSORBOARD, 'seed(%d)'%seed) )
        nmi, entropies, noises = run(torch.device(args.device), hparams['batch_norm'], hparams["n_clusters"], hparams["pca"],
                                     hparams["sobel"],  hparams["training_batch_size"], random_state=seed, dataset_path=args.dataset,
                                     use_faiss=args.use_faiss, log_dir=None)
        writer.add_histogram("pseudoclasses_entropies", np.array(entropies), 0)
        writer.add_histogram("pseudoclasses_noises", np.array(noises), 0)
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