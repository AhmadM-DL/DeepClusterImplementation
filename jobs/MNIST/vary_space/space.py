import json, logging, argparse
import sys, importlib, os

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop

import torch
from torch.utils.tensorboard import SummaryWriter

from deep_clustering_models import LeNet
from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering_net import DeepClusteringNet

from utils import set_seed
from preprocessing import l2_normalization, sklearn_pca_whitening
from clustering import sklearn_kmeans, faiss_kmeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from scipy.stats import entropy

DATASET = "MNIST"

def run(device, batch_norm, n_clusters, pca, training_batch_size, random_state, dataset_path, use_faiss):

    logging.info("Set Seed")
    set_seed(random_state)
    
    logging.info("Loading Dataset")
    mnist = MNIST(dataset_path, download=True)

    device = torch.device(device)

    logging.info("Build Model")
    model = LeNet(batch_normalization=batch_norm, device=device)

    logging.info("Decorating Dataset")
    dataset = DeepClusteringDataset(mnist)

    logging.info("Defining Transformations")
    normalize = transforms.Normalize(mean=(0.1307,),
                                     std=(0.3081,))
    loading_transform = transforms.Compose([
        transforms.Resize(44),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize])


    logging.info("  Remove Top Layer")
    if model.top_layer:
        model.top_layer = None

    if loading_transform:
        dataset.set_transform(loading_transform)

    logging.info("  Full Feedforward")
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

    logging.info("  Clustering")
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

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hyperparam', default="jobs/mnist/hyper.json", type=str, help='Path to hyperparam json file')
    parser.add_argument('--dataset', default="../datasets", type=str, help="Path to datasets")
    parser.add_argument('--device', default="cpu", type=str, help="Device to use")
    parser.add_argument('--use_faiss', action="store_true")

    args = parser.parse_args()

    logging.basicConfig(filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    hparams = json.load(open(args.hyperparam, "r"))
    
    nmis= []
    all_entropies = []
    all_noises = []
    for seed in np.range(0, 100):
        writer = SummaryWriter('space_run/seed(%d)/'%seed)
        nmi, entropies, noises = run(torch.device(args.device), hparams['batch_norm'], hparams["n_clusters"], hparams["pca"], hparams["training_batch_size"],
                                    random_state=seed, dataset_path=args.dataset, use_faiss=args.use_faiss)
        writer.add_histogram("pseudoclasses_entropies", np.array(entropies), 0)
        writer.add_histogram("pseudoclasses_entropies", np.array(noises), 0)
        nmis.append(nmi)
        all_entropies.append(np.array(entropies))
        all_noises.append(np.array(noises))

    fig = plt.figure()
    plt.hist(nmis)
    plt.xlable("NMI values")
    plt.ylable("Frequency")
    plt.title("%s NMI Distribution"%DATASET)
    plt.savefig("space_run_nmi.png")
    
    json.dump({"dataset": DATASET,
               "seed_range": [0,100],
               "hparams": hparams,
               "nmis": nmis,
               "entropies": all_entropies,
               "noises": all_noises
               }, open("./space_run.json", "w"))          

    