    """
    
    """

import sys
sys.path.append("../")
from deep_clustering_dataset import DeepClusteringDataset
from deep_clustering_net import DeepClusteringNet
from utils import set_seed

def eval_linear(model: DeepClusteringNet, conv_layer, **kwargs):

    # set seed
    set_seed(kwargs.get("random_state",0))

    # Freeze features
    model.freeze_features()

    # 

