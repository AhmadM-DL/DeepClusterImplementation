import sys
import importlib

sys.path.append('/content/DeepClusterImplementation/')

import utils
import models
import deep_cluster
import apps

importlib.reload(apps)
importlib.reload(utils)
importlib.reload(models)
importlib.reload(deep_cluster)