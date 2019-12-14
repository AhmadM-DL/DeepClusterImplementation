# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 09:37:58 2019

@author: amm90
"""
import numpy as np 
from torch.utils.data.sampler import Sampler
import pickle, os, copy, hashlib
import torch
from torch.optim import SGD
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
from PIL import Image
from PIL import ImageFile
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
from sklearn.manifold import TSNE
from scipy.stats import entropy
from PIL import Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

class UnifLabelSampler(Sampler):

    def __init__(self, images_lists, size_per_pseudolabel="average", dataset_multiplier=1):
        self.images_lists = images_lists
        self.dataset_multiplier = dataset_multiplier
        self.size_per_pseudolabel = size_per_pseudolabel
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):

        pseudolabels_sizes = [len(pseudolabel_set) for pseudolabel_set in self.images_lists]

        if(self.size_per_pseudolabel=="average"):
            pseudolabel_size = int( np.average(pseudolabels_sizes) )+1
        else:
            if(self.size_per_pseudolabel=="max"):
                pseudolabel_size = np.max(pseudolabels_sizes)
            else:
                if(self.size_per_pseudolabel=="min"):
                    pseudolabel_size = np.min(pseudolabels_sizes)
                else:
                    print("UnifLabelSampler:Error: size_per_pseudolabel should be 'average', 'max', or 'min' ")

        self.N = self.dataset_multiplier * pseudolabel_size * len(self.images_lists)
        res = np.zeros(self.N)

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                pseudolabel_size,
                replace=(len(self.images_lists[i]) <= pseudolabel_size)
            )
            res[i * pseudolabel_size: (i + 1) * pseudolabel_size] = indexes

        np.random.shuffle(res)
        return res.astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class NMIMeter(object):
    """ Computes and store the NMI values """

    def __init__(self):
        self.reset()

    def reset(self):
        self.nmi_array=[]

    def update(self, ground_truth, predictions):
        nmi = normalized_mutual_info_score(ground_truth, predictions)
        self.nmi_array.append(nmi)

    def store_as_csv(self, path):
        nmi_rows = [ {'Epoch': index, 'NMI': nmi} for (index,nmi) in enumerate(self.nmi_array)]
        nmis_df = pd.DataFrame(nmi_rows)
        nmis_df.to_csv(path)

    def load_from_csv(self, path):
        nmis_df = pd.read_csv(path, index_col=0)
        self.nmi_array = nmis_df['NMI'].values

    def avg(self):
        return np.average(self.nmi_array)


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)
            
            
class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class, rand_image_min=150, rand_image_max=200):
        self.model = model
        self.model.eval()
        self.target_class = target_class
        
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(rand_image_min, rand_image_max, (32, 32, 1)))
        #self.created_image = np.uint8(np.zeros( (32, 32, 1) ) )

    def generate(self, n_steps=150, initial_learning_rate=6, verbose=True):
        
        for i in range(1, n_steps):
          
            # Process image and return variable
            self.processed_image = self.__preprocess_image(self.created_image, False)
            
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            #optimizer = torch.optim.Adam([self.processed_image], lr=initial_learning_rate, weight_decay=1e-6)
            
            # Forward
            output = self.model(self.processed_image)
            
            # Target specific class
            class_loss = -output[0, self.target_class] + self.processed_image.sum().abs().cpu().data
            if(verbose):
              print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.cpu().data.numpy()))
            
            # Zero grads
            self.model.zero_grad()
            
            # Backward
            class_loss.backward()
            
            # Update image
            optimizer.step()
            
            # Recreate image
            self.created_image = self.__recreate_image(self.processed_image)
                
        return self.processed_image
    
    def __preprocess_image(pil_im, resize_im=0 ):
        """
            Processes image for CNNs
        Args:
            PIL_img (PIL_img): Image to process
            resize_im (bool): Resize to 224 or not
        returns:
            im_as_var (torch variable): Variable that contains processed float tensor
        """
        # mean and std list for channels (Imagenet)
        mean = [0.1307] 
        std = [0.3081]
        
        # Resize image
        if resize_im:
            pil_im.thumbnail((resize_im, resize_im))
            
        im_as_arr = np.float32(pil_im)
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
        
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
            im_as_arr[channel] -= mean[channel]
            im_as_arr[channel] /= std[channel]
            
        # Convert to float tensor
        im_as_ten = torch.from_numpy(im_as_arr).float()
        
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        im_as_ten.unsqueeze_(0)
        
        # Convert to Pytorch variable
        im_as_var = torch.autograd.Variable(im_as_ten, requires_grad=True)
        
        return im_as_var
  
    def __recreate_image(im_as_var):
        """
            Recreates images from a torch variable, sort of reverse preprocessing
        Args:
            im_as_var (torch variable): Image to recreate
        returns:
            recreated_im (numpy arr): Recreated image in array
        """
        reverse_mean = [-0.1307]
        reverse_std = [1/0.3081]
        recreated_im = copy.copy(im_as_var.data.numpy()[0])
        
        for c in range(1):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]
            
        recreated_im[recreated_im > 1] = 1
        recreated_im[recreated_im < 0] = 0
        recreated_im = np.round(recreated_im * 255)
    
        recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
        return recreated_im
    
    
def generate_random_colors(n):
    colors = {}
    for i in np.arange(n):
        numByte = str.encode(str(i+np.random.rand()))
        hashObj = hashlib.sha1(numByte).digest()
        r, g, b = hashObj[-1]/255 , hashObj[-2]/255 , hashObj[-3]/255
        colors[i] =  (r, g, b,1.0)
    return colors

def plot_t_sne_embedding_2d(tsne_results, images, clusters=None, n_clusters=None, plot_images=False, **kwargs ):

    if(tsne_results.shape[1]!=2):
        print("The provided t_sne results are of "+str(tsne_results.shape[1])+
              " components and not 2.\nThis function considers t_sne results with 2 components")
        return

    else:
        if(n_clusters):
            colors = generate_random_colors(n_clusters)

        fig = plt.figure(figsize=kwargs.get("figsize",(15,15)))
        ax = fig.add_subplot(111)
        ax.set_title(kwargs.get("title","t_SNE results of 2 components"))

        xmin = kwargs.get("xmin",None)
        xmax = kwargs.get("xmax", None)
        ymin = kwargs.get("ymin", None)
        ymax = kwargs.get("ymax", None)

        if( (xmin is not None) and (xmax is not None) and (ymin is not None) and (ymax is not None)  ):
          plt.xlim(xmin, xmax)
          plt.ylim(ymin, ymax)

        if (plot_images):
            data_points_alpha=0
        else:
            data_points_alpha=1

        ax.scatter(tsne_results[:,0], tsne_results[:,1], c= clusters, marker='o', alpha=data_points_alpha)

        ax.set_xlabel('tsne_0')
        ax.set_ylabel('tsne_1')

        if(plot_images):

            for i in np.arange(0, len(tsne_results)):
                image_data = images[i]
                im = OffsetImage(image_data, cmap='gray', zoom=0.8)

                if(n_clusters):
                    ab = AnnotationBbox(im, (tsne_results[i,0],tsne_results[i,1]), frameon=True, bboxprops= dict( edgecolor= colors[clusters[i]], facecolor = colors[clusters[i]] ) )
                else:
                    ab = AnnotationBbox(im, (tsne_results[i,0],tsne_results[i,1]), frameon=False )

                ax.add_artist(ab)

            lp = lambda i: mpatches.Patch(color=colors[i], label=str(i))

            handles = [lp(i) for i in np.unique(clusters)]

            plt.legend(handles=handles)

        return fig
    
def pil_loader(path):
    """Loads an image.
    Args:
        path (string): path to image file
    Returns:
        Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def plot_feature_space_using_tsne( features, images, labels, percent_of_data_to_plot=10, plot_images=False, **kwargs):

    tsne = TSNE(n_components=kwargs.get("n_components",2),
                n_iter=kwargs.get("n_iter", 1000),
                perplexity=kwargs.get("perplexity",40),
                verbose= kwargs.get("verbose",0)
                )

    number_of_data_to_use = int(len(features)*percent_of_data_to_plot//100)
    data_to_plot_indices = np.random.choice( len(features), size=number_of_data_to_use )

    tsne_components = tsne.fit_transform(np.array(features)[data_to_plot_indices])

    plot_t_sne_embedding_2d(tsne_results=tsne_components,
                            images= np.array(images)[data_to_plot_indices],
                            plot_images=plot_images,
                            clusters = np.array(labels)[data_to_plot_indices],
                            n_clusters = len(np.unique( np.array(labels)[data_to_plot_indices]) ),
                            **kwargs
                            )

def plot_cluster_images(cluster, images_paths, percent_of_images_to_plot=100):
    cluster_images_indices = [image_index for image_index in cluster]
    number_images_to_plot = len(cluster_images_indices) * percent_of_images_to_plot // 100
    images_to_plot_indices = np.random.choice(cluster_images_indices, number_images_to_plot, replace=False)
    plot_set_of_images(images_to_plot_indices, images_paths)


def plot_set_of_images(images_indices, images_paths, figsize=(20, 20)):
    N = len(images_indices)
    fig = plt.figure(figsize=figsize)
    images_to_plot_paths = np.array(images_paths)[images_indices]

    for i, image_path in enumerate(images_to_plot_paths):
        plt.subplot(N // 10 + 1, 10, i + 1)
        im = Image.open(image_path)
        plt.axis("off")
        plt.imshow(im)


def plot_clusters_histograms(clusters, epoch, ground_truth):
    fig = plt.figure(figsize=(20, 30))

    for (n, cluster) in enumerate(clusters[epoch]):
        images_original_classes = [ground_truth[image_index] for image_index in cluster]
        plt.subplot(len(clusters[epoch]) // 10 + 1, 10, n + 1)
        plt.xticks(rotation='horizontal')
        plt.yticks([])

        values, counts = np.unique(images_original_classes, return_counts=True)
        cluster_entropy = entropy(counts)
        # max_count_target = values[np.argmax(counts)]

        plt.ylabel("C %d E %f " % (n, cluster_entropy))
        plt.hist(np.array(images_original_classes).astype(str))


def merge_clusters_on_entropy_ground_truth(clusters, ground_truth, merging_entropy_threshold=0):
    clusters_to_merge_indices = [[] for i in range(len(np.unique(ground_truth)))]

    for (k, cluster) in enumerate(clusters):
        images_original_classes = [ground_truth[image_index] for image_index in cluster]
        values, counts = np.unique(images_original_classes, return_counts=True)
        cluster_entropy = entropy(counts)
        max_count_target = values[np.argmax(counts)]

        if (cluster_entropy <= merging_entropy_threshold):
            clusters_to_merge_indices[max_count_target].extend([k])

    clusters_to_persist_indices = set(range(len(clusters))) - set(
        np.concatenate(clusters_to_merge_indices))

    merged_clusters = [clusters[i] for i in clusters_to_persist_indices]

    for group_to_cluster in clusters_to_merge_indices:
        if group_to_cluster == []: continue
        new_cluster = []
        for cluster_index in group_to_cluster:
            new_cluster.extend(clusters[cluster_index])
        merged_clusters.append(new_cluster)

    return merged_clusters
    