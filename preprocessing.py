"""
Created on Tuesday April 20 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)

Contains methods to preprossess Deep cluster inputs.
"""
import numpy as np
import sys

from sklearn.decomposition import PCA

def l2_normalization(npdata):
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]
    return npdata

def faiss_pca_whitening(npdata, n_components):
    if "faiss" not in sys.modules:
        try:
            import faiss
        except ImportError:
            print("faiss is not installed")
            
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')
    mat = faiss.PCAMatrix (ndim, n_components, eigen_power=-0.5, )
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)
    return

def kmeans_pca_whitening(npdata, n_components, random_state=0):
    npdata =  npdata.astype('float32')
    pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
    npdata = pca.fit_transform(npdata)
    return npdata
