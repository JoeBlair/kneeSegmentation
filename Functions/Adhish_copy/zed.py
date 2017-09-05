import time
import sys
import numpy as np
import scipy as sp
import matplotlib as mpl
if sys.platform == 'darwin': mpl.use('TkAgg')

import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import scipy.optimize

# STEP 0: Load data


im = plt.imread("/home/xvt131/images.png")
#
print im.shape

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

im = rgb2gray(im)
M = im.reshape(6400, 1)


class ZCA():
    def __init__(self, bias=0.1):
        self.bias = bias

    def _flat(self, x):
        return np.reshape(x, (x.shape[0], -1))
    
    def _normalize(self, x):
        return (x.astype(np.float_) - self._mean) / self._std
    
    def fit(self, x):
        x = self._flat(x)
        self._mean = np.mean(x, axis=0, dtype=np.float64).astype(x.dtype)
        self._std = np.std(x, axis=0, dtype=np.float64).astype(x.dtype)
        x = self._normalize(x)
        try:
            # Perform dot product on GPU
            import cudarray as ca
            x_ca = ca.array(x)
            cov = np.array(ca.dot(x_ca.T, x_ca)).astype(np.float_)
        except:
            cov = np.dot(x.T, x)
        cov = cov / x.shape[0] + self.bias * np.identity(x.shape[1])
        s, v = np.linalg.eigh(cov)
        s = np.diag(1.0 / np.sqrt(s + 0.0001))
        self.whitener = np.dot(np.dot(v, s), v.T)
        return self

    def transform(self, x):
        shape = x.shape
        x = self._flat(x)
        x = self._normalize(x)
        x_white = np.dot(x, self.whitener.T)
        return np.reshape(x_white, shape)

methods = [
   ('zca', ZCA(bias=1))]
       
for name, method in methods:
    method.fit(im)
    x_white = im
    x_white = method.transform(x_white)
    if x_white.ndim == 4:
        x_white = np.transpose(x_white, (0, 2, 3, 1))
    plt.imshow(x_white, cmap="gray")
    
    plt.show()
    plt.imshow(im,  cmap="gray")
    plt.show()
