import nibabel as nib
import numpy as np
from scipy.ndimage import morphology
import os
from joblib import Parallel, delayed
import dpBrain as TD
import scipy.io as io


def get_paths(path):
    path = path
    listing = os.listdir(path)
    paths = np.empty([0])

    for i in listing:
        paths = np.hstack((paths, str(path + '/' + i)))

    return paths

def get_masks(path, path_L, path_R):

    lPath = path_L +"/" + path[-24:]
    rPath = path_R + "/" + path[-24:]

    leftMask = nib.load(lPath).get_data()
    rightMask = nib.load(rPath).get_data()
    mask = (leftMask) + rightMask
    return mask

Test = TD.get_paths("/home/xvt131/Biomediq/Data/adni/vali_mri")
Left = "/home/xvt131/Biomediq/Data/adni/vali_leftH"
Right = "/home/xvt131/Biomediq/Data/adni/vali_rightH"
for img in Test:
    Label = get_masks(img, Left, Right)
    Scan = nib.load(img).get_data()
    io.savemat("/home/xvt131/Biomediq/Data/adni/valiMat/%s" %(img[-24:-4]), mdict= {"Scan":Scan,"Label":Label} )
