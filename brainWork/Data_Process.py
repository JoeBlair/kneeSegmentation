import scipy.io as io
import numpy as np
import os
from skimage.util.shape import view_as_windows
from scipy.ndimage import morphology

def get_paths(path):

    path = path
    listing = os.listdir(path)
    paths = np.empty([0])

    for i in listing:

        paths = np.hstack((paths, str(path + '/' + i)))

    return paths



def voxel_samples(image_path, patch_sizes):
    """
    :param image_path: path to image generated by get_paths
    :param patch_sizes: a list or array of patch sizes desired for convolution. Must be odd > 1.
    :param total_sample_size: sampled patches for convolution, and corresponding labels
    :return: sampled patches for convolution, and corresponding labels
    """

    # Load image from path
    Image = io.loadmat(image_path)
    # Extract total number of voxels
    Dims = np.prod(Image['scan'].shape)
    # Create 1 mask image
    Label = (Image['Tibia'] + 2*Image['Femur'])
    np.place(Label, Label>2, [2,1])

    if Image['isright'] == 0:

        Scan = np.fliplr(Image["scan"])
        Label = np.fliplr(Label)

    else: Scan = Image["scan"]

    Label = Label.reshape(Dims)

    
    # Find indexes of voxels to sample
    N_samp = 7500

    
    Gold_ind = np.random.choice(np.squeeze(np.where(Label == 1)), N_samp, replace=False)
    Silv_ind = np.random.choice(np.squeeze(np.where(Label == 2)), N_samp, replace=False)
    Neg_Samp_ind = np.random.choice(np.squeeze(np.where(Label == 0)), 3*N_samp, replace=False)
   
    Index = np.hstack((Gold_ind, Silv_ind, Neg_Samp_ind))

    np.random.shuffle(Index)

    Patch_List = []

    # for each desired patch size, create patches
    for size in patch_sizes:
        PS = (size - 1)/2
        npad = ((PS, PS), (PS, PS), (PS,PS))
        window_size = (size, size, size)

        image_pad = np.pad(Scan, pad_width=npad, mode='constant', constant_values = 0)
        Patches = (view_as_windows(image_pad, window_size)).reshape((Dims, size, size, size))

        Patch_List.append(np.float32(Patches[Index, :,:,:].reshape(Index.shape[0], 1, size, size, size)))

    Inds =  np.asarray(np.unravel_index(Index, Scan.shape, order = 'C'))
    Positions = Inds.reshape(3,1, Inds.shape[1]).T
    Labels = Label[Index]
    
    Patch_List = tuple(Patch_List)

    return Patch_List, np.int32(Positions), np.int32(Labels)

def image_load(path, patch_size, patch_size2):

    PS = (patch_size - 1)/2
    npad = ((PS, PS), (PS, PS), (PS,PS))
    window_size = (patch_size, patch_size, patch_size)

    A = io.loadmat(path)
    Label = (A['Tibia'] + 2*A['Femur'])
    np.place(Label, Label>2, [2,1])

    if A['isright'] == 0:

        Scan = np.fliplr(A["scan"])
        Label = np.fliplr(Label)

    else: Scan = A["scan"]    


    PS2 = (patch_size2 - 1) / 2
    npad2 = ((PS2, PS2), (PS2, PS2), (PS2,PS2))
    window_size2 = (patch_size2, patch_size2, patch_size2)
    Nd = np.prod(Scan.shape)
    Label = Label.reshape(Nd)
    image_pad2 = np.pad(Scan, pad_width=npad2, mode='constant', constant_values = 0)
    Patches2 = (view_as_windows(image_pad2, window_size2)).reshape((Nd, patch_size2, patch_size2, patch_size2))

    image_pad = np.pad(Scan, pad_width=npad, mode='constant', constant_values = 0)

    Patches = (view_as_windows(image_pad, window_size)).reshape((Nd, patch_size, patch_size, patch_size))
    
    LN = np.squeeze(np.where(Label >=0))
    N = Scan.shape
    Pos = np.asarray(np.unravel_index(LN, Scan.shape, order = 'C'))
    
    Post = Pos.reshape(3, 1, Pos.shape[1]).T
    return  np.int32(Label).reshape(Nd), np.float32(Post) , np.float32(Patches).reshape(Nd, 1, patch_size, patch_size, patch_size), np.float32(Patches2).reshape(Nd, 1, patch_size2, patch_size2, patch_size2), N, Scan


def voxel_samples2(image_path, patch_sizes):
    """
    :param image_path: path to image generated by get_paths
    :param patch_sizes: a list or array of patch sizes desired for convolution. Must be odd > 1.
    :param total_sample_size: sampled patches for convolution, and corresponding labels
    :return: sampled patches for convolution, and corresponding labels
    """

    # Load image from path
    Image = io.loadmat(image_path)
    # Extract total number of voxels
    Dims = np.prod(Image['scan'].shape)
    # Create 1 mask image
    Label = (Image['Tibia'] + 2*Image['Femur'])
    np.place(Label, Label>2, [2,1])

    if Image['isright'] == 0:

        Scan = np.fliplr(Image["scan"])
        Label = np.fliplr(Label)

    else: Scan = Image["scan"]
    
    Index = np.array([])
    Label2 = (Image['Tibia'] + Image['Femur'])

    B = morphology.distance_transform_cdt(np.absolute(Label2-1))
    B = B.reshape(Dims)
    X =  np.float32(range(np.max(B), 3, -1))/range(4, np.max(B)+1, 1)
    #print max(X)
    #print np.unique(B)
    #print min(X)
    No_samples = 18000
    Index = np.array([])
    samples = np.divide(X, sum(X))*No_samples
    #print len(samples)
    for i in range(4, np.max(B)+1):
      #  print i
        Index = np.int32(np.append(Index, np.random.choice(np.squeeze(np.where(B == i)),(samples[i-4]), replace=False)))
    
    Label = Label.reshape(Dims)


    # Find indexes of voxels to sample
    Gold_ind = np.squeeze(np.where(Label == 1))
    Silv_ind = np.squeeze(np.where(Label == 2))
    Edge_ind = np.squeeze(np.where((B>0) & (B<=3)))
 #   print Edge_ind.shape
    #Neg_Samp_ind = np.random.choice(np.squeeze(np.where(Label == 0)), 3*N_samp, replace=False)

    Index = np.hstack((Gold_ind, Silv_ind, Edge_ind, Index))

    np.random.shuffle(Index)

    Patch_List = []

    # for each desired patch size, create patches
    for size in patch_sizes:
        PS = (size - 1)/2
        npad = ((PS, PS), (PS, PS), (PS,PS))
        window_size = (size, size, size)

        image_pad = np.pad(Scan, pad_width=npad, mode='constant', constant_values = 0)
        Patches = (view_as_windows(image_pad, window_size)).reshape((Dims, size, size, size))

        Patch_List.append(np.float32(Patches[Index, :,:,:].reshape(Index.shape[0], 1, size, size, size)))

    Inds =  np.asarray(np.unravel_index(Index, Scan.shape, order = 'C'))
    Positions = Inds.reshape(3,1, Inds.shape[1]).T
    Labels = Label[Index]

    Patch_List = tuple(Patch_List)

    return Patch_List, np.float32(Positions), np.int32(Labels)


def voxel_samples3(image_path, patch_sizes):
    """
    :param image_path: path to image generated by get_paths
    :param patch_sizes: a list or array of patch sizes desired for convolution. Must be odd > 1.
    :param total_sample_size: sampled patches for convolution, and corresponding labels
    :return: sampled patches for convolution, and corresponding labels
    """

    # Load image from path
    Image = io.loadmat(image_path)
    # Extract total number of voxels
    Dims = np.prod(Image['scan'].shape)
    # Create 1 mask image
    Label = (Image['Tibia'] + 2*Image['Femur'])
    np.place(Label, Label>2, [2,1])

    if Image['isright'] == 0:

        Scan = np.fliplr(Image["scan"])
        Label = np.fliplr(Label)

    else: Scan = Image["scan"]

    Index = np.array([])
    Label2 = (Image['Tibia'] + Image['Femur'])

    B = morphology.distance_transform_cdt(np.absolute(Label2-1))
    B = B.reshape(Dims)
    X =  np.float32(range(np.max(B), 0, -1))/range(1, np.max(B)+1, 1)
    #print max(X)
    #print np.unique(B)
    #print min(X)
    No_samples = 36000
    Index = np.array([])
    samples = np.divide(X, sum(X))*No_samples
    #print len(samples)
    for i in range(1, np.max(B)+1):
      #  print i
        Index = np.int32(np.append(Index, np.random.choice(np.squeeze(np.where(B == i)),(samples[i-1]), replace=False)))

    Label = Label.reshape(Dims)


    # Find indexes of voxels to sample
    Gold_ind = np.random.choice(np.squeeze(np.where(Label == 1)), 12000, replace = False)
    Silv_ind = np.random.choice(np.squeeze(np.where(Label == 2)), 12000, replace = False)
  #  Edge_ind = np.squeeze(np.where((B>0) & (B<=3)))
 #   print Edge_ind.shape
    #Neg_Samp_ind = np.random.choice(np.squeeze(np.where(Label == 0)), 3*N_samp, replace=False)

    Index = np.hstack((Gold_ind, Silv_ind, Index))

    np.random.shuffle(Index)

    Patch_List = []

    # for each desired patch size, create patches
    for size in patch_sizes:
        PS = (size - 1)/2
        npad = ((PS, PS), (PS, PS), (PS,PS))
        window_size = (size, size, size)

        image_pad = np.pad(Scan, pad_width=npad, mode='constant', constant_values = 0)
        Patches = (view_as_windows(image_pad, window_size)).reshape((Dims, size, size, size))

        Patch_List.append(np.float32(Patches[Index, :,:,:].reshape(Index.shape[0], 1, size, size, size)))

    Inds =  np.asarray(np.unravel_index(Index, Scan.shape, order = 'C'))
    Positions = Inds.reshape(3,1, Inds.shape[1]).T
    Labels = Label[Index]

    Patch_List = tuple(Patch_List)

    return Patch_List, np.float32(Positions), np.int32(Labels)


def image_load_eval(path, patch_size, patch_size2):

    PS = (patch_size - 1)/2
    npad = ((PS, PS), (PS, PS), (PS,PS))
    window_size = (patch_size, patch_size, patch_size)

    A = io.loadmat(path)

    if A['isright'] == 0:

        Scan = np.fliplr(A["scan"])

    else: Scan = A["scan"]


    PS2 = (patch_size2 - 1) / 2
    npad2 = ((PS2, PS2), (PS2, PS2), (PS2,PS2))
    window_size2 = (patch_size2, patch_size2, patch_size2)
    Nd = np.prod(Scan.shape)
    image_pad2 = np.pad(Scan, pad_width=npad2, mode='constant', constant_values = 0)
    Patches2 = (view_as_windows(image_pad2, window_size2)).reshape((Nd, patch_size2, patch_size2, patch_size2))
   
    X = np.zeros((Scan.shape))

    B = X
    B = B.reshape(Nd)

    image_pad = np.pad(Scan, pad_width=npad, mode='constant', constant_values = 0)

    Patches = (view_as_windows(image_pad, window_size)).reshape((Nd, patch_size, patch_size, patch_size))
    
    LN = np.squeeze(np.where(B ==0))
    N = Scan.shape
    
    Pos = np.asarray(np.unravel_index(LN, Scan.shape, order = 'C'))
    Post = Pos.reshape(3, 1, Pos.shape[1]).T
    return  Nd, np.float32(Post) , np.float32(Patches).reshape(Nd, 1, patch_size, patch_size, patch_size), np.float32(Patches2).reshape(Nd, 1, patch_size2, patch_size2, patch_size2), N, Scan
