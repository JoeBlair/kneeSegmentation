import theano
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import nesterov_momentum
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import downsample
import lasagne
import numpy as np
from theano.tensor import nnet
import lasagne.layers.dnn


def Conv_Layer_Long(input_var, Patch_Size, FS1, nodes, FS2, units):
    """
    :param input_var: Patches for convolution
    :param Patch_Size: size of patch (integer value)
    :param FS1: Filter size for first Conv layer
    :param nodes: number of nodes/ filters to be used
    :param FS2: Filter size for second Conv layer
    :return: hidden layer of after 2 convolutional layers
    """

    input = lasagne.layers.InputLayer((None, 1, Patch_Size, Patch_Size, Patch_Size), input_var=input_var)

    l_conv_1 = lasagne.layers.dnn.Conv3DDNNLayer(input, nodes, (FS1, FS1, FS1))

    l_conv_2 = lasagne.layers.dnn.Conv3DDNNLayer(l_conv_1, nodes, (FS2, FS2, FS2))

    l_hidden = lasagne.layers.DenseLayer(l_conv_2, num_units=units,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

    return l_hidden


def Conv_Layer_Short(input_var, Patch_Size, FS, nodes, units):
    """
    :param input_var: Patches for convolution
    :param Patch_Size: size of patch (integer value)
    :param FS1: Filter size for  Conv layer
    :param nodes: number of nodes/ filters to be used
    :return: hidden layer of after convolutional layer
    """

    input = lasagne.layers.InputLayer((None, 1, Patch_Size, Patch_Size, Patch_Size), input_var=input_var)

    l_conv = lasagne.layers.dnn.Conv3DDNNLayer(input, nodes, (FS, FS, FS))

    l_hidden = lasagne.layers.DenseLayer(l_conv, num_units=units,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

    return l_hidden


def Spatial_Layer(input_var, num_units):
   """
    :param input_var: coordinates of voxel to be classified
    :param num_units: number of units in hidden layer. NB: This must be the same as the number of units in the other layers.
    :return: hidden layer output
    """
    input2 = lasagne.layers.InputLayer((None, 1, 3), input_var=input_var)

    l_hidden = lasagne.layers.DenseLayer(input2, num_units=num_units, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

    return l_hidden

