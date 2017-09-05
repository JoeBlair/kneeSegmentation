import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import *
from lasagne.updates import *
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import downsample
import lasagne
import numpy as np
import DP1 as DP
from theano.tensor import nnet
import lasagne.layers.dnn
import os
dtensor5 = TensorType('float32', (False,)*5)
input_var = T.ftensor4('XY')
input2_var = T.ftensor4('XZ')
input3_var = T.ftensor4('YZ')
target_var = T.matrix('Y_train')
x1 = T.matrix('x1')
x2 = T.matrix('x2')
x3 = T.matrix('x3')
PS = 29
NC = 1
# Build Neural Network:
# Conv Net XY Plane


input = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv2DDNNLayer(input, 28, (5,5))

l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_1, (2,2))

l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_1, 56,(5,5))

l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_2, 112, (5,5))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_3, num_units=1792, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Conv Net patch si
input2 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input2_var)

l_conv_11 = lasagne.layers.dnn.Conv2DDNNLayer(input2, 28, (5,5))

l_maxpool_11 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_11, (2,2))

l_conv_22 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_11, 56, (5,5))
l_conv_33 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_22, 112, (5,5))

l_hidden2 = lasagne.layers.DenseLayer(l_conv_33, num_units=1792,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# conv patch yz
input3 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input3_var)

l_conv_111 = lasagne.layers.dnn.Conv2DDNNLayer(input3, 28, (5,5))

l_maxpool_111= lasagne.layers.dnn.Pool2DDNNLayer(l_conv_111, (2,2))

l_conv_222 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_111, 56,(5,5))
l_conv_333 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_222, 112, (5,5))


l_hidden3 = lasagne.layers.DenseLayer(l_conv_333, num_units=1792,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Concatenation Layer
Merged = lasagne.layers.ConcatLayer([l_hidden1, l_hidden2, l_hidden3])

#Dropout = lasagne.layers.DropoutLayer(Merged, p=0.5)
#Output Layer
output = lasagne.layers.DenseLayer(Merged, num_units=1, nonlinearity = lasagne.nonlinearities.sigmoid)

## Train
train_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input3_var:x3}, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input2_var:x3 }, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.binary_crossentropy(train_out, target_var).mean()
costV = T.nnet.binary_crossentropy(eval_out, target_var).mean()

all_grads = T.grad(cost, all_params)

# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.adam(all_grads, all_params,learning_rate=0.00001, beta1=0.9, beta2=0.999, epsilon=1e-08)

f_eval = theano.function([input_var, input2_var, input3_var], eval_out)

f_train = theano.function([input_var, input2_var, input3_var, target_var], [cost], updates=updates)

f_vali = theano.function([input_var, input2_var, input3_var, target_var], [costV])


import Evaluation as E

import DP1 as TD
import scipy.io as io

all_dice = np.array([])

with np.load('/home/xvt131/Functions/Adhish_copy/Exp101/AD.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(output, param_values)
Dice = []
for img in DP.get_paths("/home/xvt131/Functions/Adhish_copy/Validating-Rand"):

    A, B, C = TD.get_indeces(img)
    B1 = B.reshape(np.prod(B.shape))
    batch = 12000
    num_batches = A.shape[0] / batch
    Sha = B.shape

    preds = np.zeros(shape = (A.shape[0], NC ))
    for i in range(num_batches):
        idx = range(i*batch, (i+1)*batch)
        K = A[idx]
        M, N, O = TD.Patch_gen(K, 29, C)
        preds[idx] = f_eval(M,N,O)
    if num_batches*batch < A.shape[0]:
        tot = num_batches*batch
        K = A[tot:]
        M, N, O = TD.Patch_gen(K, 29, C)
        preds[tot:A.shape[0]] = f_eval(M,N,O)

    MM = np.ravel_multi_index(A.T, np.asarray(B.shape))
    Final_pred = np.zeros(B1.shape)
    Final_pred[MM] = preds
    Lab = B1.reshape(Sha)
    Segs = Final_pred.reshape(B.shape)

    Dice = np.append(Dice,  [E.Dice_score(Segs, Lab, 1)])
    print Dice

    all_dice = np.append(all_dice, Dice)
    io.savemat("/home/xvt131/Functions/Adhish_copy/ADC/%s" %(img[51:]), mdict= {"Seg":Segs,"Lab":Lab} )
