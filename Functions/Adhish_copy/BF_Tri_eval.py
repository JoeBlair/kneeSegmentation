import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import *
from lasagne.updates import *
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import pool
import lasagne
import numpy as np
import try_DP as DP
from theano.tensor import nnet
import lasagne.layers.dnn
import os
dtensor5 = TensorType('float32', (False,)*5)
input_var = T.ftensor4('XY')
input2_var = T.ftensor4('XZ')
input3_var = T.ftensor4('YZ')
target_var = T.ivector('Y_train')
x1 = T.matrix('x1')
x2 = T.matrix('x2')
x3 = T.matrix('x3')
PS = 29
NC = 2
# Build Neural Network:
# Conv Net XY Plane
input = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv2DDNNLayer(input, 20, (3,3),  nonlinearity=lasagne.nonlinearities.rectify)

l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_1, (2,2))

l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_1, 20,(5,5),  nonlinearity=lasagne.nonlinearities.rectify)
l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_2, 20, (9,9),  nonlinearity=lasagne.nonlinearities.rectify)

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (2,2,2))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_3, num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Conv Net patch si
input2 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input2_var)

l_conv_11 = lasagne.layers.dnn.Conv2DDNNLayer(input2, 20, (3,3),  nonlinearity=lasagne.nonlinearities.rectify)

l_maxpool_11 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_11, (2,2))

l_conv_22 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_11, 20, (5,5), nonlinearity=lasagne.nonlinearities.rectify)
l_conv_33 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_22,20, (9,9), nonlinearity=lasagne.nonlinearities.rectify)

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))

l_hidden2 = lasagne.layers.DenseLayer(l_conv_33, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Spatial Prior FFNet
input3 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input3_var)

l_conv_111 = lasagne.layers.dnn.Conv2DDNNLayer(input3, 20, (3,3), nonlinearity=lasagne.nonlinearities.rectify)

l_maxpool_111= lasagne.layers.dnn.Pool2DDNNLayer(l_conv_111, (2,2))

l_conv_222 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_111, 20,(5,5), nonlinearity=lasagne.nonlinearities.rectify)
l_conv_333 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_222, 20, (9,9), nonlinearity=lasagne.nonlinearities.rectify)
#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))

l_hidden3 = lasagne.layers.DenseLayer(l_conv_333, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Concatenation Layer
Merged = lasagne.layers.ConcatLayer([l_hidden1, l_hidden2, l_hidden3])

#l_dropout = lasagne.layers.DropoutLayer(Merged, p=0.5)

#Output Layer
output = lasagne.layers.DenseLayer(Merged, num_units=NC, nonlinearity = lasagne.nonlinearities.sigmoid)

## Train
train_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input3_var:x3}, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input2_var:x3 }, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(train_out, target_var).mean()
costV = T.nnet.categorical_crossentropy(eval_out, target_var).mean()

all_grads = T.grad(cost, all_params)

# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.00001, momentum=0.8)

f_eval = theano.function([input_var, input2_var, input3_var], eval_out)

f_train = theano.function([input_var, input2_var, input3_var, target_var], [cost], updates=updates)

f_vali = theano.function([input_var, input2_var, input3_var, target_var], [costV])

import Evaluation as E

import DP1 as TD
import scipy.io as io

all_dice = np.array([])

with np.load('/home/xvt131/Biomediq/Functions/Adhish_copy/Exp101/cheat.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(output, param_values)
Dice = []
for img in DP.get_paths("/home/xvt131/Biomediq/Data/kneeData/Validating-Rand"):

    A, B, C = TD.get_indeces(img)
    B1 = B.reshape(np.prod(B.shape))
    batch = 5000
    num_batches = A.shape[0] / batch
    Sha = B.shape
    
    
    preds = np.zeros(shape = (len(B1), NC ))
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

    P = np.argmax(preds, axis = -1)
    MM = np.ravel_multi_index(A.T, np.asarray(B.shape))
    print MM.shape
    print P.shape
    Final_pred = np.zeros(B1.shape)
    Final_pred[MM] = P
    
    Lab = B1.reshape(Sha)
    Segs = Final_pred.reshape(Sha)

    Dice = np.append(Dice,  [E.Dice_score(Segs, Lab, 1)])
    print Dice

    io.savemat("/home/xvt131/Biomediq/Results/valitest/%s" %(img[-32:-4]), mdict= {"Seg":Segs,"Lab":Lab} )

