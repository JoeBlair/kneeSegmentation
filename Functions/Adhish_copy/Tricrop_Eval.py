import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import *
from lasagne.updates import *
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import downsample
import lasagne
import numpy as np
import BBox as DP
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
NC = 2
# Build Neural Network:
# Conv Net XY Plane
input = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv2DDNNLayer(input, 20, (5,5))


l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_1, 40,(5,5))


l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_2, 80, (5,5))
l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_3, 80, (5,5))
l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_3, 80, (5,5))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_3, num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Conv Net patch si
input2 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input2_var)

l_conv_11 = lasagne.layers.dnn.Conv2DDNNLayer(input2, 20, (5,5))


l_conv_22 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_11, 40, (5,5))


l_conv_33 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_22, 80, (5,5))
l_conv_33 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_33, 80, (5,5))
l_conv_33 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_33, 80, (5,5))

l_hidden2 = lasagne.layers.DenseLayer(l_conv_33, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# conv patch yz
input3 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input3_var)

l_conv_111 = lasagne.layers.dnn.Conv2DDNNLayer(input3, 20, (5,5))


l_conv_222 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_111, 40,(5,5))

l_conv_333 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_222, 80, (5,5))
l_conv_333 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_333, 80, (5,5))
l_conv_333 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_333, 80, (5,5))


l_hidden3 = lasagne.layers.DenseLayer(l_conv_333, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))
 


# Concatenation Layer
Merged = lasagne.layers.ConcatLayer([l_hidden1, l_hidden2, l_hidden3])

#Output Layer
output = lasagne.layers.DenseLayer(Merged, num_units=2, nonlinearity = lasagne.nonlinearities.sigmoid)
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
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.0001, momentum=0.8)

f_eval = theano.function([input_var, input2_var, input3_var], eval_out)



import Evaluation as E

import DP1 as TD
import scipy.io as io

all_dice = np.array([])

with np.load('/home/xvt131/Functions/Adhish_copy/Exp101/light.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(output, param_values)
Dice = []
for img in DP.get_paths("/home/xvt131/Functions/Adhish_copy/Testing-Rand"):

    A, B, C = TD.get_indeces(img)
    B1 = B.reshape(np.prod(B.shape))
    batch = 12000
    num_batches = A.shape[0] / batch
    Sha = B.shape
    
    
    preds = np.zeros(shape = ( len(B1), 2 ))
    for i in range(num_batches):
        idx = range(i*batch, (i+1)*batch)
        K = A[idx]
        M, N, O = TD.Patch_gen(K, PS, C)
        preds[idx] = f_eval(M,N,O)
    if num_batches*batch < A.shape[0]:
        tot = num_batches*batch
        K = A[tot:]
        M, N, O = TD.Patch_gen(K, PS, C)
        preds[tot:A.shape[0]] = f_eval(M,N,O)
   
    P = np.argmax(preds, axis = -1)
    MM = np.ravel_multi_index(A.T, np.asarray(B.shape))
    Final_pred = np.zeros(B1.shape)
    Final_pred[MM] = P

    Lab = B1.reshape(Sha)
    Segs = Final_pred.reshape(Sha)

    Dice = np.append(Dice,  [E.Dice_score(Segs, Lab, 1)])
    print Dice

    io.savemat("/home/xvt131/Functions/Adhish_copy/testsmall/%s" %(img[51:]), mdict= {"Seg":Segs,"Lab":Lab, "Shape":Sha} )
