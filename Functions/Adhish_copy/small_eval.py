import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import sgd
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import pool
import lasagne
import numpy as np
import try_DP as DP
from theano.tensor import nnet
import lasagne.layers.dnn

dtensor5 = TensorType('float32', (False,)*5)
input_var = dtensor5('X_Train')
input2_var = dtensor5('X_Train2')
target_var = T.ivector('Y_train')
x1 = T.matrix('x1')
x2 = T.matrix('x2')

NC = 2
PS = 9
PS2 = 5

# Build Neural Network:
# Conv Net patch size 9
input = lasagne.layers.InputLayer((None, 1, PS, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv3DDNNLayer(input, 50, (2,2,2))

#l_maxpool_1 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_1, (2,2,2))

l_conv_2 = lasagne.layers.dnn.Conv3DDNNLayer(l_conv_1, 50, (2,2,2))



l_hidden1 = lasagne.layers.DenseLayer(l_conv_2, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

input = lasagne.layers.InputLayer((None, 1, PS2, PS2, PS2), input_var=input2_var)

l_conv_11 = lasagne.layers.dnn.Conv3DDNNLayer(input, 20, (2,2,2))

#l_conv_22 = lasagne.layers.dnn.Conv3DDNNLayer(l_conv_11, 40, (3,3,3))


l_hidden11 = lasagne.layers.DenseLayer(l_conv_11, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

Merged = lasagne.layers.ConcatLayer([ l_hidden1, l_hidden11])
output = lasagne.layers.DenseLayer(Merged, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

 
## Train


## Train
train_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2}, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2}, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(train_out, target_var).mean()

costV = T.nnet.categorical_crossentropy(eval_out, target_var).mean()


all_grads = T.grad(cost, all_params)


# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.000001, momentum=0.9)

f_eval = theano.function([input_var, input2_var], eval_out)
f_vali = theano.function([input_var,input2_var, target_var], [costV])

f_train = theano.function([input_var,input2_var, target_var], [cost], updates=updates)

import Evaluation as E
import DP1 as TD
import scipy.io as io

with np.load("/home/xvt131/Biomediq/Functions/Adhish_copy/Exp101/3dpop.npz") as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(output, param_values)
Dice = []
for img in DP.get_paths("/home/xvt131/Biomediq/Data/kneeData/Validating-Rand"):

    A, B, C = TD.get_indeces(img)
    B1 = B.reshape(np.prod(B.shape))
    batch = 10000
    num_batches = A.shape[0] / batch
    Sha = B.shape
    print Sha
    preds = np.zeros(shape = (A.shape[0], NC ))
    for i in range(num_batches):
        idx = range(i*batch, (i+1)*batch)
        K = A[idx]
        M, N = TD.Patch_gen_three(K,PS,PS2, C)
        preds[idx] = f_eval(M, N)
    if num_batches*batch < A.shape[0]:
        tot = num_batches*batch
        K = A[tot:]
        M, N = TD.Patch_gen_three(K, PS,PS2, C)
        preds[tot:A.shape[0]] = f_eval(M, N)

    P = np.argmax(preds, axis = -1)
    MM = np.ravel_multi_index(A.T, np.asarray(B.shape))
    Final_pred = np.zeros(B1.shape)
    Final_pred[MM] = P
    Lab = B1.reshape(Sha)
    Segs = Final_pred.reshape(Sha)

    Dice = np.append(Dice,  [E.Dice_score(Segs, Lab, 1)])
    print Dice
    io.savemat("/home/xvt131/Biomediq/Results/valitest/%s" %(img[51:]), mdict= {"Seg":Segs,"Lab":Lab} )


