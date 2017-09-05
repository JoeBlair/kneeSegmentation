

import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import *
from lasagne.updates import *
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import downsample
import lasagne
import numpy as np
import try_DP as DP
from theano.tensor import nnet
import lasagne.layers.dnn

dtensor5 = TensorType('float32', (False,)*5)
input_var = T.ftensor4('XY')
input2_var = T.ftensor4('XZ')
input3_var = T.ftensor4('YZ')
input4_var = T.ftensor4('sXY')
input5_var = T.ftensor4('sXZ')
input6_var = T.ftensor4('sYZ')
target_var = T.matrix('Y_train')
x1 = T.matrix('x1')
x2 = T.matrix('x2')
x3 = T.matrix('x3')
x4 = T.matrix('x4')
x5 = T.matrix('x5')
x6 = T.matrix('x6')

PS = 29
NC = 1
# Build Neural Network:
# Conv Net XY Plane
input = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv2DDNNLayer(input, 20, (9,9))

l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_1, (3,3))

l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_1, 40,(5,5))

l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_2, 80, (3,3))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (2,2,2))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_3, num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Conv Net patch si
input2 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input2_var)

l_conv_11 = lasagne.layers.dnn.Conv2DDNNLayer(input2, 20, (9,9))

l_maxpool_11 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_11, (3,3))

l_conv_22 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_11, 40, (5,5))
l_conv_33 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_22, 80, (3,3))

l_hidden2 = lasagne.layers.DenseLayer(l_conv_33, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Spatial Prior FFNet
input3 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input3_var)

l_conv_111 = lasagne.layers.dnn.Conv2DDNNLayer(input3, 20, (9,9))

l_maxpool_111= lasagne.layers.dnn.Pool2DDNNLayer(l_conv_111, (3,3))

l_conv_222 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_111, 20,(5,5))
l_conv_333 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_222, 20, (3,3))
#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))




l_hidden3 = lasagne.layers.DenseLayer(l_conv_333, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


input4 = lasagne.layers.InputLayer((None, 1, 9,9), input_var=input4_var)

l_conv_4 = lasagne.layers.dnn.Conv2DDNNLayer(input4, 20, (3,3))

l_hidden_4 = lasagne.layers.DenseLayer(l_conv_4, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

input5 = lasagne.layers.InputLayer((None, 1, 9,9), input_var=input5_var)

l_conv_5 = lasagne.layers.dnn.Conv2DDNNLayer(input5, 20, (3,3))

l_hidden_5 = lasagne.layers.DenseLayer(l_conv_5, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

input6 = lasagne.layers.InputLayer((None, 1, 9,9), input_var=input6_var)

l_conv_6 = lasagne.layers.dnn.Conv2DDNNLayer(input6, 20, (3,3))

l_hidden_6 = lasagne.layers.DenseLayer(l_conv_6, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))



# Concatenation Layer
Merged = lasagne.layers.ConcatLayer([l_hidden1, l_hidden2, l_hidden3, l_hidden_4, l_hidden_5, l_hidden_6])


#Output Layer
output = lasagne.layers.DenseLayer(Merged, num_units=1, nonlinearity = lasagne.nonlinearities.sigmoid)


## Train
train_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input3_var:x3, input4_var:x4, input5_var:x5, input6_var:x6}, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input3_var:x3, input4_var:x4, input5_var:x5, input6_var:x6 }, deterministic=True)

all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(train_out, target_var).mean()
costV = T.nnet.categorical_crossentropy(eval_out, target_var).mean()

all_grads = T.grad(cost, all_params)

# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.0001, momentum=0.8)


f_eval = theano.function([input_var, input2_var, input3_var, input4_var, input5_var, input6_var], eval_out)

f_train = theano.function([input_var, input2_var, input3_var, input4_var,input5_var, input6_var, target_var], [cost], updates=updates)

f_vali = theano.function([input_var, input2_var, input3_var, input4_var,input5_var, input6_var, target_var], [costV])


import Evaluation as E

import try_DP as TD
import scipy.io as io

all_dice = np.array([])

with np.load('/home/xvt131/Functions/Adhish_copy/TP_param/triplanar_Paramsmix.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(output, param_values)
Dice = []
for img in DP.get_paths("/home/xvt131/Functions/Adhish_copy/Validating-Rand"):

    A, B, C = TD.Tri_Image_Load(img)
    B1 = B.reshape(np.prod(B.shape))
    batch = 12000
    num_batches = A.shape[0] / batch
    Sha = B.shape
    
    
    preds = np.zeros(shape = ( len(B1), NC ))
    for i in range(num_batches):
        idx = range(i*batch, (i+1)*batch)
        K = A[idx]
        M, N, O , P, Q, R= TD.Patch_gen_mix(K, 29, C)
        preds[idx] = f_eval(M,N,O, P, Q, R)
    if num_batches*batch < A.shape[0]:
        tot = num_batches*batch
        K = A[tot:]
        M, N, O,P, Q, R  = TD.Patch_gen_mix(K, 29, C)
        preds[tot:A.shape[0]] = f_eval(M,N,O, P, Q, R)


    Lab = B1.reshape(Sha)
    Final_pred =preds.reshape(Sha)

    Dice = np.append(Dice,  [E.Dice_score(Final_pred, Lab, 1)])

    io.savemat("/home/xvt131/Functions/Adhish_copy/mix/%s" %(img[51:]), mdict= {"Seg":Final_pred,"Lab":Lab} )
