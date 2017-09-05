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
input2_var = T.ftensor3('X_pos')
target_var = T.ivector('Y_train')
x1 = T.matrix('x1')
x2 = T.matrix('x2')
PS = 29

# Build Neural Network:
# Conv Net XY Plane
input = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv2DDNNLayer(input, 20, (9,9))

l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_1, (3,3))

l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_1, 20,(5,5))
l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_2, 20, (3,3))


l_hidden_1 = lasagne.layers.DenseLayer(l_conv_3, num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Spatial Prior FFNet
input2 = lasagne.layers.InputLayer((None, 1, 3), input_var=input2_var)

l_hidden_spa = lasagne.layers.DenseLayer(input2, num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Concatenation Layer
Merged = lasagne.layers.ConcatLayer([l_hidden_1, l_hidden_spa])


#Output Layer
output = lasagne.layers.DenseLayer(Merged, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

## Train
train_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2} , deterministic=False)

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

f_train = theano.function([input_var, input2_var,  target_var], [cost], updates=updates)

f_vali = theano.function([input_var, input2_var, target_var], [costV])

import Evaluation as E
import try_DP as TD
import scipy.io as io


with np.load("/home/xvt131/Network_adapt/planarWI_Params_class.npz") as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(output, param_values)

for img in DP.get_paths("/home/xvt131/Functions/Adhish_copy/Validating-Rand"):
    A, B, C = TD.Tri_Image_Load(img)
    B1 = B.reshape(np.prod(B.shape))
    batch = 100
    num_batches = A.shape[0] / batch
    Sha = B.shape
    print Sha
    TibiaD = []
    FemoralD = []
    preds = np.zeros(shape = ( len(B1), 2 ))
    for i in range(num_batches):
        idx = range(i*batch, (i+1)*batch)
        K = A[idx]
        M, N, O , P= TD.Patch_gen(K, 29, C)
        preds[idx] = f_eval(N,P)
    Final_pred = np.argmax(preds, axis = -1)
    print Final_pred.shape
    Lab = B1.reshape(Sha)
    Final_pred = Final_pred.reshape(Sha)

    TibiaD += [E.Dice_score(Final_pred, Lab, 1)]
    print TibiaD
    io.savemat("/home/xvt131/Network_adapt/EvalPlan/Seg_%s" %(img[-36:-30]), mdict= {"Seg2_%s" %(img[-36:-30]):Final_pred,"Lab_%s" %(img[-36:-30]):Lab} )
