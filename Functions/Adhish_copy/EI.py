import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import *
from lasagne.updates import *
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import downsample
import lasagne
import numpy as np
import BBox as TD
from theano.tensor import nnet
import lasagne.layers.dnn
PS = 3
classes = 2
input2_var = T.ftensor3('X_pos')
target_var = T.ivector('Y_train')
x2 = T.matrix('x2')

# Spatial Prior FFNet
input2 = lasagne.layers.InputLayer((None, 1, 3), input_var=input2_var)



l_hidden_spa = lasagne.layers.DenseLayer(input2, num_units=16, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

l_hidden_spa2 = lasagne.layers.DenseLayer(l_hidden_spa, num_units=32, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

l_hidden_spa3 = lasagne.layers.DenseLayer(l_hidden_spa2, num_units=64, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))




#Output Layer
output = lasagne.layers.DenseLayer(l_hidden_spa3, num_units=classes, nonlinearity = lasagne.nonlinearities.softmax)


## Train
train_out = lasagne.layers.get_output(output, { input2_var:x2} , deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input2_var:x2}, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(train_out, target_var).mean()
costV = T.nnet.categorical_crossentropy(eval_out, target_var).mean()

all_grads = T.grad(cost, all_params)

# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.sgd(all_grads, all_params, learning_rate=0.001)

f_eval = theano.function([ input2_var], eval_out)

f_train = theano.function([input2_var,  target_var], [cost], updates=updates)

f_vali = theano.function([ input2_var, target_var], [costV])

import Evaluation as E
import scipy.io as io


with np.load("/home/xvt131/Network_adapt/WI_Params.npz") as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(output, param_values)

for img in TD.get_paths("/home/xvt131/Functions/Adhish_copy/Validating-Rand"):
    A, B, C, D = TD.Tri_Image_Load(img)
    B1 = B.reshape(np.prod(B.shape))
    batch = 10000
    num_batches = A.shape[0] / batch
    Sha = B.shape
    print Sha
    TibiaD = []
    FemoralD = []
    preds = np.zeros(shape = ( len(B1), 2 ))
    for i in range(num_batches):
        idx = range(i*batch, (i+1)*batch)
        K = D[idx]
        preds[idx] = f_eval(K)
    print Final_pred.shape
    Lab = B1.reshape(Sha)
    Final_pred = Final_pred.reshape(Sha)

    io.savemat("/home/xvt131/Network_adapt/Seg_%s" %(img[-36:-30]), mdict= {"Seg_%s" %(img[-36:-30]):Final_pred,"Lab_%s" %(img[-36:-30]):Lab} )
