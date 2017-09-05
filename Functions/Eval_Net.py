import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import sgd
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import downsample
import lasagne
import numpy as np
import Data_Process as DP
from theano.tensor import nnet
import lasagne.layers.dnn
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt

dtensor5 = TensorType('float32', (False,)*5)
input_var = dtensor5('X_Train')
input2_var = T.ftensor3('X_pos')
input3_var = dtensor5('X_Bigger')
target_var = T.ivector('Y_train')
x1 = T.matrix('x1')
x2 = T.matrix('x2')
x3 = T.matrix('x3')

# Build Neural Network:
# Conv Net patch size 9
input = lasagne.layers.InputLayer((None, 1, 9,9,9), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv3DDNNLayer(input, 50, (2,2,2))

#l_maxpool_1 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_1, (2,2,2))

l_conv_2 = lasagne.layers.dnn.Conv3DDNNLayer(l_conv_1, 50, (2,2,2))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (2,2,2))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_2, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Conv Net patch size 15
input3 = lasagne.layers.InputLayer((None, 1, 5,5,5), input_var=input3_var)

l_conv_11 = lasagne.layers.dnn.Conv3DDNNLayer(input3, 50, (2,2,2))

#l_maxpool_1 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_1, (2,2,2))

#l_conv_22 = lasagne.layers.dnn.Conv3DDNNLayer(l_conv_11, 20, (2,2,2))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))

l_hidden2 = lasagne.layers.DenseLayer(l_conv_11, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Spatial Prior FFNet
input2 = lasagne.layers.InputLayer((None, 1, 3), input_var=input2_var)

l_hidden_spa = lasagne.layers.DenseLayer(input2, num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Concatenation Layer
Merged = lasagne.layers.ConcatLayer([l_hidden_spa, l_hidden2, l_hidden1])

l_dropout = lasagne.layers.DropoutLayer(Merged, p=0.5)

#Output Layer
output = lasagne.layers.DenseLayer(l_dropout, num_units=3, nonlinearity = lasagne.nonlinearities.softmax)

## Train
train_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input3_var:x3}, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input2_var:x3 }, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(train_out, target_var).mean()

all_grads = T.grad(cost, all_params)


# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.001, momentum=0.8)

f_eval = theano.function([input_var, input2_var, input3_var], eval_out)

f_train = theano.function([input_var, input2_var, input3_var, target_var], [cost], updates=updates)


import Evaluation as E

with np.load('/home/xvt131/Functions/epoch_31_params.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(output, param_values)



Tibial_Score, Femoral_Score = E.Evaluate1("/home/xvt131/Running/train2", DP.image_load, 9, 5 ,f_eval)
#E.Segment("/home/xvt131/Running/evaluation", DP.image_load_eval, 9, 5 ,f_eval)


#for param in DP.get_paths('/home/xvt131/Functions/params_101'):

#    with np.load(param) as f:
#        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
#    lasagne.layers.set_all_param_values(output, param_values)

#    Tibial_Score, Femoral_Score = E.Evaluate1("/home/xvt131/Running/testing", DP.image_load, 9, 5 ,f_eval)
#    print param
#    print "Mean Tibia Dice Score:" , np.mean(Tibial_Score)
#    print "Mean Femur Dice Score:", np.mean(Femoral_Score)


print "Mean Tibia Dice Score:" , np.mean(Tibial_Score)
print "Mean Femur Dice Score:", np.mean(Femoral_Score)

#np.savez('model_T.npz', *lasagne.layers.get_all_param_values(output))

