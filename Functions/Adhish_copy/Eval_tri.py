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
target_var = T.ivector('Y_train')
x1 = T.matrix('x1')
x2 = T.matrix('x2')
x3 = T.matrix('x3')
PS = 29

# Build Neural Network:
# Conv Net XY Plane
input = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv2DDNNLayer(input, 20, (5,5))

l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_1, (5,5))

l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_1, 20,(5,5))
#l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_2, 20, (2,2))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (2,2,2))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_2, num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Conv Net patch si
input2 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input2_var)

l_conv_11 = lasagne.layers.dnn.Conv2DDNNLayer(input2, 20, (5,5))

l_maxpool_11 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_11, (5,5))

l_conv_22 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_11, 20, (5,5))
#l_conv_33 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_22, 20, (2,2))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))

l_hidden2 = lasagne.layers.DenseLayer(l_conv_22, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Spatial Prior FFNet
input3 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input3_var)

l_conv_111 = lasagne.layers.dnn.Conv2DDNNLayer(input3, 20, (5,5))

l_maxpool_111= lasagne.layers.dnn.Pool2DDNNLayer(l_conv_111, (5,5))

l_conv_222 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_111, 20,(5,5))
#l_conv_333 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_222, 20, (2,2))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))

l_hidden3 = lasagne.layers.DenseLayer(l_conv_222, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Concatenation Layer
Merged = lasagne.layers.ConcatLayer([l_hidden1, l_hidden2, l_hidden3])

#l_dropout = lasagne.layers.DropoutLayer(Merged, p=0.5)

#Output Layer
output = lasagne.layers.DenseLayer(Merged, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

## Train
train_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input3_var:x3}, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input2_var:x3 }, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(train_out, target_var).mean()

all_grads = T.grad(cost, all_params)

# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.00001, momentum=0.8)

f_eval = theano.function([input_var, input2_var, input3_var], eval_out)

f_train = theano.function([input_var, input2_var, input3_var, target_var], [cost], updates=updates)


import Evaluation as E

#Tibial_Score, Femoral_Score = E.Evaluate1("/home/xvt131/Running/train2", DP.image_load, 9, 5 ,f_eval)
#E.Segment("/home/xvt131/Running/evaluation", DP.image_load_eval, 9, 5 ,f_eval)


import try_DP as TD
import scipy.io as io

Tibia_Mean = []
Femoral_Mean = []
#for param in DP.get_paths('/home/xvt131/Functions/Adhish_copy/triplan_param'):

with np.load("/home/xvt131/Functions/Adhish_copy/triplanar_Params_21classR.npz") as f:
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
        M, N, O = TD.Patch_gen(K, 29, C)
        preds[idx] = f_eval(M,N,O)
    Final_pred = np.argmax(preds, axis = -1)
    print Final_pred.shape
    Lab = B1.reshape(Sha)
    Final_pred = Final_pred.reshape(Sha)

    TibiaD += [E.Dice_score(Final_pred, Lab, 1)]
    print TibiaD
    io.savemat("/home/xvt131/Functions/Adhish_copy/EvalTri/Seg_%s" %(img[-36:-30]), mdict= {"Seg2_%s" %(img[-36:-30]):Final_pred,"Lab_%s" %(img[-36:-30]):Lab} )

np.savez('Dice_Scores_Tri.npz', Tibia_Mean, Femoral_Mean)
