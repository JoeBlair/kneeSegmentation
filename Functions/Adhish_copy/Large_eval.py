import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import sgd
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import downsample
import lasagne
import numpy as np
import try_DP as DP
from theano.tensor import nnet
import lasagne.layers.dnn

dtensor5 = TensorType('float32', (False,)*5)
input_var = dtensor5('X_Train')
target_var = T.ivector('Y_train')
x1 = T.matrix('x1')

PS = 29

# Build Neural Network:
# Conv Net patch size 9
input = lasagne.layers.InputLayer((None, 1, PS, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv3DDNNLayer(input, 20, (5,5,5))

l_maxpool_1 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_1, (5,5,5))

l_conv_2 = lasagne.layers.dnn.Conv3DDNNLayer(l_maxpool_1, 20, (5,5,5))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (2,2,2))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_2, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


#Output Layer
output = lasagne.layers.DenseLayer(l_hidden1, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

## Train
train_out = lasagne.layers.get_output(output, input_var, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, input_var, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(train_out, target_var).mean()
costV = T.nnet.categorical_crossentropy(eval_out, target_var).mean()

all_grads = T.grad(cost, all_params)


# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.00001, momentum=0.8)

f_eval = theano.function([input_var], eval_out)
f_vali = theano.function([input_var, target_var], [costV])

f_train = theano.function([input_var, target_var], [cost], updates=updates)
import Evaluation as E
import try_DP as TD
import scipy.io as io

Tibia_Mean = []
Femoral_Mean = []
#for param in DP.get_paths('/home/xvt131/Functions/Adhish_copy/three_param'):

with np.load("/home/xvt131/Functions/Adhish_copy/three_Params2.npz") as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(output, param_values)

for img in DP.get_paths("/home/xvt131/Functions/Adhish_copy/Validating-Rand"):
    print img
    A, B, C = TD.Tri_Image_Load(img)
    B1 = B.reshape(np.prod(B.shape))
    batch = 100
    num_batches = A.shape[0] / batch
    Sha = B.shape
    TibiaD = []
    FemoralD = []
    preds = np.zeros(shape = ( len(B1), 2))

    for i in range(num_batches):

        idx = range(i*batch, (i+1)*batch)
        K = A[idx]
        M = TD.Patch_gen_three(K, 29, C)
        M = M.reshape(batch,  1, PS, PS, PS)
        preds[idx] = f_eval(M)
    
    Final_pred = np.argmax(preds, axis = -1)
    Lab = B1.reshape(Sha)
    Final_pred = Final_pred.reshape(Sha)

    TibiaD += [E.Dice_score(Final_pred, Lab, 1)]
    FemoralD +=  [E.Dice_score(Final_pred, Lab, 2)]
    print TibiaD
    io.savemat("/home/xvt131/Functions/Adhish_copy/EvalThree/Seg_%s" %(img[-36:-30]), mdict= {"Seg_%s" %(img[-36:-30]):Final_pred, "Label_%s" %(img[-36:-30]):Lab})
   

