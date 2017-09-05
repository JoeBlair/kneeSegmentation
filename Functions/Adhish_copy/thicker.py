import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import *
from lasagne.updates import *
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import pool
import lasagne
import numpy as np
import DP1 as DP
from theano.tensor import nnet
import lasagne.layers.dnn
dtensor5 = TensorType('float32', (False,)*5)

input_var = T.ftensor4('XY')
target_var = T.ivector('Y_train')
x1 = T.matrix('x1')
PS = 15
P2 = 3
# Build Neural Network:
# Conv Net XY Plane
input = lasagne.layers.InputLayer((None, 15, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv2DDNNLayer(input, 20, (3,3))

l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_1, (2, 2))

l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_1, 20,(3,3))

l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_2, 20, (3,3))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (2,2,2))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_3, num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))
#
## Conv Net patch si
#input2 = lasagne.layers.InputLayer((None, 10, PS, PS), input_var=input2_var)
#
#l_conv_11 = lasagne.layers.dnn.Conv2DDNNLayer(input2, 20, (5,5))
#
#l_maxpool_11 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_11, (2,2))
#
#l_conv_22 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_11, 40, (5,5))
#l_conv_33 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_22, 80, (5,5))
#
##l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))
#
#l_hidden2 = lasagne.layers.DenseLayer(l_conv_33, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))
#
#
## Spatial Prior FFNet
#input3 = lasagne.layers.InputLayer((None, 3, PS, PS), input_var=input3_var)
#
#l_conv_111 = lasagne.layers.dnn.Conv2DDNNLayer(input3, 20, (5,5))
#
#l_maxpool_111= lasagne.layers.dnn.Pool2DDNNLayer(l_conv_111, (2,2))
#
#l_conv_222 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_111, 20,(5,5))
#l_conv_333 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_222, 20, (5,5))
#
#l_hidden3 = lasagne.layers.DenseLayer(l_conv_333, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))
#
## Concatenation Layer
#Merged = lasagne.layers.ConcatLayer([l_hidden1, l_hidden2, l_hidden3])
#
#l_dropout = lasagne.layers.DropoutLayer(Merged, p=0.5)

#Output Layer
output = lasagne.layers.DenseLayer(l_hidden1, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

## Train
train_out = lasagne.layers.get_output(output, input_var, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, input_var , deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(train_out, target_var).mean()
costV = T.nnet.categorical_crossentropy(eval_out, target_var).mean()

all_grads = T.grad(cost, all_params)

# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.adadelta(all_grads, all_params, learning_rate=0.1)

f_eval = theano.function([input_var], eval_out)

f_train = theano.function([input_var, target_var], [cost], updates=updates)

f_vali = theano.function([input_var, target_var], [costV])


from confusionmatrix import ConfusionMatrix

batch_size = 100
num_epochs = 10

train_acc= []
valid_acc = []

cur_loss = 0
loss = []
valid_loss = []

Train = DP.get_paths("/home/xvt131/Biomediq/Validating-Rand")
Test = DP.get_paths("/home/xvt131/Biomediq/Validating-Rand")
import gc
for epoch in range(num_epochs):
    cur_loss = 0
    val_loss = 0
    confusion_valid = ConfusionMatrix(2)
    confusion_train = ConfusionMatrix(2)
    
    for im in Train:

        XY, XZ, Y_train  = DP.Patch_3D_para(im, PS, P2)
        num_samples_train = Y_train.shape[0]
        num_batches_train = num_samples_train // batch_size
        for i in range(num_batches_train):
            idx = range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
          
            target_batch = Y_train[idx]
            batch_loss = f_train(xy_batch ,target_batch) 
            cur_loss += batch_loss[0]/batch_size


        for i in range(num_batches_train):
            idx =  range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
            targets_batch = Y_train[idx]
            net_out = f_eval(xy_batch)
            preds = np.argmax(net_out, axis=-1)
  #          print preds
            confusion_train.batch_add(targets_batch, preds)
    loss += [cur_loss/len(Train)]


    for img in Test:
        XY, XZ, Y_test  = DP.Patch_3D_para(im, PS, P2)
        num_samples_valid = Y_test.shape[0]
        num_batches_valid = num_samples_valid // batch_size
        for i in range(num_batches_valid):
            idx = range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
            targets_batch = Y_test[idx]
            net_out = f_eval(xy_batch)
            preds = np.argmax(net_out, axis=-1)
            AB = f_vali(xy_batch, targets_batch)
            confusion_valid.batch_add(targets_batch, preds)
            val_loss += AB[0]/batch_size
    valid_loss += [val_loss/len(Test)]
    train_acc_cur = confusion_train.accuracy()
    valid_acc_cur = confusion_valid.accuracy()
    train_acc += [train_acc_cur]
    valid_acc += [valid_acc_cur]

  #  if (epoch) % 10 == 0:

   #     np.savez('/home/xvt131/Functions/Adhish_copy/TP_param/epoch_%d_paramsBTF.npz' %(epoch), *lasagne.layers.get_all_param_values(output))    


    print confusion_train
    print "Epoch %i : Train Loss %e , Train acc %f, " % (epoch+1, loss[-1], train_acc_cur,) 

#import Evaluation as E

#np.savez('/home/xvt131/Functions/Adhish_copy/TP_param/triplanar_Params.npz', *lasagne.layers.get_all_param_values(output))
#np.savez('/home/xvt131/Functions/Adhish_copy/TP_param/Loss_values_Tri.npz', loss, train_acc, valid_loss, valid_acc )

