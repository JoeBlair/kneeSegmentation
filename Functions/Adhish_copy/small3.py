import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import sgd
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import downsample
import lasagne
import numpy as np
import DP1 as DP
from theano.tensor import nnet
import lasagne.layers.dnn
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt

dtensor5 = TensorType('float32', (False,)*5)
input_var = dtensor5('X_Train')
input2_var = T.ftensor3('X_pos')
input3_var = dtensor5('X_Bigger')
target_var = T.matrix('Y_train')
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
output = lasagne.layers.DenseLayer(l_dropout, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

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
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.001, momentum=0.8)

f_eval = theano.function([input_var, input2_var, input3_var], eval_out)
f_vali = theano.function([input_var, input2_var, input3_var, target_var], [costV])
f_train = theano.function([input_var, input2_var, input3_var, target_var], [cost], updates=updates)

from confusionmatrix import ConfusionMatrix

batch_size = 100
num_epochs = 50

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
loss = []
valid_loss = []

Train = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Training-Rand")
Test = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Validating-Rand")

for epoch in range(num_epochs):
    cur_loss = 0
    val_loss = 0
    confusion_valid = ConfusionMatrix(2)
    confusion_train = ConfusionMatrix(2)

    for im in Train:
        XY, XZ, YZ, Y_train  = DP.Patch_gen_mix(im)
        num_samples_train = Y_train.shape[0]
        num_batches_train = num_samples_train // batch_size
        for i in range(num_batches_train):
            idx = range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            print xy_batch.shape
            xz_batch = XZ[idx]
            print xz_batch.shape
          
            yz_batch = YZ[idx]
            print yz_batch.shape
            target_batch =np.float32(Y_train[idx].reshape(batch_size, 1))
            print target_batch.shape
            batch_loss = f_train(xy_batch, xz_batch, yz_batch ,target_batch) #this will do the backprop pass
            cur_loss += batch_loss[0]/batch_size


        for i in range(num_batches_train):
            idx =  range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
            yz_batch = YZ[idx]
            targets_batch = np.float32(Y_train[idx].reshape(batch_size, 1))
            net_out = f_eval(xy_batch, xz_batch, yz_batch)
            preds = np.where(net_out>threshold, upper, lower)
            confusion_train.batch_add(targets_batch.reshape(batch_size, 1), preds)
    loss += [cur_loss/len(Train)]

    for img in Test:
        XY, XZ, YZ, Y_test  = DP.Patch_gen_mix(img)
        num_samples_valid = Y_test.shape[0]
        num_batches_valid = num_samples_valid // batch_size
        for i in range(num_batches_valid):
            idx = range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
            yz_batch = YZ[idx]
            targets_batch = np.int32(Y_test[idx].reshape(batch_size, 1))
            net_out = f_eval(xy_batch, xz_batch, yz_batch)
            preds = np.where(net_out>threshold, upper, lower)
            AB = f_vali(xy_batch, xz_batch, yz_batch, targets_batch)
            confusion_valid.batch_add(targets_batch.reshape(batch_size, 1), preds.reshape(batch_size, 1))
            val_loss += AB[0]/batch_size
    valid_loss += [val_loss/len(Test)]
    train_acc_cur = confusion_train.accuracy()
    valid_acc_cur = confusion_valid.accuracy()
    train_acc += [train_acc_cur]
    valid_acc += [valid_acc_cur]

    print confusion_train
    print "Epoch %i : Train Loss %e , Train acc %f, Valid Loss %f, Valid acc %f " % (epoch+1, loss[-1], train_acc_cur, valid_loss[-1], valid_acc_cur)

#import Evaluation as E

np.savez('/home/xvt131/Functions/Adhish_copy/TP_param/triplanar_Params_box_whole.npz', *lasagne.layers.get_all_param_values(output))
np.savez('/home/xvt131/Functions/Adhish_copy/TP_param/Loss_values_Tri_box_whole.npz', loss, train_acc, valid_loss, valid_acc )




