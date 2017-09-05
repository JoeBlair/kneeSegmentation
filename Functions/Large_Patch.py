import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import sgd
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import downsample
import lasagne
import numpy as np
import tri_DP as DP
from theano.tensor import nnet
import lasagne.layers.dnn
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt

dtensor5 = TensorType('float32', (False,)*5)
input_var = dtensor5('X_Train')
target_var = T.ivector('Y_train')
x1 = T.matrix('x1')

# Build Neural Network:
# Conv Net patch size 9
input = lasagne.layers.InputLayer((None, 1, 25,25,25), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv3DDNNLayer(input, 50, (5,5,5))

l_maxpool_1 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_1, (5,5,5))

l_conv_2 = lasagne.layers.dnn.Conv3DDNNLayer(l_maxpool_1, 50, (3,3,3))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (2,2,2))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_2, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


#Output Layer
output = lasagne.layers.DenseLayer(l_hidden1, num_units=3, nonlinearity = lasagne.nonlinearities.softmax)

## Train
train_out = lasagne.layers.get_output(output, input_var, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, input_var, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(train_out, target_var).mean()

all_grads = T.grad(cost, all_params)


# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.001, momentum=0.8)

f_eval = theano.function([input_var], eval_out)

f_train = theano.function([input_var, target_var], [cost], updates=updates)

from confusionmatrix import ConfusionMatrix

batch_size = 150
num_epochs = 5

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
loss = []

Train1 = DP.get_paths("/home/xvt131/Running/train1")
Train2 = DP.get_paths("/home/xvt131/Running/train2")
Train3 = DP.get_paths("/home/xvt131/Running/train3")

Test = DP.get_paths("/home/xvt131/Running/train4")

for epoch in range(num_epochs):
    cur_loss = 0
    Rand = 1
    confusion_valid = ConfusionMatrix(3)
    confusion_train = ConfusionMatrix(3)
    counter = 0
    if Rand == 1:
        Train = Train1
    elif Rand == 2:
        Train = Train2
    elif Rand == 3:
        Train = Train3
    elif Rand == 4:
        Train = Train4

    Rand += 1 
    if Rand > 4:
        Rand = 1
   
    for im in Train:
        print counter
        X_train, Y_train  = DP.Patch_3D_para(im, 25)
        num_samples_train = Y_train.shape[0]
        num_batches_train = num_samples_train // batch_size
   
        counter += 1  
        for i in range(num_batches_train):
            
            idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X_train[idx]
            target_batch = Y_train[idx]

            batch_loss = f_train(x_batch ,target_batch) #this will do the backprop pass
            cur_loss += batch_loss[0]

        loss += [cur_loss/batch_size]

        for i in range(num_batches_train):
            idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X_train[idx]
            targets_batch = Y_train[idx]
            net_out = f_eval(x_batch)
            preds = np.argmax(net_out, axis=-1)

            confusion_train.batch_add(targets_batch, preds)

    for img in Train:
        X_test, Y_test  = DP.Patch_3D_para(im,25)
        num_samples_valid = Y_test.shape[0]
        num_batches_valid = num_samples_valid // batch_size

        for i in range(num_batches_valid):
            idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X_test[idx]
            targets_batch = Y_test[idx]
            net_out = f_eval(x_batch)
            preds = np.argmax(net_out, axis=-1)

            confusion_valid.batch_add(targets_batch, preds)

    train_acc_cur = confusion_train.accuracy()
    valid_acc_cur = confusion_valid.accuracy()

    if (epoch) % 5 == 0:

        np.savez('/home/xvt131/Functions/epoch_%d_params.npz' %(epoch+1), *lasagne.layers.get_all_param_values(output))





    print confusion_train
    print "Epoch %i : Train Loss %e , Train acc %f,  Valid acc %f " % (epoch+1, loss[-1], train_acc_cur, valid_acc_cur)

import Evaluation as E

np.savez('Evaluation_Params.npz', *lasagne.layers.get_all_param_values(output))

X, Y = E.Evaluate1("/home/xvt131/Running/validating", DP.image_load, 9, 5 ,f_eval)

print "Mean Tibia Dice Score:" , np.mean(X)
print "Mean Femur Dice Score:", np.mean(Y)




