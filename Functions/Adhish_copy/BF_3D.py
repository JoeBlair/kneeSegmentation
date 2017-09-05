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

l_conv_1 = lasagne.layers.dnn.Conv3DDNNLayer(input, 20, (9,9,9))

l_maxpool_1 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_1, (3,3,3))

l_conv_2 = lasagne.layers.dnn.Conv3DDNNLayer(l_maxpool_1, 40, (5,5,5))

l_conv_3 = lasagne.layers.dnn.Conv3DDNNLayer(l_conv_2, 80, (3,3,3))


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
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.000001, momentum=0.9)

f_eval = theano.function([input_var], eval_out)
f_vali = theano.function([input_var, target_var], [costV])

f_train = theano.function([input_var, target_var], [cost], updates=updates)

from confusionmatrix import ConfusionMatrix

batch_size = 250
num_epochs = 175

train_acc = []
valid_loss = []
valid_acc = []
train_loss = []
cur_loss = 0
loss = []

Train = DP.get_paths("/home/xvt131/Functions/Adhish_copy/all_train")
Test = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Validating-Rand")


for epoch in range(num_epochs):
    cur_loss = 0
    val_loss = 0
    confusion_valid = ConfusionMatrix(2)
    confusion_train = ConfusionMatrix(2)
   # Train = np.random.choice(Train_all, 5)

    for im in Train:
        X_train, Y_train  = DP.Patch_3D_para(im, PS)
        num_samples_train = Y_train.shape[0]
        num_batches_train = num_samples_train // batch_size
   
        for i in range(num_batches_train):
            
            idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X_train[idx]
            target_batch = Y_train[idx]

            batch_loss = f_train(x_batch ,target_batch) #this will do the backprop pass
            cur_loss += batch_loss[0]/batch_size

        for i in range(num_batches_train):
            idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X_train[idx]
            targets_batch = Y_train[idx]
            net_out = f_eval(x_batch)
            preds = np.argmax(net_out, axis=-1)
            confusion_train.batch_add(targets_batch, preds)
    loss += [cur_loss/len(Train)]


#    for img in Test:
#        X_test, Y_test  = DP.Patch_3D_para(im,PS)
#        num_samples_valid = Y_test.shape[0]
#        num_batches_valid = num_samples_valid // batch_size
#
#        for i in range(num_batches_valid):
#            idx = range(i*batch_size, (i+1)*batch_size)
#            x_batch = X_test[idx]
#            targets_batch = Y_test[idx]
#            net_out = f_eval(x_batch)
#            preds = np.argmax(net_out, axis=-1)
#            AB = f_vali(x_batch, targets_batch)
#            confusion_valid.batch_add(targets_batch, preds)
#            val_loss += AB[0]/batch_size
#            confusion_valid.batch_add(targets_batch, preds)
#  
#    valid_loss += [val_loss/len(Test)]
    train_acc_cur = confusion_train.accuracy()
#    valid_acc_cur = confusion_valid.accuracy()
    train_acc += [train_acc_cur]
#    valid_acc += [valid_acc_cur]
#
#
#    if (epoch) % 10 == 0:
#
#        np.savez('/home/xvt131/Functions/Adhish_copy/3D_params/epoch_%d_params_bigger.npz' %(epoch), *lasagne.layers.get_all_param_values(output))
#
#
    print confusion_train
    print "Epoch %i : Train Loss %e , Train acc %f" % (epoch+1, loss[-1], train_acc_cur,)


np.savez('/home/xvt131/Functions/Adhish_copy/3D_params/3D_all_params11.npz', *lasagne.layers.get_all_param_values(output))

np.savez('/home/xvt131/Functions/Adhish_copy/3D_params/Loss_value_3D_all.npz', loss, train_acc)





