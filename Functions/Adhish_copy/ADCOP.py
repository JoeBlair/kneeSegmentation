import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import *
from lasagne.updates import *
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import downsample
import lasagne
import numpy as np
import BBox as DP
from theano.tensor import nnet
import lasagne.layers.dnn

dtensor5 = TensorType('float32', (False,)*5)
input_var = T.ftensor4('XY')
input2_var = T.ftensor4('XZ')
input3_var = T.ftensor4('YZ')
target_var = T.matrix('Y_train')
x1 = T.matrix('x1')
x2 = T.matrix('x2')
x3 = T.matrix('x3')
PS = 28

# Build Neural Network:
# Conv Net XY Plane
input = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv2DDNNLayer(input, 28, (5,5))

l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_1, (2,2))

l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_1, 56,(5,5))

l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_2, 112, (5,5))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_3, num_units=1792, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Conv Net patch si
input2 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input2_var)

l_conv_11 = lasagne.layers.dnn.Conv2DDNNLayer(input2, 28, (5,5))

l_maxpool_11 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_11, (2,2))

l_conv_22 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_11, 56, (5,5))
l_conv_33 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_22, 112, (5,5))

l_hidden2 = lasagne.layers.DenseLayer(l_conv_33, num_units=1792,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# conv patch yz
input3 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input3_var)

l_conv_111 = lasagne.layers.dnn.Conv2DDNNLayer(input3, 28, (5,5))

l_maxpool_111= lasagne.layers.dnn.Pool2DDNNLayer(l_conv_111, (2,2))

l_conv_222 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_111, 56,(5,5))
l_conv_333 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_222, 112, (5,5))


l_hidden3 = lasagne.layers.DenseLayer(l_conv_333, num_units=1792,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Concatenation Layer
Merged = lasagne.layers.ConcatLayer([l_hidden1, l_hidden2, l_hidden3])

#Dropout = lasagne.layers.DropoutLayer(Merged, p=0.5)
#Output Layer
output = lasagne.layers.DenseLayer(Merged, num_units=1, nonlinearity = lasagne.nonlinearities.sigmoid)

## Train
train_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input3_var:x3}, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input2_var:x3 }, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.binary_crossentropy(train_out, target_var).mean()
costV = T.nnet.binary_crossentropy(eval_out, target_var).mean()

all_grads = T.grad(cost, all_params)

# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.adam(all_grads, all_params,learning_rate=0.0000001, beta1=0.9, beta2=0.999, epsilon=1e-08)

f_eval = theano.function([input_var, input2_var, input3_var], eval_out)

f_train = theano.function([input_var, input2_var, input3_var, target_var], [cost], updates=updates)

f_vali = theano.function([input_var, input2_var, input3_var, target_var], [costV])


from confusionmatrix import ConfusionMatrix

batch_size = 250
num_epochs = 25

train_acc= []
valid_acc = []

cur_loss = 0
loss = []
valid_loss = []
threshold, upper, lower = 0.5, 1, 0
Train = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Training-Rand")
Test = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Validating-Rand")
import gc
for epoch in range(num_epochs):
    cur_loss = 0
    val_loss = 0
    confusion_valid = ConfusionMatrix(2)
    confusion_train = ConfusionMatrix(2)
    
    for im in Train:
        XY, XZ, YZ, Y_train  = DP.Patch_triplanar_para(im, PS)
        num_samples_train = Y_train.shape[0]
        num_batches_train = num_samples_train // batch_size
        for i in range(num_batches_train):
            idx = range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
            yz_batch = YZ[idx]
            target_batch =np.float32(Y_train[idx].reshape(batch_size, 1))
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
        XY, XZ, YZ, Y_test  = DP.Patch_triplanar_para(img, PS)
        num_samples_valid = Y_test.shape[0]
        num_batches_valid = num_samples_valid // batch_size
        for i in range(num_batches_valid):
            idx = range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
            yz_batch = YZ[idx]
            targets_batch = np.float32(Y_test[idx].reshape(batch_size, 1))
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

np.savez('/home/xvt131/Functions/Adhish_copy/Exp101/AD.npz', *lasagne.layers.get_all_param_values(output))
np.savez('/home/xvt131/Functions/Adhish_copy/Exp101/AD_loss.npz', loss, train_acc, valid_loss, valid_acc )




