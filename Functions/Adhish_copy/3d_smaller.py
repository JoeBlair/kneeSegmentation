import theano
from confusionmatrix import ConfusionMatrix
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import sgd
import theano.tensor as T
from theano.tensor import *
from theano.tensor.signal import pool
import lasagne
import numpy as np
import DP1 as DP
from theano.tensor import nnet
import lasagne.layers.dnn

dtensor5 = TensorType('float32', (False,)*5)
input_var = dtensor5('X_Train')
input2_var = dtensor5('X_Train2')
target_var = T.ivector('Y_train')
x1 = T.matrix('x1')
x2 = T.matrix('x2')
PS = 9
PS2 = 5

# Build Neural Network:
# Conv Net patch size 9
input = lasagne.layers.InputLayer((None, 1, PS, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv3DDNNLayer(input,50, (2,2,2))

#l_maxpool_1 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_1, (2,2,2))

l_conv_2 = lasagne.layers.dnn.Conv3DDNNLayer(l_conv_1, 50, (2,2,2))


l_hidden1 = lasagne.layers.DenseLayer(l_conv_2, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


input = lasagne.layers.InputLayer((None, 1, PS2, PS2, PS2), input_var=input2_var)

l_conv_11 = lasagne.layers.dnn.Conv3DDNNLayer(input, 20, (2,2,2))

#l_conv_22 = lasagne.layers.dnn.Conv3DDNNLayer(l_conv_11, 40, (3,3,3))


l_hidden11 = lasagne.layers.DenseLayer(l_conv_11, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

Merged = lasagne.layers.ConcatLayer([ l_hidden1, l_hidden11])


#Output Layer
output = lasagne.layers.DenseLayer(Merged, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)
 
## Train
train_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2}, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2}, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(train_out, target_var).mean()
costV = T.nnet.categorical_crossentropy(eval_out, target_var).mean()

all_grads = T.grad(cost, all_params)


# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.adadelta(all_grads, all_params, learning_rate=0.00005)

f_eval = theano.function([input_var, input2_var], eval_out)
f_vali = theano.function([input_var,input2_var, target_var], [costV])

f_train = theano.function([input_var,input2_var, target_var], [cost], updates=updates)

from confusionmatrix import ConfusionMatrix

batch_size = 100
num_epochs = 100

train_acc = []
valid_loss = []
valid_acc = []
train_loss = []
cur_loss = 0
loss = []

Train = DP.get_paths("/home/xvt131/Biomediq/Data/kneeData/allTrain")
Test = DP.get_paths("/home/xvt131/Biomediq/Data/kneeData/Validating-Rand")

for epoch in range(num_epochs):
    cur_loss = 0
    val_loss = 0
    confusion_valid = ConfusionMatrix(2)
    confusion_train = ConfusionMatrix(2)
   # Train = np.random.choice(Train_all, 5)

    for im in Train:
        X_train, XX, Y_train  = DP.Patch_3D_para(im, PS, PS2)
        num_samples_train = Y_train.shape[0]
        num_batches_train = num_samples_train // batch_size
   
        for i in range(num_batches_train):
            
            idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X_train[idx]
            X_batch = XX[idx]
            target_batch = Y_train[idx]

            batch_loss = f_train(x_batch, X_batch ,target_batch) #this will do the backprop pass
            cur_loss += batch_loss[0]/batch_size

        for i in range(num_batches_train):
            idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X_train[idx]
            X_batch = XX[idx]
            targets_batch = Y_train[idx]
            net_out = f_eval(x_batch, X_batch)
            preds = np.argmax(net_out, axis=-1)
            confusion_train.batch_add(targets_batch, preds)
    loss += [cur_loss/len(Train)]


    for img in Test:
        X_test, XX, Y_test  = DP.Patch_3D_para(im,PS, PS2)
        num_samples_valid = Y_test.shape[0]       
        num_batches_valid = num_samples_valid // batch_size

        for i in range(num_batches_valid):
            idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X_test[idx]
            X_batch = XX[idx]
            targets_batch = Y_test[idx]
            net_out = f_eval(x_batch, X_batch)
            preds = np.argmax(net_out, axis=-1)
            AB = f_vali(x_batch,X_batch, targets_batch)
            confusion_valid.batch_add(targets_batch, preds)
            val_loss += AB[0]/batch_size
            confusion_valid.batch_add(targets_batch, preds)
#  
    valid_loss += [val_loss/len(Test)]
    train_acc_cur = confusion_train.accuracy()
    valid_acc_cur = confusion_valid.accuracy()
    train_acc += [train_acc_cur]
    valid_acc += [valid_acc_cur]
#
#
#    if (epoch) % 10 == 0:
#
#        np.savez('/home/xvt131/Functions/Adhish_copy/3D_params/epoch_%d_params_bigger.npz' %(epoch), *lasagne.layers.get_all_param_values(output))
#
#
    print confusion_train
    print "Epoch %i : Train Loss %e , Train acc %f Valid Loss %e, Valid acc %f" % (epoch+1, loss[-1], train_acc_cur, valid_loss[-1], valid_acc_cur)


np.savez('/home/xvt131/Biomediq/Functions/Adhish_copy/Exp101/3Dknee.npz', *lasagne.layers.get_all_param_values(output))

np.savez('/home/xvt131/Biomediq/Functions/Adhish_copy/Exp101/3Dknee_loss', loss, train_acc, valid_loss, valid_acc)





