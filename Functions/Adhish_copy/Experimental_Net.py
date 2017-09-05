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
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
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
# Conv Net patch size 9
input = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv2DDNNLayer(input, 25, (9, 9))

#l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_1, (5, 5))

l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_1, 50, (3,3))
#l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_2, 20, (5,5))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (2,2,2))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_2, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Conv Net patch size 15
input2 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input2_var)

l_conv_11 = lasagne.layers.dnn.Conv2DDNNLayer(input2, 25, (9,9))

#l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_11, (5,5))

l_conv_22 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_11, 50, (3,3))
#l_conv_22 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_22, 20, (5,5))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))

l_hidden2 = lasagne.layers.DenseLayer(l_conv_22, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Spatial Prior FFNet
input3 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input3_var)

l_conv_111 = lasagne.layers.dnn.Conv2DDNNLayer(input3, 25, (9,9))

#l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_111, (5,5))

l_conv_222 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_111, 20, (3,3))
#l_conv_222 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_22, 20, (5,5))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))

l_hidden3 = lasagne.layers.DenseLayer(l_conv_222, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Concatenation Layer
Merged = lasagne.layers.ConcatLayer([l_hidden1, l_hidden2, l_hidden3])

l_dropout = lasagne.layers.DropoutLayer(Merged, p=0.5)

#Output Layer
output = lasagne.layers.DenseLayer(l_dropout, num_units=2, nonlinearity = lasagne.nonlinearities.sigmoid)

## Train
train_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input3_var:x3}, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input2_var:x3 }, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.binary_crossentropy(train_out, target_var).mean()


all_grads = T.grad(cost, all_params)

# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.01, momentum=0.75)

f_eval = theano.function([input_var, input2_var, input3_var], eval_out)

f_train = theano.function([input_var, input2_var, input3_var, target_var], [cost], updates=updates)

from confusionmatrix import ConfusionMatrix

batch_size = 5
num_epochs = 1

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
loss = []

Training = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Training")
Test = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Testing")

index = 0
for epoch in range(num_epochs):
    cur_loss = 0
    confusion_valid = ConfusionMatrix(2)
    confusion_train = ConfusionMatrix(2)
    if index == 0:
        Train = Training[0:10]
        index += 10
    elif index > 0:
        Train = Training[10:]
        index = 0

    for im in Train:
        XY, XZ, YZ, Y_train  = DP.Patch_triplanar_para(im, PS)
        num_samples_train = Y_train.shape[0]
        num_batches_train = num_samples_train // batch_size
        for i in range(num_batches_train):
            idx = range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
            yz_batch = YZ[idx]
            target_batch = Y_train[idx]
            print yz_batch.dtype
            print xz_batch.shape
  	    print xy_batch.shape
	    print target_batch.dtype
            batch_loss = f_train(xy_batch, xz_batch, yz_batch ,target_batch) #this will do the backprop pass
            cur_loss += batch_loss[0]
        loss += [cur_loss/batch_size]
        print loss
        for i in range(num_batches_train):
            idx = range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
            yz_batch = YZ[idx]
            targets_batch = Y_train[idx]
            net_out = f_eval(xy_batch, xz_batch, yz_batch)
            preds = np.argmax(net_out, axis=-1)

            confusion_train.batch_add(targets_batch, preds)

    for img in Train:
        XY, XZ, YZ, Y_test  = DP.Patch_triplanar_para(img, PS)
        num_samples_valid = Y_test.shape[0]
        num_batches_valid = num_samples_valid // batch_size

        for i in range(num_batches_valid):
            idx = range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
            yz_batch = YZ[idx]
            targets_batch = Y_test[idx]
            net_out = f_eval(xy_batch, xz_batch, yz_batch)
            preds = np.argmax(net_out, axis=-1)

            confusion_valid.batch_add(targets_batch, preds)

    train_acc_cur = confusion_train.accuracy()
    valid_acc_cur = confusion_valid.accuracy()



    print confusion_train
    print "Epoch %i : Train Loss %e , Train acc %f,  Valid acc %f " % (epoch+1, loss[-1], train_acc_cur, valid_acc_cur)

#import Evaluation as E

#np.savez('Evaluation_Params.npz', *lasagne.layers.get_all_param_values(output))

#X, Y = E.Evaluate2("/home/xvt131/Running/train4", DP.Tri_Image_Load, PS ,f_eval)

#print "Mean Tibia Dice Score:" , np.mean(X)
#print "Mean Femur Dice Score:", np.mean(Y)




