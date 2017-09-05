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
PS = 25

# Build Neural Network:
# Conv Net XY Plane
input = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv2DDNNLayer(input, 20, (7,7))

l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_1, (3,3))

l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_1, 20,(5,5))
l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_2, 20, (2,2))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (2,2,2))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_3, num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Conv Net patch si
input2 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input2_var)

l_conv_11 = lasagne.layers.dnn.Conv2DDNNLayer(input2, 20, (7,7))

l_maxpool_11 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_11, (3,3))

l_conv_22 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_11, 20, (5,5))
l_conv_33 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_22, 20, (2,2))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))

l_hidden2 = lasagne.layers.DenseLayer(l_conv_33, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Spatial Prior FFNet
input3 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input3_var)

l_conv_111 = lasagne.layers.dnn.Conv2DDNNLayer(input3, 20, (7,7))

l_maxpool_111= lasagne.layers.dnn.Pool2DDNNLayer(l_conv_111, (3,3))

l_conv_222 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_111, 20,(5,5))
l_conv_333 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_222, 20, (2,2))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))

l_hidden3 = lasagne.layers.DenseLayer(l_conv_333, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Concatenation Layer
Merged = lasagne.layers.ConcatLayer([l_hidden1, l_hidden2, l_hidden3])

#l_dropout = lasagne.layers.DropoutLayer(Merged, p=0.5)

#Output Layer
output = lasagne.layers.DenseLayer(Merged, num_units=3, nonlinearity = lasagne.nonlinearities.softmax)

## Train
train_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input3_var:x3}, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input2_var:x3 }, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(T.clip(train_out, 0.01, 0.999), target_var).mean()

all_grads = T.grad(cost, all_params)

# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.1, momentum=0.9)

f_eval = theano.function([input_var, input2_var, input3_var], eval_out)

f_train = theano.function([input_var, input2_var, input3_var, target_var], [cost], updates=updates)

from confusionmatrix import ConfusionMatrix

batch_size = 15
num_epochs = 50

train_acc= []
train_loss = np.array([])
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
loss = []

Train = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Training")
import gc
for epoch in range(num_epochs):
    cur_loss = 0
    #Train = np.random.choice(Train_all, 5)
    confusion_valid = ConfusionMatrix(3)
    confusion_train = ConfusionMatrix(3)
    
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
            batch_loss = f_train(xy_batch, xz_batch, yz_batch ,target_batch) #this will do the backprop pass
            cur_loss += batch_loss[0]
        loss += [cur_loss]


        for i in range(num_batches_train):
            idx = range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
            yz_batch = YZ[idx]
            targets_batch = Y_train[idx]
            net_out = f_eval(xy_batch, xz_batch, yz_batch)
            preds = np.argmax(net_out, axis=-1)
  #          print preds
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

    train_loss = np.append(train_loss, cur_loss)
    print confusion_train
    print "Epoch %i : Train Loss %e , Train acc %f,  Valid acc %f " % (epoch+1, cur_loss, train_acc_cur, valid_acc_cur) 

print train_loss
#import Evaluation as E

np.savez('Evaluation_Params.npz', *lasagne.layers.get_all_param_values(output))
np.savez('Loss_values.npz', train_loss)
#X, Y = E.Evaluate2("/home/xvt131/Running/train4", DP.Tri_Image_Load, PS ,f_eval)

#print "Mean Femur Dice Score:", np.mean(Y)




