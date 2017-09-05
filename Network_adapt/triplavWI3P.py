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
input4_var = T.ftensor3('X_pos')
input5_var = dtensor5('cube')
target_var = T.ivector('Y_train')
x1 = T.matrix('x1')
x2 = T.matrix('x2')
x3 = T.matrix('x3')
x4 = T.matrix('x4')
x5 = T.matrix('x5')
PS = 29
PS2 = 9
# Build Neural Network:
# Conv Net XY Plane
input = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv2DDNNLayer(input, 20, (9,9))

l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_1, (3,3))

l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_1, 40,(5,5))
l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_2, 80, (3,3))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (2,2,2))

l_hidden1 = lasagne.layers.DenseLayer(l_conv_3, num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Conv Net patch si
input2 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input2_var)

l_conv_11 = lasagne.layers.dnn.Conv2DDNNLayer(input2, 20, (9,9))

l_maxpool_11 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_11, (3,3))

l_conv_22 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_11, 40, (5,5))
l_conv_33 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_22, 80, (3,3))

#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))

l_hidden2 = lasagne.layers.DenseLayer(l_conv_33, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Spatial Prior FFNet
input3 = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input3_var)

l_conv_111 = lasagne.layers.dnn.Conv2DDNNLayer(input3, 20, (9,9))

l_maxpool_111= lasagne.layers.dnn.Pool2DDNNLayer(l_conv_111, (3,3))

l_conv_222 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_111, 40,(5,5))
l_conv_333 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_222, 80, (3,3))
#l_maxpool_2 = lasagne.layers.dnn.MaxPool3DDNNLayer(l_conv_2, (1,1,1))

l_hidden3 = lasagne.layers.DenseLayer(l_conv_333, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


# Spatial Prior FFNet
input4 = lasagne.layers.InputLayer((None, 1, 3), input_var=input4_var)

l_hidden_spa = lasagne.layers.DenseLayer(input4, num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# 3D smaller patch
input5= lasagne.layers.InputLayer((None, 1, PS2, PS2 ,PS2), input_var=input5_var)

l_conv_1111 = lasagne.layers.dnn.Conv3DDNNLayer(input5, 20, (3,3,3))

l_conv_2222 = lasagne.layers.dnn.Conv3DDNNLayer(l_conv_1111, 40, (2,2,2))

l_hidden4 = lasagne.layers.DenseLayer(l_conv_2222, num_units=256,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

# Concatenation Layer
Merged = lasagne.layers.ConcatLayer([l_hidden1, l_hidden2, l_hidden3, l_hidden_spa, l_hidden4])

#l_dropout = lasagne.layers.DropoutLayer(Merged, p=0.5)

#Output Layer
output = lasagne.layers.DenseLayer(Merged, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

## Train
train_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input3_var:x3, input4_var:x4, input5_var:x5}, deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input_var:x1, input2_var:x2, input2_var:x3, input4_var:x4, input5_var:x5 }, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(train_out, target_var).mean()
costV = T.nnet.categorical_crossentropy(eval_out, target_var).mean()

all_grads = T.grad(cost, all_params)

# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.00001, momentum=0.8)

f_eval = theano.function([input_var, input2_var, input3_var, input4_var, input5_var], eval_out)

f_train = theano.function([input_var, input2_var, input3_var, input4_var, input5_var,target_var], [cost], updates=updates)

f_vali = theano.function([input_var, input2_var, input3_var,input4_var, input5_var ,target_var], [costV])


from confusionmatrix import ConfusionMatrix

batch_size = 250
num_epochs = 150

train_acc= []
valid_acc = []

cur_loss = 0
loss = []
valid_loss = []

Train = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Training-Rand")
Test = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Validating-Rand")
import gc
for epoch in range(num_epochs):
    cur_loss = 0
    val_loss = 0
    confusion_valid = ConfusionMatrix(2)
    confusion_train = ConfusionMatrix(2)
    
    for im in Train:

        XY, XZ, YZ, cube, Pos , Y_train  = DP.Big_Small(im, PS, PS2)
        num_samples_train = Y_train.shape[0]
        num_batches_train = num_samples_train // batch_size
        for i in range(num_batches_train):
            idx = range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
            yz_batch = YZ[idx]
            cube_batch = cube[idx]
            pos_batch = Pos[idx]
            target_batch = Y_train[idx]
            batch_loss = f_train(xy_batch, xz_batch, yz_batch,pos_batch, cube_batch ,target_batch) #this will do the backprop pass
            cur_loss += batch_loss[0]/num_samples_train
            net_out = f_eval(xy_batch, xz_batch, yz_batch, pos_batch, cube_batch)
            preds = np.argmax(net_out, axis=-1)
            confusion_train.batch_add(target_batch, preds)
    loss += [cur_loss/batch_size]

    for img in Test:
        XY, XZ, YZ, cube, Pos, Y_test  = DP.Big_Small(img, PS, PS2)
        num_samples_valid = Y_test.shape[0]
        num_batches_valid = num_samples_valid // batch_size
        for i in range(num_batches_valid):
            idx = range(i*batch_size, (i+1)*batch_size)
            xy_batch = XY[idx]
            xz_batch = XZ[idx]
            yz_batch = YZ[idx]
            cube_batch = cube[idx]
            pos_batch = Pos[idx]
            targets_batch = Y_test[idx]
            net_out = f_eval(xy_batch, xz_batch, yz_batch, pos_batch, cube_batch)
            preds = np.argmax(net_out, axis=-1)
            AB = f_vali(xy_batch, xz_batch, yz_batch, pos_batch, cube_batch, targets_batch)
            confusion_valid.batch_add(targets_batch, preds)
            val_loss += AB[0]/num_samples_valid
    valid_loss += [val_loss/batch_size]
    train_acc_cur = confusion_train.accuracy()
    valid_acc_cur = confusion_valid.accuracy()
    train_acc += [train_acc_cur]
    valid_acc += [valid_acc_cur]

    if (epoch) % 10 == 0:

        np.savez('/home/xvt131/Network_adapt/paramstri/epoch_%d_triWI.npz' %(epoch), *lasagne.layers.get_all_param_values(output))    


    print confusion_train
    print "Epoch %i : Train Loss %e , Train acc %f, Valid Loss %f ,Valid acc %f " % (epoch+1, loss[-1], train_acc_cur, valid_loss[-1], valid_acc_cur) 

#import Evaluation as E

np.savez('triplanar_Params_WI.npz', *lasagne.layers.get_all_param_values(output))
np.savez('Loss_values_WI.npz', loss, train_acc,  valid_loss,valid_acc )
#X, Y = E.Evaluate2("/home/xvt131/Running/train4", DP.Tri_Image_Load, PS ,f_eval)

#print "Mean Femur Dice Score:", np.mean(Y)




