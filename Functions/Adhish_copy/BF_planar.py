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
target_var = T.ivector('Y_train')
x1 = T.matrix('x1')
PS = 29

# Build Neural Network:
# Conv Net XY Plane
input = lasagne.layers.InputLayer((None, 1, PS, PS), input_var=input_var)

l_conv_1 = lasagne.layers.dnn.Conv2DDNNLayer(input, 20, (9,9))

l_maxpool_1 = lasagne.layers.dnn.Pool2DDNNLayer(l_conv_1, (3,3))

l_conv_2 = lasagne.layers.dnn.Conv2DDNNLayer(l_maxpool_1, 20,(5,5))

l_conv_3 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv_2, 20, (3,3))

l_hidden_1 = lasagne.layers.DenseLayer(l_conv_3, num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


#Output Layer
output = lasagne.layers.DenseLayer(l_hidden_1, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

## Train
train_out = lasagne.layers.get_output(output, {input_var:x1} , deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input_var:x1}, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.categorical_crossentropy(train_out, target_var).mean()
costV = T.nnet.categorical_crossentropy(eval_out, target_var).mean()

all_grads = T.grad(cost, all_params)

# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate=0.000001, momentum=0.9)

f_eval = theano.function([input_var], eval_out)

f_train = theano.function([input_var, target_var], [cost], updates=updates)

f_vali = theano.function([input_var, target_var], [costV])


from confusionmatrix import ConfusionMatrix

batch_size =250
num_epochs = 175

train_acc= []
valid_acc = []

cur_loss = 0
loss = []
valid_loss = []

Train = DP.get_paths("/home/xvt131/Functions/Adhish_copy/all_train")
Test = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Validating-Rand")
import gc
for epoch in range(num_epochs):
    cur_loss = 0
    val_loss = 0
    confusion_valid = ConfusionMatrix(2)
    confusion_train = ConfusionMatrix(2)
    
    for im in Train:

        XZ, Y_train  = DP.Patch_planar_para(im, PS)
        num_samples_train = Y_train.shape[0]
        num_batches_train = num_samples_train // batch_size
        for i in range(num_batches_train):
            idx = range(i*batch_size, (i+1)*batch_size)
            xz_batch = XZ[idx]
            target_batch = Y_train[idx]
            batch_loss = f_train(xz_batch ,target_batch) #this will do the backprop pass
            cur_loss += batch_loss[0]/batch_size


        for i in range(num_batches_train):
            idx =  range(i*batch_size, (i+1)*batch_size)
            xz_batch = XZ[idx]
            targets_batch = Y_train[idx]
            net_out = f_eval( xz_batch)
            preds = np.argmax(net_out, axis=-1)
            confusion_train.batch_add(targets_batch, preds)
    loss += [cur_loss/len(Train)]

#    for img in Test:
#        XZ, Y_test  = DP.Patch_planar_para(img, PS)
#        num_samples_valid = Y_test.shape[0]
#        num_batches_valid = num_samples_valid // batch_size
#        for i in range(num_batches_valid):
#            idx = range(i*batch_size, (i+1)*batch_size)
#            xz_batch = XZ[idx]
#            targets_batch = Y_test[idx]
#            net_out = f_eval(xz_batch)
#            preds = np.argmax(net_out, axis=-1)
#            AB = f_vali( xz_batch,  targets_batch)
#            confusion_valid.batch_add(targets_batch, preds)
#            val_loss += AB[0]/batch_size
#    valid_loss += [val_loss/len(Test)]
    train_acc_cur = confusion_train.accuracy()
#    valid_acc_cur = confusion_valid.accuracy()
    train_acc += [train_acc_cur]
#    valid_acc += [valid_acc_cur]
#
#    if (epoch) % 10 == 0:
#
#        np.savez('/home/xvt131/Functions/Adhish_copy/P_params/epoch_%d_Pparams.npz' %(epoch), *lasagne.layers.get_all_param_values(output))    
#

    print confusion_train
    print epoch + 1
  #  print "Epoch %i : Train Loss %e , Train acc %f, Valid Loss %f ,Valid acc %f " % (epoch+1, loss[-1], train_acc_cur, valid_loss[-1], valid_acc_cur) 

#import Evaluation as E

np.savez('/home/xvt131/Functions/Adhish_copy/P_params/planar_Params_all.npz', *lasagne.layers.get_all_param_values(output))
np.savez('/home/xvt131/Functions/Adhish_copy/P_params/Loss_values_planar.npz', loss, train_acc )
#X, Y = E.Evaluate2("/home/xvt131/Running/train4", DP.Tri_Image_Load, PS ,f_eval)

#print "Mean Femur Dice Score:", np.mean(Y)




