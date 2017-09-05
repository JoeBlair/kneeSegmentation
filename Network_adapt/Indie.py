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
PS = 3
classes = 1
input2_var = T.ftensor3('X_pos')
target_var = T.matrix('Y_train')
x2 = T.matrix('x2')

# Spatial Prior FFNet
input2 = lasagne.layers.InputLayer((None, 1, 3), input_var=input2_var)


l_hidden_spa = lasagne.layers.DenseLayer(input2, num_units=128, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

l_hidden_spa2 = lasagne.layers.DenseLayer(l_hidden_spa, num_units=128, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))

l_hidden_spa3 = lasagne.layers.DenseLayer(l_hidden_spa2, num_units=128, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'))


#Output Layer
output = lasagne.layers.DenseLayer(l_hidden_spa3, num_units=classes, nonlinearity = lasagne.nonlinearities.sigmoid)

## Train
train_out = lasagne.layers.get_output(output, { input2_var:x2} , deterministic=False)

#Evaluate
eval_out = lasagne.layers.get_output(output, {input2_var:x2}, deterministic=True)


all_params = lasagne.layers.get_all_params(output, trainable=True)
cost = T.nnet.binary_crossentropy(train_out, target_var).mean()
costV = T.nnet.binary_crossentropy(eval_out, target_var).mean()

all_grads = T.grad(cost, all_params)

# Set the update function for parameters 
# you might wan't to experiment with more advanded update schemes like rmsprob, adadelta etc.
updates = lasagne.updates.sgd(all_grads, all_params, learning_rate=0.001)

f_eval = theano.function([ input2_var], eval_out)

f_train = theano.function([input2_var,  target_var], [cost], updates=updates)

f_vali = theano.function([ input2_var, target_var], [costV])


from confusionmatrix import ConfusionMatrix

batch_size = 5
num_epochs = 10

train_acc= []
valid_acc = []

cur_loss = 0
loss = []
valid_loss = []
threshold, upper, lower = 0.5, 1, 0

Train = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Training-Rand")
Test = DP.get_paths("/home/xvt131/Functions/Adhish_copy/Validating-Rand")
for epoch in range(num_epochs):
    print epoch
    cur_loss = 0
    val_loss = 0
    confusion_valid = ConfusionMatrix(classes)
    confusion_train = ConfusionMatrix(classes)
    B = np.array([])
    Pos = np.empty((0, 1,3))

    for im in Train:
        Post, Y_train  = DP.strat_sample(im)
        B = np.float32(np.append(B, Y_train))
        Pos = np.float32(np.vstack((Pos, Post)))
    random = np.arange(len(B))
    Y_train = B[random].reshape( len(random), 1) 
    Pos = Pos[random]
    print Pos.shape
    print Y_train.shape
    num_samples_train = Y_train.shape[0]
    num_batches_train = num_samples_train // batch_size
    for i in range(num_batches_train):
        idx = range(i*batch_size, (i+1)*batch_size)
        pos_batch = Pos[idx]
        target_batch = Y_train[idx]
        batch_loss = f_train(pos_batch ,target_batch.reshape(1,batch_size)) #this will do the backprop pass
        cur_loss += batch_loss[0]/num_samples_train
             

        net_out = f_eval( pos_batch)
        preds = np.where(net_out>threshold, upper, lower)
        confusion_train.batch_add(targets_batch.reshape(1,batch_size), preds)
    loss += [cur_loss/batch_size]

    for img in Test:
        Pos, Y_test  = DP.strat_sample(im)
        num_samples_valid = Y_test.shape[0]
        num_batches_valid = num_samples_valid // batch_size
        for i in range(num_batches_valid):
            idx = range(i*batch_size, (i+1)*batch_size)
            pos_batch = Pos[idx]
            targets_batch = Y_test[idx]
            net_out = f_eval(pos_batch)
            preds = np.where(net_out>threshold, upper, lower)
            AB = f_vali( pos_batch,  targets_batch.reshape(1,batch_size))
            val_loss += AB[0]/num_samples_valid
    valid_loss += [val_loss/batch_size]
    train_acc_cur = confusion_train.accuracy()
    train_acc += [train_acc_cur]

   # if (epoch) % 10 == 0:

    #    np.savez('/home/xvt131/Functions/Adhish_copy/plan_param/epoch_%d_PIparams.npz' %(epoch), *lasagne.layers.get_all_param_values(output))    


    print confusion_train
    print "Epoch %i : Train Loss %e , Train acc %f " % (epoch+1, loss[-1], train_acc_cur) 

#import Evaluation as E

np.savez('WI_Params.npz', *lasagne.layers.get_all_param_values(output))
np.savez('Loss_values_WI.npz', loss, train_acc,  valid_loss,valid_acc )
#X, Y = E.Evaluate2("/home/xvt131/Running/train4", DP.Tri_Image_Load, PS ,f_eval)

#print "Mean Femur Dice Score:", np.mean(Y)




