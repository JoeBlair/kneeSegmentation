import numpy as np
import Data_Process as DP
import time
import lasagne
from lasagne.layers import get_all_param_values
import scipy.io as io


def Dice_score(Preds, Truth, Value):

    dice = np.sum(np.logical_and(Preds == Value, Truth == Value))*2.0 / (np.sum(Preds==Value) + np.sum(Truth==Value))

    return dice


def Evaluate(paths, data_load_fun, Patches):

    Paths = DP.get_paths(paths)
    Gold_Dice = np.zeros(len(Paths))
    Silver_Dice = np.zeros(len(Paths))
    Image_Eval_time = np.zeros(len(Paths))
    Segmentations = []
    i = 0
    for img in Paths:

        Start = time.time()

        Lab, Posi, Pat, Sha, Ori =  data_load_fun(img, Patches)

        Pred_probs = eval_func(Pat[0], Posi, Pat[1])

        Final_pred = np.argmax(Pred_probs, axis = -1).reshape(Sha)
        Lab = Lab.reshape(Sha)

        Gold_Dice[i] = Dice_score(Final_pred, Lab, 1)

        Silver_Dice[i] = Dice_score(Final_pred, Lab, 2)

        Image_Eval_time[i] = time.time() - Start
        
        Segmentations.append(Final_pred)
        print i
        i += 1

    return Gold_Dice, Silver_Dice, Image_Eval_time, Segmentations


def Save_Network_Outputs(output, results_list):

    timestr = time.strftime("%m%d-%H%M")

    np.savez(str("Segment_Out_Params_" + timestr + ".npz"), *lasagne.layers.get_all_param_values(train_out))

    np.savez(str("Segment_Results_" + timestr + ".npz"), results_list)

def Evaluate1(paths, data_load_fun, Patch_Size,Patch2, eval_func):

    Paths = DP.get_paths(paths)
    Gold_Dice = np.zeros(len(Paths))
    Silver_Dice = np.zeros(len(Paths))
    Image_Eval_time = np.zeros(len(Paths))
    i = 0
    for img in Paths:
        
        Image = io.loadmat(img)
        Lab, Posi, Pat ,Pat2, Sha, Ori =  data_load_fun(img, Patch_Size, Patch2)
        batch_size = 2048
        preds = np.zeros(shape = ( len(Lab), 3))
        size_im = len(Lab)
        num_batches = size_im // batch_size
        for p in range(num_batches):

            idx = range(p*batch_size, (p+1)*batch_size)
            patch_batch = Pat[idx]
            pos_batch = Posi[idx]
            patch2_batch = Pat2[idx]
           
            preds[idx] = eval_func(patch_batch, pos_batch, patch2_batch)
        
        Final_pred = np.argmax(preds, axis = -1).reshape(Sha)

        Lab = Lab.reshape(Sha)

        if Image['isright'] == 0:

            Final_pred = np.fliplr(Final_pred)
            Lab = np.fliplr(Lab)        

      #  np.savez("Segmentatin_%s" %(img[-7:-4]), Final_pred, Lab)
        io.savemat("/home/xvt131/Functions/Seg_%s" %(img[-7:-4]), mdict= {"Seg_%s" %(img[-7:-4]):Final_pred, "Lab_%s" %(img[-7:-4]):Lab})
        print i

        Gold_Dice[i] = Dice_score(Final_pred, Lab, 1)

        Silver_Dice[i] = Dice_score(Final_pred, Lab, 2)
        
        i += 1

    return Gold_Dice, Silver_Dice

def Evaluate2(paths, data_load_fun, Patch_Size, eval_func):

    Paths = DP.get_paths(paths)
    Gold_Dice = np.zeros(len(Paths))
    Silver_Dice = np.zeros(len(Paths))
    Image_Eval_time = np.zeros(len(Paths))
    i = 0
    
    for img in Paths:
        
        Image = io.loadmat(img)
        Sha = Image['scan'].shape
        XY, XZ, YZ, Lab =  data_load_fun(img, Patch_Size, Patch2)
        batch_size = 2048
        preds = np.zeros(shape = ( len(Lab), 3))
        size_im = len(Lab)
        num_batches = size_im // batch_size
        for p in range(num_batches):

            idx = range(p*batch_size, (p+1)*batch_size)
            XY_batch = XY[idx]
            XZ_batch = XZ[idx]
            YZ_batch = YZ[idx]

            preds[idx] = eval_func(XY_batch, XZ_batch, YZ_batch)

        Final_pred = np.argmax(preds, axis = -1).reshape(Sha)

        Lab = Lab.reshape(Sha)

        if Image['isright'] == 0:

            Final_pred = np.fliplr(Final_pred)
            Lab = np.fliplr(Lab)

      #  np.savez("Segmentatin_%s" %(img[-7:-4]), Final_pred, Lab)
        io.savemat("/home/xvt131/Functions/Seg_%s" %(img[-7:-4]), mdict= {"Seg_%s" %(img[-7:-4]):Final_pred, "Lab_%s" %(img[-7:-4]):Lab})
        print i

        Gold_Dice[i] = Dice_score(Final_pred, Lab, 1)

        Silver_Dice[i] = Dice_score(Final_pred, Lab, 2)

        i += 1

    return Gold_Dice, Silver_Dice
def Segment(paths, data_load_fun,  Patch_Size,Patch2, eval_func):

    Paths = DP.get_paths(paths)
    i = 0
    for img in Paths:
        Image = io.loadmat(img)
        Nd, Posi, Pat ,Pat2, Sha, Scan =  data_load_fun(img, Patch_Size, Patch2)
        batch_size = 2048
        preds = np.zeros(shape = ( Nd, 3))
        size_im = Nd
        num_batches = size_im // batch_size
        for p in range(num_batches):

            idx = range(p*batch_size, (p+1)*batch_size)
            patch_batch = Pat[idx]
            pos_batch = Posi[idx]
            patch2_batch = Pat2[idx]


            
            preds[idx] = eval_func(patch_batch, pos_batch, patch2_batch)

        Final_pred = np.argmax(preds, axis = -1).reshape(Sha)
        if Image['isright'] == 0:
            
            Final_pred = np.fliplr(Final_pred)
        
        #np.savez("Segmentation_%s" %(img[-7:-4]), Final_pred)
        io.savemat("/home/xvt131/Functions/Eval_Segmentations/Seg_%s" %(img[-7:-4]), mdict= {"Seg_%s" %(img[-7:-4]):Final_pred})
