import Data_Process as DP
import numpy as np
import time

Start = time.time()
A, B, C = DP.voxel_samples("/home/xvt131/Running/training/P41", [5, 9])
print time.time() - Start
print len(A)
print B.shape
print C.shape
