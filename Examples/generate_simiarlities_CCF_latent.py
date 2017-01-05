

##########################################################################################
# This is an example scipt to test CCF model using data on HDFS
##########################################################################################

import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from operator import add
from sklearn.metrics import pairwise_kernels


if __name__ == '__main__':
    Inf = np.load('/a/joe_data/MSD_trained_models/CCF-muse-MSD-step10_rank_15_15_lambda_1.0_1.0_alpha_5e-07_5e-07_iterations_50_50.log_Inf.npy')
    sourceSong = Inf[13456].reshape(1,15)
    targetSong = Inf
    
    print sourceSong.shape, targetSong.shape
    
    mySimilarities = pairwise_kernels(sourceSong,
                     targetSong,
                     metric='cosine')
    
    print mySimilarities.shape
    mySimilarities[0,13456] = -100.
    print mySimilarities[0,:].argsort()[::-1][:10]
