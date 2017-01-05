
##########################################################################################
# This is an example scipt to plot the P-R curve from CCF model trained on binary ratings
##########################################################################################

# common modules
import os
import sys
from os.path import expanduser
import itertools
import numpy as np
# pyspark modules
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
# in-house modules
from CCF import CCF



if __name__ == '__main__':

    from museUtility import getDatabaseKey, myPlot, mySigmoid
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    from sklearn import metrics
    import matplotlib
    from MuseUtil.museUtility import myPrecisionatK, myAbsolutePrecisionatK, mymeanAbsolutePrecisionatK
    import pandas as pd

    font = {'family' : 'Times',
            'weight' : 'bold',
            'size'   : '14'}
    
    matplotlib.rc('font', **font) 

    #matplotlib.rc('font', serif='Times New Roman') 
    #matplotlib.rc('text', usetex='false') 
    matplotlib.rcParams.update({'font.size': 14})
    # sigmoid mapping                                                                                                                                                                                                                         
    def sigmoidmapping(x):
        x = 1 -  ( mySigmoid( 2.0*(np.abs(x-1.0))**1.5 ) - 0.5 ) / (1.0-0.5) * (1.0-0.0)
        return x
        #return 1 -  ( mySigmoid( 8.0*(np.abs(x-1.0)/0.1)**1.5 ) - 0.5 ) / (1.0-0.5) * (1.0-0.0) 


    #user_batch = ['50k', '50000']
    #dataFile = '/home/ec2-user/yahoo_music_data/train_0.txt_%s_sorted' % user_batch[1]
    #Umf = np.load('/home/ec2-user/log/CCF-muse-yahoo-%s-step1_rank_15_15_lambda_1.0_1.0_alpha_1e-05_1e-05_iterations_50_50.log_Umf.npy' % user_batch[0] )
    #Inf = np.load('/home/ec2-user/log/CCF-muse-yahoo-%s-step1_rank_15_15_lambda_1.0_1.0_alpha_1e-05_1e-05_iterations_50_50.log_Inf.npy' % user_batch[0] )
    
    dataFile = '/home/ec2-user/MSD_data/train_data_sorted.csv_test'
    dataFile_train = '/home/ec2-user/MSD_data/train_data_sorted.csv_train'
    Umf = np.load('/home/ec2-user/MSD_log/binary_implicit_CCF/CCF-muse-MSD-step10_rank_15_15_lambda_1.0_1.0_alpha_5e-07_5e-07_iterations_50_50.log_Umf.npy')
    Inf = np.load('/home/ec2-user/MSD_log/binary_implicit_CCF/CCF-muse-MSD-step10_rank_15_15_lambda_1.0_1.0_alpha_5e-07_5e-07_iterations_50_50.log_Inf.npy')

    numMusics = Inf.shape[0]
    numUsers = Umf.shape[0]
    dfMusic = pd.read_csv(dataFile, delim_whitespace=True, names = ['user_id', 'song_id', 'rating'] )
    dfMusic_train = pd.read_csv(dataFile_train, delim_whitespace=True, names = ['user_id', 'song_id', 'rating'] )


    #hotUser = [1,2,3,4,5,6,10,1000,2000,3000,4000,5000]
    #hotUser = [1,]
    hotUser = np.random.choice(numUsers, 100 ).tolist()
    print 'Users under test: ', hotUser

    numHistoricalSong = 0
    y_true = np.array([])
    y_scores = np.array([])
    predictions_total = np.array([])
    random_predictions_total = np.array([])
    predictions_total_raw = np.array([])
    random_predictions_total_raw = np.array([])

    mAp = 0.0
    mynumUsers = len(hotUser)

    for ptr_hot_user in hotUser:

        historicalSongs =  dfMusic.loc[ dfMusic['user_id'] == ptr_hot_user  ]['song_id'].tolist()
        historicalSongs_train =  dfMusic_train.loc[ dfMusic_train['user_id'] == ptr_hot_user  ]['song_id'].tolist()
        numSongsThisUserRated = len(historicalSongs) + len(historicalSongs_train)
        if numSongsThisUserRated<20: continue
        print 'this user rated how many songs?', numSongsThisUserRated


        numHistoricalSong += len(historicalSongs)
        randomSelectedSongs = np.random.choice(numMusics, len(historicalSongs) ).tolist()
        for _ptr in randomSelectedSongs:
            if _ptr in historicalSongs_train or _ptr in historicalSongs:
                randomSelectedSongs.remove(_ptr)

        #print historicalSongs, historicalSongs_train, randomSelectedSongs

        hotUser = ptr_hot_user

    
        predictions = Umf[hotUser].dot(np.transpose( Inf[historicalSongs] ) ).flatten()
        #print predictions.shape, predictions


        random_predictions = Umf[hotUser].dot(np.transpose( Inf[randomSelectedSongs] ) ).flatten()
        #print random_predictions.shape, random_predictions


        predictions_total_raw = np.concatenate( (predictions_total_raw,predictions), axis = 0 )
        random_predictions_total_raw = np.concatenate( (random_predictions_total_raw, random_predictions), axis = 0 )
        
        predictions = sigmoidmapping(predictions)
        random_predictions = sigmoidmapping(random_predictions)
    
        #print predictions.shape, random_predictions.shape


        y_true_ptr = np.concatenate( (np.ones(len(predictions)) , np.zeros(len(random_predictions)) ), axis = 0)
        #print y_true_ptr.shape
        y_scores_ptr = np.concatenate( (predictions, random_predictions), axis = 0)
        #print y_scores_ptr.shape


        # precision@k, ap@k, and map@k                                                                                                                                                                                             
        #p_at_K = myPrecisionatK( y_scores_ptr, y_true_ptr, 10)
        #ap_at_k = myAbsolutePrecisionatK( y_scores_ptr, y_true_ptr, 10 )
        #print 'p@10 = %s ap@10 = %s' % (p_at_K, ap_at_k)
        #mAp += ap_at_k
        
        y_true = np.concatenate( (y_true, y_true_ptr), axis=0)
        y_scores = np.concatenate( (y_scores, y_scores_ptr), axis=0)
        predictions_total = np.concatenate( (predictions_total,predictions), axis = 0 )
        random_predictions_total = np.concatenate( (random_predictions_total, random_predictions), axis = 0 )

    print numHistoricalSong
    #mAp /= mynumUsers
    #print 'mAp @ 10 = %s' % (mAp)

    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    print 'AUC is %s ' % metrics.auc(fpr, tpr)
    

    plt.figure(1,facecolor='white', figsize=(14,10) )
    plt.subplot(221)
    myPlot(xrange(numHistoricalSong), predictions_total_raw , 'Music id', 'Implicit rating [arbitrary units]', 'Binary rating vs Music id', 'used', -0.5, 3.5)
    myPlot(xrange(numHistoricalSong), random_predictions_total_raw , '', '', '', 'random selected',-0.5,3.5)

    plt.subplot(222)
    myPlot(xrange(numHistoricalSong), predictions_total , 'Music id', 'Probability to use', 'Probability vs Music id', 'used', -0.5 , 1.5)
    myPlot(xrange(numHistoricalSong), random_predictions_total , '', '', '', 'random selected',-0.5,1.5)

    plt.subplot(223)
    myPlot(recall , precision, 'Recall', 'Precision', 'P-R curve', '',-0.2,1.2)
    plt.tight_layout()

    plt.subplot(224)
    myPlot( fpr, tpr , 'False Positive', 'True Positive', 'ROC curve', '',-0.2,1.2)
    plt.tight_layout()

    plt.show()


