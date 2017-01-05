
##########################################################################################
# This is an example scipt to test CCF model using data on HDFS
##########################################################################################

import os
import sys
from os.path import expanduser
import itertools
import numpy as np
from operator import add

# pyspark modules
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
# in-house modules
from MuseModels.CCF import CCF


def logThis(fileName, content, mode):
    print content
    if mode == 'w':
        with open(fileName, mode) as myFile:
            myFile.write(content)
            myFile.write('\n')
    elif mode == 'a':
        with open(fileName, mode) as myFile:
            myFile.write(content)
            myFile.write('\n')



if __name__ == '__main__':
    if len(sys.argv)>1:
        myStep = int(sys.argv[1])
        myPreviousStep = int(myStep)-1
    else:
        myStep = 0
        myPreviousStep = 0

    myRank = [15]
    myLambda = [1.0]
    myAlpha = [5e-07] # [0.00001] # 0.0001 sweet spot                                                                                                                                                                
    myIteration = [50]
    homeDir = expanduser("~")
    configModel = '_rank_%s_%s_lambda_%s_%s_alpha_%s_%s_iterations_%s_%s' % (myRank[0], myRank[-1], myLambda[0], myLambda[-1], myAlpha[0], myAlpha[-1], myIteration[0], myIteration[-1])
    logFileFolder = homeDir + '/MSD_log/binary_implicit_CCF_new'
    if not os.path.exists(logFileFolder) or not os.path.isdir(logFileFolder):
        os.makedirs(logFileFolder)
    logFileName= logFileFolder + '/CCF-muse-MSD_predictions_test' + '.log'
    # you can input some starter model for the CF                                                                                                                                                          
    starterModelUmf = logFileFolder + '/CCF-muse-MSD-step%s' % myStep + configModel + '.log_Umf.npy'
    starterModelInf = logFileFolder + '/CCF-muse-MSD-step%s' % myStep + configModel + '.log_Inf.npy'
    

    #conf = SparkConf().setMaster("local[*]").setAppName("CCF-muse-HDFS")
    conf = SparkConf().set("spark.deploy.defaultCores", 2).set("spark.cores.max", 10000000).set("spark.executor.memory", "8G")\
        .setMaster("spark://10.118.132.141:7077").setAppName("CCF-muse-HDFS")
    sc = SparkContext(conf=conf)
    
    # yahoo data
    #ratingRDD = sc.textFile("hdfs://ec2-52-41-137-208.us-west-2.compute.amazonaws.com:9000/yahoo_music_data/test_0.txt").map(lambda r: r.split()).map(lambda r: ( (int(r[0]), int(r[1])) , int(r[2]) )  ).repartition(1000)

    # MSD data
    ratingRDD = sc.textFile("hdfs://ec2-52-41-137-208.us-west-2.compute.amazonaws.com:9000/MSD_data/train_data_sorted.csv_test").map(lambda r: r.split()).map(lambda r: ( (int(r[0]), int(r[1])) , 1.0 )  ).repartition(100)
    

    ratingRDD.cache()

    logString =  'Debug: Testing RDD mem level: ' + str( ratingRDD.getStorageLevel() )
    logThis(logFileName, logString, 'w')

    logString = 'Debug: Testing RDD num of partitions: ' + str( ratingRDD.getNumPartitions() )
    logThis(logFileName, logString, 'a')

    logString = 'Debug: what is mystep %s' % myStep
    logThis(logFileName, logString, 'a')
    
    numTestSamples = float(ratingRDD.count())
    
    logString = 'Debug: now it needs a starting point'
    logThis(logFileName, logString, 'a')
    myCCF = CCF(logFileName = logFileName, starterModel = [ starterModelUmf, starterModelInf ], sc = sc )
    
    X_test_predicted = myCCF.predict(ratingRDD)
    
    predictionsAndRatings = X_test_predicted.join(ratingRDD).values()
    logString = 'Debug: RMSE of the model is %s' % np.sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(numTestSamples))
    logThis(logFileName, logString, 'a')
    
    #meanRating = ratingRDD.map(lambda x: x[1]).mean()
    meanRating = 0.5
    baselineRmse = np.sqrt(ratingRDD.map(lambda x: (meanRating - x[1]) ** 2).reduce(add) / numTestSamples)
    logString = 'Debug: RMSE of the reference (average rating) is %s' % baselineRmse
    logThis(logFileName, logString, 'a')
    


