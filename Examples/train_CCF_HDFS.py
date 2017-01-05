
##########################################################################################
# This is an example scipt to train CCF model using data on HDFS
##########################################################################################

import os
import sys
from os.path import expanduser
import itertools



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
    myAlpha = [0.00001] # 0.0001 sweet spot                                                                                                                                                                  
    myIteration = [50]
    homeDir = expanduser("~")
    configModel = '_rank_%s_%s_lambda_%s_%s_alpha_%s_%s_iterations_%s_%s' % (myRank[0], myRank[-1], myLambda[0], myLambda[-1], myAlpha[0], myAlpha[-1], myIteration[0], myIteration[-1])
    logFileFolder = homeDir + '/yahoo_log'
    if not os.path.exists(logFileFolder) or not os.path.isdir(logFileFolder):
        os.makedirs(logFileFolder)
    logFileName = logFileFolder + '/CCF-muse-yahoo-step%s' % myStep + configModel + '.log'
    # you can input some starter model for the CF                                                                                                                                                            
    starterModelUmf = logFileFolder + '/CCF-muse-yahoo-step%s' % myPreviousStep + configModel + '.log_Umf.npy'
    starterModelInf = logFileFolder + '/CCF-muse-yahoo-step%s' % myPreviousStep + configModel + '.log_Inf.npy'
    

    #conf = SparkConf().setMaster("local[*]").setAppName("CCF-muse-HDFS")
    conf = SparkConf().set("spark.deploy.defaultCores", 2).set("spark.cores.max", 10000000).set("spark.executor.memory", "8G")\
        .setMaster("spark://10.118.132.141:7077").setAppName("CCF-muse-HDFS")
    sc = SparkContext(conf=conf)
    ratingRDD = sc.textFile("hdfs://ec2-52-41-137-208.us-west-2.compute.amazonaws.com:9000/yahoo_music_data/train_0.txt").map(lambda r: r.split()).map(lambda r: ( (int(r[0]), int(r[1])) , float(r[2])  )  ).repartition(1000)
    ratingRDD.cache()

    logString =  'Debug: Training RDD mem level: ' + str( ratingRDD.getStorageLevel() )
    logThis(logFileName, logString, 'w')

    logString = 'Debug: Training RDD num of partitions: ' + str( ratingRDD.getNumPartitions() )
    logThis(logFileName, logString, 'a')

    logString = 'Debug: what is mystep %s' % myStep
    logThis(logFileName, logString, 'a')
    
    logString = 'Debug: now it does not need a starting point'
    logThis(logFileName, logString, 'a')

    if myStep > 0:
        logString = 'Debug: now it needs a starting point'
        logThis(logFileName, logString, 'a')
        myCCF = CCF(logFileName = logFileName, starterModel = [ starterModelUmf, starterModelInf ], sc = sc )
    else:
        myCCF = CCF(logFileName = logFileName, starterModel = None , sc = sc )

    for _rank, _lmbda, _numIter, _alpha in itertools.product(myRank, myLambda, myIteration, myAlpha):
        logString = "Debug: New model config, we train at rank = %d, lambda = %f, num_iterations = %d, alpha = %f.\n" % (_rank, _lmbda, _numIter, _alpha)
        logThis(logFileName, logString, 'a')

        status = myCCF.train(ratingRDD, rank= _rank , iterations= _numIter, lambda_= _lmbda, alpha= _alpha )

        if len(myRank) == 1 and len(myLambda) == 1 and len(myAlpha) == 1 and len(myIteration) == 1:
            myCCF.saveModel()


    
    #userRDD = text_file.map(lambda r: r[0] )
    #musicRDD = text_file.map(lambda r: r[1] )
    #ratingRDD = text_file.map(lambda r: r[2] )

    #print 'test zhou', userRDD.max(), userRDD.min(), userRDD.distinct().count()
    #print 'test zhou', musicRDD.max(), musicRDD.min(), musicRDD.distinct().count()
    #print 'test zhou', ratingRDD.max(), ratingRDD.min(), ratingRDD.distinct().count()

