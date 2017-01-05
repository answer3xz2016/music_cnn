
##########################################################################################
# This is an example scipt to train CCF model
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


def convertRatingToDecimal(r):
    local_map_user_id = map_user_id_br.value
    local_map_song_id = map_song_id_br.value
    return (local_map_user_id[r[0]], local_map_song_id[r[1]]) , r[2]

def convertRatingToBinary(r):
    local_map_user_id = map_user_id_br.value
    local_map_song_id = map_song_id_br.value
    return (local_map_user_id[r[0]], local_map_song_id[r[1]]) , 1.0 

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

def myTrain_Test_Split_Default(ratingsRDD, testSize):
    X_train, X_test = ratingsRDD.randomSplit([1-testSize, testSize],17)
    return X_train, X_test


if __name__ == '__main__':

    if len(sys.argv)>1:
        myStep = int(sys.argv[1])
        myPreviousStep = int(myStep)-1
    else:
        myStep = 0
        myPreviousStep = 0

    homeDir = expanduser("~")
    museDir = '/home/ec2-user/muse'
    jarPath= homeDir + "/postgresql-9.4.1208.jre7.jar" # this file has to be there on every worker node
    urlPath = "jdbc:postgresql://dmdbmusic.cydl1z5zjcto.us-west-2.rds.amazonaws.com/dmdbmusic?user=postgres&password=j0yfulc4r"
    myRank = [10]
    myLambda = [1.0]
    myAlpha = [0.0001] # 0.0001 sweet spot
    myIteration = [50]
    fractionOfTestSample = 0.33
    configModel = '_rank_%s_%s_lambda_%s_%s_alpha_%s_%s_iterations_%s_%s' % (myRank[0], myRank[-1], myLambda[0], myLambda[-1], myAlpha[0], myAlpha[-1], myIteration[0], myIteration[-1])
    numPartitions = 1000  # 1000 sweet spot
    logFileFolder = homeDir + '/log'
    if not os.path.exists(logFileFolder) or not os.path.isdir(logFileFolder):
        os.makedirs(logFileFolder)
    logFileName = logFileFolder + '/CCF-muse-1M-binary-step%s' % myStep + configModel + '.log'
    logFileNamePrediction = logFileFolder + '/CCF-muse-1M_predictions' + '.log'
    testBatchSize = None #None  # 1491952
    if testBatchSize:
        testBatchSizeStr = 'LIMIT %s' % testBatchSize
    else:
        testBatchSizeStr = ''
    # you can input some starter model for the CF                                                                                                                                                       
    starterModelUmf = logFileFolder + '/CCF-muse-1M-binary-step%s' % myPreviousStep + configModel + '.log_Umf.npy' 
    starterModelInf = logFileFolder + '/CCF-muse-1M-binary-step%s' % myPreviousStep + configModel + '.log_Inf.npy' 
    trainOrPredict = 'train'
    localOrRemoteCluster = 'remote'
    if trainOrPredict == 'predict':
        localOrRemoteCluster = 'local'
        numPartitions = 4
        starterModelUmf = logFileFolder + '/CCF-muse-1M-binary-step%s' % myStep + configModel + '.log_Umf.npy'
        starterModelInf = logFileFolder + '/CCF-muse-1M-binary-step%s' % myStep + configModel + '.log_Inf.npy'
  
    GeneratingUserSongIdMap = False
        

    #========================================================================================================================
    #========================================================================================================================
    #========================================================================================================================
    #========================================================================================================================




    if localOrRemoteCluster == 'local':
        #local cluster
        # this line below has go before spark conf
        os.environ['SPARK_CLASSPATH'] = jarPath
        conf = SparkConf().setMaster("local[*]").setAppName("CCF-muse")                                                                                                        
    
    else:
        #remote cluster
        conf = SparkConf().set("spark.deploy.defaultCores", 2).set("spark.cores.max", 10000000).set("spark.executor.memory", "8G").set("spark.executor.extraClassPath", jarPath).setMaster("spark://10.118.132.141:7077").setAppName("CCF-muse")

    sc = SparkContext(conf=conf)
    sqlctx = SQLContext(sc)
    
    query = '(SELECT user_id,song_id,rating, artist_7digitalid, artist_familiarity, artist_hotttnesss, danceability, energy, loudness,tempo FROM master_music_rating_total ORDER BY user_id %s ) tmp' % testBatchSizeStr

    df = sqlctx.read.format('jdbc').options( source="jdbc", url=urlPath, dbtable=query,
                                            partitionColumn='artist_7digitalid',
                                            lowerBound = -1,
                                            upperBound = 817066,
                                            numPartitions = numPartitions,
                                           ).load().repartition(numPartitions)



    df.registerTempTable("dfMaster")
    df.cache()
    df.show()

    

    dfUser = sqlctx.sql("SELECT DISTINCT user_id FROM dfMaster ORDER BY user_id") # let's sort the user ID and song ID here so that the indexing is always the same
    dfSong = sqlctx.sql("SELECT DISTINCT song_id FROM dfMaster ORDER BY song_id") # let's sort the user ID and song ID here so that the indexing is always the same
    list_user_id = dfUser.map(lambda r: r.user_id ).collect()
    list_song_id = dfSong.map(lambda r: r.song_id).collect()
    numUsers = len( list_user_id )
    numMusics = len( list_song_id )
    # here we encode the user id and song id from categorical to numeric features                                                                                                                
    map_user_id  = dict(zip( list_user_id, xrange(0,numUsers) ))
    map_song_id = dict(zip( list_song_id, xrange(0,numMusics) ))
    # broadcast it, so that every rdd block can see this map                                                                                                                                     
    map_user_id_br = sc.broadcast(map_user_id)
    map_song_id_br = sc.broadcast(map_song_id)
   
    
    # here we want to write out a userId <-> userId_index and songId <-> songId_index map to database
    ###############################################################################
    #### save these two maps to SQL datastore
    if GeneratingUserSongIdMap:
        tuple_user_id = [(k,v) for k,v in map_user_id.items() ] 
        tuple_song_id = [(k,v) for k,v in map_song_id.items() ]
        
        df_map_user_id = sc.parallelize(tuple_user_id).toDF(['user_id','user_id_index'])
        df_map_song_id = sc.parallelize(tuple_song_id).toDF(['song_id','song_id_index'])
                
        df_map_user_id.write.jdbc("jdbc:postgresql://dmdbmusic.cydl1z5zjcto.us-west-2.rds.amazonaws.com/dmdbmusic?user=postgres&password=j0yfulc4r","users_profile",mode = "overwrite")
        df_map_song_id.write.jdbc("jdbc:postgresql://dmdbmusic.cydl1z5zjcto.us-west-2.rds.amazonaws.com/dmdbmusic?user=postgres&password=j0yfulc4r","song_profile",mode = "overwrite")
    
        sys.exit(0)
    ###############################################################################



 
    ratingRDD = df.select('user_id', 'song_id', 'rating').map(convertRatingToBinary) # here we map ratings to 1 or 0,  listened or not listened
    ratingRDD.cache()
    

    if trainOrPredict == 'train':
        # training
        logString =  'Debug: Training RDD mem level: ' + str( ratingRDD.getStorageLevel() )
        logThis(logFileName, logString, 'w')
        logString = 'Debug: Training RDD num of partitions: ' + str( ratingRDD.getNumPartitions() )
        logThis(logFileName, logString, 'a')


        logString = 'Debug: what is mystep %s' % myStep
        logThis(logFileName, logString, 'a')

        if myStep > 0:
            logString = 'Debug: now it needs a starting point' 
            logThis(logFileName, logString, 'a')
            myCCF = CCF(logFileName = logFileName, starterModel = [ starterModelUmf, starterModelInf ], sc = sc )
        else:
            logString = 'Debug: now it does not need a starting point'
            logThis(logFileName, logString, 'a')
            myCCF = CCF(logFileName = logFileName, starterModel = None , sc = sc )


        for _rank, _lmbda, _numIter, _alpha in itertools.product(myRank, myLambda, myIteration, myAlpha): 
            logString = "Debug: New model config, we train at rank = %d, lambda = %f, num_iterations = %d, alpha = %f.\n" % (_rank, _lmbda, _numIter, _alpha)
            logThis(logFileName, logString, 'a')
        
            status = myCCF.train(ratingRDD, rank= _rank , iterations= _numIter, lambda_= _lmbda, alpha= _alpha )
            #myCCF.train(ratingRDD, rank=3 , iterations=1, lambda_= 0.1, alpha=0.001)
        
            if status:
                _alpha *= 0.1
                logString = "Debug: Previous config blowed up, change to 10% alpha, New model config, we train at rank = %d, lambda = %f, num_iterations = %d, alpha = %f.\n" % (_rank, _lmbda, _numIter, _alpha)
                logThis(logFileName, logString, 'a')
                status = myCCF.train(ratingRDD, rank= _rank , iterations= _numIter, lambda_= _lmbda, alpha= _alpha )

            myCCF.showModel()
            
            if len(myRank) == 1 and len(myLambda) == 1 and len(myAlpha) == 1 and len(myIteration) == 1:
                myCCF.saveModel()
            else:
                myCCF.saveModel('_rank_%d_lambda_%f_num_iterations_%d_alpha_%f_' % (_rank, _lmbda, _numIter, _alpha) )


    else:
        # prediction
        X_train, X_test = myTrain_Test_Split_Default(ratingRDD, fractionOfTestSample)
        logString =  'Debug: Testing RDD mem level: ' + str( X_test.getStorageLevel() )
        logThis(logFileNamePrediction, logString, 'w')
        logString = 'Debug: Testing RDD num of partitions: ' + str( X_test.getNumPartitions() )
        logThis(logFileNamePrediction, logString, 'a')

        numTestSamples = X_test.count()
        logString = 'Debug: Testing RDD contains %s samples' %  numTestSamples
        logThis(logFileNamePrediction, logString, 'a')
        
        myCCF = CCF(logFileName = logFileNamePrediction, starterModel = [ starterModelUmf, starterModelInf ] )
        X_test_predicted = myCCF.predict(X_test)
        
        predictionsAndRatings = X_test_predicted.join(X_test).values()
        logString = 'Debug: RMSE of the model is %s' % np.sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(numTestSamples))
        logThis(logFileNamePrediction, logString, 'a')

        meanRating = X_test.map(lambda x: x[1]).mean()
        baselineRmse = np.sqrt(X_test.map(lambda x: (meanRating - x[1]) ** 2).reduce(add) / numTestSamples)
        logString = 'Debug: RMSE of the referebce (average rating) is %s' % baselineRmse
        logThis(logFileNamePrediction, logString, 'a')
