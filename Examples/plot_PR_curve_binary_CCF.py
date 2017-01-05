
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


if __name__ == '__main__1':

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
    fractionOfTestSample = 0.001
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
    trainOrPredict = 'predict'
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
        
        myCCF = CCF(logFileName = logFileNamePrediction, starterModel = [ starterModelUmf, starterModelInf ], sc = sc )
        
        # here we add same amount (numTestSamples) of user_id,song_id pairs that has no ratings yet, not used yet
        keys_user_id_song_id = ratingRDD.map(lambda r: r[0] ).collect()
        keys_user_id_song_id_br = sc.broadcast(keys_user_id_song_id)

        norating_user_id_index = np.random.choice(numUsers, numTestSamples) 
        norating_song_id_index = np.random.choice(numMusics,numTestSamples)
        noratings = zip(norating_user_id_index, norating_song_id_index)
        noratingsRDD = sc.parallelize(noratings).map(lambda r: ((r[0],r[1]),0.0) ).filter(lambda r: r[0] not in keys_user_id_song_id_br.value )

        total_Used_NotUsed = sc.union([ noratingsRDD , X_test])

        logString = 'Debug: total number of test samples %s norating %s rating %s' % ( total_Used_NotUsed.count(), noratingsRDD.count(), X_test.count()  ) 
        logThis(logFileNamePrediction, logString, 'a')   
        

        X_test_predicted = myCCF.predict(total_Used_NotUsed)
        
        predictionsAndRatings = X_test_predicted.join(total_Used_NotUsed).values()
        predictionsAndRatingsArray = predictionsAndRatings.collect()
        predictionsAndRatingsArray = np.array(predictionsAndRatingsArray)
        logString = 'Debug: Prediction-Real'
        logThis(logFileNamePrediction, logString, 'a')

        

        with open(logFileNamePrediction,'a') as f_handle:
            np.savetxt(f_handle,predictionsAndRatingsArray)


        #logString = 'Debug: RMSE of the model is %s' % np.sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(numTestSamples))
        #logThis(logFileNamePrediction, logString, 'a')
        #meanRating = X_test.map(lambda x: x[1]).mean()
        #baselineRmse = np.sqrt(X_test.map(lambda x: (meanRating - x[1]) ** 2).reduce(add) / numTestSamples)
        #logString = 'Debug: RMSE of the referebce (average rating) is %s' % baselineRmse
        #logThis(logFileNamePrediction, logString, 'a')



if __name__ == '__main__':
    import psycopg2
    from museUtility import getDatabaseKey, myPlot, mySigmoid
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    from sklearn import metrics
    import matplotlib
    from MuseUtil.museUtility import myPrecisionatK, myAbsolutePrecisionatK, mymeanAbsolutePrecisionatK

    font = {'family' : 'Times',
            'weight' : 'bold',
            'size'   : '14'}
    
    matplotlib.rc('font', **font) 

    #matplotlib.rc('font', serif='Times New Roman') 
    #matplotlib.rc('text', usetex='false') 
    matplotlib.rcParams.update({'font.size': 14})
    # sigmoid mapping                                                                                                                                                                                                                         
    def sigmoidmapping(x):
        x = 1 -  ( mySigmoid( 80*(np.abs(x-1.0))**2.5 ) - 0.5 ) / (1.0-0.5) * (1.0-0.0)
        return x

    
    myMuseBase = '/home/ec2-user/muse'
    myKeyDir = myMuseBase + '/WebUI/keys'
    myPostgresKey = myKeyDir + "/.credentials.cred"
    USER_CRED_FILE = os.path.abspath(myPostgresKey)
    print USER_CRED_FILE
    Umf = np.load('/home/ec2-user/log/CCF-muse-1M-binary-step13_rank_10_10_lambda_1.0_1.0_alpha_0.0001_0.0001_iterations_50_50.log_Umf.npy')
    Inf = np.load('/home/ec2-user/log/CCF-muse-1M-binary-step13_rank_10_10_lambda_1.0_1.0_alpha_0.0001_0.0001_iterations_50_50.log_Inf.npy')
    numMusics = Inf.shape[0]


    conn = psycopg2.connect(getDatabaseKey(dbcred_file=USER_CRED_FILE))
    cursor = conn.cursor()
    #cursor.execute("SELECT DISTINCT user_id, COUNT(rating)  FROM master_music_rating_total GROUP BY user_id ORDER BY COUNT(rating) DESC LIMIT 10")
    #queryResult = cursor.fetchall()
    #hotUser = queryResult[0]
    #print queryResult
    #sys.exit(0)
    #hotUser = '316110734d8da7478cc33237458814f770a9eb7a'
    hotUser = ['316110734d8da7478cc33237458814f770a9eb7a',
               'ad4b2717e89766b66b96fe52a38736116e315874',
              # '3233c598c50f0ddbd351504e773cd51de79691db',
              # 'd30e18323f15426c3cdc8585252ed34459916f51', 
              # '016a24e91a72c159a5048ab1b9b2ba5ce761b526', 
              # '0f8308935bcbb9a1e04ebb7c4d41c037e5f23b90', 
              # '9b0f827c8bad0cf089b0d778307e1b390f463730', 
              # '03ad93fdb01506ce205f4708decf8e4b1ae90fff', 
              # '70060f2eba3f2486a7a147546adf4e6b1660e295', 
              # '7d90be8dfdbde170f036ce8a4b915440137cb11c',
]
    
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
        cursor.execute("SELECT song_id, song_id_index FROM song_profile WHERE song_id IN (SELECT DISTINCT song_id FROM master_music_rating_total WHERE user_id = '%s' ) " % ptr_hot_user  )
        queryResult = cursor.fetchall()  
        historicalSongs = [i[1] for i in queryResult]
        numHistoricalSong += len(historicalSongs)
        randomSelectedSongs = np.random.choice(numMusics, len(historicalSongs) )
        
        cursor.execute("SELECT user_id_index, user_id FROM users_profile WHERE user_id IN %s" , (tuple([ptr_hot_user]),) )
        queryResult = cursor.fetchall()
        hotUser = [i[0] for i in queryResult]

    
        predictions = Umf[hotUser].dot(np.transpose( Inf[historicalSongs] ) ).flatten()
        print predictions.shape


        random_predictions = Umf[hotUser].dot(np.transpose( Inf[randomSelectedSongs] ) ).flatten()
        print random_predictions.shape


        predictions_total_raw = np.concatenate( (predictions_total_raw,predictions), axis = 0 )
        random_predictions_total_raw = np.concatenate( (random_predictions_total_raw, random_predictions), axis = 0 )
        
        predictions = sigmoidmapping(predictions)
        random_predictions = sigmoidmapping(random_predictions)
    
        print predictions.shape, random_predictions.shape


        y_true_ptr = np.concatenate( (np.ones(len(predictions)) , np.zeros(len(random_predictions)) ), axis = 0)
        print y_true_ptr.shape
        y_scores_ptr = np.concatenate( (predictions, random_predictions), axis = 0)
        print y_scores_ptr.shape


        # precision@k, ap@k, and map@k                                                                                                                                                                                             
        p_at_K = myPrecisionatK( y_scores_ptr, y_true_ptr, 10)
        ap_at_k = myAbsolutePrecisionatK( y_scores_ptr, y_true_ptr, 10 )
        print 'p@10 = %s ap@10 = %s' % (p_at_K, ap_at_k)
        mAp += ap_at_k
        
        y_true = np.concatenate( (y_true, y_true_ptr), axis=0)
        y_scores = np.concatenate( (y_scores, y_scores_ptr), axis=0)
        predictions_total = np.concatenate( (predictions_total,predictions), axis = 0 )
        random_predictions_total = np.concatenate( (random_predictions_total, random_predictions), axis = 0 )

    print numHistoricalSong
    mAp /= mynumUsers
    print 'mAp @ 10 = %s' % (mAp)

    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    print 'AUC is %s ' % metrics.auc(fpr, tpr)
    

    plt.figure(1,facecolor='white', figsize=(14,10) )
    plt.subplot(221)
    myPlot(xrange(numHistoricalSong), predictions_total_raw , 'Music id', 'Implicit rating [arbitrary units]', 'Binary rating vs Music id', 'used', 0.5, 3.5)
    myPlot(xrange(numHistoricalSong), random_predictions_total_raw , '', '', '', 'random selected',0.5,3.5)

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

    np.save('ground_truth', y_true)
    np.save('predicted_score', y_scores)

