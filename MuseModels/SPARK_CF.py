#
# # Binary Recommendation System Implemented using SPARK RDD
#     Author: Dr. Z. Xing
#     Contact: joe.xing@nextev.com
#     Company: NextEV, Data Science Group
# 
# ### Introduction:
# 
#     This is an SPARK implementation of the Binary Recommender system using implicity user and potentially item profile data
# 
# ### Reference
#     Here is some reference papers that describes research on Binary Recommender:
#     
#     Here is the document that shows the feature data in our music database at AWS RDS
#     https://nextev.githost.io/data-science/Documents/blob/master/Music_database_definitions.docx
# 
#     

# #### Define some userful utility functions

# In[1]:

# Function that creates a feature vector from a (user,item,rating) tuple
def createFMData(arr):
    all_users = usersBroadcastVar.value
    all_items = itemsBroadcastVar.value
    numusers = len(all_users)
    numitems = len(all_items)
    useridx = {}
    itemidx = {}
    
    for i in range(0,numusers):
        useridx[int(all_users[i])] = i
    for i in range(0,numitems):
        itemidx[int(all_items[i])] = i
        
    x = [0 for i in range(0,numusers+numitems+1)] # +1 is for the rating at the end
    user_id = int(arr[0])
    item_id = int(arr[1])
    rating  = arr[2]
    if useridx.has_key(user_id):
        x[useridx[user_id]] = 1.0
    if itemidx.has_key(item_id):
        x[numusers+itemidx[item_id]] = 1.0
    x[-1] = rating
    return x

# Utility function for encode categorical features into numeric data
def encodeCategoricalFeatures(df, columnNames):
    # Encode the user_id and song_id to numeric values so that they can be used by Spark mllib
    for i in columnNames:
        indexer = StringIndexer(inputCol = i, outputCol= i + "_index")
        indexer.setHandleInvalid("skip")
        df = indexer.fit(df).transform(df)
    return df

def myTrain_Test_Split(ratingsRDD, myUsersMap, myItemsMap, totalSampleSize, test_size):
    myUsersMapCopy = myUsersMap.copy()
    myItemsMapCopy = myItemsMap.copy()
    myData = ratingsRDD.collect()
    testSampleSize = int(test_size*totalSampleSize)
    indices = sample(xrange(totalSampleSize),testSampleSize)
    X_train, X_test = [],[]
    for index in indices:
        if myItemsMapCopy[ myData[index][1] ] > 1 and myUsersMapCopy[ myData[index][0]  ] > 1:
            X_test.append( myData[index] )
            myItemsMapCopy[ myData[index][1] ] -= 1
            myUsersMapCopy[ myData[index][0]  ] -= 1
        else:
            X_train.append( myData[index] )
            
    X_train.extend( list(np.delete(np.array(myData),indices, axis=0 )) )
    return sc.parallelize(X_train), sc.parallelize(X_test)

def myTrain_Test_Split_Default(ratingsRDD, testSize):
    X_train, X_test = ratingsRDD.randomSplit([1-testSize, testSize],17)
    return X_train, X_test
    
def myTrain_Test_Split_Best(ratingsRDD):
    numPartitions = 4
    training = ratingsRDD.filter(lambda x: x[0] < 6)       .repartition(numPartitions)       .cache()

    validation = ratingsRDD.filter(lambda x: x[0] >= 6 and x[0] < 8)       .repartition(numPartitions)       .cache()

    test = ratingsRDD.filter(lambda x: x[0] >= 8).values().cache()

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()
    
    return training, validation, test

def myLoadPostgreDataPandas( local_or_cluster='local', homeDir = '~' ):
    
    if local_or_cluster == 'cluster':
        # run on clusters
        jarPath= homeDir + "/postgresql-9.4.1208.jre7.jar"
        conf = SparkConf().set("spark.deploy.defaultCores", 2).set("spark.cores.max", 10000000).set("spark.executor.memory", "8G").set("spark.executor.extraClassPath", jarPath).setMaster("spark://10.118.132.141:7077").setAppName("Binary_recommender")
    else:
        # run on local machine
        jarPath= homeDir + "/postgresql-9.4.1208.jre7.jar"
        os.environ['SPARK_CLASSPATH'] = jarPath
        conf = SparkConf().setMaster("local[*]").setAppName("Binary_recommender")                                                                                                                                                                                  
    sc = SparkContext(conf=conf)
    sqlctx = SQLContext(sc)
    #df = sqlctx.read.format('jdbc').options( source="jdbc", url=urlPath, dbtable=query,         
    #                                        partitionColumn='artist_7digitalid', 
    #                                        lowerBound = -1,
    #                                        upperBound = 817066,
    #                                        numPartitions = 200, 
    #                                       ).load()#.repartition(200)
    return sc, sqlctx

def convertRatingToBinary(r):
    local_map_user_id = map_user_id_br.value
    local_map_song_id = map_song_id_br.value
    return local_map_user_id[r[0]], local_map_song_id[r[1]], 1.0

def convertRatingToDecimal(r):
    local_map_user_id = map_user_id_br.value
    local_map_song_id = map_song_id_br.value
    #return local_map_user_id[ r[0] ], local_map_song_id[ r[1] ], RatingScale if r[2]> RatingScale else r[2]
    return local_map_user_id[ r[0] ], local_map_song_id[ r[1] ], r[2]

    #return r[0] , r[1] , 10 if r[2]>10 else r[2]

def convertRatingToDecimal2(r):
    local_map_user_id = map_user_id_br.value
    local_map_song_id = map_song_id_br.value
    local_map_user_rating_max = map_user_rating_max_br.value
    #local_map_user_rating_min = map_user_rating_min_br.value

    maxRateThisUser = local_map_user_rating_max[ r[0] ]
    minRateThisUser = 1.0
    maxRate = RatingScale
    minRate = RatingScaleMin
    if maxRateThisUser <= minRateThisUser:
        score = 1
    else:
        score = (r[2]-minRateThisUser)/(maxRateThisUser-minRateThisUser)*(maxRate-minRate) + minRate
    

    return local_map_user_id[ r[0] ], local_map_song_id[ r[1] ], score


def computeRmse(model, data, n , sc):
    """
    Compute RMSE (Root Mean Squared Error).
    Assume data RDD is typical triplet format: userId -> itemId -> rating
    """
    truth = data.map( lambda x: ((x[0], x[1]), x[2]) )
    truth.cache()
    ##print 'test zhou 0.....', truth.count() , '............', truth.take(10)

    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictions.cache()
    # here let's rescale predicted ratings to 0-10 scale
    maxPrediction = predictions.map(lambda x: x[2]).max()
    minPrediction = predictions.map(lambda x: x[2]).min()
    maxRate = RatingScale
    minRate = RatingScaleMin
    ##print 'test zhou 1......', predictions.count(), '............', predictions.take(10)

    #predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), (x[2]-minPrediction)/(maxPrediction-minPrediction)*(maxRate-minRate)+minRate  )).join(data.map(lambda x: ((x[0], x[1]), x[2]))).values()


    #predictedRating = predictions.map(lambda x: ((x[0], x[1]), (x[2]-minPrediction)/(maxPrediction-minPrediction)*(maxRate-minRate)+minRate  ) )
    predictedRating = predictions.map(lambda x: ((x[0], x[1]), x[2]  ) )
    predictedRating.cache()
    ##predictedRating.checkpoint()
    ##print 'test zhou 2.......', predictedRating.count(), '............', predictedRating.take(10)


    


    predictionsAndRatings = predictedRating.join(truth).values()
    #predictionsAndRatings = sc.union(predictedRating, truth)
    predictionsAndRatings.cache()
    #print 'test zhou 3........', predictionsAndRatings.count(), '............', predictionsAndRatings.take(10)
    #predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])).join(data.map(lambda x: ((x[0], x[1]), x[2]))).values()
    
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))
    #return 1.0

def computeRecall(model, data, n , sc):
    truth = data.map( lambda x: ((x[0], x[1]), x[2]) )
    truth.cache()
    numTP = truth.count()
    print 'test zhou 0.....', numTP , '............', truth.take(10)  
    userRdd = data.map( lambda x: x[0])
    songRdd = data.map( lambda x: x[1])
    userRdd.cache()
    songRdd.cache()
    
    #dummy, userSample = userRdd.randomSplit([0.9, 0.1],17)
    #dummy, songSample = songRdd.randomSplit([0.9, 0.1],17)
    userSample = userRdd 
    songSample = songRdd
    numUserSample = userSample.count()
    numSongSample = songSample.count()
    print 'test zhou 0-1.....', numUserSample , '............', numSongSample
    
    userSongRdd = userSample.cartesian(songSample).map(lambda x: ( (x[0],x[1]), 0 ) )
    #userSongRdd.cache() #??????
    #notUsed = sc.parallelize( userSongRdd.takeSample(False, numTP, 3) )
    fractionToSample = float(numTP) / (numUserSample*numSongSample) * 0.5
    dummy, notUsed = userSongRdd.randomSplit([1-fractionToSample , fractionToSample],17)
    

    notUsed.cache()
    print 'test zhou 0-2.....', notUsed.count() , '............', notUsed.take(10)
    notUsed = notUsed.subtractByKey(truth)
    notUsed.cache()
    print 'test zhou 1.....', notUsed.count() , '............', notUsed.take(10)
    Used = truth
    total_Used_NotUsed = sc.union([notUsed,Used])
    total_Used_NotUsed.cache()
    print 'test zhou 2.....', total_Used_NotUsed.count() , '............', total_Used_NotUsed.take(10)

    predictions = model.predictAll( total_Used_NotUsed.map(lambda x: x[0] ) )
    predictions.cache()
    
    print 'test zhou 2-1.....', predictions.count() , '............', predictions.take(10)

    maxPrediction = predictions.map(lambda x: x[2]).max()
    minPrediction = predictions.map(lambda x: x[2]).min()
    maxRate = 1.0
    minRate = 0.0

    predictedRating = predictions.map(lambda x: ((x[0], x[1]), (x[2]-minPrediction)/(maxPrediction-minPrediction)*(maxRate-minRate)+minRate  ) )
    
    predictionsAndRatings = predictedRating.join(total_Used_NotUsed).values()
    predictionsAndRatings.cache()
    print 'test zhou 3.....', predictionsAndRatings.count() , '............', predictionsAndRatings.take(10)
    
    y_scores = predictionsAndRatings.map(lambda x: x[0]).collect()
    y_true = predictionsAndRatings.map(lambda x: x[1]).collect()
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    
    
    return precision, recall, thresholds

def logThis(fileName, content, mode):
    if mode == 'w':
        with open(fileName, mode) as myFile:
            myFile.write(content)
    elif mode == 'a':
        with open(fileName, mode) as myFile:
            myFile.write(content)


# In[2]:

#%matplotlib inline


# In[3]:

import numpy as np
from sklearn.metrics import precision_recall_curve
import pandas
import os, sys
from os.path import expanduser
import datetime
import time
from random import sample
from math import sqrt
from operator import add
import itertools
from pyspark.mllib.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SQLContext, Row
from pyspark import SparkConf, SparkContext
import pyspark
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
sns.set_style("whitegrid")
sns.set_context("talk")
# Import my Util modules

#pythonsrc = os.getcwd() # for jupyter notebook
#pythonsrc = os.path.abspath(__file__) # for python
#pythonsrc = os.path.join(pythonsrc,'../Util')
#pythonsrc = os.path.abspath( pythonsrc )
#sys.path.append( pythonsrc )


# In[4]:
BinaryRating = False
GeneratingModel = False

homeDir = expanduser("~")
ts = time.time()
timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')

if GeneratingModel:
    fileName = homeDir + "/myCollaborativeFilterNoTest"
else:
    fileName = homeDir + "/myCollaborativeFilter"

logFileName = fileName + '_' + timeStamp + '.log'
urlPath = "jdbc:postgresql://dmdbmusic.cydl1z5zjcto.us-west-2.rds.amazonaws.com/dmdbmusic?user=postgres&password=j0yfulc4r"
RatingScale = 5.0
RatingScaleMin = 1.0
clusterOrLocalMode = 'cluster'
numIterations = 17

    

if BinaryRating:
    testBatchSize = ''
    nonNegativeRating = True
    fractionOfTestSample = 0.01
else:
    testBatchSize = ''
    nonNegativeRating = False
    fractionOfTestSample = 0.33

# In[5]:

# Get the SPARK SQL dataframe of the ratings and music metadata from AWS RDS
sc, sqlctx = myLoadPostgreDataPandas(local_or_cluster = clusterOrLocalMode , homeDir = homeDir  )

query = '(SELECT DISTINCT user_id FROM master_music_rating_total ) tmp1'
df_user_id = sqlctx.read.format('jdbc').options( source="jdbc", url=urlPath, dbtable=query).load()
df_user_id.cache()
list_user_id = df_user_id.map(lambda r: r.user_id ).collect()

query = '(SELECT DISTINCT song_id FROM master_music_rating_total ) tmp2'
df_song_id = sqlctx.read.format('jdbc').options( source="jdbc", url=urlPath, dbtable=query).load()
df_song_id.cache()
list_song_id = df_song_id.map(lambda r: r.song_id).collect()

numUsers = len( list_user_id )
numMusics = len( list_song_id )

# here we encode the user id and song id from categorical to numeric features
map_user_id  = dict(zip( list_user_id, xrange(0,numUsers) ))
map_song_id = dict(zip( list_song_id, xrange(0,numMusics) ))
# broadcast it, so that every rdd block can see this map
map_user_id_br = sc.broadcast(map_user_id)
map_song_id_br = sc.broadcast(map_song_id)


###############################################################################
#### save these two maps to SQL datastore
if GeneratingModel:
    tuple_user_id = [(k,v) for k,v in map_user_id.items() ] 
    tuple_song_id = [(k,v) for k,v in map_song_id.items() ]

    df_map_user_id = sc.parallelize(tuple_user_id).toDF(['user_id','user_id_index'])
    df_map_song_id = sc.parallelize(tuple_song_id).toDF(['song_id','song_id_index'])

    #print 'zhou test', df_map_user_id.take(10), df_map_song_id.take(10)

    df_map_user_id.write.jdbc("jdbc:postgresql://dmdbmusic.cydl1z5zjcto.us-west-2.rds.amazonaws.com/dmdbmusic?user=postgres&password=j0yfulc4r","users_profile",mode = "overwrite")
    df_map_song_id.write.jdbc("jdbc:postgresql://dmdbmusic.cydl1z5zjcto.us-west-2.rds.amazonaws.com/dmdbmusic?user=postgres&password=j0yfulc4r","song_profile",mode = "overwrite")

    #print 'zhou test', len(map_user_id_br.value), len(map_song_id_br.value), map_user_id_br.value, map_song_id_br.value

###############################################################################




tableName = '(SELECT user_id,song_id,rating, artist_7digitalid FROM master_music_rating_total ORDER BY user_id %s ) tmp' % testBatchSize
query = tableName
df = sqlctx.read.format('jdbc').options( source="jdbc", url=urlPath, dbtable=query,
                                            partitionColumn='artist_7digitalid',
                                            lowerBound = -1,
                                            upperBound = 817066,
                                            numPartitions = 22,
                                           ).load()


df.registerTempTable("df")
df.cache()

rddUserRatingMaxMin = df.select('user_id','rating').map(tuple)
rddUserRatingMaxMin.cache()
list_user_rating_max = rddUserRatingMaxMin.reduceByKey(max).collect()
map_user_rating_max = { itr[0]:itr[1] for itr in list_user_rating_max}
map_user_rating_max_br = sc.broadcast(map_user_rating_max)

# Create the triplet RDD

if BinaryRating:
    ratingsRDD =  df.select('user_id','song_id','rating').map(tuple).map(convertRatingToBinary)
else:
    ratingsRDD =  df.select('user_id','song_id','rating').map(tuple).map(convertRatingToDecimal)

ratingsRDD.cache()



# In[8]:

# How many ratings from how many users on how many music tracks
numRatings = ratingsRDD.count()
numPartitions = ratingsRDD.getNumPartitions()
logString = "Got %d ratings from %d users on %d music tracks distributed across %d partitions.\n" \
% (numRatings, numUsers, numMusics, numPartitions)
logThis(logFileName, logString, 'w')
print logString



# In[ ]:
#X_train, X_test = myTrain_Test_Split(ratingsRDD,user_indexToCount_MapLocal,song_indexToCount_MapLocal,numRatings,0.33)

# In[ ]:

#print X_train.count(), X_test.count()


# In[9]:
if GeneratingModel:
    X_train = ratingsRDD                                                                                                                                                                   
    X_test = ratingsRDD    
else:
    X_train, X_test = myTrain_Test_Split_Default(ratingsRDD, fractionOfTestSample)


X_train.cache() # cache RDD as it will be used for training
X_test.cache() # cache RDD as it will be used for training


#X_train.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
#X_test.persist(pyspark.StorageLevel.MEMORY_AND_DISK)


# In[10]:

nTrainSample = X_train.count()
nTestSample = X_test.count()
logString = "We split the sample to %d training samples, %d test samples.\n" % (nTrainSample, nTestSample)
logThis(logFileName, logString, 'a')
print logString


# In[11]:

ranks = [10] # latent feature dimensionality
lambdas = [1.0] # 0.01 for implicity, 1.0 for explicit
numIters = xrange(numIterations,numIterations+1,1) # 17 is the upper limit to not cause stack overflow issue.....18
bestModel = None
bestValidationRmse = float("inf")
bestRank = 0
bestLambda = -1.0
bestNumIter = -1
bestPrecision = None
bestRecell = None
bestThresholds = None

#print 'test zhou', X_train.take(1000)



for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters): 
    logString = "We train at rank = %d, lamda = %f, num_iterations = %d.\n" % (rank, lmbda, numIter)
    logThis(logFileName, logString, 'a')
    print logString
    
    model = ALS.train(X_train, rank, numIter, lmbda, 22, nonnegative = nonNegativeRating)
    #model = ALS.trainImplicit(X_train, rank, numIter, lmbda, 22, nonnegative=False)

    if BinaryRating:
        precision, recall, thresholds = computeRecall(model, X_test, nTestSample, sc)
        bestPrecision = precision
        bestRecell =  recall
        bestThresholds =  thresholds
    else:
        validationRmse = computeRmse(model, X_test, nTestSample, sc)
        logString = "RMSE (validation) = %f for the model trained with " % validationRmse + "rank = %d, lambda = %.4f, and numIter = %d.\n" % (rank, lmbda, numIter)
        logThis(logFileName, logString, 'a')
        print logString
    
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter
        

# In[12]:

if BinaryRating:
    logString = 'precisions : %s \n' % bestPrecision
    logString += 'recall : %s \n' % bestRecell
    logString += 'thresholds: %s \n' % bestThresholds
    logThis(logFileName, logString, 'a')
    print logString
    precisionArray = np.array(bestPrecision)
    recallArray = np.array(bestRecell)
    precisionArray = precisionArray.reshape( len(precisionArray) ,1 )
    recallArray = recallArray.reshape( len(recallArray)  , 1)
    np.savetxt( fileName+ '_' + timeStamp + '.csv',  np.concatenate(( precisionArray , recallArray  ), axis=1) ) 

else:
    testRmse = bestValidationRmse
    #testRmse = computeRmse(bestModel, X_test, nTestSample)
    logString = 'Best Model RMSE = %s.\n' % testRmse
    logThis(logFileName, logString, 'a')
    print logString
    # compare the best model with a naive baseline that always returns the mean rating
    meanRating = X_test.map(lambda x: x[2]).mean()
    baselineRmse = sqrt(X_test.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / nTestSample)
    #baselineRmse = 2.904
    logString = "Trivial prediction based on mean ratings gives RMSE = %s. \n" % baselineRmse
    logThis(logFileName, logString, 'a')
    print logString

    if baselineRmse:
        improvement = (baselineRmse - testRmse) / baselineRmse * 100
        logString =  "The best model improves the baseline by %.2f " % (improvement) + "%. \n"
        logThis(logFileName, logString, 'a')
        print logString


if GeneratingModel:
    # Save and load model
    logString = 'Predict products for users: %s %s %s %s %s \n' % ( bestModel.recommendProducts(2,5), bestModel.recommendProducts(3,5) , bestModel.recommendProducts(0,5), bestModel.recommendProducts(1000,5), bestModel.recommendProducts(5000,5) )
    logThis(logFileName, logString, 'a')
    print logString
    bestModel.save(sc, fileName+ '_' + timeStamp + '.dat')                                                                                                                            
    

# In[15]:

sc.stop()


# In[ ]:



