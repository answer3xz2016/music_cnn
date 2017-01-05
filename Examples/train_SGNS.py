
##########################################################################################
# This is an example scipt to train Word2Vec (SGNS) model 
##########################################################################################

import os
import sys
from os.path import expanduser
import itertools



# pyspark modules
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import Word2Vec

# in-house modules



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

    homeDir = expanduser("~")
    logFileFolder = homeDir + '/log'
    if not os.path.exists(logFileFolder) or not os.path.isdir(logFileFolder):
        os.makedirs(logFileFolder)
    logFileName = logFileFolder + '/SGNS-muse'  + '.log'
    

    #conf = SparkConf().setMaster("local[*]").setAppName("SGNS-muse")
    conf = SparkConf().set("spark.deploy.defaultCores", 2).set("spark.cores.max", 10000000).set("spark.executor.memory", "8G")\
        .setMaster("spark://10.118.132.141:7077").setAppName("SGNS-muse")
    sc = SparkContext(conf=conf)
    ratingRDD = sc.textFile("hdfs://ec2-52-41-137-208.us-west-2.compute.amazonaws.com:9000/wiki_data/test.jian.seg").map(lambda r: r.split(" ")).repartition(100000)
    ratingRDD.cache()

    logString =  'Debug: Training RDD mem level: ' + str( ratingRDD.getStorageLevel() )
    logThis(logFileName, logString, 'w')

    logString = 'Debug: Training RDD num of partitions: ' + str( ratingRDD.getNumPartitions() )
    logThis(logFileName, logString, 'a')
    
    word2vec = Word2Vec()
    word2vec.setVectorSize(400)
    word2vec.setMinCount(5)
    
    
    model = word2vec.fit(ratingRDD)

    logString = 'Debug: Finished training' 
    logThis(logFileName, logString, 'a')

    mode.save(sc, "hdfs://ec2-52-41-137-208.us-west-2.compute.amazonaws.com:9000/wiki_data/test.model")
    
