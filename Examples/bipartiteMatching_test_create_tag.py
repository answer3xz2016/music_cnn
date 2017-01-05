# -*- coding: utf-8 -*-


"""
# Dr. Zhou Xing (2016) NextEV USA
# joe.xing@nextev.com
# 
# Run bipartite matching
# and generate a list of
# best matches to each trackID
# 
# 
# Copyright 2016, Dr. Zhou Xing
# 
"""

import os, sys
import csv
import sqlite3
import gensim
import subprocess as sub
import pandas as pd

def returnConnLocalDB(catID):
     myFileName = "/a/muse_nebula_shared_data/ximalaya_song_similarity_cat_%s_album.csv" % catIDPtr
     myFile = open(myFileName, 'r')
     myContent = myFile.readlines()
     myContent = [ _.strip().split(' ')[0]   for _ in myContent ]
     return myContent
     
def generateRecommendedItems(catIDPtr,myContent):
     conn = sqlite3.connect('/a/muse_nebula_shared_data/ximalaya_dimension_table.db')
     cursor = conn.cursor()
     myFileName = "/a/muse_nebula_shared_data/ximalaya_song_similarity_cat_%s_tag.csv" % catIDPtr
     
     with open(myFileName, 'w') as myFile:
          for idSource in myContent:
               cursor.execute("SELECT track_tags, subordinated_album_id FROM SONG_METADATA_PRODUCTION WHERE id = %s " % idSource)
               returnResults =  cursor.fetchall()
               tagsSource = returnResults[0][0].split(',')
               albumSource = returnResults[0][1]
               finalResults = []
               for ptr in xrange(len(tagsSource)):     
               # same tag recommendations
                    #print tagsSource[ptr]
                    cursor.execute("SELECT id  FROM SONG_METADATA_PRODUCTION WHERE id != ?  AND track_tags = ?   ORDER BY play_count DESC LIMIT 10" , [ idSource, tagsSource[ptr] ]  )
                    returnResults =  cursor.fetchall()
                    returnResults = [ _[0] for _ in returnResults ]
                    finalResults += returnResults
               finalResults =  [idSource] +  finalResults
               csv_out=csv.writer(myFile, delimiter=' ')
               csv_out.writerow(finalResults)
               
          

if __name__ == "__main__":

     #myContentMasterMap = getGoogleNews()
     
     #modelFileName = "/a/joe_data/ximalaya_data/ximalaya_model"
     #modelFileName = "/a/joe_data/English_wikipedia/trained_model_binary"
     #print "Loading model........."
     #myModel = gensim.models.Word2Vec.load(modelFileName)
     #print "Model loaded!"

     #getUserInput(myContentMasterMap , myModel)
     # we are going to generate list of IDs belong to each category
     # news, 1
     # music, 2
     # audio book, 3
     # kids, 6
     # healthy life,7
     # business, 8
     # comedy show, 12
     # IT, 18
     # car, 21
     # movie, 23

     catStart, catEnd = int(sys.argv[1]), int(sys.argv[2])
     
     myCategoryList = [1,2,3,6,7,8,12,18,21,23]
     for catIDPtr in myCategoryList[catStart:catEnd]: # cat 2, cat 6
          myContent = returnConnLocalDB(catIDPtr)
          print 'How many items for catID %s: %s '% (catIDPtr, len(myContent) )
          generateRecommendedItems(catIDPtr,myContent)

          
     
     sys.exit(0)
     
     
