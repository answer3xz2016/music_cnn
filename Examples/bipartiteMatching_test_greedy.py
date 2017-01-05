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

def myBipartiteMatching( catIDInXimalaya, topN ):
     conn = sqlite3.connect('/a/muse_nebula_shared_data/ximalaya_dimension_table.db')
     cursor = conn.cursor()
     cursor.execute("SELECT id  FROM SONG_METADATA_PRODUCTION WHERE category = %s ORDER BY play_count DESC LIMIT %s" % (catIDInXimalaya, topN) )
     idForThisCat = [ _[0] for _ in cursor.fetchall() ]
     return idForThisCat
                         
def returnConnLocalDB(catID):
     myFileName = "/a/muse_nebula_shared_data/ximalaya_song_similarity_cat_%s_album.csv" % catID
     myFile = open(myFileName, 'r')
     myContentTotal = myFile.readlines()
     
     myFileName = "/a/muse_nebula_shared_data/ximalaya_song_similarity_cat_%s_tag.csv" % catID
     if  os.path.exists(myFileName):
          myFile = open(myFileName, 'r')
          myContentTotal2 = myFile.readlines()
          myContentTotal2Map = {}
          for _ in myContentTotal2 :
                myContentTotal2Map[ int(_.strip().split(' ')[0]) ] = _.strip().split(' ')[1:]
                                    
          
     myContent = [ int(_.strip().split(' ')[0])   for _ in myContentTotal ]
     #myContentSimilar = [ _.strip().split(' ')[1:]   for _ in myContentTotal ]
     myContentSimilar = [ []   for _ in myContentTotal ]

     if 'myContentTotal2Map' in locals():
          for idx in xrange(len(myContent)):
               if myContentTotal2Map.has_key( myContent[idx]  ):
                    myContentSimilar[idx] += myContentTotal2Map[  myContent[idx]    ]

          del myContentTotal2Map
     return myContent, myContentSimilar 

def getXimalaya():
     fileName = "/a/joe_data/ximalaya_data/complete_database/ximalaya_tracks.csv_title_seg"
     myFile = open(fileName)
     myContentMaster = myFile.readlines()
     myContentMasterMap = {}
     for idx, sen in enumerate( myContentMaster ):
          sentence = sen.strip().split('\xef\xbc\x8c')
          myTrackID = int( sentence[0].replace(' ', '') )
          mySentence = sentence[1].strip().split(' ')
          myContentMasterMap[myTrackID] = mySentence
     return myContentMasterMap


if __name__ == "__main__":

     batchIDStart, batchIDEnd =  int(sys.argv[1]), int(sys.argv[2])
     
     modelFileName = "/a/joe_data/ximalaya_data/ximalaya_model"
     print "Loading model........."
     myModel = gensim.models.Word2Vec.load(modelFileName)
     print "Model loaded!"

     print "Loading 5M tracks from ximalaya"
     myContentMasterMap = getXimalaya()
     print "5M tracks from ximalya loaded!"

     topN = 20000
     countThresholdForMatchedItems = 50
     
     myCategoryList = [1,2,3,6,7,8,12,18,21,23]
     for catIDPtr in myCategoryList[batchIDStart:batchIDEnd]: # cat 1,2,3,6,7

          myContent, myContentSimilar = returnConnLocalDB(catIDPtr)
          print 'How many items for catID %s: %s '% (catIDPtr, len(myContent) )

          # this will generate a top-N list in this catID, and use this list to find matched ones
          # instead of using 'myContentSimilar'
          print 'search for topN items in this catID for bipartite matching'
          myContentSimilarGlobal = myBipartiteMatching(catIDPtr , topN   )
          print 'done search for topN items in this catID for bipartite matching'
          
          myCount = 0
          for idx, sen1 in enumerate(myContent):
               myCount += 1
               #if myCount >=2: break # need to get rid of this line to scan through all the 20 k items
               
               myTrackID = sen1
               mySentence1 = myContentMasterMap[myTrackID]
               numOfNodesInA  =  len(mySentence1)
              
               word_list = mySentence1 
               title_score_list = []
               print myTrackID, ' '.join(word_list) ,'\n' 
               
               myCount2 = 0
               # quadratic timing comes in ....
               #for sen2 in  myContentSimilar[idx] :
               for sen2 in myContentSimilarGlobal:     
                    myCount2 += 1                   
                    myTrackID2 = int(sen2)
                    if not myContentMasterMap.has_key(myTrackID2):
                         print "This should not happen, master trackID map cannot find this trackID: %s" % myTrackID2
                         continue
                    mySentence2 = myContentMasterMap[myTrackID2]
                    title_word_list = mySentence2
                    numOfNodesInB  =  len(mySentence2)
                                        
                    visited = [0] * len(title_word_list) 
	            total_score = 0 
                    for word1 in word_list: 
                         best_score = -10000.0
	                 ch = -1
	                 for j in range(len(title_word_list)): 
		              if visited[j] == 0:  
		                   try:
                                        #print type(word1.decode('utf-8')), type(title_word_list[j].decode('utf-8'))
	                                sim_score = myModel.similarity( word1.decode('utf-8') , title_word_list[j].decode('utf-8') )
                                        #print word1.decode('utf-8'), title_word_list[j].decode('utf-8'), sim_score
			                if sim_score > best_score: 
				             best_score = sim_score
		        	             ch = j 
		                   except(KeyError):
                                        #print 'should not happen a lot'
			                pass

	                 if ch == -1: 
		              total_score += 0 
                         else: 
		              total_score += best_score 
		              visited[ch] = 1
                    title_score_list.append(( ' '.join(title_word_list)  , total_score, myTrackID2 ))
                   
               sorted_title_list = sorted(title_score_list, key=lambda tup: tup[1], reverse=True)
               finalResultToReturn = []
               for i in xrange(len( sorted_title_list)):
                    if i >= countThresholdForMatchedItems: break
                    
                    title,score, matchedID = sorted_title_list[i]	
	            #print matchedID , title, score, '\n'
                    
                    finalResultToReturn.append(matchedID)

               if myTrackID in finalResultToReturn:
                    finalResultToReturn.remove(myTrackID)

               myFileName = "/a/muse_nebula_shared_data/ximalaya_song_similarity_cat_%s_match.csv" % catIDPtr
               with open(myFileName, 'a') as myFile:
                    csv_out=csv.writer(myFile, delimiter=' ')
                    csv_out.writerow([myTrackID]+finalResultToReturn)
                    
                    
