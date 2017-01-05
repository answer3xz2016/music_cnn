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


def myBipartiteMatching(catIDInHouse, catNameInHouse, catIDInXimalaya, searchKeyWords , topN ):
     conn = sqlite3.connect('/a/muse_nebula_shared_data/ximalaya_dimension_table.db')
     cursor = conn.cursor() 
     cursor.execute("SELECT id  FROM SONG_METADATA_PRODUCTION WHERE category = %s ORDER BY play_count DESC LIMIT %s" % (catIDInXimalaya, topN) )
     idForThisCat = [ _[0] for _ in cursor.fetchall() ]
     return idForThisCat

def generateRecommendedItems(catIDPtr,myContent):
     conn = sqlite3.connect('/a/muse_nebula_shared_data/ximalaya_dimension_table.db')
     cursor = conn.cursor()

     # here catIDPtr can be in-house cat ID
     # similar album tracks
     myFileNameAlbum = "/a/muse_nebula_shared_data/ximalaya_song_similarity_cat_%s_album.csv" % catIDPtr
     # similar tag tracks
     myFileNameTag = "/a/muse_nebula_shared_data/ximalaya_song_similarity_cat_%s_tag.csv" % catIDPtr
     with open(myFileNameAlbum, 'w') as myFileAlbum, open(myFileNameTag, 'w') as myFileTag:
          for idSource in myContent:
               cursor.execute("SELECT track_tags, subordinated_album_id FROM SONG_METADATA_PRODUCTION WHERE id = %s " % idSource)
               returnResults =  cursor.fetchall()
               tagsSource = returnResults[0][0].split(',')
               albumSource = returnResults[0][1]
               
               # same album recommendations
               cursor.execute("SELECT id  FROM SONG_METADATA_PRODUCTION WHERE subordinated_album_id = %s AND id != %s ORDER BY play_count DESC " % (albumSource,idSource) )
               returnResults =  cursor.fetchall()
               returnResults = [ _[0] for _ in returnResults ]
               finalResults =  [idSource] +  returnResults
               csv_out=csv.writer(myFileAlbum, delimiter=' ')
               csv_out.writerow(finalResults) 

               # same tag recommendations
               finalResults = []
               for ptr in xrange(len(tagsSource)):
                    cursor.execute("SELECT id  FROM SONG_METADATA_PRODUCTION WHERE id != ?  AND track_tags = ?   ORDER BY play_count DESC LIMIT 10" , [ idSource, tagsSource[ptr] ]  )
                    returnResults =  cursor.fetchall()
                    returnResults = [ _[0] for _ in returnResults ]
                    finalResults += returnResults
               finalResults =  [idSource] +  finalResults
               csv_out=csv.writer(myFileTag, delimiter=' ')
               csv_out.writerow(finalResults)
                                             

def getXimalaya():
     fileName = "/a/joe_data/ximalaya_data/complete_database/ximalaya_tracks.csv_title_seg"
     myFile = open(fileName)
     myContentMaster = myFile.readlines()
     myContentMasterMap = {}
     for idx, sen in enumerate( myContentMaster ):
          sentence = sen.split('\xef\xbc\x8c')
          myTrackID = int( sentence[0].replace(' ', '') )
          mySentence = sentence[1].strip().split(' ')
          myContentMasterMap[myTrackID] = mySentence
     return myContentMasterMap

def generateBestMatchedItems(catIDPtr, myContentMasterMap, myModel, myContent, bestMatchedID):
     
     myFileNameMatch = "/a/muse_nebula_shared_data/ximalaya_song_similarity_cat_%s_match.csv" % catIDPtr
     with open(myFileNameMatch, 'w') as myFileMatch:
          for ptr in bestMatchedID:
               sourceTitle = myContentMasterMap[ptr]
               print 'Now we find matched items for track ID', ptr, ' '.join(sourceTitle)
               bestMatchedIDForThisTrack = getUserInput(myContentMasterMap,myModel, bestMatchedID, sourceTitle , userInteractive = 2 , topNTrue = True, similarityThreshold = None, countThreshold = 50 )
               
               if ptr in bestMatchedIDForThisTrack:
                    bestMatchedIDForThisTrack.remove(ptr)
               finalResults =  [ptr] +  bestMatchedIDForThisTrack
               csv_out=csv.writer(myFileMatch, delimiter=' ')
               csv_out.writerow(finalResults)
       
               
          
def getUserInput(myContentMasterMap , myModel, listOfSearchable, nextEVTopic, userInteractive = 0, topNTrue = True, similarityThreshold = 0.50, countThreshold = None ):

     #print 'search for how many tracks?' , len(listOfSearchable)

     if userInteractive == 0:
          mySentence1 = raw_input("Please enter your sentence: ")
          mySentence1 = mySentence1.split(' ')
          mySentence1 = [ _.decode('utf-8') for _ in mySentence1 ]
          
     elif userInteractive == 1:
          mySentence1 = [ nextEVTopic  ]

     else:
          mySentence1 = nextEVTopic
          mySentence1 = [ _.decode('utf-8') for _ in mySentence1 ]
          
          
     if topNTrue:
          myContent = listOfSearchable

     else:
          myContent = myContentMasterMap.keys()
     
     word_list = mySentence1 
     title_score_list = []
     myCount2 = 0
     for sen2 in myContent  :
          myCount2 += 1
          #if myCount2>=2: break
          myTrackID2 = int(sen2)
          if not myContentMasterMap.has_key(myTrackID2):
               print "This should not happen, master trackID map cannot find this trackID: %s" % myTrackID2
               continue
          mySentence2 = myContentMasterMap[myTrackID2]
          title_word_list = mySentence2
          
          visited = [0] * len(title_word_list) 
	  total_score = 0
          #print 'loop over each sentence: ', ' '.join(title_word_list)
          
          for word1 in word_list:
               #print 'key word: ', word1
               best_score = -10000.0
	       ch = -1
	       for j in range(len(title_word_list)):
                    #print 'search word: ',  title_word_list[j]
		    if visited[j] == 0:  
		         try:
                              #print type(word1), type(title_word_list[j].decode('utf-8') )
	                      sim_score = myModel.similarity(word1, title_word_list[j].decode('utf-8') )
                              #print word1 , ' -- ', title_word_list[j],'', sim_score
			      if sim_score > best_score: 
				   best_score = sim_score
		        	   ch = j 
		         except(KeyError):
                              #print 'should not happen very often'
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
          title,score, matchedID = sorted_title_list[i]
          if similarityThreshold:
	       if score < similarityThreshold: break
          if countThreshold:
               if i >= countThreshold: break
                    
               
	  #print matchedID , title, score, '\n'
          finalResultToReturn.append(matchedID)


          
     return finalResultToReturn         

     #myFileName = "/a/muse_nebula_shared_data/ximalaya_song_similarity_cat_%s_match.csv" % catIDPtr
     #with open(myFileName, 'a') as myFile:
     #     csv_out=csv.writer(myFile, delimiter=' ')
     #     csv_out.writerow([myTrackID]+finalResultToReturn)
     
    

               


if __name__ == "__main__":

     print "loading ximalaya: trackID --> segmented title "
     myContentMasterMap = getXimalaya()
     print "loading finished!"

     print "loading NLP trained model"
     modelFileName = "/a/joe_data/ximalaya_data/ximalaya_model"
     print "Loading model........."
     myModel = gensim.models.Word2Vec.load(modelFileName)
     print "Model loaded!"

     # NextEV in-house tags !!!
     # bring on your imaginations!!!
     # we are going to generate list of IDs belong to each category
     # we start from 100
     # trump, 100

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

     #myImaginationMap = { 100 : [u'Trump (特朗普)', 1, u'特朗普']  }
     myImaginationMap = { 101 : [u'New energy (新能源)', 1, u'新能源']  }
     
     
     topN = 200000
     myCategoryList = myImaginationMap.keys()
     similarityThreshold = None
     topNForThisCategory = 5000
     
     for catIDPtr in myCategoryList:
          print 'getting topN ID from the relavant categories!'
          # internal ID -> internal topic name -> coarse ximalaya catID -> actual topic name to match similarity
          myContent = myBipartiteMatching(catIDPtr, myImaginationMap[catIDPtr][0], myImaginationMap[catIDPtr][1], myImaginationMap[catIDPtr][2] , topN   )
          print 'finished getting topN ID from the relavant categories: ', len(myContent)
          nextEVTopic = myImaginationMap[catIDPtr][2]

          # use cat name to match
          bestMatchedID = getUserInput(myContentMasterMap,myModel,myContent, nextEVTopic, userInteractive = 1 , topNTrue = True, similarityThreshold = similarityThreshold, countThreshold = topNForThisCategory)

          
          # input cat name to match
          #getUserInput(myContentMasterMap,myModel,myContent, nextEVTopic, userInteractive = 0 , topNTrue = True, similarityThreshold = similarityThreshold)
          
          # generate best matched track ID list
          
          generateBestMatchedItems(catIDPtr, myContentMasterMap, myModel, myContent, bestMatchedID)
                                                                                          
          # generate similar album, similar tag track ID lists
          #generateRecommendedItems(catIDPtr,bestMatchedID)

