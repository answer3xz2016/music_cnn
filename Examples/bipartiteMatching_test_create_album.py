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
     conn = sqlite3.connect('/a/muse_nebula_shared_data/ximalaya_dimension_table.db')
     cursor = conn.cursor() 
     cursor.execute("SELECT id FROM SONG_METADATA_PRODUCTION WHERE category = %s ORDER BY play_count DESC" % catID)
     idForThisCat = [ _[0] for _ in cursor.fetchall() ] # category = 1 -> news
     return idForThisCat

def generateRecommendedItems(catIDPtr,myContent):
     conn = sqlite3.connect('/a/muse_nebula_shared_data/ximalaya_dimension_table.db')
     cursor = conn.cursor()
     myFileName = "/a/muse_nebula_shared_data/ximalaya_song_similarity_cat_%s_album.csv" % catIDPtr
     
     with open(myFileName, 'w') as myFile:
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
               csv_out=csv.writer(myFile, delimiter=' ')
               csv_out.writerow(finalResults) 
          

def getGoogleNews():
     fileName = "/a/muse_nebula_shared_data/google_news.csv_backup"
     df = pd.read_csv(fileName , quotechar='"' , header = 0  )
     myContentMaster = df['title'].tolist()
     myContentMasterMap = {}
     for idx, sen in enumerate(myContentMaster):
          mySentence = sen.strip().split(' ') 
          myContentMasterMap[idx] = mySentence
     return myContentMasterMap

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


def getUserInput(myContentMasterMap , myModel):
     mySentence1 = raw_input("Please enter your sentence: ")
     mySentence1 = mySentence1.split(' ')
               
     numOfNodesInA  =  len(mySentence1)
     bestMatchScore = []
     bestMatchID = []
          
     myCount2 = 0
     myContent = myContentMasterMap.keys()
     # quadratic timing comes in ....
     for sen2 in myContent:
          myCount2 += 1
          #if myCount2 >= 300: break
               
          myTrackID2 = sen2
          
          mySentence2 = myContentMasterMap[myTrackID2]
          numOfNodesInB  =  len(mySentence2)
               
          myWeights = []
               
          for _ptr1 in mySentence1:
               for _ptr2 in mySentence2:
                    try:
                         mySim = myModel.similarity(_ptr1.decode('utf-8'), _ptr2.decode('utf-8'))
                    except:
                         mySim = 0

                    #print  _ptr1, myTrackID2,  _ptr2, mySim
                    mySim = int(round(mySim*100.))
                    myWeights.append(mySim)
                         
          # calling max weighted bipartite matching
          commandString = '~/match %s %s ' % ( numOfNodesInA, numOfNodesInB ) +  " ".join(map(str,myWeights))
          Sig = sub.check_output(commandString, stderr=sub.STDOUT,shell=True)
          Sig = Sig.strip()
          if not myCount2 % 1000:
               print 'Max match between %s and %s is %s ' % (0, myTrackID2, Sig)
          bestMatchScore.append( Sig )
          bestMatchID.append( myTrackID2 ) 
               
     sortedIndex = sorted(range(len(bestMatchScore)), key=lambda k: bestMatchScore[k])
     sortedIndex  = sortedIndex[::-1][:50]
     for _index in sortedIndex:
          bestMatchIndexPtr = bestMatchID[_index]
          bestMatchScorePtr = bestMatchScore[_index]
          for _ in myContentMasterMap[bestMatchIndexPtr]: print _,
          print 'match score: ', bestMatchScorePtr, bestMatchIndexPtr
          print '\n'

               


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

     topN = 20000
     myCategoryList = [1,2,3,6,7,8,12,18,21,23]
     for catIDPtr in myCategoryList[1:2]:
          myContent = returnConnLocalDB(catIDPtr)
          myContent = myContent[:topN]
          print 'How many items for catID %s: %s '% (catIDPtr, len(myContent) )
          generateRecommendedItems(catIDPtr,myContent)

          
     sys.exit(0)
     
     myCount = 0
     for sen1 in myContent:
          #sen1 = 3375665
          myCount += 1
          if myCount >=2: break
          
          myTrackID = sen1
          mySentence1 = myContentMasterMap[myTrackID]
          
          numOfNodesInA  =  len(mySentence1)
          bestMatchScore = []
          bestMatchID = []
          
          myCount2 = 0
          # quadratic timing comes in ....
          for sen2 in myContent:
               myCount2 += 1
               #if myCount2 >= 300: break
               
               myTrackID2 = sen2
               if myTrackID2 == myTrackID:
                    continue
               mySentence2 = myContentMasterMap[myTrackID2]
               numOfNodesInB  =  len(mySentence2)
               
               myWeights = []
               
               for _ptr1 in mySentence1:
                    for _ptr2 in mySentence2:
                         try:
                              mySim = myModel.similarity(_ptr1.decode('utf-8'), _ptr2.decode('utf-8'))
                         except:
                              mySim = 0

                         #print myTrackID, _ptr1, myTrackID2,  _ptr2, mySim
                         mySim = int(round(mySim*100.))
                         myWeights.append(mySim)
                         
               # calling max weighted bipartite matching
               commandString = '~/match %s %s ' % ( numOfNodesInA, numOfNodesInB ) +  " ".join(map(str,myWeights))
               Sig = sub.check_output(commandString, stderr=sub.STDOUT,shell=True)
               Sig = Sig.strip()
               #print 'Max match between %s and %s is %s ' % (myTrackID, myTrackID2, Sig)
               bestMatchScore.append( Sig )
               bestMatchID.append( myTrackID2 ) 
               
          # sort the matched list
          sourceItems = myContentMasterMap[myTrackID]
          print "Source news: ", sen1
          for _ in sourceItems:
               print _,
          print '\n'
          
          sortedIndex = sorted(range(len(bestMatchScore)), key=lambda k: bestMatchScore[k])
          sortedIndex  = sortedIndex[::-1][:50]
          for _index in sortedIndex:
               bestMatchIndexPtr = bestMatchID[_index]
               bestMatchScorePtr = bestMatchScore[_index]
               for _ in myContentMasterMap[bestMatchIndexPtr]: print _,
               print 'match score: ', bestMatchScorePtr, bestMatchIndexPtr
               print '\n'
