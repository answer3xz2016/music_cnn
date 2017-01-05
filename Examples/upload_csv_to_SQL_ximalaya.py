# -*- coding: utf-8 -*- 

"""
# Dr. Zhou Xing (2016) NextEV USA
# joe.xing@nextev.com
# 
# Generate our own PostgreSQL DataBase
# 
# ......
# ......
# 
# Copyright 2016, Dr. Zhou Xing
# 
"""

import MuseUtil.museConfig as museConfig
import MuseUtil.museUtility as museUtility
import psycopg2
import os, sys, glob
import csv
import sqlite3

def returnConnRemoteDB():
     credentialFile = os.path.abspath(museConfig.myPostgresKeyXimalaya)
     conn = psycopg2.connect(museUtility.getDatabaseKey(dbcred_file=credentialFile))
     cursor = conn.cursor()
     query = """CREATE SCHEMA IF NOT EXISTS XIMALAYA;"""                                                                                         
     cursor.execute(query)
     conn.commit()   
     return conn, cursor

def returnConnLocalDB(dataBasePath = '/a/muse_nebula_shared_data/ximalaya_dimension_table.db' ):
     conn = sqlite3.connect(dataBasePath)
     cursor = conn.cursor() 
     return conn, cursor

def createTableRemoteLocal(conn, cursor, tableName):
     # Create table now
     
     sql = """
     DROP TABLE IF EXISTS %s; """ % tableName
     cursor.execute(sql)
     conn.commit()

     sql = """
     CREATE TABLE %s(  
     id integer,
     kind TEXT ,
     track_title TEXT ,
     track_tags TEXT,
     track_intro TEXT,
     category integer,
     cover_url_small TEXT,
     cover_url_middle TEXT,
     cover_url_large TEXT,
     announcer_id integer,
     announcer_nickname TEXT,
     announcer_avatar_url TEXT,
     announcer_is_verified TEXT,
     duration FLOAT,
     play_count integer ,
     favorite_count integer,
     comment_count integer,
     download_count integer,
     play_url_32 TEXT,
     play_size_32 integer,
     play_url_64 TEXT,
     play_size_64 integer,
     download_url TEXT,
     subordinated_album_id integer,
     subordinated_album_album_title TEXT,
     subordinated_album_cover_url_small TEXT,
     subordinated_album_cover_url_middle TEXT,
     subordinated_album_cover_url_large TEXT,
     source integer,
     updated_at bigint,
     created_at bigint,
     PRIMARY KEY (id)
     );
     """ % tableName
     cursor.execute(sql)
     conn.commit()
     
     

# this below is MySQL/SQLite syntax
def insertIntoTable(conn, cursor, fileName):          
     with open(fileName, 'r') as f:
          reader = csv.reader(f,delimiter='|')
          totalCount = 0
          badCount = 0
          for data in reader:
               totalCount += 1
               #print len(data), data
               if len(data) != 31:
                    badCount+=1
                    print 'bad/total %s/%s' % (badCount,totalCount)
                    
               else:
                    data = [ _ptr.decode('utf-8') for _ptr in data]
                    query = 'insert into SONG_METADATA_PRODUCTION values ({0})'
                    query = query.format(','.join( ["?"]*len(data) ))
                    #print type(data), len(data), data, query
                    cursor.execute(query,data)
               if totalCount % 1000 == 0:
                    print 'commit once'
                    conn.commit()
                    
          print 'bad/total %s/%s' % (badCount,totalCount)
          print 'commit finally'
          conn.commit()



def bulkInsertToTable(conn, cursor, fileName):
     # chunk copy from a file
     #f = open('/Users/joe.xing/Desktop/work/muse/ximalaya_data_new.csv', 'r')                                               
     #fileName = sys.argv[1]
     f = open(fileName, 'r')
     cursor.copy_from(f, 'XIMALAYA.SONG_METADATA_PRODUCTION', sep='|')
     f.close()
     conn.commit()


def addIndexToTable(conn, cursor, remote = False):
     if not remote:
          query = " DROP INDEX IF EXISTS idx_id "
     else:
          query = ''' DROP INDEX IF EXISTS  XIMALAYA.idx_id ;
          DROP INDEX IF EXISTS  XIMALAYA.idx_catid ; '''
     cursor.execute(query)

     if not remote:
          query = "CREATE INDEX idx_id ON SONG_METADATA_PRODUCTION ('id','category','announcer_id','play_count', 'duration', 'track_tags', 'created_at', 'updated_at',  'subordinated_album_id')"
     else:
          query = ''' CREATE INDEX idx_id ON XIMALAYA.SONG_METADATA_PRODUCTION (id) ;
          CREATE INDEX idx_catid ON XIMALAYA.SONG_METADATA_PRODUCTION (category) ;
          '''
     cursor.execute(query)
     conn.commit()



def insertSongSimilarityToRDS(fileNameList, tableName, conn, cursor, notCreateNewTable = True , overrideRecords = False ) :

     tableName = "XIMALAYA.similarity_%s" % tableName
     sql = """
     DROP TABLE IF EXISTS %s ;
     CREATE TABLE %s(
     cat_id integer,
     track_id integer,
     track_rank integer,
     similar_tracks int[],
     PRIMARY KEY (cat_id, track_id)
     );
     """ % ( tableName, tableName )
     if not notCreateNewTable:
          cursor.execute(sql)
          conn.commit()
     
     for fileName in fileNameList:
          catId = int(os.path.basename(fileName).split('_')[4])     
          myFile = open(fileName,'r')
          myContent = myFile.readlines()
          myCount = 0
          for myRank, MyTrackID in enumerate(myContent):
               myCount += 1
               sourceID = int(MyTrackID.strip().split()[0])
               likedIDs = map( int, MyTrackID.strip().split()[1:] )

               #print catId, myRank, sourceID , likedIDs

               # override existing catID -- sourceID records
               if overrideRecords:
                    #print 'deleting one record'
                    myOverrideQuery = ''' DELETE FROM %s 
                    WHERE cat_id = %s AND track_id = %s ;  ''' % ( tableName, catId, sourceID )
                    cursor.execute(myOverrideQuery)
                    #conn.commit()
               #print 'inserting one record'
               q = 'INSERT INTO %s VALUES ('  % tableName
               q += ' %s , %s , %s ' % (catId, sourceID, myRank )
               q += ", " + "'{" + ','.join(map(str,likedIDs)) + "}'"
               q += ')'
               cursor.execute(q)
               if myCount % 500 == 0:
                     print 'commit once  !!!'
                     conn.commit()
          print 'commit one last time for one catID !!!'
          conn.commit()
               
def uploadSongSimilarityCSVToRDS(conn, cursor, csvFolder = '/a/muse_nebula_shared_data/', catID = None , myImaginationMap = None, uploadMatchedItemsOnly = True, newCatID = False):
     csvFiles = []
     if catID:
          fileNamePattern = 'ximalaya_song_similarity_cat_%s_*.csv' % catID
     else:
          fileNamePattern = 'ximalaya*.csv'
     for root, dirs, files in os.walk(csvFolder):
          #print root, dirs, files
          # ignore the files under 'back_ip' folder
          if 'back_up' in root:
               continue
          myFiles = glob.glob(os.path.join(root, fileNamePattern))
          csvFiles += myFiles
     csvFiles.sort()

     # update the category table: catID --> catName
     if myImaginationMap and myImaginationMap.has_key(catID):
          myCatName = myImaginationMap[catID][0]
          myCatID = catID
         
          # insert new records
          if newCatID:
               myQuery = ''' INSERT INTO XIMALAYA.categories                                                                                                                                            
               VALUES ( %s , '%s' ) ;  ''' % ( myCatID, myCatName)
               cursor.execute(myQuery)
               conn.commit()
          else:
               myQuery = ''' UPDATE XIMALAYA.categories 
               SET name = '%s'
               WHERE id = %s ;  ''' % (myCatName, myCatID)
               cursor.execute(myQuery)
               conn.commit()
     
     album_List, tag_List, match_List = [], [], []
     for fileNamePtr in csvFiles:
          if 'album' in fileNamePtr:
               album_List.append(fileNamePtr)
          if 'tag' in fileNamePtr:
               tag_List.append(fileNamePtr)
          if 'match' in fileNamePtr:
               match_List.append(fileNamePtr)

     

     if uploadMatchedItemsOnly:
          insertSongSimilarityToRDS(match_List, 'match', conn, cursor, notCreateNewTable = True, overrideRecords = True)
     else:
          insertSongSimilarityToRDS(album_List, 'album', conn, cursor, notCreateNewTable = True)
          insertSongSimilarityToRDS(tag_List, 'tag', conn, cursor, notCreateNewTable = True)
          insertSongSimilarityToRDS(match_List, 'match', conn, cursor, notCreateNewTable = True)

     
def uploadDataFromSQLiteToPostgres(LastTimeUploadItems, notCreateNewTable):
     def encode_string(s):
          return "'" + s.replace("'","''") + "'"

     conn_remote, cursor_remote = returnConnRemoteDB()
     conn_local, cursor_local = returnConnLocalDB()
     print 'start to read in the local db file'
     cursor_local.execute('''SELECT * from  SONG_METADATA_PRODUCTION ''')
     myContent =  cursor_local.fetchall()
     print 'finish reading the local db file'
     
     tableNameInRDS = 'XIMALAYA.SONG_METADATA_PRODUCTION_BACKUP'
     if not notCreateNewTable:
          createTableRemoteLocal(conn_remote, cursor_remote, tableNameInRDS)

     myCount = 0
     for _ in myContent:
          myCount += 1
          if myCount <= LastTimeUploadItems:
               continue
          q = 'INSERT INTO %s VALUES ('  % tableNameInRDS
          myLine = [ encode_string( _ptr.encode('utf-8') ) if isinstance(_ptr, unicode) else str(_ptr)  for _ptr in _ ]
                    
          q += ','.join(myLine)
          q += ')'
          
          cursor_remote.execute(q)

          print 'row %s' % myCount
          if myCount % 1000 == 0:
               print 'commit once @ %s.....' % myCount
               conn_remote.commit()   

     print 'commit last time....... %s' % myCount
     conn_remote.commit()     
     
     conn_local.close()
     conn_remote.close()
               

if __name__ == "__main__":

     # local cursor to local db file
     #conn, cursor = returnConnLocalDB()

     # remote cursor to RDS
     conn, cursor = returnConnRemoteDB() 

     # create master table for all 5 M songs of Ximalaya
     #createTableRemoteLocal(conn, cursor, tableName = "SONG_METADATA_PRODUCTION")       
     #insertIntoTable(conn, cursor, fileName = '/data/ximalaya_combined_data/combined_ximalaya_data_new.csv')
     #insertIntoTable(conn, cursor, fileName = '/data/ximalaya_data_old/ximalaya_data.csv_000005198500_000005199000')

     # add index to table, either local or remote
     #addIndexToTable(conn, cursor, remote = True)


     # upload the song-similarity csv files to sql table now
     myImaginationMap = {
          100 : [u'Trump (特朗普)', 1, u'特朗普'] ,
          101 : [u'New energy (新能源)', 1, u'新能源'],
          1 : [u'News (新闻)', 1 , u'新闻'],
          2 : [u'Music (音乐)', 2 , u'音乐'],
          3 : [u'Audio book （有声书）', 3 , u'有声书'],
          6 : [u'Kids （儿童）', 6 , u'儿童'],
          7 : [u'Healthy life （养生）', 7 , u'健康'],
          8 : [u'Business （商务）', 8 , u'商务'],
          12 : [u'Comedy show （喜剧）', 12 , u'喜剧'],
          18 : [u'IT (科技)', 18 , u'科技'],
          21 : [u'Cars （汽车）', 21 , u'汽车'],
          23 : [u'Movie （电影）', 23 , u'电影'],
          
     }

     for catIDPtr in myImaginationMap.keys():
          if catIDPtr == 101  or  catIDPtr == 100:
               continue
          uploadSongSimilarityCSVToRDS(conn, cursor, catID = catIDPtr, myImaginationMap = myImaginationMap, uploadMatchedItemsOnly = True , newCatID = False)

     
     #uploadSongSimilarityCSVToRDS(conn, cursor, catID = 101, myImaginationMap = myImaginationMap, uploadMatchedItemsOnly = False , newCatID = True )

     
     # close connection now
     conn.close()


     # upload data from SQLite to Postgres SQL
     #uploadDataFromSQLiteToPostgres(0, False)
     
