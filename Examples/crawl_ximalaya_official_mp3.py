# -*- coding: utf-8 -*
"""
# Dr. Zhou Xing (2016) NextEV USA
# joe.xing@nextev.com
# 
# Download mp3 files from vendor ximalaya
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
import os, sys
import csv
import sqlite3
import urllib

def returnConnRemoteDB():
     credentialFile = os.path.abspath(museConfig.myPostgresKeyXimalaya)
     conn = psycopg2.connect(museUtility.getDatabaseKey(dbcred_file=credentialFile))
     cursor = conn.cursor()
     return conn, cursor

def returnConnLocalDB():
     conn = sqlite3.connect('/a/joe_data/ximalaya_data/complete_database/test.db')
     cursor = conn.cursor() 
     return conn, cursor

def downloadMP3(conn, cursor, dataPath, topN=10):
    query = """
    SELECT id , track_title, play_url_64, download_url FROM SONG_METADATA_PRODUCTION
    WHERE category = 2 AND track_tags = '欧美' 
    ORDER BY play_count DESC
    LIMIT ?
    """
    cursor.execute(query, (topN,) )
    results =  cursor.fetchall()
    trackID_To_Mp3 ={}
    for _ptr in results:
        _fileName = str(_ptr[0]) 
        trackID_To_Mp3[ _fileName  ] =  _ptr[2]
        
    for key, value in trackID_To_Mp3.iteritems():
         urllib.urlretrieve (value, dataPath + "%s.mp3" % key)

def checkTrackId(conn, cursor, trackID):
     query = """                                                                                                                                                                          
     SELECT id , track_title, track_tags , play_url_64 FROM SONG_METADATA_PRODUCTION                                                                                                                        
     WHERE id IN (%s)                                                                                                                         
     """ % ','.join('?'*len(trackID))
     cursor.execute(query, trackID )
     results =  cursor.fetchall()
     for _ptr in results:
          for _ in _ptr: print _ ,
          print '\n'
         
if __name__ == '__main__':
    conn, cursor = returnConnLocalDB()
    dataPath = '/a/joe_data/ximalaya_data/complete_database/wav_files/'
    # down load the mp3 files
    #downloadMP3(conn, cursor, dataPath, topN=10)
    # check the song titles of the downloaded sample dataset
    checkTrackId(conn, cursor, [1069558
                                ,139679
                                ,16197350
                                ,16197447
                                ,164011
                                ,188178
                                ,207502
                                ,211819
                                ,230212
                                ,715973])
