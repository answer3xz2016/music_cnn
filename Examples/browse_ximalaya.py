"""
# Dr. Zhou Xing (2016) NextEV USA
# joe.xing@nextev.com
# 
# Check our own PostgreSQL DataBase
# 
# ......
# ......
# 
# Copyright 2016, Dr. Zhou Xing
# 
"""


import os
import sys
import psycopg2
import MuseUtil.museConfig as museConfig
import MuseUtil.museUtility as museUtility
from collections import deque

credential_file = os.path.abspath(museConfig.myPostgresKeyXimalaya)
conn = psycopg2.connect(museUtility.getDatabaseKey(dbcred_file=credential_file))
cursor = conn.cursor()

def checkTablesInDataBase():
    cursor.execute("SELECT relname FROM pg_class WHERE relkind='r' and relname !~ '^(pg_|sql_)';")
    print 'What are the tables in the database: ',cursor.fetchall()

    #cursor.execute("SELECT COUNT(id), MAX(id), MIN(id) FROM XIMALAYA.SONG_METADATA ")
    #print cursor.fetchall()
    
    #cursor.execute("SELECT COUNT(id), MAX(id), MIN(id) FROM XIMALAYA.SONG_METADATA_TOTAL ")
    #print cursor.fetchall()

    #cursor.execute("SELECT COUNT(id), MAX(id), MIN(id) FROM XIMALAYA.SONG_METADATA_PRODUCTION ")
    #print cursor.fetchall()



def checkSongSimilarityTables():
    print "Query table for album similarity"
    cursor.execute('''
    SELECT cat_id, track_id, track_rank,  similar_tracks FROM XIMALAYA.similarity_album
    WHERE cat_id = 100
    ORDER BY track_rank
    LIMIT 1
    ''')
    for _ in cursor.fetchall():
        print _

    print 'similar tags'
    cursor.execute(''' 
    SELECT cat_id, COUNT ( track_id ) FROM XIMALAYA.similarity_tag
    GROUP BY cat_id ''')
    for _ in cursor.fetchall():
        print _

    print 'similar album'
    cursor.execute('''SELECT cat_id, COUNT ( track_id ) FROM XIMALAYA.similarity_album                                                                                                             
    GROUP BY cat_id''')
    for _ in cursor.fetchall():
        print _

    print 'best match'

    cursor.execute('''SELECT cat_id, COUNT ( track_id ) FROM XIMALAYA.similarity_match                                                                 
    GROUP BY cat_id''')
    for _ in cursor.fetchall():
        print _

    print 'best match'
    
    cursor.execute('''SELECT cat_id, track_id, track_rank, similar_tracks FROM XIMALAYA.similarity_match                                                                      
    WHERE track_id = 8153998 AND cat_id = 1
    
    ''')
    for _ in cursor.fetchall():
        print _

    print 'cat id -> cat name'
    cursor.execute('''SELECT id, name FROM XIMALAYA.categories  ''')
    for _ in cursor.fetchall():
        print _[0], _[1].decode('utf-8'), '\n'
        

def checkMasterDimensionTable():
    print "Query master table for the real stuff...."
    cursor.execute('''
    SELECT id, track_title, cover_url_large,  play_url_64 FROM XIMALAYA.SONG_METADATA_PRODUCTION
    LIMIT 10
    ''')
    for _ in cursor.fetchall():
        for __ in _: print __,
        print '\n'

    cursor.execute('''
    SELECT COUNT(id) FROM XIMALAYA.SONG_METADATA_PRODUCTION
    ''')
    print cursor.fetchall()


def checkUserAffinityTable():
    cursor.execute('''
    SELECT DISTINCT user_id FROM XIMALAYA.user_affinity LIMIT 10
    ''')
    for _ in cursor.fetchall():
        for __ in _: print __,
        print '\n'

    cursor.execute(
        '''
        SELECT track_id FROM ximalaya.user_affinity
        WHERE affinity = 0 AND user_id = %s
        GROUP BY track_id
        ORDER BY MAX(time) DESC
        LIMIT 3
        '''
        , ('zhou.xing.2',) )
    # right now it's hard coded to return the latest 3 trackID for this user
    listOfLatestViewedTracks = [query_tuple[0] for query_tuple in cursor.fetchall()]
    print listOfLatestViewedTracks
    
    # Maintain a list of queues and dequeue each one in a round-robin
    # fashion until all matched/associated tracks have been added to RPC payload.
    listQueues = [deque() for k in xrange(len(listOfLatestViewedTracks))]
    for i, myTrackID in enumerate(listOfLatestViewedTracks):
        print type(myTrackID)
        cursor.execute(
            '''
            WITH list_tracks AS (
            SELECT unnest(similar_tracks) FROM ximalaya.similarity_match
            WHERE track_id = %s
            ORDER BY cat_id
            DESC
            )
            SELECT id, track_title, cover_url_large, play_url_64, announcer_nickname FROM list_tracks JOIN ximalaya.song_metadata_production as metadata
            ON list_tracks.unnest = metadata.id
            ''', ( myTrackID ,) )
                                                                
  
        for _ in cursor.fetchall():
            for __ in _: print __,
            print '\n'

    
if __name__ == "__main__":
    checkTablesInDataBase()
    #checkSongSimilarityTables()
    checkUserAffinityTable()
    #checkMasterDimensionTable()
    conn.close()

