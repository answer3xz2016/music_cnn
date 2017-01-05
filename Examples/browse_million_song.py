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
import sys, csv
import psycopg2
import MuseUtil.museConfig as museConfig
import MuseUtil.museUtility as museUtility


credential_file = os.path.abspath(museConfig.myPostgresKey)
conn = psycopg2.connect(museUtility.getDatabaseKey(dbcred_file=credential_file))
cursor = conn.cursor()

def getTableNames():
    cursor.execute("SELECT relname FROM pg_class WHERE relkind='r' and relname !~ '^(pg_|sql_)';")
    print 'What are the tables in the database: ',cursor.fetchall()

def convertMDSFromRDSToHDFS():

    #cursor.execute("SELECT user_id , song_id, rating FROM master_music_rating_total LIMIT 10")
    cursor.execute('''SELECT DISTINCT  t1.user_id,t1.song_id,t1.rating  FROM master_music_rating_total AS t1, spotify_url AS t2 WHERE t1.song_id = t2.song_id  ''')
    #cursor.execute('''SELECT COUNT ( DISTINCT t1.song_id) FROM master_music_rating_total AS t1, spotify_url AS t2 WHERE t1.song_id = t2.song_id  ''')
    
    # convert RDS to txt data and throw it then into HDFS
    with open('/home/ec2-user/MSD_data/train_data.csv', 'w') as myFile:
        for i in cursor.fetchall():
            csv_out=csv.writer(myFile, delimiter=' ')
            csv_out.writerow(i) 

def createMSDPopularSong():
    cursor.execute('''                                                                                                                                                                             
    DROP TABLE IF EXISTS MSD_popular_songs;                                                                                                                                                        
    CREATE TABLE MSD_popular_songs AS                                                                                                                                                              
    SELECT  t1.song_id, COUNT (t1.user_id)  FROM master_music_rating_total AS t1, spotify_url AS t2                                                                                                
    WHERE t1.song_id = t2.song_id                                                                                                                                                               
    GROUP BY t1.song_id                                                                                                                                                                            
    ORDER BY COUNT (t1.user_id)                                                                                                                                                                    
    DESC ;'''
    )
    conn.commit()

def browseDataBase():
    cursor.execute("SELECT COUNT( DISTINCT song_id) FROM spotify_url ")
    for i in cursor.fetchall():
        print i

    cursor.execute("SELECT user_id_index FROM MSD_user_profile WHERE user_id = 'ffff07d7d9bb187aa58c7b81b3d3f35e7cf7c0ee' ")
    for i in cursor.fetchall():
        print i

    cursor.execute("SELECT song_id_index FROM MSD_song_profile WHERE song_id = 'SOZZZWN12AF72A1E29' ")
    for i in cursor.fetchall():
        print i

    cursor.execute("SELECT song_id, count FROM MSD_popular_songs ORDER BY count DESC LIMIT 10")
    for i in cursor.fetchall():
        print i

    

def sortTheLocalMSDData():
    trainFileName = '/home/ec2-user/MSD_data/train_data.csv'
    myFile = open(trainFileName,'r')
    myContents = myFile.readlines()
    userList = []
    songList = []
    ratingList = []
    for ptr in myContents:
        triplet = ptr.strip().split()
        userList.append(triplet[0])
        songList.append(triplet[1])
        ratingList.append(triplet[2])
    
    sortedUserList = sorted(list(set(userList)))
    sortedSongList = sorted(list(set(songList)))
    numUsers = len(sortedUserList)
    numMusics = len(sortedSongList)
    
    map_user_id  = dict(zip( xrange(0,numUsers) , sortedUserList  ))
    map_song_id = dict(zip( xrange(0,numMusics) , sortedSongList   ))
    map_user_id_reverse  = dict(zip( sortedUserList , xrange(0,numUsers)   ))
    map_song_id_reverse = dict(zip( sortedSongList ,  xrange(0,numMusics)   ))


    sortedFileName = '/home/ec2-user/MSD_data/train_data_sorted.csv'
    userFileName = '/home/ec2-user/MSD_data/user_map.csv'
    songFileName = '/home/ec2-user/MSD_data/song_map.csv'
    with open(sortedFileName, 'w') as myFile:
        for idx, r in enumerate(ratingList) :
            csv_out=csv.writer(myFile, delimiter=' ')
            csv_out.writerow( [ map_user_id_reverse[userList[idx]] ] + [ map_song_id_reverse[songList[idx]] ]  + [ r ]  )
    with open(userFileName, 'w') as myFile:
        for keys, values in map_user_id.iteritems() :
            csv_out=csv.writer(myFile, delimiter=' ')
            csv_out.writerow( [ keys ] + [ values ]  )
    with open(songFileName, 'w') as myFile:
        for keys, values in map_song_id.iteritems() :
            csv_out=csv.writer(myFile, delimiter=' ')
            csv_out.writerow( [ keys ] + [ values ]  )


if __name__ == '__main__':
    #convertMDSFromRDSToHDFS()
   
    #getTableNames()
    #browseDataBase()
    
    sortTheLocalMSDData()
    #createMSDPopularSong()
    conn.close()
