
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
import os
import pandas.io.sql as psql
import csv

credentialFile = os.path.abspath(museConfig.myPostgresKey)
conn = psycopg2.connect(museUtility.getDatabaseKey(dbcred_file=credentialFile))
cursor = conn.cursor()


def uploadSpotifyUrlMetaData():
    # Create table now
    sql = """
    DROP TABLE IF EXISTS spotify_url;
    CREATE TABLE spotify_url(  
    song_id text, 
    track_title_7 text, 
    track_id_7 text,
    track_artist_name_7 text,
    track_artist_id_7 text,
    track_artist_url_7 text,
    track_release_title_7 text,
    track_release_id_7 text,
    track_release_image text,
    PRIMARY KEY (song_id)
    );
    """
    psql.execute(sql,conn)
    conn.commit()

    cur = conn.cursor()
    with open('/Users/joe.xing/Desktop/work/muse/spotify_7digital_data.csv', 'r') as f:
        reader = csv.reader(f)
        data = next(reader)
        query = 'insert into spotify_url values ({0})'
        query = query.format(','.join( ['%s'] * len(data)))
        print type(data), len(data), data, query
        cur.execute(query, data)
        count = 0
        for data in reader:
            count += 1
            cur.execute(query, data)
            if count % 100 == 0:
                print 'commit once'
                conn.commit()
        

                                                
                #cur.copy_from(f, 'spotify_url', sep=',')
                #f.close()

        conn.commit()
        conn.close()


def uploadMSDUserAndSongIndexToID():
    sql = """
    DROP TABLE IF EXISTS MSD_user_profile;                                                                                                                            
    CREATE TABLE MSD_user_profile(                                                                                                                                                                 
    user_id_index integer,                                                                                                                                                           
    user_id text,                                                                                                                                                                            
    PRIMARY KEY (user_id_index)                                                                                                                                                                    
    );                     

    DROP TABLE IF EXISTS MSD_song_profile;
    CREATE TABLE MSD_song_profile(
    song_id_index integer,
    song_id text,
    PRIMARY KEY (song_id_index)
    );
    """
    psql.execute(sql,conn)
    conn.commit()
    with open('/home/ec2-user/MSD_data/user_map.csv', 'r') as f:
        cursor.copy_from(f, 'MSD_user_profile', sep=' ')                                                                                                                                           
        
    with open('/home/ec2-user/MSD_data/song_map.csv', 'r') as f:
        cursor.copy_from(f, 'MSD_song_profile', sep=' ')
    conn.commit()

if __name__ == '__main__':
    uploadMSDUserAndSongIndexToID()
