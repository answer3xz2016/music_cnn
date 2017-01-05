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
import csv
import MuseUtil.museConfig as museConfig
import MuseUtil.museUtility as museUtility


credential_file = os.path.abspath(museConfig.myPostgresKeyXimalaya)
conn = psycopg2.connect(museUtility.getDatabaseKey(dbcred_file=credential_file))
cursor = conn.cursor()

cursor.execute("SELECT relname FROM pg_class WHERE relkind='r' and relname !~ '^(pg_|sql_)';")
print 'What are the tables in the database: ',cursor.fetchall()

cursor.execute("SELECT COUNT(id), MAX(id), MIN(id) FROM XIMALAYA.SONG_METADATA WHERE category = 2")
print cursor.fetchall()


cursor.execute("SELECT track_title,track_tags,track_intro,announcer_nickname,subordinated_album_album_title FROM XIMALAYA.SONG_METADATA")

with open('ximalaya_text_corpus.csv','a') as out:
    count = 0
    for i in cursor.fetchall():
        csv_out=csv.writer(out, delimiter=' ')
        csv_out.writerow(i)
        count += 1

     
cursor.execute("SELECT track_title,track_tags,track_intro,announcer_nickname,subordinated_album_album_title FROM XIMALAYA.SONG_METADATA_TOTAL")

with open('ximalaya_text_corpus.csv','a') as out:
    count = 0
    for i in cursor.fetchall():
        csv_out=csv.writer(out, delimiter=' ')
        csv_out.writerow(i)
        count += 1


conn.close()

