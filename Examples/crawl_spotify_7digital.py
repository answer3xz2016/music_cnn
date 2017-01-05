
try:
    import py7D
    import spotipy
except:
    print 'Cannot load API for 7digital and Spotify'

import os
import csv
import psycopg2
import MuseUtil.museConfig as museConfig
import MuseUtil.museUtility as museUtility
import time

def getSpotify(songID, songTitle, artistName ): # title, artist, songID
    print "Try to use spotify first"
    spotify = spotipy.Spotify()
    
    songTitle_decode = museUtility.encodeString(songTitle)
    artistName_decode = museUtility.encodeString(artistName)
    
    results = spotify.search(q='track:%s artist:%s' % (songTitle_decode,artistName_decode),type='track', limit =1)
    if not results['tracks']['items']:
        print "Try search spotify again with only song title"
        results = spotify.search(q='track:%s' % (songTitle_decode),type='track', limit =1)
    print 'wait for sometime.....'
    time.sleep(1)
    track_title_7 = results['tracks']['items'][0]['name']
    track_id_7 = results['tracks']['items'][0]['id']
    track_artist_name_7 = results['tracks']['items'][0]['artists'][0]['name']
    track_artist_id_7 = results['tracks']['items'][0]['artists'][0]['id']
    track_artist_url_7 = results['tracks']['items'][0]['external_urls']['spotify']
    track_release_title_7 = results['tracks']['items'][0]['album']['name']
    track_release_id_7 = results['tracks']['items'][0]['album']['id']
    track_release_image = results['tracks']['items'][0]['album']['images'][0]['url']
    needToReturn =[songID, track_title_7,track_id_7,track_artist_name_7,track_artist_id_7,
                             track_artist_url_7,track_release_title_7,track_release_id_7, 
                             track_release_image ]   
            
    
    return needToReturn

def get7Digital(songID, songTitle, artistName):# title, artist, songID
    print "Try to use 7 digital now"
    
    songTitle_decode = museUtility.encodeString(songTitle)
    artistName_decode = museUtility.encodeString(artistName)
    response = py7D.request('track', 'search', q=artistName_decode + ' ' + songTitle_decode, pageSize = 1 )
    tracks = response['response']['searchResults']['searchResult']
    track = tracks
    track_title_7 = track['track']['title']
    track_id_7 = track['track']['@id']
    track_artist_name_7 = track['track']['artist']['name']
    track_artist_id_7 = track['track']['artist']['@id']
    track_artist_url_7 = track['track']['artist']['url'] # where you can play the song
    track_release_title_7 = track['track']['release']['title']
    track_release_id_7 = track['track']['release']['@id']
    track_release_image = track['track']['release']['image']
    needToReturn =[songID, track_title_7,track_id_7,track_artist_name_7,track_artist_id_7,
                             track_artist_url_7,track_release_title_7,track_release_id_7, 
                             track_release_image]             
                
                
    return needToReturn






USER_CRED_FILE = os.path.abspath(museConfig.myPostgresKey)
conn = psycopg2.connect(museUtility.getDatabaseKey(dbcred_file=USER_CRED_FILE))
cursor = conn.cursor()
    

fileName_songId_list = './spotify_7digital_song_id_list.csv'
if not os.path.exists(fileName_songId_list): 
    cursor.execute("SELECT DISTINCT song_id  FROM master_music_rating_total ORDER BY song_id")
    query = cursor.fetchall()
    queryResult = [ i[0] for i in query]   
    with open(fileName_songId_list,'w') as out:
        for i in query:
            csv_out=csv.writer(out)
            csv_out.writerow(i)
        

fileName_no_match_songId_list = './spotify_7digital_no_match_song_id_list.csv'
fileName_data = './spotify_7digital_data.csv'

if os.path.exists(fileName_songId_list): 
    print 'Now we start to crawl data!'
    with open(fileName_songId_list, 'r') as csvfile:
        songList = csv.reader(csvfile)
        count = 0
        for row in songList:
            row = row[0]
            count += 1
            
            cursor.execute("SELECT DISTINCT title, artist_name FROM master_music_rating_total WHERE song_id = '%s'" % row)
            query = cursor.fetchall()
            songTitle = query[0][0]
            artistName = query[0][1]
            
            with open(fileName_data, "a") as datafile:
                try:
                    resultsFromAPI = getSpotify(row,songTitle, artistName)
                    csv_out=csv.writer(datafile)
                    csv_out.writerow(resultsFromAPI) 
                except:
                    try:
                        resultsFromAPI = get7Digital(row,songTitle, artistName)
                        csv_out=csv.writer(datafile)
                        csv_out.writerow(resultsFromAPI)
                    except:
                        print 'no match !', row
                        with open(fileName_no_match_songId_list, "a") as nomatchfile:
                            nomatchfile.write(row+'\n')
            
            print 'song #%s' % count
            #if count >= 3:
            #    break
