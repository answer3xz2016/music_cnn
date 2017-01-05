import numpy as np
import pandas as pd
import sys, os
import random

if __name__ == "__main__":
  fileName = '/home/ec2-user/MSD_data/train_data_sorted.csv'
  fileNameTrain = fileName+'_train' 
  fileNameTest = fileName+'_test'
  df = pd.read_csv(fileName,
                   delim_whitespace=True, 
                   header = None, 
                   names = ['user_id', 'song_id', 'rating'] )
  userList = df['user_id'].unique()
  songList = df['song_id'].unique()
  numUser = len(userList)
  numSong = len(songList)
  print 'num of users: ', numUser
  print 'num of songs: ', numSong


  print 'building map for user count'
  if not os.path.exists('/home/ec2-user/MSD_data/song_usercounts.csv'):
    with open('/home/ec2-user/MSD_data/song_usercounts.csv', 'w') as f:
      for song_ptr in songList:
        usersForThisSong =  df.loc[ df['song_id'] == song_ptr ]['user_id'].tolist()
        numUsersForThisSong = len(usersForThisSong)
        print str(song_ptr) + ' ' + str(numUsersForThisSong)
        f.write( str(song_ptr) + ' ' + str(numUsersForThisSong) + '\n' )
  with open('/home/ec2-user/MSD_data/song_usercounts.csv', 'r') as f:
    myContent = f.readlines()
    mapNumUsersPerSong = { int(_ptr.strip().split()[0]) : int(_ptr.strip().split()[1]) for _ptr in myContent }
  print 'map building done'

  fraction_for_test_sample = 0.2
  f_train = open(fileNameTrain , 'w') 
  f_test = open(fileNameTest ,'w')

  myCount = 1
  for u_ptr in userList:
    songsForThisUser =  df.loc[ df['user_id'] == u_ptr ]['song_id'].tolist()
    numSongsForThisUser = len(songsForThisUser)
    numSongsToTestForThisUser =  int(numSongsForThisUser * fraction_for_test_sample)
    randomSelectedSongID = random.sample( songsForThisUser , numSongsToTestForThisUser )

    trainSongList = []
    for _songUser in songsForThisUser:
      if  not _songUser in randomSelectedSongID:
        trainSongList.append(_songUser)

    #print trainSongList, randomSelectedSongID , songsForThisUser
    #print u_ptr, len(trainSongList), len(randomSelectedSongID), len(songsForThisUser)
    if ( len(trainSongList) + len(randomSelectedSongID) ) != len(songsForThisUser):
      print 'this should not happen!!!! test + train != total'
    for testSong in randomSelectedSongID:
      ratingPtr = df.loc[ (df['song_id'] == testSong) & (df['user_id'] == u_ptr)  ]['rating'].tolist()[0]
      if mapNumUsersPerSong[testSong] > 1:
        f_test.write( str(u_ptr) + ' ' + str(testSong) + ' ' + str(ratingPtr) +  '\n' )
        mapNumUsersPerSong[testSong] -= 1
      else:
        f_train.write( str(u_ptr) + ' ' + str(testSong) + ' ' + str(ratingPtr) + '\n' )
        continue

    for trainSong in trainSongList:
      ratingPtr = df.loc[ (df['song_id'] == trainSong) & (df['user_id'] == u_ptr)  ]['rating'].tolist()[0]
      f_train.write( str(u_ptr) + ' ' + str(trainSong) + ' ' + str(ratingPtr) + '\n' )

    
    
  print 'done!'
