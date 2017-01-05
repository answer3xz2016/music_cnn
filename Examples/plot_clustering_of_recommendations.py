from matplotlib import pyplot as plt
import re
import sys
import matplotlib
import numpy as np
import pandas as pd
import six
import random
from matplotlib import colors
from sklearn.decomposition import PCA
matplotlib.rc('font', serif='Times New Roman') 
matplotlib.rc('text', usetex='false') 
matplotlib.rcParams.update({'font.size': 14})


fileSongToGenre = '/home/ec2-user/yahoo_music_data/song-attributes.txt'    
fileGenreHierarchy = '/home/ec2-user/yahoo_music_data/genre-hierarchy.txt'    


dfSongToGen = pd.read_csv(fileSongToGenre,delim_whitespace =True, names= ['song_id', 'album_id', 'artist_id', 'genre_id' ] )
dfGenHiera = pd.read_csv(fileGenreHierarchy, delim_whitespace =True, names=['genre_id', 'parent_genre_id', 'level', 'genre_name'] )


topN = 1000
Umf = np.load('/home/ec2-user/log/CCF-muse-yahoo-step7_rank_10_10_lambda_1.0_1.0_alpha_1e-05_1e-05_iterations_50_50.log_Umf.npy')
Inf = np.load('/home/ec2-user/log/CCF-muse-yahoo-step7_rank_10_10_lambda_1.0_1.0_alpha_1e-05_1e-05_iterations_50_50.log_Inf.npy')

userID = random.randint(0, 200000) 
songsIDThatUserLike = Umf[userID].dot(Inf.T)
songsIDThatUserLikeTopN = np.argsort(songsIDThatUserLike)[::-1][:topN]
mapSongIdToIndex = dict(zip(songsIDThatUserLikeTopN , xrange(len(songsIDThatUserLikeTopN))  ))
#print songsIDThatUserLikeTopN # top N recom song ID

masterDf = dfSongToGen[ dfSongToGen['song_id'].isin(songsIDThatUserLikeTopN) ]
masterDf = masterDf.join(dfGenHiera, on='genre_id', how='inner', lsuffix='left', rsuffix='right' )
masterDf = masterDf[['genre_id', 'song_id', 'level']]

for _ptr in xrange(3):
    for index, row in masterDf.iterrows():
        if row['level']>1:
            gid = masterDf.loc[index, 'genre_id']
            new_gid =  dfGenHiera.loc[ dfGenHiera['genre_id'] == gid ]['parent_genre_id'].values[0]
            masterDf.loc[index, 'genre_id'] = new_gid
            masterDf.loc[index, 'level'] -= 1
        
masterDf = masterDf.join(dfGenHiera, on='genre_id', how='inner', lsuffix='left', rsuffix='right')[['genre_id', 'song_id', 'genre_name']]

colorCodesForGenre = masterDf['genre_id'].unique()
colorCodesForGenreTotal = dfGenHiera.loc[ dfGenHiera['level'] == 1 ]['genre_id'].unique()
colors_ = list(six.iteritems(colors.cnames))
lenCodes = len(colorCodesForGenreTotal)
colors_ = colors_[:lenCodes]
myMapGenreToColor = dict(zip(colorCodesForGenreTotal, colors_ ))


#latentFeatures = Inf[songsIDThatUserLikeTopN]
latentFeatures = Inf
pca = PCA(n_components=2)
pcaOfLatentFeatures = pca.fit(latentFeatures).transform(latentFeatures)
pcaOfLatentFeatures = pcaOfLatentFeatures[songsIDThatUserLikeTopN]
print pcaOfLatentFeatures.shape

plt.figure(figsize=(16,8), facecolor='white' )
plt.subplot(121)

for genre_id_ in colorCodesForGenre :
    if genre_id_ == 0:
        continue
    _my_genre_name = masterDf.loc[ masterDf['genre_id'] == genre_id_ ]['genre_name'].unique()[0]
    _my_song_id = masterDf.loc[ masterDf['genre_id'] == genre_id_ ]['song_id'].values
    _my_song_id_index = [ mapSongIdToIndex[i]  for i in _my_song_id ]
    plt.scatter(pcaOfLatentFeatures[_my_song_id_index,0], pcaOfLatentFeatures[_my_song_id_index,1], c = myMapGenreToColor[genre_id_], label=_my_genre_name, s=100 )

plt.ylim(ymin=-1.0, ymax=1.0)
plt.xlim(xmin=-1.0, xmax=1.0)
plt.title('Clustering of top-N (%s) recommended songs for user ID %s' % (topN, userID), fontsize = 15 )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.show()



latentFeaturesUser = [61228, 51055, 181652, 117588]
latentFeaturesUserRnd = [ random.randint(0, 200000) for _itr in xrange(10000) ]

pcaUser = PCA(n_components=2)
pcaOfLatentFeaturesUser = pcaUser.fit(Umf).transform(Umf)


plt.figure(figsize=(16,8), facecolor='white' )
plt.subplot(121)
plt.scatter(pcaOfLatentFeaturesUser[latentFeaturesUserRnd,0], pcaOfLatentFeaturesUser[latentFeaturesUserRnd,1], c = 'black' , s=10 )
plt.scatter(pcaOfLatentFeaturesUser[latentFeaturesUser,0], pcaOfLatentFeaturesUser[latentFeaturesUser,1], c = 'red' , s=100 )
plt.show()
