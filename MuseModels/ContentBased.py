# -*- coding: utf-8 -*-

##########################################################################################                                                                                                            
# This is an implementation of a content-based recommendation model using word embeddings   
# Author: Dr. Z. Xing
# Email: joe.xing@nextev.com                                                                                                                               
########################################################################################## 

import numpy as np
import csv, os
import pandas as pd
import datetime
import random

class ContentBased(object):
    def __init__(self, dimenLatent=400):
        self.wordModel = None
        self.dimenLatent = dimenLatent
        
    def dowloadXimalayaContentFeatures(self, cursor, tablename, filename, mustHaveTags = True):
        '''
        Generate a local file of data from the remote data store
        '''
        print 'Downloading Ximalaya data from AWS RDS database'
        #cursor.execute("SELECT relname FROM pg_class WHERE relkind='r' and relname !~ '^(pg_|sql_)';")
        #print 'What are the tables in this database: ',cursor.fetchall()

        if mustHaveTags:    
            cursor.execute("SELECT * FROM %s WHERE track_tags != '' " % tablename )
        else:
            cursor.execute("SELECT * FROM %s  " % tablename )

        # remote database from PostgresSQL in RDS, ascii code output
        #with open(filename,'w') as out:
        #    count = 0
        #    for i in cursor.fetchall():
        #        csv_out=csv.writer(out, delimiter='|')
        #        csv_out.writerow(i)
        #        count += 1
        
        # local database from MySQL, unicode output
        with open(filename,'w') as out: 
            count = 0
            fetch = cursor.fetchmany
            while True:
                count += 1    
                print 'fetch #%s' % count
                rows = fetch(10000)
                if not rows: break
                for row in rows:
                    row = [ _ptr.encode('utf-8').replace('\n', ' ') if isinstance(_ptr,unicode) else _ptr for _ptr in row]
                    csv_out=csv.writer(out, delimiter='|')                                                                                                                                       
                    csv_out.writerow(row)                                                                                                                                                     
                    
        
        
    def genCatTagTitleFile(self, catList, tagList, titleList, introList,nicknameList, album_titleList, nameToCatId_raw, dataFileName, basePath = 'Downloads/'):
        '''
        Run the CRF tokenizer to segment the words, you need to have the Stanford NLP's segmenter for this
        '''
        with open(dataFileName+'_cat','w') as out:
            for i in catList:
                csv_out=csv.writer(out, delimiter=' ')
                csv_out.writerow([i.encode('utf-8')])  
                
        with open(dataFileName+'_tag','w') as out:
            for i in tagList:
                csv_out=csv.writer(out, delimiter=' ')
                csv_out.writerow(i)  
        
        with open(dataFileName+'_title','w') as out:
            for i in titleList:
                csv_out=csv.writer(out, delimiter=' ')
                csv_out.writerow([i])  
                
        with open(dataFileName+'_title_intro_nickname_albumname','w') as out:
            for _ptr in xrange(len(titleList)):
                csv_out=csv.writer(out, delimiter=' ')
                outputList =  tagList[_ptr]  + [ titleList[_ptr] ] + [ introList[_ptr] ] + [nicknameList[_ptr] ]+[album_titleList[_ptr]]
                csv_out.writerow(outputList)  
        
        with open(dataFileName+'_cat_total','w') as out:
            _my_key, _my_value = nameToCatId_raw.keys(), nameToCatId_raw.values()
            for _ptr in xrange(len(_my_key)):
                csv_out=csv.writer(out, delimiter=' ')
                outputList =  [ _my_value[_ptr] ] + [ _my_key[_ptr].encode('utf-8') ]
                csv_out.writerow(outputList)  
                      
                
        os.system("~/%sstanford-segmenter-2015-12-09/segment.sh pku %s UTF-8 0 > %s" % ( basePath, dataFileName+'_cat', dataFileName+'_cat_seg') )
        os.system("~/%sstanford-segmenter-2015-12-09/segment.sh pku %s UTF-8 0 > %s" % (basePath,dataFileName+'_tag', dataFileName+'_tag_seg') )
        os.system("~/%sstanford-segmenter-2015-12-09/segment.sh pku %s UTF-8 0 > %s" % (basePath,dataFileName+'_title', dataFileName+'_title_seg') )
        os.system("~/%sstanford-segmenter-2015-12-09/segment.sh pku %s UTF-8 0 > %s" % (basePath,dataFileName+'_title_intro_nickname_albumname', dataFileName+'_title_intro_nickname_albumname_seg') )
        os.system("~/%sstanford-segmenter-2015-12-09/segment.sh pku %s UTF-8 0 > %s" % (basePath,dataFileName+'_cat_total', dataFileName+'_cat_total_seg') )


        
        
    def loadWordRepresentation(self, modelFileName):
        try:
            import gensim
        except:
            print 'You need gensim to load the trained model'
            return
        self.wordModel = gensim.models.Word2Vec().load(modelFileName)
        
    
    
    def compareTagCat(self, dataFileName, catIdToIndex, myMapCatToColor, catIdToName, catIndexToId):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        
        f = open(dataFileName+'_tag_seg')
        myTag = f.readlines()
        myTag = [ i.rstrip() for i in myTag]
        
        f = open(dataFileName+'_cat_seg')
        myCat = f.readlines()
        myCat = [ i.rstrip() for i in myCat]
        
        f = open(dataFileName+'_title_seg')
        myTitle = f.readlines()
        myTitle = [ i.rstrip() for i in myTitle]
        
        # hack: let's use title, instead of tags
        #myTag = myTitle
        #print len(myTag), myTag[0]
        
        wordEmdArray = []
        listMaxSim = {}
        cateEmbedding = {}
        
        for _index, _cat in enumerate(myCat):
            cats = _cat.split() # could have multiple categories
            tags = myTag[_index].split() # could have multiple tags
            _num_cat = len(cats)
            
            maxSim = -100.0
            maxSimArray =  None
            aveSimArray = 0.0
            temp = 0.0
            for _catPtr in cats: # multiple cat words for each song
                for _tagPtr in tags: # multiple tag words for each song
                    try:
                        #print _catPtr, '-' ,_tagPtr, ' ',
                        sim = self.wordModel.similarity( _catPtr.decode('utf-8') , _tagPtr.decode('utf-8') )
                        if sim>maxSim: 
                            maxSim = sim
                            maxSimArray = self.wordModel[_tagPtr.decode('utf-8')]
                    except:
                        maxSimArray = self.wordModel[_catPtr.decode('utf-8')] # need to fix this later, if tags does not have word embedding
                 
                if not cateEmbedding.has_key( catIndexToId[_index] ):
                    if _num_cat <= 1:
                        cateEmbedding[ catIndexToId[_index] ] = self.wordModel[_catPtr.decode('utf-8')]
                    else:
                        temp +=  self.wordModel[_catPtr.decode('utf-8')]
                else:
                    pass
            
            if 'numpy' in str(type(temp)):
                temp /= _num_cat
                cateEmbedding[ catIndexToId[_index] ] = temp
            #print 'Max sim : ', maxSim 
            
            if not listMaxSim.has_key( catIndexToId[_index] ):
                listMaxSim[ catIndexToId[_index] ] = [maxSim]
            else:    
                listMaxSim[ catIndexToId[_index] ].append(maxSim)
            
            
            for _tagPtr in tags: # multiple tag words for each song
                try:
                    #print _tagPtr
                    _embedding_tag_words = self.wordModel[_tagPtr.decode('utf-8')]
                    aveSimArray += _embedding_tag_words
                    #print aveSimArray
                except:
                    #print 'not embedding for this word!!!'
                    pass
            
             
            maxSimArray /= np.linalg.norm(maxSimArray)
            maxSimArray = maxSimArray.tolist()
            
            if 'float' in str(type(aveSimArray)):
                aveSimArray = np.ones(400)
            aveSimArray /= np.linalg.norm(aveSimArray)
            aveSimArray = aveSimArray.tolist()
            #print len(aveSimArray)
            
            #wordEmdArray.append(maxSimArray)
            wordEmdArray.append(aveSimArray)
        
        wordEmdArray = np.array(wordEmdArray)
        lenSongs = wordEmdArray.shape[0]
        
        
        cateEmbedding_keys = cateEmbedding.keys()
        for _ptr in cateEmbedding_keys:
            _temp_array = cateEmbedding[_ptr].reshape(1, self.dimenLatent)
            _temp_array /= np.linalg.norm(_temp_array)
            wordEmdArray = np.concatenate( [ wordEmdArray, _temp_array ] , axis=0)
        
        pca = PCA(n_components=2)
        low_dim_embs = pca.fit(wordEmdArray[:,:]).transform(wordEmdArray[:,:])
        
        #print low_dim_embs.shape #, listMaxSim
        myCatPlot = listMaxSim.keys()
        mySimPlot = []
        for _ptr in myCatPlot:
            _meanSim = np.mean(np.array(listMaxSim[_ptr]))
            _maxSim = np.max(np.array(listMaxSim[_ptr]))
            _minSim = np.min(np.array(listMaxSim[_ptr]))
            _cout = len(listMaxSim[_ptr])
            mySimPlot.append(_meanSim)
            print 'Category: %2d %s \t\t maxSim: Mean %.3f Max %.3f Min %.3f \t Count: %d' % (_ptr, catIdToName[_ptr] , _meanSim,_maxSim, _minSim, _cout)
         
        
        plt.figure(figsize=(16,8), facecolor='white' )
        plt.subplot(121)
        

        for _catId in catIdToIndex.keys():
            plt.scatter(low_dim_embs[catIdToIndex[_catId],0], low_dim_embs[catIdToIndex[_catId],1], c = myMapCatToColor[_catId], s=50, label=catIdToName[_catId]  )
            _index_tail = cateEmbedding_keys.index(_catId)
            print _index_tail
            plt.plot( low_dim_embs[_index_tail+lenSongs , 0] ,  low_dim_embs[_index_tail+lenSongs, 1] , marker= '*', markersize=20, color=myMapCatToColor[_catId] )
        
        plt.ylim(ymin=-1.0, ymax=1.0)
        plt.xlim(xmin=-1.0, xmax=1.0)
        plt.title('Clustering of audio items based on their word-embedded features' )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        plt.tight_layout()
        plt.show()


                            
        #self.wordModel.most_similar
    
    def fetchData(self, dataFileName, basePath):
        ''' 
        Draw a random subsample from the whole dataset
        '''
        # randome draw 10,000 samples from the full dataset
        n = 2096024 #number of records in file
        s = 10000 #desired sample size
        skip = sorted(random.sample(xrange(n),n-s))

        #df = pd.read_csv(dataFileName, sep = '|', header=None, names = ['id','kind','track_title','track_tags','track_intro','category','cover_url_small','cover_url_middle','cover_url_large','announcer_id','announcer_nickname','announcer_avatar_url','announcer_is_verified','duration','play_count','favorite_count','comment_count','download_count','play_url_32','play_size_32','play_url_64','play_size_64','download_url','subordinated_album_id','subordinated_album_album_title','subordinated_album_cover_url_small','subordinated_album_cover_url_middle','subordinated_album_cover_url_large','source integer','updated_at','created_at'], skiprows=skip ) 

        df = pd.read_csv(dataFileName, sep = '|', header=None, names = ['id','kind','track_title','track_tags','track_intro','category','cover_url_small','cover_url_middle','cover_url_large','announcer_id','announcer_nickname','announcer_avatar_url','announcer_is_verified','duration','play_count','favorite_count','comment_count','download_count','play_url_32','play_size_32','play_url_64','play_size_64','download_url','subordinated_album_id','subordinated_album_album_title','subordinated_album_cover_url_small','subordinated_album_cover_url_middle','subordinated_album_cover_url_large','source integer','updated_at','created_at'] ) # nrows = 50000 

        #print df

        #nameToCatId = {u'热门' :0,u'资讯':1,u'音乐': 2, u'有声书': 3, u'娱乐': 4, u'儿童': 6,  u'健康养生': 7, u'商业财经': 8, u'历史人文': 9,u'情感生活': 10,u'其他': 11, u'相声评书': 12,u'教育培训': 13,
        #               u'百家讲坛': 14,u'广播剧': 15,u'戏曲': 16,u'电台': 17,u'IT科技': 18, u'汽车': 21, u'旅游': 22,u'电影': 23,u'动漫游戏': 24, u'脱口秀': 28, u'3D体验馆': 29, 
        #               u'名校公开课': 30, u'时尚生活': 31, u'小语种': 32, u'诗歌': 34,u'英语': 38, }
        
        nameToCatId_raw = {u'音乐': 2, u'有声书': 3, u'儿童': 6,  u'健康养生': 7, u'商业财经': 8, u'相声评书': 12, u'汽车': 21,u'电影': 23 }
        nameToCatId = {u'音乐 (Music)': 2,u'有声书 (Audio book)': 3, u'儿童 (Kids)': 6,  u'健康养生 (Healthy life)': 7, u'商业财经 (Business)': 8, u'相声评书 (Comedy)': 12, u'汽车 (Car)': 21, u'电影 (Movie)': 23}
        catIdToName_raw  = dict(zip( nameToCatId_raw.values() , nameToCatId_raw.keys() ))
        catIdToName  = dict(zip( nameToCatId.values() , nameToCatId.keys() ))
        _my_cat = df.loc[ df['category'].isin( nameToCatId.values() ) ][ 'category' ].values
        _my_tags = df.loc[ df['category'].isin( nameToCatId.values() ) ][ 'track_tags' ].values
        _my_title = df.loc[ df['category'].isin( nameToCatId.values() )  ]['track_title'].values
        _my_intro = df.loc[ df['category'].isin( nameToCatId.values() )  ]['track_intro'].values
        _my_nickname = df.loc[ df['category'].isin( nameToCatId.values() )  ]['announcer_nickname'].values
        _my_album_title = df.loc[ df['category'].isin( nameToCatId.values() )  ]['subordinated_album_album_title'].values
        #for __index, __ptr in enumerate(_my_nickname):
        #    print __index, ' | ',  _my_cat[__index], ' | ', _my_tags[__index], ' | ',_my_title[__index], ' | ', _my_intro[__index], ' | ', _my_nickname[__index] , ' | ',  _my_album_title[__index]
        catList, tagList, titleList, introList, nicknameList, album_titleList, catIDList,  = [], [], [], [], [], [], []
        for _index, _ptr in enumerate(_my_cat): 
            # cat 20 does not have a key, cat 11 is "others", let's get rid of them both, cat 0 is hot, remove it for now
            #if _ptr == 20 or _ptr == 11 or _ptr ==0 :
            #    continue
            catList.append(catIdToName_raw[_ptr])
            catIDList.append(_ptr)
            #print _my_tags[_index]
            tagList.append( _my_tags[_index].split(',')  )
            titleList.append( _my_title[_index] )
            introList.append( _my_intro[_index] )
            nicknameList.append( _my_nickname[_index] )
            album_titleList.append( _my_album_title[_index] )
        
        uniqueCatList = catIdToName.keys() 
        catIdToIndex = {}
        catIndexToId = {}
        for _ptr in uniqueCatList:
            myIndices = []
            for _catIndex, _catPtr in enumerate(catIDList):
                if _catPtr == _ptr:
                    myIndices.append(_catIndex)
                    catIndexToId[_catIndex] = _ptr
            catIdToIndex[_ptr] = myIndices
       
        colors_ = list(six.iteritems(colors.cnames))
        colors_ = ['b','g','r','m','y', 'k', '#fdb462', '#CC79A7']
        
        lenCodes = len(uniqueCatList)
        colors_ = colors_[:lenCodes]
        myMapCatToColor = dict(zip(uniqueCatList, colors_ ))
        

        

        #return catIdToIndex, myMapCatToColor, catIdToName, catIndexToId, catList,tagList,titleList,introList,nicknameList,album_titleList, nameToCatId_raw, catIDList

        self.genCatTagTitleFile(catList,tagList,titleList,introList,nicknameList,album_titleList, nameToCatId_raw, dataFileName, basePath=basePath )
        self.groudTruthCategory(dataFileName,catIDList)

        print 'how many unique cat ID are we selecting ? ', lenCodes
        print 'how many unique cat ID did we select ? ', len(set(catIDList))


    def fetchDataGeneral(self, dataFileName, basePath):
        df = pd.read_csv(dataFileName, sep = '|', header=None, names = ['id','kind','track_title','track_tags','track_intro','category','cover_url_small','cover_url_middle','cover_url_large','announcer_id','announcer_nickname','announcer_avatar_url','announcer_is_verified','duration','play_count','favorite_count','comment_count','download_count','play_url_32','play_size_32','play_url_64','play_size_64','download_url','subordinated_album_id','subordinated_album_album_title','subordinated_album_cover_url_small','subordinated_album_cover_url_middle','subordinated_album_cover_url_large','source integer','updated_at','created_at'] )
        _my_title = df[['id','track_title']].values

        with open(dataFileName+'_title','w') as out  :
            for i in _my_title:
                csv_out=csv.writer(out, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
                csv_out.writerow( [i[0]] + ['   \xef\xbc\x8c  '] + [i[1]])

        os.system("~/%sstanford-segmenter-2015-12-09/segment.sh pku %s UTF-8 0 > %s" % (basePath,dataFileName+'_title', dataFileName+'_title_seg') )
        
        
    def plotCat(self, dataFileName, catIDList, catIdToIndex, myMapCatToColor):
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        
        f = open(dataFileName+'_cat_seg')
        myCat = f.readlines()
        myCat = [ i.rstrip() for i in myCat]
        
        wordEmdArray = []
        for i in myCat:
            cats = i.split()
            firstCat = cats[0]
            catArray = self.wordModel[firstCat.decode('utf-8')]
            catArray /= np.linalg.norm(catArray)
            catArray = catArray.tolist()
            wordEmdArray.append(catArray)
        wordEmdArray = np.array(wordEmdArray)
        #print np.linalg.norm(wordEmdArray[1000])
        #for i in self.wordModel.most_similar( myCat[100].decode('utf-8') , topn=20): print i[0], i[1]

        pca = PCA(n_components=2)
        
        #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 1000
        #print wordEmdArray[:plot_only,:], np.isfinite(wordEmdArray[:plot_only,:]).all(), np.isnan(wordEmdArray[:plot_only,:]).any(), np.isinf(wordEmdArray[:plot_only,:]).any()
        #low_dim_embs = tsne.fit_transform(wordEmdArray[:plot_only,:])
        
        
        low_dim_embs = pca.fit(wordEmdArray[:,:]).transform(wordEmdArray[:,:])
        print low_dim_embs.shape
        
        for _catId in catIdToIndex.keys():
            plt.scatter(low_dim_embs[catIdToIndex[_catId],0], low_dim_embs[catIdToIndex[_catId],1], c = myMapCatToColor[_catId], s=100 )
        plt.show()
        
    def predictCategory(self, dataFileName):
        f_total = open(dataFileName+'_title_intro_nickname_albumname_seg') # this one contains tags + others
        f = open(dataFileName+'_tag_seg') # this one contains only tags
        f_real = open(dataFileName+'_cat_seg') # 
        
        myTags = f.readlines()
        myTags = [ i.rstrip() for i in myTags ]
        myTitleIntroNicknameAlbum = f_total.readlines()
        myTitleIntroNicknameAlbum = [ i.rstrip() for i in myTitleIntroNicknameAlbum]
        myTrueCat = f_real.readlines()
        myTrueCat = [ i.rstrip() for i in myTrueCat ]

        
        f = open(dataFileName+'_cat_total_seg')
        myAllCats = f.readlines()
        myAllCats = [i.rstrip() for i in myAllCats]
        
        predictedCat = []
        for _index, _words in enumerate(myTags):
            words = _words.split() # could have multiple categories
            maxSim = -1000.
            maxCat = None
            maxCatId = -1
            for _cat in myAllCats:
                cats = _cat.split()
                
                for i in words:
                    for j in cats[1:]:
                        try:
                            sim = self.wordModel.similarity( i.decode('utf-8') , j.decode('utf-8') )
                            if sim > maxSim:
                                maxSim = sim
                                maxCat = _cat
                                maxCatId = cats[0]
                        except:
                            pass

            print maxSim, maxCat, maxCatId, myTrueCat[_index]
            #if maxSim > 0.3: # use all other stuff in addition to tags
            if maxSim > 0.0: # just use tags
                predictedCat.append( maxCatId   )
                continue
            
            words = myTitleIntroNicknameAlbum[_index].split() # could have multiple categories                                                                                             
            print 'going to search whole text feature set with size: ', len(words)
            maxSim = -1000.
            maxCat = None
            maxCatId = -1
            for _cat in myAllCats:
                cats = _cat.split()
                for i in words:
                    for j in cats[1:]:
                        try:
                            sim = self.wordModel.similarity( i.decode('utf-8') , j.decode('utf-8') )
                            if sim > maxSim:
                                maxSim = sim
                                maxCat = _cat
                                maxCatId = cats[0]
                        except:
                            pass
            print maxSim, maxCat, maxCatId, myTrueCat[_index]
                            
                # use n_similarty
                #myWords = [ __ptr.decode('utf-8') for __ptr in words ]
                #myCats =  [ __ptr.decode('utf-8') for __ptr in cats[1:] ]
                #try:
                #    sim = self.wordModel.n_similarity( myWords, myCats )
                #    if sim > maxSim:
                #        maxSim = sim 
                #        maxCat = _cat 
                #        maxCatId = cats[0]
                #except:
                #    pass
            
            predictedCat.append( maxCatId   ) 
                 
        with open(dataFileName+'_predicted_cat','w') as out:
            for i in predictedCat:
                csv_out=csv.writer(out, delimiter=' ')
                csv_out.writerow([i])   
                
    def groudTruthCategory(self, dataFileName,catIDList):
        with open(dataFileName+'_true_cat','w') as out:
            for i in catIDList:
                csv_out=csv.writer(out, delimiter=' ')
                csv_out.writerow([i]) 
                
    def estimatePrecisionOfCategorization(self, dataFileName): 
        f_true = open(dataFileName+'_true_cat')
        f_predict = open(dataFileName+'_predicted_cat')
        trueCat = f_true.readlines()
        predictCat = f_predict.readlines()
        #print trueCat
        trueCat = np.array([ float(i.rstrip()) for i in trueCat])
        predictCat = np.array([float(i.rstrip()) for i in predictCat])
        correctNum =  float(np.count_nonzero(trueCat==predictCat))
        totalNum = float(len(trueCat))
        precision =  correctNum / totalNum
        precision_error = np.sqrt(correctNum*(totalNum-correctNum)/(totalNum)**3) # binomial error sqrt(npq)/n, sqrt(pq/n)
        print "%.3f +/- %.3f " % (precision, precision_error)
        
    def splitToTrainTestSample(self, dataFileName):
        
        f_train = open(dataFileName+'_train', 'a')
        f_test = open(dataFileName+'_test', 'a')
        
        f_predict = open(dataFileName+'_predicted_cat')
        predictCat = f_predict.readlines()
        predictCat = pd.Series([int(i.rstrip()) for i in predictCat])
        
        
        df = pd.read_csv(dataFileName, sep = '|', header=None, names = ['id','kind','track_title','track_tags','track_intro','category','cover_url_small','cover_url_middle','cover_url_large','announcer_id','announcer_nickname','announcer_avatar_url','announcer_is_verified','duration','play_count','favorite_count','comment_count','download_count','play_url_32','play_size_32','play_url_64','play_size_64','download_url','subordinated_album_id','subordinated_album_album_title','subordinated_album_cover_url_small','subordinated_album_cover_url_middle','subordinated_album_cover_url_large','source integer','updated_at','created_at'] )      
        df['created_at_datetime'] = df['created_at'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1e3) )
        nameToCatId_raw = {u'音乐': 2, u'有声书': 3, u'儿童': 6,  u'健康养生': 7, u'商业财经': 8, u'相声评书': 12, u'汽车': 21,u'电影': 23 }
        nameToCatId = {u'音乐 (Music)': 2,u'有声书 (Audio book)': 3, u'儿童 (Kids)': 6,  u'健康养生 (Healthy life)': 7, u'商业财经 (Business)': 8, u'相声评书 (Comedy)': 12, u'汽车 (Car)': 21, u'电影 (Movie)': 23}
        catIdToName_raw  = dict(zip( nameToCatId_raw.values() , nameToCatId_raw.keys() ))
        catIdToName  = dict(zip( nameToCatId.values() , nameToCatId.keys() ))
        dfCatUnderTest = df.loc[ df['category'].isin( nameToCatId.values() ) ].copy() # need the copy here to avoid chain assignements
        
       
        dfCatUnderTest['category_predict'] = predictCat.values
         
        _myUserId = dfCatUnderTest[ 'announcer_id' ].values
        userList = np.unique(_myUserId)
        
        #print dfCatUnderTest[ ['announcer_id', 'id', 'duration','play_count', 'created_at_datetime', 'category', 'category_predic'] ]
        for userPtr in userList:
            dfThisUser = dfCatUnderTest.loc[ dfCatUnderTest['announcer_id'] == userPtr][ ['announcer_id', 'id', 'duration','play_count', 'created_at_datetime', 'category', 'category_predict'] ]
            totalSample = len(dfThisUser)
            
            if totalSample<10: # is a user upload less than 10 items, ignore him
                continue
            
            catForThisUser = dfThisUser['category'].values[0]
            #print catForThisUser
            
            test_sample = dfThisUser[0:1].copy()
            train_sample = dfThisUser[1:].copy()
            numTrainSample = len(train_sample)
            numTestSample = len(test_sample)
            
            # randome same number of items that are not uploaded by this user
            train_sample_false = dfCatUnderTest.loc[ (dfCatUnderTest['announcer_id']!=userPtr ) & (dfCatUnderTest['category']!=catForThisUser) ][ ['announcer_id', 'id', 'duration','play_count', 'created_at_datetime', 'category', 'category_predict'] ].sample(n=numTrainSample).copy()
            test_sample_false = dfCatUnderTest.loc[ (dfCatUnderTest['announcer_id']!=userPtr ) & (dfCatUnderTest['category']!=catForThisUser) ][ ['announcer_id', 'id', 'duration','play_count', 'created_at_datetime', 'category', 'category_predict'] ].sample(n=numTestSample).copy()
            
            #print train_sample, '\n\n', train_sample_false
            #print '\n\n\n\n\n\n\n'
            #print test_sample, '\n\n', test_sample_false
            
            train_sample['id'] = train_sample['id'].apply(lambda x: 1.0 )
            train_sample_false['id'] = train_sample_false['id'].apply(lambda x: 0.0 )
            train_sample_false['announcer_id'] = train_sample_false['announcer_id'].apply(lambda x: userPtr )
            test_sample['id'] = test_sample['id'].apply(lambda x: 1.0 )
            test_sample_false['id'] = test_sample_false['id'].apply(lambda x: 0.0 )
            test_sample_false['announcer_id'] = test_sample_false['announcer_id'].apply(lambda x: userPtr )
            #print train_sample[['announcer_id', 'id', 'duration','play_count', 'created_at_datetime', 'category_predict']]
            #print train_sample_false[['announcer_id', 'id', 'duration','play_count', 'created_at_datetime', 'category_predict']]
            
            train_sample[['announcer_id', 'id', 'duration','play_count', 'created_at_datetime', 'category_predict']].to_csv(f_train, sep = ' ', header = False, index= False)
            train_sample_false[['announcer_id', 'id', 'duration','play_count', 'created_at_datetime', 'category_predict']].to_csv(f_train, sep = ' ', header = False, index= False)
            test_sample[['announcer_id', 'id', 'duration','play_count', 'created_at_datetime', 'category_predict']].to_csv(f_test, sep = ' ', header = False, index= False)
            test_sample_false[['announcer_id', 'id', 'duration','play_count', 'created_at_datetime', 'category_predict']].to_csv(f_test, sep = ' ', header = False, index= False)
            
            #print numTrainSample, numTestSample
            

    def learnUserAffinity(self, dataFileName):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import log_loss
        from sklearn.metrics import precision_recall_curve
        from sklearn import metrics
        import matplotlib.pyplot as plt
        from MuseUtil.museUtility import myPlot
        
        f_train = dataFileName+'_train'
        f_test = dataFileName+'_test'
        dfTrain = pd.read_csv(f_train, delim_whitespace = True, names=['announcer_id', 'id', 'duration','play_count', 'created_at_datetime', 'category_predict'], parse_dates=['created_at_datetime'])
        dfTest = pd.read_csv(f_test, delim_whitespace = True, names=['announcer_id', 'id', 'duration','play_count', 'created_at_datetime', 'category_predict'], parse_dates=['created_at_datetime'])
        
        dfTrain['day_of_week'] = dfTrain['created_at_datetime'].apply(lambda x: x.dayofweek )
        dfTrain['hour_of_day'] = dfTrain['created_at_datetime'].apply(lambda x: x.hour )
        dfTrain['year'] = dfTrain['created_at_datetime'].apply(lambda x: x.year )
        
        dfTest['day_of_week'] = dfTest['created_at_datetime'].apply(lambda x: x.dayofweek )
        dfTest['hour_of_day'] = dfTest['created_at_datetime'].apply(lambda x: x.hour )
        dfTest['year'] = dfTest['created_at_datetime'].apply(lambda x: x.year )
        
        _myUserId = dfTrain[ 'announcer_id' ].values
        userList = np.unique(_myUserId)
        
        trueLikeDislike = []
        predictedLikeDislike = []
        
        for user_ptr in userList:
            dfTrainThisUser = dfTrain.loc[ dfTrain['announcer_id'] == user_ptr ].copy()
            dfTestThisUser = dfTest.loc[ dfTest['announcer_id'] == user_ptr ].copy()
            
            xTrain, yTrain = dfTrainThisUser[['duration','play_count', 'day_of_week', 'hour_of_day', 'year', 'category_predict']].values, dfTrainThisUser[['id']].values
            yTrain = yTrain.ravel()
            
            xTest, yTest = dfTestThisUser[['duration','play_count', 'day_of_week', 'hour_of_day', 'year', 'category_predict']].values, dfTestThisUser[['id']].values
            yTest = yTest.ravel()
            
    
            clf = RandomForestClassifier(n_estimators=25)
            #print xTrain.shape, yTrain.shape
            clf.fit(xTrain, yTrain)
            clf_probs = clf.predict_proba(xTest)[:, 1]
            #print yTest, clf_probs
            
            trueLikeDislike.extend(yTest)
            predictedLikeDislike.extend(clf_probs)
            #sig_score = log_loss(yTest, clf_probs)
            #print sig_score

        precision, recall, thresholds = precision_recall_curve(trueLikeDislike, predictedLikeDislike)
        fpr, tpr, thresholds = metrics.roc_curve(trueLikeDislike, predictedLikeDislike)
        print 'AUC is %s ' % metrics.auc(fpr, tpr)
        
    
        myFig = plt.figure(figsize=(8,6), facecolor='white')
        myPlot( fpr, tpr , 'False Positive', 'True Positive', 'ROC curve', '',-0.2,1.2)
        myFig.savefig('temp.eps', format='eps')
        plt.tight_layout()
        plt.show()
        


if __name__ == '__main__':  
    import os
    import sys
    import psycopg2, sqlite3
    import csv
    import MuseUtil.museConfig as museConfig
    import MuseUtil.museUtility as museUtility
    import pandas as pd
    import numpy as np
    from matplotlib import colors
    import six


    def returnConnRemoteDB():
        credentialFile = os.path.abspath(museConfig.myPostgresKeyXimalaya)
        conn = psycopg2.connect(museUtility.getDatabaseKey(dbcred_file=credentialFile))
        cursor = conn.cursor()
        query = """CREATE SCHEMA IF NOT EXISTS XIMALAYA;"""                                                                                         
        cursor.execute(query)
        conn.commit()   
        return conn, cursor

    def returnConnLocalDB():
        #conn = sqlite3.connect('/home/ec2-user/ximalaya_data/test.db')
        conn = sqlite3.connect('/a/muse_nebula_shared_data/ximalaya_dimension_table.db')
        cursor = conn.cursor() 
        return conn, cursor    
   
    conn, cursor = returnConnLocalDB()

    # (1). start create an instance for our content-based analysis
    model = ContentBased()
    
    # (2). (optional step) download ximalaya features into a local file

    # down from RDS PostgreSQL
    #model.dowloadXimalayaContentFeatures(cursor, 'XIMALAYA.SONG_METADATA', 'ximalaya_hot_tracks.csv', mustHaveTags = True)
    #model.dowloadXimalayaContentFeatures(cursor, 'XIMALAYA.SONG_METADATA_TOTAL', 'ximalaya_tracks.csv', mustHaveTags = False)

    # down from MySQL
    #model.dowloadXimalayaContentFeatures(cursor, 'SONG_METADATA_PRODUCTION', '/a/joe_data/ximalaya_data/complete_database/ximalaya_tracks.csv', mustHaveTags = False)
    
        

    # (3). (need this step for similarity calculation) load word embedding model
    # gensim trained model
    #model.loadWordRepresentation('/home/ec2-user/ximalaya_data/trained_model/gensim/ximalaya_model')
    # cuda c++ trained model
    #model.loadWordRepresentation('/home/ec2-user/ximalaya_data/trained_model/gpu_cpp/gpu_cpp_binary_model')
    # gensim trained with wikipedia
    #model.loadWordRepresentation('/home/ec2-user/ximalaya_data/trained_model/gensim/with_wikipeadia/ximalaya_model')
    # cuda c++ trained model with wikipedia
    #model.loadWordRepresentation('/home/ec2-user/ximalaya_data/trained_model/gpu_cpp/with_wikipedia/gpu_cpp_model_with_wikipedia')


    # (3.5) fetch data and create some list of features
    # getting local data ready now, can be run on hot tracks or the regular tracks
    #dataFileName = '/home/ec2-user/ximalaya_data/tag_generation/ximalaya_tracks.csv'
    #dataFileName = '/home/ec2-user/ximalaya_data/tag_generation/ximalaya_tracks_must_have_tags.csv'
    #dataFileName = '/home/ec2-user/ximalaya_data/tag_generation/ximalaya_hot_tracks.csv'
    dataFileName = '/a/joe_data/ximalaya_data/complete_database/ximalaya_tracks.csv'

    #catIdToIndex, myMapCatToColor, catIdToName, catIndexToId, \
    #catList,tagList,titleList,introList,nicknameList,\
    #album_titleList, nameToCatId_raw, catIDList = model.fetchData(dataFileName, '') # second parameter is for CRF tokenizer

    # symplified version of fetch data, will create tag, title, intro text files for only the selected categories
    #model.fetchData(dataFileName, '') # second parameter is for CRF tokenizer

    # (3.6)  more general version of fetchData, all categories
    model.fetchDataGeneral(dataFileName, '')
    sys.exit(0)
    

    # (4) (not needed any more, optional) generate 3 files for cat, tags and title and 1 file for combined all the features, segmented words each line for each song   
    #model.genCatTagTitleFile(catList,tagList,titleList,introList,nicknameList,album_titleList, nameToCatId_raw, dataFileName, basePath='' )
    #model.groudTruthCategory(dataFileName,catIDList)
    ####model.plotCat(dataFileName, catIDList, catIdToIndex, myMapCatToColor)
   
    

    # (5) plot cluster only using tag information
    #model.compareTagCat(dataFileName, catIdToIndex, myMapCatToColor, catIdToName, catIndexToId)
    
    
    # (6) predict the category of the tracks
    #model.predictCategory(dataFileName)
    
        

    # (7) compute precision of categorization
    #model.estimatePrecisionOfCategorization(dataFileName)
            

    # (8) split sample into train and test sample
    #model.splitToTrainTestSample(dataFileName=dataFileName)
    
    # (9) learn the user affinity vector
    #model.learnUserAffinity(dataFileName=dataFileName)
    
    
