# -*- coding: utf-8 -*
import MuseMusicVendorAPI.ximalaya as ximalaya
import MuseUtil.museConfig as museConfig
import csv
import sys

def isTokenExpired(response):
    if isinstance(response, dict) and response.has_key('error_no') and response['error_no']==206:
        return True
    else:
        return False
    
def print_category():
    response = ximalaya.requestXimalaya('categories','list')
    if isTokenExpired(response):
        ximalaya.getAccessToken(museConfig.myMacAddress, forceGenerate = True)
        return
    print  '%s, %s, %s' % ('category_name', 'id', 'order_num')
    for i in response:
        print '%s, %s, %s' % ( i['category_name'], i['id'], i['order_num'] )


def print_tags():
    print 'category_id, type, tag_name ' 
    for i in xrange(39):
        for j in xrange(2):
            response = ximalaya.requestXimalaya('tags','list', category_id=i, type=j)
            if isTokenExpired(response):
                ximalaya.getAccessToken(museConfig.myMacAddress, forceGenerate = True)
                return
            
            for _ptr in response:
                print '%s, %s, %s ' % (i, j, _ptr['tag_name'])


def print_ximalaya_human_recommendations():
    response = ximalaya.requestXimalaya('categories','human_recommend')
    if isTokenExpired(response):
        ximalaya.getAccessToken(museConfig.myMacAddress, forceGenerate = True)
        return
    print 'category_id, category_name, human_recommend_category_name' 
    for _ptr in response:
        print '%s, %s, %s ' % (_ptr['id'], _ptr['category_name'], _ptr['human_recommend_category_name'])
        
def print_ximalaya_relative_alnum_recommendations(albumId):
    response = ximalaya.requestXimalaya('albums','relative_album', albumId=albumId)
    if isTokenExpired(response):
        ximalaya.getAccessToken(museConfig.myMacAddress, forceGenerate = True)
        return
    print response
    #print 'category_id, category_name, human_recommend_category_name' 
    #for _ptr in response:
    #    print '%s, %s, %s ' % (_ptr['id'], _ptr['category_name'], _ptr['human_recommend_category_name'])
        
def print_ximalaya_relative_track_recommendations(trackId):
    response = ximalaya.requestXimalaya('tracks','relative_album', trackId=trackId)
    if isTokenExpired(response):
        ximalaya.getAccessToken(museConfig.myMacAddress, forceGenerate = True)
        return
    relativeAlbums = response['reletive_albums']
    for _album in relativeAlbums:
        print _album['id'], ' | ', _album['album_title'], ' | ', _album['album_tags'], ' | ', _album['album_intro'], ' | ', _album['play_count'], ' | '

def print_track_related_info(_track, category_id=-1):    
    myvalues = [_track['id'],
    _track['kind'], 
    _track['track_title'].replace("\r\n","").replace("|",""), 
    _track['track_tags'].replace("\r\n","").replace("|",""),
    _track['track_intro'].replace("\r\n","").replace("|",""),
    #category_id,
    _track['category_id'],            
    _track['cover_url_small'], 
    _track['cover_url_middle'],
    _track['cover_url_large'],
    _track['announcer']['id'],
    _track['announcer']['nickname'].replace("\r\n","").replace("|",""),
    _track['announcer']['avatar_url'],
    _track['announcer']['is_verified'],
     _track['duration'],
     _track['play_count'],
     _track['favorite_count'],
     _track['comment_count'],
     _track['download_count'],
     _track['play_url_32'],
     _track['play_size_32'],
     _track['play_url_64'],
     _track['play_size_64'],
     _track['download_url'],
     _track['subordinated_album']['id'],
     _track['subordinated_album']['album_title'].replace("\r\n","").replace("|",""),
     _track['subordinated_album']['cover_url_small'],
     _track['subordinated_album']['cover_url_middle'],
     _track['subordinated_album']['cover_url_large'],
     _track['source'],
     _track['updated_at'],
     _track['created_at'],
     ]
    
                   
    myvalues = [unicode(s).encode("utf-8") for s in myvalues]
    return myvalues
    
            
            
                
        
def print_hot_tracks(category_id=0, page=1, count=20):
    response = ximalaya.requestXimalaya('tracks','hot', category_id=category_id, page=page, count=count)
    if isTokenExpired(response):
        ximalaya.getAccessToken(museConfig.myMacAddress, forceGenerate = True)
        response = ximalaya.requestXimalaya('tracks','hot', category_id=category_id, page=page, count=count)
    
    hotTracks = response['tracks']

    with open('ximalaya_data.csv','a') as out:
        for _track in hotTracks:
            csv_out=csv.writer(out, delimiter='|')
            myvalues = print_track_related_info(_track, category_id)
            csv_out.writerow(myvalues)
    
    
        
            

def print_tracks_get_batch(ids, fileName):
    
    response = ximalaya.requestXimalaya('tracks','get_batch', ids=ids )

    if isTokenExpired(response):
        ximalaya.getAccessToken(museConfig.myMacAddress, forceGenerate = True)
        response = ximalaya.requestXimalaya('tracks','get_batch', ids=ids )

    if not response:
        return

    batchTracks = response['tracks']
    
    with open(fileName,'w') as out:
        for _track in batchTracks:
            csv_out=csv.writer(out, delimiter='|')
            myvalues = print_track_related_info(_track)
            csv_out.writerow(myvalues)
        


if __name__ == '__main__':
    fileName = sys.argv[1]
    startId = sys.argv[2]
    endId = sys.argv[3]    
    _start_id = int(startId)
    _end_id = int(endId)
    _batch_size = 1
    numPadZeros = 12
    
    listOfTracks = []
    for _track_id in xrange( _start_id , _end_id , _batch_size):
        listOfTracks.append( str(_track_id) )
    listOfTracks = ','.join(listOfTracks)

    startIdStr = str(startId).zfill(numPadZeros)
    endIdStr = str(endId).zfill(numPadZeros)
    fileName += '_%s_%s' % (startIdStr,endIdStr)

    print_tracks_get_batch(ids=listOfTracks, fileName = fileName)


    #print_tracks_get_batch(ids=_track_id, fileName = fileName)
    #reader=csv.reader(open(fileName, 'r'), delimiter='|')
    #writer=csv.writer(open(fileName+'new', 'w'), delimiter='|')
    #mykeys = set()
    #for row in reader:
    #    if row[0] not in mykeys:
    #        writer.writerow(row)
    #        mykeys.add( row[0] )            







            #for _track_id in xrange(_start_id, 20000000, _batch_size):
            #print_category()                                                                                                                                                                                
            #print_tags()                                                                                                                                                                                    
            #print_ximalaya_human_recommendations()                                                                                                                                                                  
            #print_ximalaya_relative_track_recommendations(16849833)                                                                                                                                                 
            #for cat_ptr in xrange(40):                                                                                                                                                                              
            #    for page_ptr in xrange(1,10,1):                                                                                                                                                                     
            #        print cat_ptr, page_ptr                                                                                                                                                                         
            #        print_hot_tracks(category_id=cat_ptr, page=page_ptr)                                                                                                                                            
            # 30000 -> 20000000                                                                                                                                                                                      
            #_start_id = 16327173 # 16202628 # 16004279 # 16000000       


