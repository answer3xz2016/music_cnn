
import MuseMusicVendorAPI.ximalaya as ximalaya
import csv
from operator import itemgetter

def query_one_album_id(album_id):
    queried_data = ximalaya.request('playlist','album', device='android', albumId = album_id )
    if int(queried_data['ret'])  != 0 :
        # not found
        return None
    if not queried_data.has_key('data'):
        # empty album
        return None
    list_of_tracks = queried_data['data']
    return list_of_tracks

column_names = [        
    'trackId',
    'uid',
    'title',
    'albumTitle',
    'albumId',
    'albumImage',
    'duration',
    'isPaid',
    'playUrl32',
    'playUrl64',
    'downloadUrl',
    'playPathAacv164',
    'playPathAacv224',
]


with open('ximalaya_data.csv','a') as out:
    for album_id in xrange(303585,200000,-1):
        print 'album id trying right now is: %s' % album_id
        list_of_tracks = query_one_album_id(album_id)
        if not list_of_tracks:
            continue
        for i in list_of_tracks:
            #myvalues = itemgetter(*column_names)(i)
            myvalues = []
            for mykey in column_names:
                if not i.has_key(mykey):
                    i[mykey] = 'NA'
                myvalues.append(i[mykey])
                
            myvalues = [unicode(s).encode("utf-8") for s in myvalues]
            csv_out=csv.writer(out)
            csv_out.writerow(myvalues)
        
