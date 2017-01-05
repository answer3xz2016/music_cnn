
import ConfigParser
import numpy as np


def getDatabaseKey(dbcred_file):
    conf = ConfigParser.ConfigParser()
    conf.read(dbcred_file)
    host = conf.get('database_creds','host')
    port = conf.get('database_creds','port')
    user = conf.get('database_creds','user')
    database = conf.get('database_creds','database')
    password = conf.get('database_creds','password')
    conn_str =  """dbname='{database}' user='{user}' host='{host}' port='{port}' password='{password}'""".format(database=database, host=host, port=port, user=user, password=password )
    return conn_str

def myPlot(x,y, xtitle, ytitle, plttitle, label, ymin, ymax):
    from matplotlib import pyplot as plt
    plt.plot(x, y, 'o-', label=label)
    if xtitle: plt.xlabel(xtitle, fontsize=18)
    if ytitle: plt.ylabel(ytitle, fontsize=18)
    plt.ylim(ymin=ymin, ymax=ymax)
    #plt.xlim([0.0, 1.0])
    if plttitle: plt.title(plttitle)
    #plt.legend(loc="lower left", prop={'size':10})
    plt.legend(loc="upper right", prop={'size':10})
    

def mySigmoid(x):
    return 1.0 / ( 1.0 + np.exp(-1.0*x) )



def encodeString(songTitle):
    if '(' in songTitle and ')' in songTitle:
        songTitle_decode = songTitle[:songTitle.rfind('(')].rstrip()
    else:
        songTitle_decode = songTitle
        
    if '[' in songTitle_decode and ']' in songTitle_decode:
        songTitle_decode = songTitle_decode[:songTitle_decode.rfind('[')].rstrip()
       
    songTitle_decode = songTitle_decode.replace("'","")    
                
    print 'after stripping for () and single quote', songTitle_decode
    return songTitle_decode  


def myPrecisionatK( y_scores, y_true, rank ):
    # rank has to be larger than 0, >=1
    sorted_index = np.argsort(y_scores)[::-1]
    sorted_index_topK = sorted_index[0:rank]
    my_precision_at_k = np.sum( y_true[ sorted_index_topK ] ) / (rank)
    return my_precision_at_k


def myAbsolutePrecisionatK(y_scores, y_true, rank ):
    # rank has to be larger than 0, >=1
    sorted_index = np.argsort(y_scores)[::-1]
    sorted_index_topK = sorted_index[0:rank]
    sorted_index_after_topK = sorted_index[rank:]
    num_selected = np.sum( y_true[sorted_index_topK]  )
    num_missed = np.sum( y_true[sorted_index_after_topK] )
    #print num_selected, num_missed
    n_u = np.min( [num_selected+num_missed, rank] )
    #print n_u
    sum_precision = 0.0
    for rank_ptr in xrange(rank):
        #print myPrecisionatK(y_scores, y_true, rank_ptr+1 ) , y_true[ sorted_index_topK[ rank_ptr ] ]
        sum_precision += myPrecisionatK(y_scores, y_true, rank_ptr+1 ) * y_true[ sorted_index_topK[rank_ptr] ]
    ap =  sum_precision / n_u
    #print ap
    return ap

def mymeanAbsolutePrecisionatK(y_scores, y_true, rank ):
    # rank has to be larger than 0, >=1
    pass


