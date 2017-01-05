

from matplotlib import pyplot as plt
import re
import sys
import matplotlib
import numpy as np
matplotlib.rc('font', serif='Times New Roman') 
matplotlib.rc('text', usetex='false') 
matplotlib.rcParams.update({'font.size': 14})



def myPlot(x,y, label):
    
    plt.plot(x, y, 'o-', label=label)
    plt.xlabel('Number of iterations')
    #plt.ylabel('Approximate of Cost function of RBM [Arbitrary Units]')
    plt.ylabel('RMSE')
    plt.ylim(ymin=0, ymax=1)
    #plt.xlim([0.0, 1.0])
    plt.title('Cost function')
    #plt.legend(loc="lower left", prop={'size':10})
    plt.legend(loc="upper right", prop={'size':10})
    

logFiles = []
for itr in xrange(1):    
    #logFiles.append( '/home/ec2-user/log/RBM-muse-1M-step%s_rank_10_10_lambda_1.0_1.0_alpha_0.2_0.2_iterations_5000_5000.log' % itr )
    logFiles.append( '/home/ec2-user/log/RBM-muse-1M-step%s_rank_10_10_lambda_1.0_1.0_alpha_0.0005_0.0005_iterations_5000_5000.log' % itr )

lines = []

for logFile in logFiles:
    with open(logFile) as f:
        lines += f.readlines()

plt.figure(figsize=(12,9), facecolor='white' )

if len(logFiles)>1:
    counts = 0
    epochs = []
    costs = []
    for lptr in lines:
        if 'New model config' in lptr and 'myLabel' not in locals():
            rank , lambda_, num_iterations, alpha =  re.findall(r"[-+]?\d*\.\d+|\d+", lptr)
            myLabel = "rank = %s lambda = %s alpha = %s" % (rank , lambda_, alpha)
        if 'epoch' in lptr:
            cost = re.findall(r"[-+]?\d*\.\d+|\d+", lptr)
            epochs.append( counts )
            costs.append( cost[1]  )
            counts += 1
    myPlot(epochs, costs, label = myLabel)

else:

    for lptr in lines:
        if 'New model config' in lptr:
            rank , lambda_, num_iterations, alpha =  re.findall(r"[-+]?\d*\.\d+|\d+", lptr)
            if 'epochs' in locals() and 'costs' in locals():
                myPlot(epochs, costs, label = myLabel)
            myLabel = "rank = %s num_iterations = %s alpha = %s" % (rank , num_iterations, alpha)
            epochs = []
            costs = []
        if 'epoch' in lptr:
            #cost = re.findall(r"[-+]?\d*\.\d+|\d+", lptr)
            cost = lptr.split()
            epochs.append( float(cost[2]) )
            costs.append( np.sqrt(float(cost[8])/1000.)  )
            #print cost[2], cost[8]

    if epochs and costs:
        myPlot(epochs, costs, label = myLabel)

plt.show()
