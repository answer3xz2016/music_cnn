
##########################################################################################
# This is the implementation of a context-aware Collaborative Filtering model 
##########################################################################################

import numpy as np
from scipy.sparse import csr_matrix
from operator import add
from functools import partial
from collections import OrderedDict


class ImplicitCCF(object):
    def __init__(self, rank=10, maxIter=10, regParam=0.1, logFileName = 'CCF.log', starterModel = None, sc = None ):
        self.weights_Umf = None
        self.weights_Inf = None
        self.fileName = logFileName
        self.starterModel = starterModel
        self.sc = sc

    def showModel(self):
        logString = 'Debug: Here is the trained model Umf :' + str( self.weights_Umf.shape ) + '\n' + str( self.weights_Umf.shape )
        self.logThis(logString, 'a')
        logString = 'Debug: Here is the trained model Inf :' + str( self.weights_Inf.shape ) + '\n' + str( self.weights_Inf.shape )
        self.logThis(logString, 'a')

    def saveModel(self, modelConfig= ''):
        np.save(self.fileName + modelConfig  + '_Umf.npy', self.weights_Umf)
        np.save(self.fileName + modelConfig  + '_Inf.npy', self.weights_Inf)
        
    def logThis(self, content, mode):
        print content
        if mode == 'w':
            with open(self.fileName, mode) as myFile:
                myFile.write(content)
                myFile.write('\n')
        elif mode == 'a':
            with open(self.fileName, mode) as myFile:
                myFile.write(content)
                myFile.write('\n')

    def recommendUsers(self, my_song_id_index, topNForEachSong ):
        if not self.starterModel:
            return None
        Umf = np.load(self.starterModel[0]) # 0 element of starterModel is Umf
        Inf = np.load(self.starterModel[1]) # 1 element of startedModel is Inf 
        predictedRatings = Inf[int(my_song_id_index)].dot(np.transpose(Umf))
        recommenedUserList = np.argsort(predictedRatings)[::-1][:topNForEachSong]
        return recommenedUserList
        
    def recommendProducts(self, my_user_id_index ,topNForEachUser ):
        if not self.starterModel:
            return None
        Umf = np.load(self.starterModel[0]) # 0 element of starterModel is Umf
        Inf = np.load(self.starterModel[1]) # 1 element of startedModel is Inf 
        predictedRatings = Umf[int(my_user_id_index)].dot(np.transpose(Inf))
        recommenedSongList = np.argsort(predictedRatings)[::-1][:topNForEachUser]
        return recommenedSongList
        
    def predict(self, ratings):
        if not self.starterModel:
            logString = 'Debug: No intial Umf and Inf provided for prediction, will return an ERROR status' 
            self.logThis(logString, 'a')
        else:
            Umf = np.load(self.starterModel[0]) # 0 element of starterModel is Umf                                                                                                                         
            Inf = np.load(self.starterModel[1]) # 1 element of startedModel is Inf                                                                                                                         
            
        def computePredictedRating(r):
            localUmf =  UmfBr.value
            localInf =  InfBr.value
            predictedRating = localUmf[ r[0][0] ].dot( localInf[ r[0][1]  ] )
            return (r[0][0], r[0][1]), predictedRating
    
        # predict function starts here                                                                                                                                                                 
        m = ratings.map(lambda r: r[0][0] ).distinct().count() # num of users                                                                                                                              
        n = ratings.map(lambda r: r[0][1] ).distinct().count() # num of items                                                                                                              
        logString = 'Debug: Testing on number of users: %s and items: %s' % (m,n)
        self.logThis(logString, 'a')
        
        UmfBr = self.sc.broadcast(Umf)
        InfBr = self.sc.broadcast(Inf)
        
        return ratings.map(computePredictedRating)
        
        




    def train(self, ratings, rank, iterations=5, lambda_=0.01, alpha=0.01, blocks=1, nonnegative=False, implicityGain = 40.0 ):
        # these are the functions are going to be pushed to into spark executors, so cannot use class memeber functions......
        def buildSparseMatrixIndex(ratingsPerPartition):
            setUmf = list(OrderedDict.fromkeys(  [ i[0][0] for i in ratingsPerPartition  ]  ) )
            setInf = list(OrderedDict.fromkeys(  [ i[0][1] for i in ratingsPerPartition  ]  ) )
            mapUmf = dict(zip( setUmf , xrange(0, len(setUmf)) ) )
            mapInf = dict(zip( setInf , xrange(0, len(setInf)) ) )
            
            rowIndex, colIndex, ratingValues = [],[],[]
            for ptrRating in ratingsPerPartition:
                rowIndex.append( mapUmf[ptrRating[0][0]] )
                colIndex.append( mapInf[ptrRating[0][1]] )
                ratingValues.append( ptrRating[1] )

            return rowIndex, colIndex, ratingValues

        def cofiCostFunc(r, UmfBr, InfBr, lambdaBr, alphaBr, implicityGainBr):
            #print '\n\n Debug: I am entering the function running in each partition'
            myUmf = UmfBr.value
            myInf = InfBr.value
            lambda_ = lambdaBr.value
            alpha = alphaBr.value
            implicityGain = implicityGainBr.value

            ratingsPerPartition = list(r) # this dumps all the ratings within this partition block into a list
            #print '\n\nDebug: List of ratings in this partition ', len(ratingsPerPartition)
            # we need preserve the order of U and I indices, so we cannot use set
            indexUmf =  list(OrderedDict.fromkeys(  [ i[0][0] for i in ratingsPerPartition  ]  ))
            indexInf =  list(OrderedDict.fromkeys(  [ i[0][1] for i in ratingsPerPartition  ]  ))

            UmfPart = myUmf[ indexUmf ]
            InfPart = myInf[ indexInf ]
            #print '\n\nDebug: Umf part: ', UmfPart, UmfPart.shape
            #print '\n\nDebug: Inf part: ', InfPart, InfPart.shape
            
            # R = U * I^T
            InfPart_T = np.transpose(InfPart)
            Rmn = np.dot( UmfPart , InfPart_T )
            #print '\n\nDebug: Rmn part: ', Rmn, Rmn.shape
            
            # R = R .* A : Index of all the ratings
            rowIndex, colIndex, ratingValues = buildSparseMatrixIndex(ratingsPerPartition)
            A = csr_matrix(( [1.0]*len(rowIndex) , ( rowIndex , colIndex ) ) )
             
            # !!! binarize the ratings >0 to 1, == 0 to 0
            Y = csr_matrix(( [1.0]*len(ratingValues) , ( rowIndex , colIndex ) ) )
            #print '\n\nDebug raw rating matrix :', Y.toarray()

            # !!!!! here we do not mask out all the 0 rating ones for implicit ratings
            #R = A.multiply(Rmn)
            R = Rmn

            #print '\n\nR matrix after point multiply: ', R.toarray()

            # confidence matrix 1+gain*rating
            c_ui = csr_matrix(( list(np.array(ratingValues)*implicityGain) , ( rowIndex , colIndex ) ) )
            c_ui = c_ui.toarray()
            c_ui += 1.0

            #Delta = R - Y
            Delta =  R - Y.toarray()

            #print '\n\nDebug: Delta matrix after point-multiply: ', Delta.toarray()
            
            # cost function J
            #J =  sum(sum((X * Theta' - Y).^2 .* R)) / 2. + ...                                                                                                                      
            #  lambda/2. * ( sum(sum(Theta.^2)) +  sum(sum(X.^2)) );                 
            #J = ( Delta.multiply(Delta).sum()  ) / 2.0   #+ lambda_ / 2.0 * ( np.sum(np.square(UmfPart)) + np.sum(np.square(InfPart))   )

            errorSuarede = np.multiply(Delta,Delta)

            J = (  np.multiply( errorSuarede, c_ui ).sum()  ) / 2.0   #+ lambda_ / 2.0 * ( np.sum(np.square(UmfPart)) + np.sum(np.square(InfPart))   )
            
            #print '\n\nDebug: cost function of the squred terms in this partition is: ', J

            # (R - Y) * U + lambda * I
            #X_grad = (X*Theta' - Y).*R * Theta + lambda*X  ; 
            # here the bug could be adding lambda*InfPart to each partition, and then this component get double-counted, so let's take it out in each partition, add it in the end when collecting
            InfPart_grad =  np.multiply(c_ui, Delta).transpose().dot( UmfPart )  #+ lambda_ * InfPart
            #print '\n\nDebug: InfPart_grad, ', InfPart_grad, InfPart_grad.shape

            # (R - Y) * I + lambda * U
            # Theta_grad = (Theta*X' - Y').*R' * X + lambda*Theta;
            UmfPart_grad =  np.multiply(c_ui, Delta).dot( InfPart ) #+ lambda_ * UmfPart
            #print '\n\nDebug: UmfPart_grad, ', UmfPart_grad, UmfPart_grad.shape

            # update the weights
            #InfPart -= alpha * InfPart_grad
            #UmfPart -= alpha * UmfPart_grad
            #print '\n\n Debug: New updated InfPart ', InfPart
            #print '\n\n Debug: New updated UmfPart ', UmfPart

            wt_Umf = [ ( (0,i) , UmfPart_grad[idx]) for idx, i in enumerate(indexUmf) ]
            wt_Inf = [ ( (1,i) , InfPart_grad[idx]) for idx, i in enumerate(indexInf) ]
            wt_J = [ (2,J), ]
            #print 'test zhou', wt_Umf , '\n\n', wt_Inf, '\n\n', wt_J 
            wt_total = wt_Umf+wt_Inf+wt_J
            
            #print '\n\nDebug: these are the serialized weights that we are going to return for thi partition:', wt_total
            #return  ( (0,0), np.array([ [1,2], [10,100]]) ) , ( (0,1), np.array([3,4])) 
            return wt_total

        # train function starts here    
        m = ratings.map(lambda r: r[0][0] ).max() + 1 # num of users
        n = ratings.map(lambda r: r[0][1] ).max() + 1 # num of items
        logString = 'Debug: Training on number of users: %s and items: %s' % (m,n)
        self.logThis(logString, 'a')


        logString =  "Debug: Do we have an initial guess of Umf and Inf ? %s" % (self.starterModel)
        self.logThis(logString, 'a')

        # randomly initialize the weights if no starter model is given
        if not self.starterModel:
            Umf = np.random.rand(m,rank)
            Inf = np.random.rand(n,rank)
        else:
            Umf = np.load(self.starterModel[0]) # 0 element of starterModel is Umf
            Inf = np.load(self.starterModel[1]) # 1 element of startedModel is Inf


            
 
        #print '\n\nDebug, here is the randomly generated Umf and Inf', Umf, Inf
        lambdaBr = self.sc.broadcast(lambda_)
        alphaBr = self.sc.broadcast(alpha)
        implicityGainBr = self.sc.broadcast(implicityGain)
        cost_function = None

        for i in range(0,iterations):
            logString =  "Debug: Start training epoc #%s" % i
            self.logThis(logString, 'a')
            UmfBr = self.sc.broadcast(Umf)
            InfBr = self.sc.broadcast(Inf)
            #print 'Debug: Before updating', Umf, Inf
            cost_function = lambdaBr.value / 2.0 * ( np.sum(np.square(Umf)) + np.sum(np.square(Inf))   )    
            #print 'Debug: Before updating, cost function on the regularized terms ', cost_function
            weights = ratings.mapPartitions(partial(cofiCostFunc,UmfBr=UmfBr,InfBr=InfBr,lambdaBr=lambdaBr,alphaBr=alphaBr, implicityGainBr = implicityGainBr )).reduceByKey(add).collect()
            #weights = ratings.mapPartitions(lambda x: x).glom().collect()
            #print 'Debug: Here are the collected weights: ', weights
            for wt_ptr in weights:
                if isinstance( wt_ptr[0] , tuple):
                    if wt_ptr[0][0] == 0:
                        #print 'Debug: updating Umf', wt_ptr[0][0]
                        # remember to add -alpha*lambda*Umf back here we did not add it in each partition for duplication issues
                        Umf[ wt_ptr[0][1] ]  -=  alphaBr.value * wt_ptr[1] + alphaBr.value * lambdaBr.value * Umf[ wt_ptr[0][1] ]
                    if wt_ptr[0][0] == 1:
                        #print 'Debug: updating Inf', wt_ptr[0][0]
                        Inf[ wt_ptr[0][1]  ] -= alphaBr.value * wt_ptr[1] + alphaBr.value * lambdaBr.value * Inf[ wt_ptr[0][1]  ]
                if isinstance( wt_ptr[0] , int ):
                    cost_function += wt_ptr[1]
                    logString = 'Debug: epoch %s the total cost function is %f' %  ( i ,cost_function )
                    self.logThis(logString, 'a')

                    
            # convert updated weights to Umf and Inf, for the next iteration
            
            if cost_function == float('Inf'):
                logString = 'Debug: cost function blows up to Inf, quit the whole learning cycle now !'
                self.logThis(logString, 'a')
                return 1  # 1 is the error code which means training failed

            #print '\n\nAfter updating', Umf, Inf
            UmfBr.unpersist()
            InfBr.unpersist()

        self.weights_Umf = Umf
        self.weights_Inf = Inf
        return 0




