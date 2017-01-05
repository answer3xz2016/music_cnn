import numpy as np
from scipy.sparse import csr_matrix
from operator import add
from functools import partial
from collections import OrderedDict


class RBM:
  
  def __init__(self, logFileName = 'CCF.log', starterModel = None, sc = None):
    self.weights = None
    self.fileName = logFileName
    self.starterModel = starterModel
    self.sc = sc


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


  def train(self, ratings, num_hidden=10, learning_rate = 0.1, max_epochs = 1000):
    """
    Train the machine.
    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.    
    We need make data as RDD in the context of SPARK framework, so data is a RDD now!

    """
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
    
    def _mylogistic(x):
      return 1.0 / (1 + np.exp(-x))

    def rbmPosHidden(r, weightsBr):
      myweights = weightsBr.value
      ratingsPerPartition = list(r) # this dumps all the ratings within this partition block into a list                                                                                              
      #print '\n\nDebug: List of ratings in this partition ', len(ratingsPerPartition)                                                                                                                
      # we need preserve the order of U and I indices, so we cannot use set                                                                                                                           
      indexUmf =  list(OrderedDict.fromkeys(  [ i[0][0] for i in ratingsPerPartition  ]  ))
      indexInf =  list(OrderedDict.fromkeys(  [ i[0][1]+1 for i in ratingsPerPartition  ]  )) # add 1 to n_music for the biasing unit

      # subset of weights for only the music in this partition
      myweightsPart = myweights[  indexInf ] # no bias unit in weights
      rowIndex, colIndex, ratingValues = buildSparseMatrixIndex(ratingsPerPartition)
      A = csr_matrix(( [1.0]*len(rowIndex) , ( rowIndex , colIndex ) ) )
      Y = csr_matrix((ratingValues, ( rowIndex , colIndex ) ) )
      mydata = Y # no bias unit in data  

      # Clamp to the data and sample from the hidden units.                                                                                                                                                
      # (This is the "positive CD phase", aka the reality phase.)                                                                                                                                          
      pos_hidden_activations = mydata.dot(myweightsPart )  # n_u x ( n_h+1 )
      wt_pos_hidden_activations = [ ( (i,) , pos_hidden_activations[idx]) for idx, i in enumerate(indexUmf) ]
      #print 'test zhou', wt_pos_hidden_activations
      return wt_pos_hidden_activations

    def rbmPosAssociation(r, pos_hidden_probsBr):
      my_pos_hidden_probs = pos_hidden_probsBr.value # n_u x n_h+1
      ratingsPerPartition = list(r)
      indexUmf =  list(OrderedDict.fromkeys(  [ i[0][0] for i in ratingsPerPartition  ]  ))
      indexInf =  list(OrderedDict.fromkeys(  [ i[0][1] for i in ratingsPerPartition  ]  ))
      rowIndex, colIndex, ratingValues = buildSparseMatrixIndex(ratingsPerPartition)
      Y = csr_matrix((ratingValues, ( rowIndex , colIndex ) ) )
      mydata = Y.transpose()
      my_pos_hidden_probsPart = my_pos_hidden_probs[  indexUmf ]
      pos_associations = mydata.dot(my_pos_hidden_probsPart)
      wt_pos_associations = [ ( (i,) , pos_associations[idx]) for idx, i in enumerate(indexInf) ]
      return wt_pos_associations

    def rbmCostFunction( r, weightsBr , pos_hidden_statesBr ):
      myweights = weightsBr.value
      mypos_hidden_state = pos_hidden_statesBr.value
      ratingsPerPartition = list(r)
      indexUmf =  list(OrderedDict.fromkeys(  [ i[0][0] for i in ratingsPerPartition  ]  ))
      indexInf =  list(OrderedDict.fromkeys(  [ i[0][1]+1 for i in ratingsPerPartition  ]  ))
      rowIndex, colIndex, ratingValues = buildSparseMatrixIndex(ratingsPerPartition)
      A = csr_matrix(( [1.0]*len(rowIndex) , ( rowIndex , colIndex ) ) )
      Y = csr_matrix((ratingValues, ( rowIndex , colIndex ) ) )
      mydata = Y.toarray()
 

      neg_visible_activations = np.dot(mypos_hidden_state[indexUmf], myweights[indexInf].T) # n_u x m+1 
      neg_visible_probs = _mylogistic(neg_visible_activations) # 1 x m+1                                                                                                                                  
 
      neg_visible_probs = A.multiply(neg_visible_probs)

      #print 'test zhou !!!! ', mydata, neg_visible_probs.toarray() 
      error = np.sum(( mydata - neg_visible_probs.toarray() ) ** 2) 
      return [ ( (0,) , error  ) ]

    # this is where train function starts
    num_examples = ratings.map(lambda r: r[0][0] ).max() + 1 # num of users                                                                                                                                
    num_visible = ratings.map(lambda r: r[0][1] ).max() + 1 # num of items
    num_hidden= num_hidden
    logString = 'Debug: Training on number of users: %s and items: %s' % (num_examples , num_visible)
    self.logThis(logString, 'a')
    
    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using                                                                                                                          
    # a Gaussian distribution with mean 0 and standard deviation 0.1.                                                                                                                                      
    weights = 0.1 * np.random.randn(num_visible, num_hidden)
    # Insert weights for the bias units into the first row and first column.                                                                                                                               
    weights = np.insert(weights, 0, 0, axis = 0)
    weights = np.insert(weights, 0, 0, axis = 1)
    #weights = np.array(
    #  [[ 0.,          0. ,         0.        ],
    #   [ 0. ,        -0.02258248,  0.1466121 ],
    ##   [ 0.  ,        0.03929332 , 0.11875235],
    #   [ 0.   ,       0.07519332, -0.12009325],
    #   [ 0.    ,     -0.05472227,  0.18818327],
    #   [ 0.     ,    -0.14657284,  0.11834431],
    #   [ 0.      ,    0.16617544, -0.07278836],])

    learningRateBr = self.sc.broadcast(learning_rate)
    cost_function = None


    # Insert bias units of 1 into the first column.
    #data = np.insert(data, 0, 1, axis = 1)

    
    for epoch in range(max_epochs):     
      logString =  "Debug: Start training epoc #%s" % epoch
      self.logThis(logString, 'a')
    
      weightsBr = self.sc.broadcast(weights)
      #logString =  "Debug: weights are %s" % weightsBr.value
      #self.logThis(logString, 'a')

      pos_hidden_activations = np.zeros([num_examples, num_hidden+1])
      PosHidden = ratings.mapPartitions(partial(rbmPosHidden,weightsBr=weightsBr )).reduceByKey(add).collect()
      for PosHidden_ptr in PosHidden:
        pos_hidden_activations[ PosHidden_ptr[0][0] ] = PosHidden_ptr[1]
        
      for row_ptr in xrange(pos_hidden_activations.shape[0]):
        pos_hidden_activations[row_ptr] += weightsBr.value[0,:]


      #logString =  "Debug: pos_hidden_activations are %s" % pos_hidden_activations
      #self.logThis(logString, 'a')

      # Clamp to the data and sample from the hidden units. 
      # (This is the "positive CD phase", aka the reality phase.)
      ##pos_hidden_activations = np.dot(data, self.weights)  # n_u x ( n_h+1 )     
      pos_hidden_probs = _mylogistic(pos_hidden_activations) #  n_u x ( n_h+1 )

      #logString =  "Debug: pos_hidden_probs are %s" % pos_hidden_probs
      #self.logThis(logString, 'a')
     
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, num_hidden+1) # n_u x ( n_h+1 )     
      # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
      # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
      # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
     
      pos_hidden_probsBr = self.sc.broadcast(pos_hidden_probs)
      pos_hidden_statesBr = self.sc.broadcast(pos_hidden_states)

      pos_associations = np.zeros([ num_visible+1 , num_hidden+1])
      ##pos_associations = np.dot(data.T, pos_hidden_probs) # m+1 X n_h+1
      PosAssociation = ratings.mapPartitions(partial(rbmPosAssociation, pos_hidden_probsBr=pos_hidden_probsBr  )).reduceByKey(add).collect()
      for PosAssociation_ptr in PosAssociation:
        pos_associations[ PosAssociation_ptr[0][0]+1 ] = PosAssociation_ptr[1]
      pos_associations[0] = np.ones(num_examples).dot(pos_hidden_probsBr.value)

      #logString =  "Debug: pos_associations are %s" % pos_associations
      #self.logThis(logString, 'a')

      #print 'test zhou', pos_associations.shape, pos_associations

      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      
      # we have to break neg_visible_activations into small chunks as it does not fit into memeory
      neg_associations = np.zeros([ num_visible+1 , num_hidden+1])
      neg_hidden_probs_assembly = np.zeros([ num_examples , num_hidden+1])

      for ptr_user in xrange(num_examples):
        neg_visible_activations = np.dot(pos_hidden_states[ptr_user], weightsBr.value.T) # 1 x m+1 
        neg_visible_probs = _mylogistic(neg_visible_activations) # 1 x m+1 
        #print 'test zhou ** ', neg_visible_probs.shape
        neg_visible_probs[0] = 1 # Fix the bias unit. # 1 x m+1 
        neg_hidden_activations = np.dot(neg_visible_probs, weightsBr.value) # 1 x n_h+1
        neg_hidden_probs = _mylogistic(neg_hidden_activations) # 1 x n_h+1
        neg_hidden_probs_assembly[ptr_user] = neg_hidden_probs

      #print 'test zhou ** ', neg_hidden_probs_assembly.shape 
      #logString =  "Debug: neg_hidden_probs are %s" % neg_hidden_probs_assembly
      #self.logThis(logString, 'a')


      for ptr_music in xrange(num_visible+1):
        neg_visible_activations = np.dot(pos_hidden_states, weightsBr.value[ptr_music].T).T # 1 x n_u
        neg_visible_probs = _mylogistic(neg_visible_activations) # 1 x n_u
        #neg_visible_probs[0,0] = 1 # Fix the bias unit. # 
        if ptr_music == 0:
          neg_associations[ptr_music] = np.ones([1,num_examples]).dot(neg_hidden_probs_assembly)  #  1 x n_u  n_u x n_h+1
        else:
          neg_associations[ptr_music] = neg_visible_probs.dot(neg_hidden_probs_assembly) #  1 x n_u   n_u x n_h+1

      #logString =  "Debug: neg_associations are %s" % neg_associations
      #self.logThis(logString, 'a')

      #neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs) #
      #neg_visible_activations = np.dot(pos_hidden_states, self.weights.T) # n_u x m+1
      #neg_visible_probs = self._logistic(neg_visible_activations) # n_u x m+1
      #neg_visible_probs[:,0] = 1 # Fix the bias unit. # n_u x m+1
      #neg_hidden_activations = np.dot(neg_visible_probs, self.weights) # n_u x n_h+1
      #neg_hidden_probs = self._logistic(neg_hidden_activations) # n_u x n_h+1
      # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
      # themselves.
      #neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs) # m+1 X n_h+1

      # Update weights.
      #print 'test zhou: ', (pos_associations - neg_associations).shape, (pos_associations - neg_associations)
      
      #error = ratings.mapPartitions(partial(rbmCostFunction,weightsBr=weightsBr )).reduceByKey(add).collect()

      # this part only works for small data set
      error = ratings.mapPartitions(partial(rbmCostFunction,weightsBr=weightsBr, pos_hidden_statesBr = pos_hidden_statesBr )).reduceByKey(add).collect()
      error = error[0][1]
      #neg_visible_activations = np.dot(pos_hidden_states, weightsBr.value.T) # n_u x m+1                                                                                                          
      #neg_visible_probs = _mylogistic(neg_visible_activations) # 1 x m+1                
      #ratingsPerPartition = list(ratings.collect())
      #indexUmf =  list(OrderedDict.fromkeys(  [ i[0][0] for i in ratingsPerPartition  ]  ))
      #indexInf =  list(OrderedDict.fromkeys(  [ i[0][1] for i in ratingsPerPartition  ]  ))
      #rowIndex, colIndex, ratingValues = buildSparseMatrixIndex(ratingsPerPartition)
      #Y = csr_matrix((ratingValues, ( rowIndex , colIndex ) ) )

      
      #error = np.sum((data - neg_visible_probs) ** 2)
      #print("Epoch %s: error is %s" % (epoch, error))

      logString =  "Debug: epoch %s the total cost function is %s" % (epoch, error)
      self.logThis(logString, 'a')
      #########################

      weightsBr.unpersist()
      pos_hidden_probsBr.unpersist()
      pos_hidden_statesBr.unpersist()

      weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

    self.weights = weights
      
      
  def saveModel(self, modelConfig= ''):
    np.save(self.fileName + modelConfig  + '.npy', self.weights)
  

  def run_visible(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.
    
    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.
    
    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """
    
    num_examples = data.shape[0]
    
    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1
  
    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  # TODO: Remove the code duplication between this method and `run_visible`?
  def run_hidden(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of hidden units, to get a sample of the visible units.
    Parameters
    ----------
    data: A matrix where each row consists of the states of the hidden units.
    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states
    
  def daydream(self, num_samples):
    """
    Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
    (where each step consists of updating all the hidden units, and then updating all of the visible units),
    taking a sample of the visible units at each step.
    Note that we only initialize the network *once*, so these samples are correlated.
    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """

    # Create a matrix, where each row is to be a sample of of the visible units 
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0,1:] = np.random.rand(self.num_visible)

    # Start the alternating Gibbs sampling.
    # Note that we keep the hidden units binary states, but leave the
    # visible units as real probabilities. See section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines"
    # for more on why.
    for i in range(1, num_samples):
      visible = samples[i-1,:]

      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)      
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1

      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states

    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:,1:]        
      
  #def _logistic(self, x):
  #  return 1.0 / (1 + np.exp(-x))

#if __name__ == '__main__':
#  r = RBM(num_visible = 6, num_hidden = 2)
#  training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
#  r.train(training_data, max_epochs = 50)
#  print(r.weights)
#  user = np.array([[0,0,0,1,1,0]])
#  print(r.run_visible(user))
