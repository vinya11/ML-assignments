import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    N= X.shape[0]
    
    err = np.power(np.linalg.norm(np.matmul(X,w)-y),2)/N
    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here                    #
  #####################################################	
  
  w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),y)
  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #
  #####################################################	
    # print(X.shape)
    
    D = X.shape[1]	
    Z = np.matmul(X.T,X).astype(np.float64)
    # print(lambd)
    p1 = lambd*np.identity(D)
    p2 =np.add(Z,p1)
    p3 = np.linalg.inv(p2)
    p4 = np.matmul(X.T,y)
    w = np.matmul(p3,p4)
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################
   
    bestlambda=np.inf
    
    min_err = np.inf
    
    for i in range(0,15):
      lambd = 1/np.power(2,i)
      
      w = regularized_linear_regression(Xtrain,ytrain,lambd)
      err = mean_square_error(w, Xval, yval)
      if err<min_err:
        min_err = err
        bestlambda = lambd

    return bestlambda
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################
    Y = np.copy(X)		
    w,h = X.shape
    for i in range(2,p+1):
      X =np.insert(X,X.shape[1],np.power(Y,i).T,axis=1)
    return X

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

