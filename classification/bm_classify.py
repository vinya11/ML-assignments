import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    

    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
    
    # print(y)
    #TODO: insert bias as w0
    # w = np.insert(w, 0, b, axis=1)
    if loss == "perceptron":
        y_re = y.copy()
        y_re = np.where(y_re==0,-1,y_re)
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
		# derivative of the perceptron loss at 0)      # 
        ################################################
        for iter in range(max_iterations):
            dot_prod = np.dot(X,w)+b
            fx = np.where(dot_prod<0,0.0,1.0)
            error = fx-y
            error=np.where(dot_prod==0,-1,error)
            dw = (1.0/N)*(np.matmul(error.T,X)) 
            db = np.mean(error,axis=0)
        
            w = w-step_size*(dw)
            b = b-(step_size*db)
        
    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################
        for iter in range(max_iterations):
        
            y_hat = sigmoid(np.matmul(X,w)+b)
            # loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
            
            grad_w = (1.0/N)*(np.dot(X.T,(y_hat-y)) ) 
            
            grad_b = np.mean(y_hat-y)

            # print(grad_b.shape)
        
            w = w-step_size*grad_w
            b= b-step_size*grad_b



        

    else:
        raise "Undefined loss function."
    # print(f"b {b}")

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################
    value = 1.0/(1+np.exp(-z))
    
    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape
        
    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    preds = np.where(np.add(np.matmul(X,w),b) >0,1.0,0.0)
  

    assert preds.shape == (N,) 
    return preds


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
	
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape
    
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0
    one_hot_y = np.zeros((N,C))
    one_hot_y[np.arange(y.size),y] =1
    one_hot_y = one_hot_y.astype(int)
    print(one_hot_y.dtype)
  
    
    
    
    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################
            inp = np.dot(X[n],w.T)+b
            num = np.exp(inp-np.max(inp,axis=0))
            denom = np.sum(num)
            probability = num/denom
            probability[y[n]] -=1
            W_deriv = np.multiply(probability.reshape(-1,1),X[n].reshape(-1,1).T)
            w -= step_size*W_deriv
            b -= step_size*(probability) 
                    
    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        for it in range(max_iterations):
            inp = np.dot(X,w.T)+b
            probabilities = softmax(inp)
            derivation =probabilities-one_hot_y
            w -= (step_size/N)*np.dot(X.T,derivation).T
            b-= (step_size/N)*np.sum(derivation,axis=0)
            

    else:
        
        
        raise "Undefined algorithm."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b

def softmax(z):

    num = np.exp(z - np.max(z, axis=1)[:,None])
    denom =np.sum(num,axis=1,keepdims=True)
    return num/denom
def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    preds = np.argmax(np.dot(X,w.T)+b.T,axis=1)
    

    
    assert preds.shape == (N,)
    return preds




        