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
   
    #TODO: insert bias as w0
    # w = np.insert(w, 0, b, axis=1)
    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
		# derivative of the perceptron loss at 0)      # 
        ################################################
        for iter in range(max_iterations):
            dot_prod = np.dot(X,w)+b
            fx = np.where(dot_prod <0,0.0,1.0)
            
            
            error = y-fx
            loss_func = (1.0/N)*max(0,-np.matmul(dot_prod,error)) 
            # print(error.shape)
            # print(X.shape)
            derivative_w = (1.0/N)*(np.matmul(error.T,X)) 
            dw = np.where(loss_func==0,1,derivative_w)
            # print(derivative_w)
            derivative_b = error
            db = np.mean(np.where(loss_func==0,1,derivative_b))
            # print(db.item())
        
            w = w+(dw)
            b = b+(db.item())
 
            # print(type(b))
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
    print(f"b {b}")

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
            # P = np.zeros((C,1))
            # k_sum=0
            # w_not_yn = np.delete(w.copy(), y[n], axis=0)
            # b_not_yn = np.delete(b.copy(), y[n], axis=0)
            # print(w.shape)
            # print(w_not_yn.shape)
            # k_sum = np.sum(np.exp(np.dot(w_not_yn,X[n].T)+b_not_yn),axis=0)
            
            # w_yn = w[int(y[n])]
            # b_yn = b[int(y[n])]
            dot  = np.dot(w,X[n].T)+b
            # zyn=np.dot(w_yn.T,X[n])
            # zyn_dash = zyn - np.max(zyn)
            # for k in range(C):
                
            #     zk = np.dot(w[k].T,X[n])+b[k]
            #     zk_dash = zk - np.max(zk)
            #     k_sum += np.exp(zk_dash)
            k_sum = np.sum(np.exp(dot-np.max(dot)),axis=0)
            for k in range(C):
                zk = np.dot(w[k],X[n].T)+b[k]
                sf = softmax(zk,k_sum)
                # bk = b[k] - b_yn
                # bk_dash = bk - np.max(bk)
                # P_k_b = np.exp(bk_dash)/(1+k_sum)
                # P_k_x = np.exp(zk_dash-zyn_dash)/(1+k_sum)
                if k!=y[n]:
                    w[k] -=  step_size*sf*X[n]
                    b[k] -= step_size*sf
                    # w[k] = w[k] -(step_size* P_k_x*X[n]*1.0/N)
                    # b[k] = b[k] - (step_size*P_k_b*1.0/N)
                else:
                    w[k] -= step_size*(sf-1)*X[n]
                    b[k] -= step_size*(sf-1)
                    # w[k] = w[k] - (step_size*(P_k_x-1)*X[n]*1.0/N)
                    # b[k] = b[k] - (step_size*(P_k_b-1)*1.0/N)

            
            

        
        
        print(b.shape)
        print(C,D)
    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        for it in range(max_iterations):
            probabilities = np.empty((N,C))
            num = np.dot(X,w.T)+b
            num_dash = np.exp(num-np.max(num))
            denom = np.sum(np.exp(num_dash))
            probabilities = num_dash/denom
            probabilities[np.arange(N),y.flatten().astype(int)] -=1
            w -= 1.0/N*step_size*np.dot(probabilities.T,X)
            b-= 1.0/N*step_size*np.sum(probabilities,axis=0)

    else:
        
        
        raise "Undefined algorithm."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b

def softmax(z,sum):
    num = np.exp(z-np.max(z))
    
    return num/sum
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
    preds = np.argmax(np.dot(X,w.T)+b,axis=1)

    
    assert preds.shape == (N,)
    return preds




        