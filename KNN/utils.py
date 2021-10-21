import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    tp=0
    fp=0
    fn = 0
    real_labels= np.array(real_labels,dtype=float)
    predicted_labels=np.array(predicted_labels,dtype=float)
    assert len(real_labels) == len(predicted_labels)
    fp = np.sum((real_labels==0) & (predicted_labels==1))
    fn = np.sum((real_labels ==1) & (predicted_labels==0))
    tp = np.sum((real_labels==predicted_labels) & (real_labels==1))
    return tp/(tp+0.5*(fp+fn))

class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        abs_cube = 0
        point1 = np.array(point1,dtype=float)
        point2 = np.array(point2,dtype=float)
        abs_cube = np.sum(np.power(np.absolute(point1-point2),3))
        return (np.power(abs_cube,1/3))

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        
        point1=np.array(point1,dtype=float)
        point2 = np.array(point2,dtype=float)
        dist_squared = np.sum(np.power(point1-point2,2))
        return (np.power(dist_squared,1/2))

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        
        numerator = 0
        for x1 in range(0,len(point1)):
            numerator+=point1[x1]*point2[x1]
        x1_norm = 0
        x2_norm = 0
        for x1 in range(0,len(point1)):
            x1_norm+= np.power(point1[x1],2)
            x2_norm+= np.power(point2[x1],2)
        if x1_norm == 0 or x2_norm == 0:
            return 1
        else:
            cosine_similarity = numerator/(np.power(x1_norm,0.5)*np.power(x2_norm,0.5))
            return 1-cosine_similarity

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        self.max_f1_score = np.NINF
        
        
        #for every value of k
        for k in range(1,30,2):
            for func_name,func in distance_funcs.items():
                knn_object = KNN(k, func)
                #training
                knn_object.train(x_train,y_train)
                #predictions are made by by searching through the entire training set for the K most similar instances (the neighbors) and summarizing the output variable for those K instances
                #prediction on validation dataset
                prediction = knn_object.predict(x_val)
                
                model_score = f1_score(y_val,prediction)
                
                if self.max_f1_score<model_score:
                    self.max_f1_score = model_score
                    self.best_k = k
                    self.best_distance_function = func_name
                    self.best_model = knn_object
                    self.best_scaler = None


    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        
        self.max_f1_score = np.NINF
        for scaler_name,scaler in scaling_classes.items():
            s = scaler()
            scaler_x_train = s(x_train)
            scaler_x_val = s(x_val)
        #for every value of k
            for k in range(1,30,2):
                for func_name,func in distance_funcs.items():
                    knn_object = KNN(k, func)
                    #training
                    knn_object.train(scaler_x_train,y_train)
                    #predictions are made by by searching through the entire training set for the K most similar instances (the neighbors) and summarizing the output variable for those K instances
                    #prediction on validation dataset
                    normalized_prediction = knn_object.predict(scaler_x_val)
                    model_score = f1_score(y_val,normalized_prediction)
                    if self.max_f1_score<model_score:
                        self.max_f1_score = model_score
                        self.best_k = k
                        self.best_distance_function = func_name
                        self.best_model = knn_object
                        self.best_scaler = scaler_name
        


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalized_features = list()
        features = np.array(features,dtype=float)
        for x in features:
            # print(x)
            
            norm_x = list()
            sq_l2_norm = np.linalg.norm(x)
            for x1 in x:
                if x1==0:
                    norm_x.append(0)
                else:
                    norm_x.append(x1/sq_l2_norm)
            normalized_features.append(norm_x)
        return normalized_features


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        
        features = np.array(features,dtype=float)
        min_l = features.min(axis=0)
        max_l = features.max(axis=0)
        cols = features.shape[1]
        for i in range(cols):
            den = max_l[i]-min_l[i]
            features[:,i] = (features[:,i]- min_l[i])/den if den>0 else 0
        
        return features.tolist()
