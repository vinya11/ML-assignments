import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    centers = list()
    center_values = list()
    p = generator.randint(0, n) #this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
	# according to some distribution, first generate a random number r between 0 and
	# 1 using generator.rand(), then find the the smallest index n so that the 
	# cumulative probability from example 1 to example n is larger than r.
    #############################################################################
    centers.append(p)
    center_values.append(x[p])
    # datapoints = x.copy()
    
    # datapoints = np.delete(datapoints,p,0)
    for k in range(1,n_cluster):
        r = generator.rand()
        # print(f"r {r}")
        sub = x-np.array(center_values)[:, np.newaxis]
        l2_norm = np.linalg.norm(sub,axis=2)**2
        # if k==1:
        #     sum_all = np.sum(l2_norm)
        # else:
        #     sum_all = np.sum(l2_norm,axis=0)
        min_distances = np.min(l2_norm,axis=0)
        min_distances = min_distances/np.sum(min_distances)
        # print(min_distances.shape)
        cummulative = 0
        for d in range(len(min_distances)):
            cummulative+=min_distances[d]
            if cummulative>=r:
                centers.append(d)
                center_values.append(x[d])
                # datapoints = np.delete(datapoints,d,0)
                break
 

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape
        gamma = np.zeros((N,))
        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        # print(self.centers)
        centers = x[self.centers,:]
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        prev_J =0
        J=0
        for iter in range(self.max_iter):
            prev_J=J
            J=0
            g_k = dict()
            for n in range(N):
                #cluster indexes of n data samples
                gamma[n] = np.argmin(np.power(np.linalg.norm(x[n]-centers,axis=1),2))
            for k in range(self.n_cluster):
                g_k[k] = np.argwhere(gamma==k).reshape(-1)
                centers[k]= np.sum(x[g_k[k]],axis=0)/len(g_k[k])
            for k in range(self.n_cluster):
                dist = x[g_k[k]]-centers[k]
                J += np.linalg.norm(dist)**2
            if 1.0/N*abs(J - prev_J) < self.e:
                break
            
        return centers,gamma,iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        centroid_labels = np.zeros(self.n_cluster)
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        # print(f"y{y}")
        means = KMeans( self.n_cluster, self.max_iter, self.e, self.generator)
        centroids,gamma,i = means.fit(x,centroid_func)
        for k in range(self.n_cluster):
            cluster_assignments= np.argwhere(gamma==k).reshape(-1)
            unique, counts = np.unique(y[cluster_assignments], return_counts=True)
            count_assignments = dict(zip(unique, counts))
            centroid_labels[k] = max(count_assignments,key=count_assignments.get)
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        predicted_labels = np.zeros(N,)
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        # print(self.centroid_labels)
        for n in range(N):
            idx = np.argmin(np.linalg.norm(x[n]-self.centroids,axis=1)**2) 
            predicted_labels[n] = self.centroid_labels[idx]
        return predicted_labels



def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    im_shape = image.shape
    image = image.reshape(image.shape[0]*image.shape[1],image.shape[2])
    for pixel_idx in range(image.shape[0]):
        # print(pixel)
        sub = np.linalg.norm(image[pixel_idx]-code_vectors,axis=0)**2
        image[pixel_idx] = image[np.argmin(sub)]

    # sub = image-code_vectors[:, np.newaxis]
    # l2_norm = np.linalg.norm(sub,axis=2)**2
    # nearest_centroid_idx = np.argmin(l2_norm,axis=0)
    # print(nearest_centroid_idx.shape)
    # image = image[nearest_centroid_idx]
    # print(image)
    image = image.reshape(im_shape)
    return image 
    
