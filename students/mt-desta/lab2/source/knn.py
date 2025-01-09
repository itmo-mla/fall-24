import numpy as np
from scipy.spatial.distance import cdist

class my_KNN:
    def __init__(self,neighbours=1,mode='simple',h = 2):
        self.neighbours = neighbours #K
        self.h = 2 # bandwidth
        self.mode = mode #mode

    def gaussian_kernel(self,pd, h):
        return np.exp(-0.5 * ( pd / h ) ** 2)
    
    def parzen_window(self,dist,closest_neighbours):
            if self.mode == "simple":
                return np.ones((closest_neighbours, )) 
            elif self.mode == "parzen_variable":
                h = np.sort(dist,axis=0)[self.neighbours]
                return 1 / np.sqrt(2 * np.pi) * self.gaussian_kernel(closest_neighbours,h) 
            elif self.mode == "parzen_fixed":
                return 1 / np.sqrt(2 * np.pi) * self.gaussian_kernel(closest_neighbours,self.h) 
    
    def fit(self, X , y):
        self.X_train = X
        self.y_train = y
    
    def predict_single(self, x):
            dist = cdist(self.X_train,x.reshape((1,-1))) # pairwise distance

            closest_neighbour_ids = np.argsort(dist,axis=0)[:self.neighbours] # get the ids of the first K nearest neighbours
            
            closest_neighbours = dist[closest_neighbour_ids].flatten() # get the distances from those ids
            labels = self.y_train[closest_neighbour_ids].flatten() # get associated labels for those ids

            weights = np.zeros((self.neighbours, len(np.unique(self.y_train)))) # init weights

            closest_neighbours = self.parzen_window(dist,closest_neighbours)

            weights[ np.arange(self.neighbours),labels] = closest_neighbours # update weights
        
            return np.argmax(np.sum(weights,axis=0)).astype(np.int32) 

    def predict(self, X_test):
        pred = np.empty(X_test.shape[0])

        for i in range(len(X_test)):        
            pred[i] = self.predict_single(X_test[i])

        return pred       
