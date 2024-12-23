import numpy as np 
class KNN:
    def __init__(self, k:int, x_train,y_train ):
        self.k = k
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)


    def predict(self, x_test):
        x_test = np.array(x_test)
        if len(x_test.shape) == 1:
            x_test = x_test.reshape(1, -1)
        distances = np.sqrt(np.sum(x_test**2, axis=1).reshape(-1,1) + \
                                np.sum(self.x_train**2, axis=1).reshape(1,-1)- \
                                    2*np.dot(x_test,self.x_train.T))
       
        sorted_distances_idx = np.argsort(distances, axis=1)[:,:self.k+1]
        min_dist_points = distances[np.arange(distances.shape[0])[:,None], sorted_distances_idx[:,:self.k]]
        k_max_neib = distances[np.arange(distances.shape[0])[:,None], sorted_distances_idx[:, -1].reshape(distances.shape[0], 1)]
        kernel_weights = np.exp(-((min_dist_points/k_max_neib)**2)/2)
        labels = self.y_train[sorted_distances_idx[:,:self.k]]
        classes = np.unique(self.y_train)
        weighted_votes = np.zeros((x_test.shape[0], classes.size))
        for idx, cls in enumerate(classes):
            cls_mask = (labels == cls)
            weighted_votes[:, idx] = np.sum(kernel_weights * cls_mask, axis=1)
        
        return classes[np.argmax(weighted_votes, axis=1)]
   
def LOO(X, y, k_neib):
    metrics = np.zeros(len(k_neib))
    for index, k in enumerate(k_neib):
        for i in range(len(y)):
            x_test = X.iloc[i]
            y_test = y.iloc[i]
            X_train = X[X.index != i]
            y_train = y[y.index != i]
            KNNclass = KNN(k,X_train, y_train)
            y_pred = KNNclass.predict(x_test)[0]
            if y_pred == y_test:
                metrics[index] += 1
    metrics = 1 - metrics/len(y)
    return k_neib[np.argmin(metrics)], metrics
            





