import numpy as np 

'''def information_gain(y, y_left, y_right, gain_type):
    if gain_type == 'd':'''

def cl_entropy(y):
    proba = np.unique(y, return_counts = True)[1] / len(y)
    proba = proba[proba > 0]
    return -np.sum(proba * np.log2(proba))

def calc_entr_criterion(y, y_l,y_r ):
    main_en = cl_entropy(y)
    y_l_en = cl_entropy(y_l) * len(y_l) / len(y)
    y_r_en = cl_entropy(y_r)* len(y_r) / len(y)
    return main_en - y_l_en - y_r_en

def mse(y):
    return np.mean((y - np.mean(y)**2))


class ID3:
    def __init__(self, gain_type = None,max_depth = None, min_samples_leaf = None, ID3_type = 'classification' ):

        self.gain_type = gain_type
        self.ID3_type= ID3_type
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.depth_left = 0
        self.depth_right = 0

    def criterion_gain(self, y, left_ = None, reight_ = None):
        if self.gain_type == 'entropy':
            main_en = cl_entropy(y)
            y_l_en = cl_entropy(y[left_]) * len(y[left_]) / len(y)
            y_r_en = cl_entropy(y[reight_]) * len(y[reight_]) / len(y)
            return main_en - y_l_en - y_r_en
        if self.gain_type == 'don':
            count = 0
            for i in range(len(y)): #слишком долго
                for j in range(i + 1, len(y)):
                    if left_[i] != left_[j] and y[i] != y[j]:
                        count += 1 
            return count
        

    def fit(self, X_train, y_train):
        self.num_classes = np.unique(y_train)
        self.dec_tree = self._tree_creation(X_train, y_train)

    def _tree_creation(self, X_train, y_train):
        if self.ID3_type == 'classification':
            if len(np.unique(y_train)) == 1 or self.depth_left >= self.max_depth or self.depth_right >= self.max_depth:
                return {'Тип развилки': 'leaf', 'values': np.bincount(y_train), 'value': np.bincount(y_train).argmax() }
        else:
            if len(np.unique(y_train)) == 1 or self.depth_left >= self.max_depth or self.depth_right >= self.max_depth:
                return {'Тип развилки': 'leaf',  'value': np.mean(y_train) }
        
        criterion_split, tr, feature = self.splitter(X_train, y_train)
        X = X_train[:, feature]
        not_nones = ~np.isnan(X) #обработка пропусков, тут не None
        left_ = (X <= tr) & not_nones
        reight_ = (X > tr) & not_nones
        self.depth_left += 1
        self.depth_right += 1
        to_left_fr_node = self._tree_creation(X_train[left_], y_train[left_] )
        to_right_fr_node = self._tree_creation(X_train[reight_], y_train[reight_])
        return {
            "Тип развилки": "node",
            "feature": feature,
            "threshold": tr,
            "left": to_left_fr_node,
            "right": to_right_fr_node,
            'q_v': np.sum(left_) / np.sum(not_nones),
            'criterion_split': criterion_split
        }

    def splitter(self, X_train, y_train):
        criterion = 0
        split_tr = None
        for feature in range(X_train.shape[1]):
            X = X_train[:, feature]
            not_nones = ~np.isnan(X)
            tresholds = np.unique(X[not_nones])   #обработка пропусков, ненавижу y_train
            for tr in tresholds:
                left_ = (X <= tr) & not_nones  #обработка пропусков
                reight_ = (X > tr) &  not_nones  #обработка пропусков
                if len(y_train[left_]) ==  0 or len(y_train[reight_]) == 0:
                    continue
                if self.ID3_type == 'classification':
                    criterion_split = self.criterion_gain(y_train, left_, reight_)
                    if criterion < criterion_split:
                        criterion = criterion_split
                        split_tr = tr
                        split_ind = feature
                else:
                    criterion_split = (len(y_train[left_]) * mse(y_train[left_]) / len(y_train)) + \
                                      (len(y_train[reight_]) * mse(y_train[reight_]) / len(y_train))
                    if criterion_split < criterion:
                        criterion = criterion_split
                        split_tr = tr
                        split_ind = feature

                

        return criterion_split, split_tr, split_ind
    

    def predict(self, X, tree_ = None):
        y = np.zeros(X.shape[0])  
        if tree_ == None:
            tree_n = self.dec_tree
        else:
            tree_n = tree_
        for i, x in enumerate(X):  
            tree = tree_n
            while tree["Тип развилки"] != "leaf":
                feature = tree["feature"]
                threshold = tree["threshold"]
                q_v = tree['q_v']
                if np.isnan(x[feature]):
                    left_value = self.calc_values_nan(tree['left'], x)
                    right_value =  self.calc_values_nan(tree['right'], x)
                    y[i] = q_v * left_value + (1 - q_v) * right_value
                    break 
                if x[feature] <= threshold:
                    tree = tree["left"]  
                else:
                    tree = tree["right"] 
            y[i] = tree["value"]  
        return y

    def calc_values_nan(self, l_tree, x):
        while l_tree["Тип развилки"] != "leaf":
            feature = l_tree["feature"]
            threshold = l_tree["threshold"]
            if np.isnan(x[feature]) or x[feature] <= threshold:
                l_tree = l_tree["left"]
            else:
                l_tree = l_tree["right"]
        return l_tree["value"]
    
    def pruning(self, X, y, tree=False):
        if tree is False:
            tree = self.dec_tree
        if tree['Тип развилки'] == 'leaf':
            return tree
        data = X[:, tree['feature']]
        left_mask = (data <= tree['threshold']) | np.isnan(data)
        right_mask = ~left_mask
        tree['left'] = self.pruning(X[left_mask], y[left_mask], tree['left'])
        tree['right'] = self.pruning(X[right_mask], y[right_mask], tree['right'])
        if len(y) == 0:
            return {'Тип развилки': 'leaf', 'value': 0}
           
        if self.ID3_type == 'classification':
            classes, counts = np.unique(y, return_counts=True)
            mc = classes[counts.argmax()]
            r_v = np.sum(self.predict(X, tree_=tree) != y)
            rL_v = np.sum(self.predict(X, tree_=tree['left']) != y)
            rR_v = np.sum(self.predict(X, tree_=tree['right']) != y)
            rc_v = np.sum(y != mc)
        else:
            mc = np.mean(y)
            r_v = np.sum((self.predict(X, tree_=tree) - y)**2)
            rL_v = np.sum((self.predict(X, tree_=tree['left']) - y)**2)
            rR_v = np.sum((self.predict(X, tree_=tree['right']) - y)**2)
            rc_v = np.sum((mc - y)**2)

        errors = [rc_v, r_v, rL_v, rR_v]
        minimal = np.argmin(errors)

        if minimal == 0:
            return {'Тип развилки': 'leaf', 'value': mc}
        elif minimal == 1:
            return tree
        elif minimal == 2:
            return tree['left']
        elif minimal == 3:
            return tree['right']


