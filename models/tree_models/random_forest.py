import numpy as np
from collections import Counter

class Node:
    def __init__(self,feature=None ,threshold=None ,left=None, right=None,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value 
        
    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    # This will be the base decision tree for use in the Random Forest
    def __init__(self, max_depth=None, min_samples_split=2, criterion='entropy', 
                 max_features=None, task='classification'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.task = task
        self.root = None
    
    def fit(self,X,y,feature_indices=None):
        self.n_classes = len(np.unique(y)) if self.task == 'classification' else None
        self.feature_indices = feature_indices
        self.root = self._grow_tree(X,y,depth=0)
        return self
    
    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # stopping criteria
        if (depth >= self.max_depth if self.max_depth else False) or \
            n_labels == 1 or \
            n_samples < self.min_samples_split:
                return self._create_leaf(y)
            
        # find the best split 
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return self._create_leaf(y)
        
        # split the dataset
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        # grow children recursively
        left = self._grow_tree(X[left_idxs],y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold,
                    left=left,right=right)
        
    def _create_leaf(self,y):
        if self.task == 'classification':
            # most common class
            value = Counter(y).most_common(1)[0][0]
        else:
            # mean of target values 
            value = np.mean(y)
        return Node(value=value)
    
    def _best_split(self,X,y):
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None, None
        
        # Determine which features to consider
        if self.feature_indices is not None:
            features_to_try = self.feature_indices
        elif self.max_features:
            n_features_to_try = min(self.max_features, n_features)
            features_to_try = np.random.choice(n_features, n_features_to_try,replace=False)
        else:
            features_to_try = range(n_features)
            
        
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        # Calculate parent impurity/error
        if self.task == 'classification':
            parent_impurity = self._calculate_impurity(y)
        else:
            parent_impurity = self._calculate_mse(y)
            
        # Try each feature
        for feature in features_to_try:
            thresholds = np.unique(X[:, feature])
            
            # Try each threshold
            for threshold in thresholds:
                # Split
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                y_left, y_right = y[left_mask], y[right_mask]
                n_left, n_right = len(y_left), len(y_right)
                
                if self.task == 'classification':
                    left_impurity = self._calculate_impurity(y_left)
                    right_impurity = self._calculate_impurity(y_right)
                else:
                    left_impurity = self._calculate_mse(y_left)
                    right_impurity = self._calculate_mse(y_right)
                    
                # Weighted average of the child impurities
                child_impurity = (n_left/n_samples) * left_impurity + \
                    (n_right/n_samples) * right_impurity
                    
                # Information gain
                gain = parent_impurity - child_impurity
                
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _calculate_impurity(self, y):
        if len(y) == 0:
            return 0
        
        proportions = np.bincount(y.astype(int))/len(y)
        
        if self.criterion == 'gini':
            return 1 - np.sum(proportions ** 2)
        elif self.criterion == 'entropy':
            proportions = proportions[proportions > 0] # avoid log(0)
            return -np.sum(proportions * np.log2(proportions))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
        
    def _calculate_mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y-np.mean(y))**2)
    
    def predict(self, X):
        # Predict for multiple samples
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 max_features='sqrt', criterion='entropy', task='classification', 
                 bootstrap=True, oob_score=False, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion if task == 'classification' else 'mse'
        self.task = task
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        
        self.trees = []
        self.oob_score_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Determine the max_features
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * n_features)
        else:
            max_features = n_features
        
        max_features = max(1, min(max_features, n_features))
        
        # store for OOB scoring
        if self.oob_score:
            oob_predicitions = [[] for _ in range(n_samples)]
            
        # Build trees
        self.trees = []
        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(indices))
            else:
                indices = np.arange(n_samples)
                oob_indices = []
            
            X_sample = X[indices]
            y_sample = y[indices]
            
            # random feature selection for this tree
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            
            # Build tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                task=self.task
            )
            
            tree.fit(X_sample, y_sample,feature_indices=feature_indices)
            self.trees.append(tree)
            
            # OOB predictions 
            if self.oob_score and len(oob_indices)> 0:
                oob_preds = tree.predict(X[oob_indices])
                for idx, pred in zip(oob_indices, oob_preds):
                    oob_predicitions[idx].append(pred)
                    
        if self.oob_score:
            self._calculate_oob_score(y, oob_predicitions)
        
        return self

    def _calculate_oob_score(self, y, oob_predictions):
        oob_preds = []
        valid_indices = []
        
        for i, preds in enumerate(oob_predictions):
            if len(preds) > 0:
                if self.task == 'classification':
                    oob_pred = Counter(preds).most_common(1)[0][0]
                else:
                    oob_pred = np.mean(preds)
                oob_preds.append(oob_pred)
                valid_indices.append(i)
                
        if len(oob_preds) > 0:
            y_valid = y[valid_indices]
            if self.task == 'classification':
                self.oob_score_ = np.mean(np.array(oob_preds) == y_valid)
            else:
                self.oob_score_ = -np.mean((np.array(oob_preds) - y_valid) ** 2)
                
    def predict(self, X):
        X = np.array(X)
        
        if self.task == 'classification':
            all_predicitions = np.array([tree.predict(X) for tree in self.trees])
            
            predictions = []
            for i in range(X.shape[0]):
                predictions.append(Counter(all_predicitions[:, i]).most_common(1)[0][0])
            return np.array(predictions)
        else:
            all_predictions = np.array([tree.predict(X) for tree in self.trees])
            return np.mean(all_predictions, axis=0)
    
    def predict_proba(self, X):
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification")
        
        X = np.array(X)
        n_samples = X.shape[0]
        
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        classes = np.unique(all_predictions)
        n_classes = len(classes)
        
        probabilities = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            counts = Counter(all_predictions[:, i])
            for j,cls in enumerate(classes):
                probabilities[i,j] = counts.get(cls, 0) / self.n_estimators
                
        return probabilities
    
    def score(self, X,y):
        # Calculate accuracy if classification and R^2 for regression
        predictions = self.predict(X)
        
        if self.task == 'classification':
            return np.mean(predictions == y)
        else:
            ss_res = np.sum((y-predictions)**2)
            ss_tot = np.sum((y-np.mean(y))**2)
            return 1-(ss_res/ss_tot)
            