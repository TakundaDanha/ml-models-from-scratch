import numpy as np
from collections import Counters

class Node:
    def __init__(self, feature=None, threshold=None, left=None,right=None,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None
    
class RegressionTree:
    # This is going to fit to residuals/ pseudo-residuals
    
    def __init__(self, max_depth=3,min_samples_split=2,min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self._grow_tree(X, y, depth=0)
    
    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Stoppin criteria
        if(self.max_depth is not None and depth >= self.max_depth) or \
            n_samples < self.min_samples_split or \
                len(np.unique(y)) == 1:
                    return self._create_leaf(y)
                
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return self._create_leaf(y)
        
        # Split dataset
        left_idxs = X[:, best_feature] <= best_feature
        right_idxs = ~left_idxs
        
        # Check min_samples_leaf
        if np.sum(left_idxs) < self.min_samples_leaf or \
            np.sum(right_idxs) < self.min_samples_leaf:
                return self._create_leaf(y)
            
        # Grow children recursively
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return Node(feature=best_feature, threshold=best_threshold,left=left, right=right)
    
    def _create_leaf(self, y):
        value = np.mean(y)
        return Node(value=value)
    
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None, None
        
        # Current MSE
        parent_mse = self._criterion_mse(y)
        
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        # Try each feature
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            # Try each threshold
            for threshold in thresholds:
                #split 
                left_idxs = X[:, feature] <= threshold
                right_idxs = ~left_idxs
                
                if np.sum(left_idxs) < self.min_samples_leaf or \
                    np.sum(right_idxs) < self.min_samples_leaf:
                        continue
                    
                y_left, y_right = y[left_idxs], y[right_idxs]
                n_left, n_right = len(y_left), len(y_right)
                
                # Calc MSE for children
                left_mse = self._calculate_mse(y_left)
                right_mse = self._calculate_mse(y_right)
                
                # Weighted average of child MSE
                child_mse = (n_left/ n_samples) * left_mse + \
                    (n_right/n_samples)*right_mse
                    
                gain = parent_mse - child_mse
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    
    def _calculate_mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y-np.mean(y))**2)
    
    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        

class GradientBoostingRegressor:
    """
Gradient Boosting for Regression.

Parameters:
-----------
n_estimators : int, default=100
    Number of boosting iterations (trees to build)

learning_rate : float, default=0.1
    Shrinkage parameter (eta). Controls contribution of each tree.
    Lower values need more estimators but can generalize better.

max_depth : int, default=3
    Maximum depth of individual trees. Shallow trees work best.

min_samples_split : int, default=2
    Minimum samples required to split an internal node

min_samples_leaf : int, default=1
    Minimum samples required in a leaf node

subsample : float, default=1.0
    Fraction of samples to use for fitting each tree.
    Values < 1.0 result in Stochastic Gradient Boosting.

loss : str, default='mse'
    Loss function to optimize. Options: 'mse', 'mae'

random_state : int, optional
    Random seed for reproducibility
"""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                 loss='mse', random_state=None):
         self.n_estimators = n_estimators
         self.learning_rate = learning_rate
         self.max_depth = max_depth
         self.min_samples_split = min_samples_split
         self.min_samples_leaf = min_samples_leaf
         self.subsample = subsample
         self.loss = loss
         self.random_state = random_state
        
         self.trees = []
         self.init_value = None
         self.train_scores = []  # Track training loss over iterations
         
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        
        # Step 1: Initialize model with constant value
        self.init_value = self._initialize_model(y)
        
        ## Current predictions, start with initial value for all samples
        current_predictions = np.full(n_samples, self.init_value)
        
        # Step 2: Iteratively add trees
        self.trees = []
        
        for iteration in range(self.n_estimators):
            # Step 2.1: Compute pseudo-residuals (negative gradient)
            residuals = self._compute_residuals(y, current_predictions)
            
            # Subsampling for Stochastic Gradient Boosting
            if self.subsample < 1.0:
                sample_size = int(self.subsample * n_samples)
                sample_indices = np.random.choice(n_samples, sample_size, replace=False)
                X_sample = X[sample_indices]
                residuals_sample = residuals[sample_indices]
                
            else:
                X_sample = X
                residuals_sample = residuals
                
            # Step 2.2: Fit a tree to the residuals
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_sample, residuals_sample)
            
            # Step 2.3 and 2.4: Update predicitons with learning rate 
            # For MSE we can use the tree predicitons directly
            tree_predicitions = tree.predict(X)
            current_predictions += self.learning_rate * tree_predicitions
            
            # Store tree
            self.trees.append(tree)
            
            # Track training score
            train_loss = self._calculate_loss(y, current_predictions)
            self.train_scores.append(train_loss)
            
        return self
    
    def _initialize_model(self, y):
        if self.loss == 'mse':
            return np.mean(y)
        elif self.loss == 'mae':
            return np.median(y)
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
        
    def _compute_residuals(self, y, predictions):
        if self.loss == 'mse':
            return y - predictions
        elif self.loss == 'mae':
            return np.sign(y-predictions)
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
    
    def _calculate_loss(self, y, predictions):
        if self.loss == 'mse':
            return np.mean((y - predictions) ** 2)
        elif self.loss == 'mae':
            return np.mean(np.abs(y - predictions))
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
    
    def predict(self, X):
        X = np.array(X)
        
        # Start with initial value
        predictions = np.full(X.shape[0], self.init_value)
        
        # Add contribution from each tree
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions

    def staged_predict(self, X):
        X = np.array(X)
        predictions = np.full(X.shape[0], self.init_value)
        
        yield predictions.copy()
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
            yield predictions.copy()

    def score(self, X, y):
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
        
        
class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1,max_depth=3,
                 min_samples_split=2, min_samples_leaf=1,subsample=1.0,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        
        self.trees = []
        self.init_value = None
        self.classes_ = None
        self.train_scores = []
        
    def fit(self,X,y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        X = np.array(X)
        y = np.array(y)
        
        # Store classes
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Only supports Binary classification problems")
        
        # Ensure labels are 0 and 1
        if not np.all(np.isin(y, [0,1])):
            y = np.where(y == self.classes_[0],0,1)
            
        n_samples = X.shape[0]
        
        current_predictions = np.full(n_samples, self.init_value)
        
        # Iteratively add trees
        self.trees = []
        
        for iteration in range(self.n_estimators):
            # Convert to probabilities
            probabilities = self._sigmoid(current_predictions)
            
            # Compute pseudo_residuals (gradient of log loss)
            residuals = y - probabilities
            
            # Subsampling
            if self.subsample < 1.0:
                sample_size = int(self.subsample * n_samples)
                sample_indices = np.random.choice(n_samples, sample_size, replace=False)
                X_sample = X[sample_indices]
                residuals_sample = residuals[sample_indices]
            else:
                X_sample = X
                residuals_sample = residuals
                
            # Fit tree to residuals
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample,residuals_sample)
            
            # Update predictions
            tree_predictions = tree.predict(X)
            current_predictions += self.learning_rate * tree_predictions
            
            # Store tree
            self.trees.append(tree)
            
            # Track training score (log loss)
            train_loss = self._calculate_loss(y, current_predictions)
            self.train_scores.append(train_loss)
            
        return self
    
    def _initialize_model(self, y):
        p = np.mean(y)
        # avoid log(0) or log(inf)
        p = np.clip(p, 1e-15,1 - 1e-15)
        return np.log(p/(1-p))
    
    def _sigmoid(self, x):
        # Sigmoid with numerical stability
        return 1 / (1 + np.exp(-np.clip(x,-500,500)))
    
    def _calculate_loss(self, y, log_odds):
        probabilities = self._sigmoid(log_odds)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(probabilities)  + (1 - y)*np.log(1 - probabilities))
    
    def predict_proba(self, X):
        X = np.array(X)
        
        # Start with initial value
        log_odds = np.full(X.shape[0], self.init_value)
        
        # Add contribution from each tree
        for tree in self.trees:
            log_odds += self.learning_rate*tree.predict(X)
            
        # Convert to probabilities
        prob_class_1 = self._sigmoid(log_odds)
        prob_class_0 = 1 - prob_class_1
        
        return np.column_stack([prob_class_0, prob_class_1])
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        # Map back to original classes
        return self.classes_[predictions]

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)