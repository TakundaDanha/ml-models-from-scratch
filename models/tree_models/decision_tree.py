import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


class Node:
    """Node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Class value if leaf node


class DecisionTreeClassifier:
    """
    Decision Tree Classifier using entropy and information gain.
    
    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of the tree. None means unlimited.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    criterion : str, default='entropy'
        Function to measure split quality ('entropy' or 'gini').
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_features = None
        self.n_classes = None
        self.feature_names = None
        self.class_names = None
        
    def fit(self, X, y):
        """
        Build decision tree from training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        X = np.array(X)
        y = np.array(y)
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        # If no valid split found, create leaf
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Split the data
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        # Recursively build left and right subtrees
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)
    
    def _best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(self.n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, y, X_column, threshold):
        """Calculate information gain for a split."""
        # Parent impurity
        parent_impurity = self._impurity(y)
        
        # Split the data
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs
        
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0
        
        # Calculate weighted average of children impurity
        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        impurity_left = self._impurity(y[left_idxs])
        impurity_right = self._impurity(y[right_idxs])
        child_impurity = (n_left / n) * impurity_left + (n_right / n) * impurity_right
        
        # Information gain
        gain = parent_impurity - child_impurity
        return gain
    
    def _impurity(self, y):
        """Calculate impurity (entropy or gini) of a node."""
        if self.criterion == 'entropy':
            return self._entropy(y)
        elif self.criterion == 'gini':
            return self._gini(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _entropy(self, y):
        """Calculate entropy of labels."""
        counts = np.bincount(y)
        probabilities = counts[counts > 0] / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _gini(self, y):
        """Calculate Gini impurity of labels."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _most_common_label(self, y):
        """Return the most common label in y."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _traverse_tree(self, x, node):
        """Traverse the tree to make a prediction for a single sample."""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def print_tree(self, node=None, depth=0, prefix="Root: "):
        """
        Print the decision tree structure.
        
        Parameters:
        -----------
        node : Node, default=None
            Starting node (uses root if None)
        depth : int, default=0
            Current depth in tree
        prefix : str, default="Root: "
            Prefix for the current node
        """
        if node is None:
            node = self.root
        
        if node.value is not None:
            class_name = self.class_names[node.value] if self.class_names is not None else node.value
            print(f"{'  ' * depth}{prefix}Predict {class_name}")
        else:
            feature_name = self.feature_names[node.feature] if self.feature_names is not None else f"Feature {node.feature}"
            print(f"{'  ' * depth}{prefix}{feature_name} <= {node.threshold:.3f}")
            self.print_tree(node.left, depth + 1, "Left: ")
            self.print_tree(node.right, depth + 1, "Right: ")
    
    def plot_tree(self, feature_names=None, class_names=None, figsize=(20, 10)):
        """
        Visualize the decision tree using matplotlib.
        
        Parameters:
        -----------
        feature_names : list, default=None
            Names of features
        class_names : list, default=None
            Names of classes
        figsize : tuple, default=(20, 10)
            Figure size
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
        
        self.feature_names = feature_names
        self.class_names = class_names
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Calculate tree structure
        self._node_positions = {}
        self._calculate_positions(self.root, 0.5, 1.0, 0.5)
        
        # Draw tree
        self._draw_node(ax, self.root)
        
        plt.tight_layout()
        plt.show()
    
    def _calculate_positions(self, node, x, y, width, depth=0):
        """Calculate positions for all nodes in the tree."""
        if node is None:
            return
        
        self._node_positions[id(node)] = (x, y)
        
        if node.value is None:  # Not a leaf
            y_child = y - 0.15
            width_child = width / 2
            
            # Left child
            x_left = x - width / 4
            self._calculate_positions(node.left, x_left, y_child, width_child, depth + 1)
            
            # Right child
            x_right = x + width / 4
            self._calculate_positions(node.right, x_right, y_child, width_child, depth + 1)
    
    def _draw_node(self, ax, node, parent_pos=None):
        """Recursively draw nodes and edges."""
        if node is None:
            return
        
        x, y = self._node_positions[id(node)]
        
        # Draw edge from parent
        if parent_pos is not None:
            ax.plot([parent_pos[0], x], [parent_pos[1], y], 'k-', linewidth=1.5, zorder=1)
        
        # Determine node color and text
        if node.value is not None:  # Leaf node
            class_name = self.class_names[node.value] if self.class_names is not None else str(node.value)
            text = f"Class:\n{class_name}"
            box_color = '#90EE90'  # Light green
        else:  # Decision node
            feature_name = self.feature_names[node.feature] if self.feature_names is not None else f"X[{node.feature}]"
            text = f"{feature_name}\n<= {node.threshold:.3f}"
            box_color = '#ADD8E6'  # Light blue
        
        # Draw node box
        box_width = 0.12
        box_height = 0.08
        box = FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.01",
            facecolor=box_color,
            edgecolor='black',
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y, text, ha='center', va='center', fontsize=9, 
                weight='bold', zorder=3)
        
        # Recursively draw children
        if node.value is None:
            self._draw_node(ax, node.left, (x, y))
            self._draw_node(ax, node.right, (x, y))