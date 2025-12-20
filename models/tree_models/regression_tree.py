import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # FIXED: was missing = value
        
class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split  # FIXED: was min_samples_splits (typo)
        self.root = None
        self.n_features = None
        self.feature_names = None
        # REMOVED: criterion, n_classes, class_names (not needed for regression)
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.n_features = X.shape[1]
        # REMOVED: self.n_classes (not needed for regression)
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_samples == 0:  # ADDED: safety check
            leaf_value = np.mean(y) if len(y) > 0 else 0
            return Node(value=leaf_value)
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Split the data
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        # ADDED: Safety check - ensure split actually separates data
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Recursively build the left and right subtrees
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)
    
    def _best_split(self, X, y):
        # Find the best feature and threshold to split on
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
        
        # Check if we found a valid split with positive gain
        if best_gain <= 0 or best_gain == -1:
            return None, None
                        
        return best_feature, best_threshold
    
    def _information_gain(self, y, X_column, threshold):
        # Calculate the information gain using MSE for a split
        parent_mse = self._mse(y)
        
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs
        
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0
        
        # Calculate the weighted average of children MSE
        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        mse_left = self._mse(y[left_idxs])
        mse_right = self._mse(y[right_idxs])
        child_mse = (n_left / n) * mse_left + (n_right / n) * mse_right
        
        gain = parent_mse - child_mse
        return gain
    
    def _traverse_tree(self, x, node):
        # Traverse the tree to make a prediction for a single sample
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def _mse(self, y):
        """Calculate Mean Squared Error."""
        if len(y) == 0:
            return 0
        
        mean = np.mean(y)
        mse = np.mean((y - mean) ** 2)
        return mse
        
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
            print(f"{'  ' * depth}{prefix}Predict {node.value:.3f}")
        else:
            feature_name = self.feature_names[node.feature] if self.feature_names is not None else f"Feature {node.feature}"
            print(f"{'  ' * depth}{prefix}{feature_name} <= {node.threshold:.3f}")
            self.print_tree(node.left, depth + 1, "Left: ")
            self.print_tree(node.right, depth + 1, "Right: ")
    
    def plot_tree(self, feature_names=None, figsize=(20, 10)):
        """
        Visualize the decision tree using matplotlib.
        
        Parameters:
        -----------
        feature_names : list, default=None
            Names of features
        figsize : tuple, default=(20, 10)
            Figure size
        """
        self.feature_names = feature_names
        
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
            x_left = x - width / 2
            self._calculate_positions(node.left, x_left, y_child, width_child, depth + 1)
            
            # Right child
            x_right = x + width / 2
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
            text = f"Value:\n{node.value:.3f}"
            box_color = '#90EE90'  # Light green
        else:  # Decision node
            feature_name = self.feature_names[node.feature] if self.feature_names is not None else f"X[{node.feature}]"
            text = f"{feature_name}\n<= {node.threshold:.3f}"
            box_color = '#ADD8E6'  # Light blue
        
        # Draw node box
        box_width = 0.12
        box_height = 0.08
        box = FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2),
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


# Example usage
if __name__ == "__main__":
    # Simple regression example
    print("=== REGRESSION TREE EXAMPLE ===")
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    y = np.array([2.5, 3.1, 3.8, 6.2, 7.1, 7.8, 10.2, 11.5])
    
    reg = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    reg.fit(X, y)
    predictions = reg.predict(X)
    
    print("True values:     ", y)
    print("Predicted values:", np.round(predictions, 2))
    print("\nTree structure:")
    reg.feature_names = ['X']
    reg.print_tree()
    
    # Calculate MSE
    mse = np.mean((y - predictions) ** 2)
    print(f"\nTraining MSE: {mse:.4f}")