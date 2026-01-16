import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class SVMClassifier:
    def __init__(self, C=1.0, kernel='rbf',gamma='auto',degree=3,coef0=0.0,
                 tol=1e-3,max_iter=1000, epsilon=1e-8):
        """
        Parameters:
        -----------
        C : float, regularization parameter (soft-margin trade-off)
        kernel : str, kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        gamma : float or 'auto', kernel coefficient for rbf, poly and sigmoid
        degree : int, degree for polynomial kernel
        coef0 : float, independent term in kernel function
        tol : float, tolerance for stopping criterion
        max_iter : int, maximum number of iterations
        epsilon : float, small value to avoid division by zero
        """
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.epsilon = epsilon
        
        # Model parameters set during training
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_indices = None
        self.X_train = None
        self.y_train = None
        self.trained = False
        
        # Training history and statistics
        self.training_history = {
            'iterations': [],
            'objective': [],
            'n_support_vectors': [],
            'margin': []
        }
    
    def _kernel(self, x1, x2):
        # Compute kernel function K(x1,x2)
        if self.kernel_type == 'linear':
            return np.dot(x1,x2)
        elif self.kernel_type == 'poly':
            return (np.dot(x1,x2) + self.coef0)** self.degree
        elif self.kernel_type == 'rbf':
            gamma = self.gamma if self.gamma != 'auto' else 1.0/x1.shape[0]
            return np.exp(-gamma *np.linalg.norm(x1-x2)**2)
        elif self.kernel_type == 'sigmoid':
            gamma = self.gamma if self.gamma != 'auto' else 1.0/x1.shape[0]
            return np.tanh(gamma * np.dot(x1,x2)+self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")
        
    def _compute_kernel_matrix(self, X1, X2=None):
        # Compute the Gram matrix K[i,j] = K(X1[i],X2[j])
        if X2 is None:
            X2 = X1
        n1,n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1,n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i,j] = self._kernel(X1[i],X2[j])
        
        return K
    
    def _decision_function(self, X):
        # Compute decision function w^T x + b
        if not self.trained:
            raise ValueError("Model must be trained before making a predicition")
        
        K = self._compute_kernel_matrix(X, self.support_vectors)
        return np.dot(K, self.alpha * self.support_vector_labels) + self.b
    
    def _compute_objective(self):
        # Compute dual objective function
        K = self._compute_kernel_matrix(self.X_train)
        return (np.sum(self.alpha)-
                0.5*np.sum(self.alpha *self.y_train)[:, None]*
                (self.alpha * self.y_train)[None, :] * K)
        
    def fit(self, X, y, verbose=False):
        X = np.array(X)
        y = np.array(y)
        
        if len(np.unique(y)) != 2:
            raise ValueError("SVM is a binary classifier, use labels -1 and 1")
        
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, [-1,1]):
            label_map = {unique_labels[0]: -1, unique_labels[1]: 1}
            y = np.array([label_map[label] for label in y])
            
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y
        
        # Initialize alpha
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Set gamma if auto
        if self.gamma == 'auto':
            self.gamma = 1.0/n_features
            
        # Compute kernel Matrix
        K = self._compute_kernel_matrix(X)
        
        # SMO algorithm
        iteration = 0
        while iteration < self.max_iter:
            alpha_prev = np.copy(self.alpha)
            
            for j in range(n_samples):
                # Pick a random i != j
                i = self._get_random_index(j, n_samples)
                
                # Calculate error
                E_i = self._decision_function(X[i:i+1])[0] - y[i]
                E_j = self._decision_function(X[j:j+1])[0] - y[j]
                
                # Check KKT conditions
                if not self._violates_kkt(i, E_i):
                    continue
                
                # Calculate bound L and H
                if y[i] != y[j]:
                    L = max(0, self.alpha[j] - self.alpha[i])
                    H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                else:
                    L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                    H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                if L == H:
                    continue
                
                # Compute eta
                eta = 2 * K[i,j] - K[i,i] - K[j,j]
                
                if eta >= 0:
                    continue
                
                # Update alpha_j
                alpha_j_new = self.alpha[j] - (y[j] * (E_i - E_j))/ eta
                alpha_j_new = np.clip(alpha_j_new, L, H)
                
                if abs(alpha_j_new - self.alpha[j]) < self.epsilon:
                    continue
                
                # Update alpha_i
                alpha_i_new = self.alpha[i] + y[i] * y[j] * (self.alpha[j] - alpha_j_new)
                
                # Update the bias term
                b1 = self.b - E_i - y[i] * (alpha_i_new - self.alpha[i]) * K[i,i] - \
                    y[j] * (alpha_j_new - self.alpha[j]) * K[i,j]
                b2 = self.b - E_i - y[i] * (alpha_i_new - self.alpha[i]) * K[i,j] - \
                    y[j] * (alpha_j_new - self.alpha[j]) * K[j,j]
                    
                if 0 < alpha_i_new < self.C:
                    self.b = b1
                elif 0 < alpha_j_new < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2)/2
                    
                # Update the alphas
                self.alpha[i] = alpha_i_new
                self.alpha[j] = alpha_j_new
                
            # Check convergence
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                if verbose:
                    print(f" Converged at iteration {iteration}")
                break
                
            # Store training history
            if iteration % 10 == 0:
                sv_indices = self.alpha > self.epsilon
                margin = self._compute_margin() if np.any(sv_indices) else 0
                    
                self.training_history['iterations'].append(iteration)
                self.training_history['objective'].append(self._compute_objective())
                self.training_history['n_support_vectors'].append(np.sum(sv_indices))
                self.training_history['margin'].append(margin)
            iteration +=1
            
        # Extract support vectors
        sv_indices = self.alpha > self.epsilon
        self.support_vector_indices = np.where(sv_indices)[0]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.alpha = self.alpha[sv_indices]
            
        self.trained = True
            
        if verbose:
            print(f"Training completed in {iteration} iterations")
            print(f"Number of support vectors: {len(self.support_vectors)}")
            print(f"Margin: {self._compute_margin():.4f}")
            
        return self
    
    def _get_random_index(self, exclude, n):
        idx = exclude
        while idx == exclude:
            idx = np.random.randint(0,n)
        return idx
    
    def _violates_kkt(self, i, E_i):
        r = E_i *self.y_train[i]
        return ((r < -self.tol and self.alpha[i] < self.C) or
                (r > self.tol and self.alpha[i] > 0))
        
    def _compute_margin(self):
        if not self.trained or len(self.support_vectors) == 0:
            return 0
        
        # For linear kernel, margin = 2 / ||w||
        if self.kernel == 'linear':
            w = np.dot(self.alpha * self.support_vector_labels, self.support_vectors)
            return 2.0 / (np.linalg.norm(w) + self.epsilon)
        else:
            # For non-linear kernels, approx using distance of support vectors
            distances = np.abs(self._decision_function(self.support_vectors))
            return 2 * np.min(distances[distances > self.epsilon]) if len(distances) > 0 else 0
        
    def predict(self, X):
        return np.sign(self._decision_function(X))
    
    def predict_proba(self, X):
        decision = self._decision_function(X)
        prob_positive = 1/(1 + np.exp(-decision))
        return np.column_stack([1-prob_positive, prob_positive])
    
    def score(self, X,y):
        return np.mean(self.predict(X) == y)
    
    def get_params(self):
     #Get model parameters
     return {
         'C': self.C,
         'kernel': self.kernel_type,
         'gamma': self.gamma,
         'degree': self.degree,
         'n_support_vectors': len(self.support_vectors) if self.trained else 0,
         'margin': self._compute_margin() if self.trained else 0,
         'b': self.b
     }


class SVMRegressor:   
    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', gamma='auto', 
                 degree=3, coef0=0.0, tol=1e-3, max_iter=1000):
        self.C = C
        self.epsilon = epsilon
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        
        self.alpha = None
        self.alpha_star = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_targets = None
        self.X_train = None
        self.y_train = None
        self.trained = False
        
    def _kernel(self, x1, x2):
        # Compute kernel function
        if self.kernel_type == 'linear':
            return np.dot(x1, x2)
        elif self.kernel_type == 'poly':
            return (np.dot(x1, x2) + self.coef0) ** self.degree
        elif self.kernel_type == 'rbf':
            gamma = self.gamma if self.gamma != 'auto' else 1.0 / x1.shape[0]
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel_type == 'sigmoid':
            gamma = self.gamma if self.gamma != 'auto' else 1.0 / x1.shape[0]
            return np.tanh(gamma * np.dot(x1, x2) + self.coef0)
    
    def _compute_kernel_matrix(self, X1, X2=None):
        # Compute Gram matrix
        if X2 is None:
            X2 = X1
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel(X1[i], X2[j])
        return K
    
    def fit(self, X, y, verbose=False):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        self.X_train = X
        self.y_train = y
        
        if self.gamma == 'auto':
            self.gamma = 1.0 / n_features
        
        # Initialize dual variables
        self.alpha = np.zeros(n_samples)
        self.alpha_star = np.zeros(n_samples)
        self.b = 0
        
        K = self._compute_kernel_matrix(X)
        
        # Simplified training (gradient descent on dual)
        for iteration in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)
            alpha_star_prev = np.copy(self.alpha_star)
            
            for i in range(n_samples):
                # Compute prediction
                pred = np.sum((self.alpha - self.alpha_star) * K[:, i]) + self.b
                error = pred - y[i]
                
                # Update alpha and alpha_star
                if error > self.epsilon:
                    if self.alpha_star[i] < self.C:
                        self.alpha_star[i] = min(self.C, self.alpha_star[i] + 0.01 * abs(error))
                        self.alpha[i] = max(0, self.alpha[i] - 0.01 * abs(error))
                elif error < -self.epsilon:
                    if self.alpha[i] < self.C:
                        self.alpha[i] = min(self.C, self.alpha[i] + 0.01 * abs(error))
                        self.alpha_star[i] = max(0, self.alpha_star[i] - 0.01 * abs(error))
            
            # Update bias
            sv_indices = ((self.alpha > 0) & (self.alpha < self.C)) | \
                        ((self.alpha_star > 0) & (self.alpha_star < self.C))
            if np.any(sv_indices):
                sv_idx = np.where(sv_indices)[0]
                predictions = np.dot(K[sv_idx, :], self.alpha - self.alpha_star)
                self.b = np.mean(y[sv_idx] - predictions)
            
            # Check convergence
            if (np.linalg.norm(self.alpha - alpha_prev) < self.tol and
                np.linalg.norm(self.alpha_star - alpha_star_prev) < self.tol):
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        # Extract support vectors
        sv_indices = (self.alpha > 1e-5) | (self.alpha_star > 1e-5)
        self.support_vectors = X[sv_indices]
        self.support_vector_targets = y[sv_indices]
        self.alpha = self.alpha[sv_indices]
        self.alpha_star = self.alpha_star[sv_indices]
        
        self.trained = True
        
        if verbose:
            print(f"Number of support vectors: {len(self.support_vectors)}")
        
        return self
    
    def predict(self, X):
        # Predict continuous values
        if not self.trained:
            raise ValueError("Model must be trained first")
        
        X = np.array(X)
        K = self._compute_kernel_matrix(X, self.support_vectors)
        return np.dot(K, self.alpha - self.alpha_star) + self.b
    
    def score(self, X, y):
        # Compute R² score
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
        
    
class SVMVisualizer:
    def __init__(self, model):
        self.model = model
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    def plot_decision_boundary_2d(self, X, y, title="SVM Decision Boundary", 
                                   resolution=500, save_path=None):
        """Plot decision boundary for 2D data"""
        if X.shape[1] != 2:
            raise ValueError("This method requires 2D data")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        
        # Predict on mesh
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        ax.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], 
                   colors=[self.colors[0], self.colors[1]])
        ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], 
                  alpha=0.5, linestyles=['--', '-', '--'])
        
        # Plot data points
        for label in np.unique(y):
            mask = y == label
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c=[self.colors[int(label)]], 
                      label=f'Class {int(label)}',
                      edgecolors='k', s=100, alpha=0.7)
        
        # Highlight support vectors
        if hasattr(self.model, 'support_vectors') and self.model.support_vectors is not None:
            ax.scatter(self.model.support_vectors[:, 0], 
                      self.model.support_vectors[:, 1],
                      s=300, linewidths=2, facecolors='none', 
                      edgecolors='red', label='Support Vectors')
        
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def plot_decision_boundary_3d(self, X, y, title="3D SVM Decision Boundary", 
                                   resolution=50, save_path=None):
        """Plot decision boundary for 3D data"""
        if X.shape[1] != 3:
            raise ValueError("This method requires 3D data")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot data points
        for label in np.unique(y):
            mask = y == label
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                      c=[self.colors[int(label)]], 
                      label=f'Class {int(label)}',
                      s=100, alpha=0.6, edgecolors='k')
        
        # Highlight support vectors
        if hasattr(self.model, 'support_vectors') and self.model.support_vectors is not None:
            ax.scatter(self.model.support_vectors[:, 0],
                      self.model.support_vectors[:, 1],
                      self.model.support_vectors[:, 2],
                      s=300, facecolors='none', edgecolors='red',
                      linewidths=2, label='Support Vectors')
        
        # Create decision surface (simplified - show slices)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        z_mid = (X[:, 2].min() + X[:, 2].max()) / 2
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        zz = np.full_like(xx, z_mid)
        
        points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        Z = self.model.predict(points).reshape(xx.shape)
        
        ax.plot_surface(xx, yy, zz, facecolors=plt.cm.RdYlBu(Z), 
                       alpha=0.3, rstride=1, cstride=1)
        
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_zlabel('Feature 3', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def plot_high_dimensional_pca(self, X, y, title="SVM Decision Boundary (PCA)", 
                                   save_path=None):
        """Visualize high-dimensional data using PCA reduction to 2D"""
        if X.shape[1] <= 2:
            return self.plot_decision_boundary_2d(X, y, title, save_path=save_path)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Train a temporary SVM on PCA-transformed data for visualization
        temp_model = SVMClassifier(
            C=self.model.C,
            kernel=self.model.kernel_type,
            gamma=self.model.gamma,
            degree=self.model.degree
        )
        temp_model.fit(X_pca, y)
        
        # Temporarily swap models
        original_model = self.model
        self.model = temp_model
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create mesh
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        
        Z = temp_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1],
                   colors=[self.colors[0], self.colors[1]])
        ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1],
                  alpha=0.5, linestyles=['--', '-', '--'])
        
        for label in np.unique(y):
            mask = y == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[self.colors[int(label)]],
                      label=f'Class {int(label)}',
                      edgecolors='k', s=100, alpha=0.7)
        
        if temp_model.support_vectors is not None:
            ax.scatter(temp_model.support_vectors[:, 0],
                      temp_model.support_vectors[:, 1],
                      s=300, linewidths=2, facecolors='none',
                      edgecolors='red', label='Support Vectors')
        
        variance_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({variance_explained[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({variance_explained[1]:.1%} variance)', fontsize=12)
        ax.set_title(f'{title}\n(Original dimensions: {X.shape[1]})', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        
        # Restore original model
        self.model = original_model
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if not hasattr(self.model, 'training_history'):
            print("No training history available")
            return
        
        history = self.model.training_history
        if len(history['iterations']) == 0:
            print("Training history is empty")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Objective function
        axes[0, 0].plot(history['iterations'], history['objective'], 
                       'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Dual Objective')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Number of support vectors
        axes[0, 1].plot(history['iterations'], history['n_support_vectors'], 
                       'g-', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Number of SVs')
        axes[0, 1].set_title('Support Vectors Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Margin
        axes[1, 0].plot(history['iterations'], history['margin'], 
                       'r-', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Margin')
        axes[1, 0].set_title('Margin Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        Training Summary
        {'='*40}
        Final Objective: {history['objective'][-1]:.4f}
        Support Vectors: {history['n_support_vectors'][-1]}
        Final Margin: {history['margin'][-1]:.4f}
        Total Iterations: {history['iterations'][-1]}
        
        Kernel: {self.model.kernel_type}
        C parameter: {self.model.C}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                       family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_support_vectors_analysis(self, X, y, save_path=None):
        """Analyze support vectors distribution"""
        if not self.model.trained:
            print("Model must be trained first")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Distribution of alpha values
        axes[0].hist(self.model.alpha, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Alpha Values')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Lagrange Multipliers')
        axes[0].axvline(self.model.C, color='r', linestyle='--', 
                       label=f'C = {self.model.C}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Support vector contribution
        contributions = self.model.alpha * self.model.support_vector_labels
        axes[1].bar(range(len(contributions)), np.abs(contributions), 
                   alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Support Vector Index')
        axes[1].set_ylabel('|α_i * y_i|')
        axes[1].set_title('Support Vector Contributions')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_margin_analysis(self, X, y, save_path=None):
        """Visualize margin and support vectors in detail"""
        if X.shape[1] != 2:
            print("Margin analysis visualization requires 2D data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get decision function values
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        
        Z = self.model._decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot filled contours
        levels = np.linspace(-2, 2, 20)
        contourf = ax.contourf(xx, yy, Z, levels=levels, cmap='RdYlBu', alpha=0.6)
        plt.colorbar(contourf, ax=ax, label='Decision Function Value')
        
        # Plot margin boundaries
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'blue'],
                  linewidths=[2, 3, 2], linestyles=['--', '-', '--'])
        
        # Plot data points
        for label in np.unique(y):
            mask = y == label
            ax.scatter(X[mask, 0], X[mask, 1],
                      c=[self.colors[int(label)]],
                      label=f'Class {int(label)}',
                      s=100, edgecolors='k', alpha=0.7)
        
        # Highlight support vectors
        if self.model.support_vectors is not None:
            ax.scatter(self.model.support_vectors[:, 0],
                      self.model.support_vectors[:, 1],
                      s=400, facecolors='none', edgecolors='red',
                      linewidths=3, label='Support Vectors')
        
        margin = self.model._compute_margin()
        ax.set_title(f'Margin Analysis\nMargin width: {margin:.4f}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_regression_fit(self, X, y, save_path=None):
        """Plot regression fit (for 1D or 2D input)"""
        if not isinstance(self.model, SVMRegressor):
            print("This method is for SVMRegressor only")
            return
        
        if X.shape[1] == 1:
            # 1D regression plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            X_sorted = np.sort(X, axis=0)
            y_pred = self.model.predict(X_sorted)
            
            ax.scatter(X, y, c='blue', s=50, alpha=0.6, label='Data')
            ax.plot(X_sorted, y_pred, 'r-', linewidth=2, label='SVR Prediction')
            
            # Plot epsilon tube
            ax.plot(X_sorted, y_pred + self.model.epsilon, 'k--', 
                   alpha=0.5, label=f'ε-tube (ε={self.model.epsilon})')
            ax.plot(X_sorted, y_pred - self.model.epsilon, 'k--', alpha=0.5)
            ax.fill_between(X_sorted.ravel(), 
                           y_pred - self.model.epsilon,
                           y_pred + self.model.epsilon,
                           alpha=0.2, color='gray')
            
            # Highlight support vectors
            if self.model.support_vectors is not None:
                ax.scatter(self.model.support_vectors, 
                          self.model.support_vector_targets,
                          s=200, facecolors='none', edgecolors='red',
                          linewidths=2, label='Support Vectors')
            
            ax.set_xlabel('Feature', fontsize=12)
            ax.set_ylabel('Target', fontsize=12)
            ax.set_title('SVR Fit with ε-insensitive Tube', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            # Multi-dimensional: show actual vs predicted
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            y_pred = self.model.predict(X)
            
            # Actual vs Predicted
            axes[0].scatter(y, y_pred, alpha=0.6, s=50)
            min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            axes[0].set_xlabel('Actual', fontsize=12)
            axes[0].set_ylabel('Predicted', fontsize=12)
            axes[0].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Residuals
            residuals = y - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.6, s=50)
            axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[1].axhline(y=self.model.epsilon, color='k', linestyle=':', alpha=0.5)
            axes[1].axhline(y=-self.model.epsilon, color='k', linestyle=':', alpha=0.5)
            axes[1].set_xlabel('Predicted', fontsize=12)
            axes[1].set_ylabel('Residuals', fontsize=12)
            axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_kernel_comparison(self, X, y, kernels=['linear', 'poly', 'rbf'], 
                               save_path=None):
        """Compare different kernels"""
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X)
        else:
            X_plot = X
        
        n_kernels = len(kernels)
        fig, axes = plt.subplots(1, n_kernels, figsize=(6*n_kernels, 5))
        
        if n_kernels == 1:
            axes = [axes]
        
        for idx, kernel in enumerate(kernels):
            model = SVMClassifier(C=self.model.C, kernel=kernel, 
                                gamma=self.model.gamma)
            model.fit(X_plot, y)
            
            # Create mesh
            x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
            y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                np.linspace(y_min, y_max, 300))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            axes[idx].contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1],
                              colors=[self.colors[0], self.colors[1]])
            axes[idx].contour(xx, yy, Z, colors='k', levels=[0],
                             linewidths=2)
            
            for label in np.unique(y):
                mask = y == label
                axes[idx].scatter(X_plot[mask, 0], X_plot[mask, 1],
                                c=[self.colors[int(label)]],
                                s=50, edgecolors='k', alpha=0.7)
            
            if model.support_vectors is not None:
                axes[idx].scatter(model.support_vectors[:, 0],
                                model.support_vectors[:, 1],
                                s=200, facecolors='none', edgecolors='red',
                                linewidths=2)
            
            accuracy = model.score(X_plot, y)
            axes[idx].set_title(f'{kernel.upper()} Kernel\nAcc: {accuracy:.2%}\nSVs: {len(model.support_vectors)}',
                              fontweight='bold')
            axes[idx].set_xlabel('Feature 1')
            axes[idx].set_ylabel('Feature 2')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class SVMStatistics:
    """Statistical analysis tools for SVMs"""
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """Compute confusion matrix"""
        labels = np.unique(np.concatenate([y_true, y_pred]))
        n_labels = len(labels)
        cm = np.zeros((n_labels, n_labels), dtype=int)
        
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
        
        return cm, labels
    
    @staticmethod
    def classification_report(y_true, y_pred):
        """Generate classification report"""
        cm, labels = SVMStatistics.confusion_matrix(y_true, y_pred)
        
        report = {}
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            report[label] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': int(cm[i, :].sum())
            }
        
        # Overall metrics
        accuracy = np.trace(cm) / cm.sum()
        report['accuracy'] = accuracy
        report['confusion_matrix'] = cm
        report['labels'] = labels
        
        return report
    
    @staticmethod
    def cross_validate(model, X, y, cv=5, verbose=True):
        """Perform k-fold cross-validation"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        fold_size = n_samples // cv
        scores = []
        
        for fold in range(cv):
            # Split data
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < cv - 1 else n_samples
            
            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train and evaluate
            model_copy = SVMClassifier(C=model.C, kernel=model.kernel_type,
                                      gamma=model.gamma, degree=model.degree)
            model_copy.fit(X_train, y_train)
            score = model_copy.score(X_test, y_test)
            scores.append(score)
            
            if verbose:
                print(f"Fold {fold + 1}/{cv}: Accuracy = {score:.4f}")
        
        scores = np.array(scores)
        if verbose:
            print(f"\nCross-Validation Results:")
            print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm, labels = SVMStatistics.confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels, yticklabels=labels,
               xlabel='Predicted Label',
               ylabel='True Label',
               title='Confusion Matrix')
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_roc_curve(model, X, y, save_path=None):
        """Plot ROC curve for binary classification"""
        if len(np.unique(y)) != 2:
            print("ROC curve is for binary classification only")
            return
        
        # Get decision function scores
        scores = model._decision_function(X)
        
        # Compute ROC curve
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        tpr_list, fpr_list = [], []
        
        for threshold in thresholds:
            y_pred = np.where(scores >= threshold, 1, -1)
            
            tp = np.sum((y_pred == 1) & (y == 1))
            fp = np.sum((y_pred == 1) & (y == -1))
            tn = np.sum((y_pred == -1) & (y == -1))
            fn = np.sum((y_pred == -1) & (y == 1))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Compute AUC
        auc = np.trapz(tpr_list, fpr_list)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr_list, tpr_list, 'b-', linewidth=2, label=f'ROC curve (AUC = {abs(auc):.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return abs(auc)