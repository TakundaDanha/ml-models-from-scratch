import numpy as np

class RidgeRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_param=1.0):
        """
        Ridge Regression using Gradient Descent with L2 Regularization.
        Model: y = mx + b
        Loss: Mean Squared Error + L2 Penalty
        
        L2 Regularization prevents overfitting by penalizing large weights.
        """
        
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.lambda_param = lambda_param  # Regularization strength
        self.m = None
        self.b = None
        self.losses = []
        
    def compute_gradients(self, X, y, y_pred):
        """
        Compute gradients with L2 regularization:
        
        ∂E/∂m = -(2/n) * Σ[x(y - ŷ)] + 2λm
        ∂E/∂b = -(2/n) * Σ(y - ŷ)
        
        Key difference from Linear Regression:
        - Weight gradient includes +2λm (regularization term)
        - Bias gradient unchanged (bias is not regularized)
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Input features
        y : array, shape (n_samples,)
            True values
        y_pred : array, shape (n_samples,)
            Predicted values
            
        Returns:
        --------
        dm : array, shape (n_features,)
            Gradient w.r.t. weights
        db : float
            Gradient w.r.t. bias
        """
        
        n = X.shape[0]
        error = y - y_pred
        error = error.reshape(-1, 1)  # Reshape for broadcasting
        
        # Gradient w.r.t. m (with L2 regularization)
        dm = (-2 / n) * np.sum(X * error, axis=0) + 2 * self.lambda_param * self.m
        
        # Gradient w.r.t. b (no regularization on bias)
        db = (-2 / n) * np.sum(error)
        
        return dm, db
    
    def compute_loss(self, y, y_pred):
        """
        Compute Mean Squared Error with L2 Regularization Penalty.
        
        E = (1/n) * Σ(y - ŷ)² + λ * Σ(mⱼ²)
        
        Components:
        - MSE term: measures prediction error
        - L2 penalty: sum of squared weights (||m||²)
        
        Parameters:
        -----------
        y : array, shape (n_samples,)
            True values
        y_pred : array, shape (n_samples,)
            Predicted values
            
        Returns:
        --------
        loss : float
            Total loss (MSE + L2 penalty)
        """
        n = y.shape[0]
        
        # Mean Squared Error
        mse = (1 / n) * np.sum((y - y_pred) ** 2)
        
        # L2 Regularization Penalty: λ * ||m||²
        l2_penalty = self.lambda_param * np.sum(self.m ** 2)
        
        # Total loss
        total_loss = mse + l2_penalty
        
        return total_loss
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        ŷ = mx + b
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        return np.dot(X, self.m) + self.b
    
    def fit(self, X, y):
        """
        Fit the ridge regression model using gradient descent.
        
        Algorithm:
        1. Initialize m and b to zeros
        2. For each iteration:
           a. Compute predictions: ŷ = mx + b
           b. Compute loss: E = MSE + λ||m||²
           c. Compute gradients (with L2 penalty on weights)
           d. Update parameters:
              - m = m(1 - 2αλ) - α∇ₘE  [includes shrinkage]
              - b = b - α∇ᵦE           [no regularization]
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.m = np.zeros(n_features)
        self.b = 0
        
        # Gradient Descent with Regularization
        for i in range(self.n_iters):
            # Forward pass
            y_pred = self.predict(X)
            
            # Compute loss (includes L2 penalty)
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # Compute gradients (with L2 regularization)
            dm, db = self.compute_gradients(X, y, y_pred)
            
            # Update parameters
            # Weight update includes shrinkage: m = m(1 - 2αλ) - α * gradient
            self.m = self.m - self.lr * dm
            self.b = self.b - self.lr * db
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                mse = (1 / n_samples) * np.sum((y - y_pred) ** 2)
                l2_pen = self.lambda_param * np.sum(self.m ** 2)
                print(f"Iteration {i+1}/{self.n_iters} - Loss: {loss:.4f} (MSE: {mse:.4f}, L2: {l2_pen:.4f})")
        
        return self
        
    def get_params(self):
        """
        Get the fitted parameters.
        
        Returns:
        --------
        params : dict
            Dictionary containing:
            - 'm': weights (with L2 shrinkage applied)
            - 'b': bias (not regularized)
            - 'lambda': regularization strength
        """
        return {
            'm': self.m, 
            'b': self.b,
            'lambda': self.lambda_param
        }
    
    def get_weight_magnitude(self):
        """
        Calculate L2 norm of weights.
        
        Useful for understanding the effect of regularization.
        Lower values indicate more regularization/shrinkage.
        
        Returns:
        --------
        l2_norm : float
            L2 norm of weights: √(Σmⱼ²)
        """
        return np.sqrt(np.sum(self.m ** 2))