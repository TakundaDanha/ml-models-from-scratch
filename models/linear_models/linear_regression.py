import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Linear Regression using Gradient Descent.
        Model: y = mx + b
        Loss: Mean Squared Error
        """
        
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.m = None
        self.b = None
        self.losses = []
        
    def compute_gradients(self, X, y, y_pred):
        """
        Compute gradients:
        dm = -(2/n) * Σ[x(y - ŷ)]  # vectorized over features
        db = -(2/n) * Σ(y - ŷ)
        """
        
        n = X.shape[0]
        error = y - y_pred
        error = error.reshape(-1, 1)  # Reshape for broadcasting
        
        dm = (-2 / n) * np.sum(X * error, axis=0)
        db = (-2 / n) * np.sum(error)
        
        return dm, db
    
    def compute_losses(self, y, y_pred):
        """
        Compute Mean Squared Error.
        
        MSE = (1/n) * Σ(y - ŷ)²
        """
        n = y.shape[0]
        return (1 / n) * np.sum((y - y_pred) ** 2)
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
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
        Fit the linear regression model using gradient descent.
        
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
        
        # Gradient Descent
        for i in range(self.n_iters):
            y_pred = self.predict(X)
            
            # Compute loss
            loss = self.compute_losses(y, y_pred)
            self.losses.append(loss)
            
            # Compute Gradient
            dm, db = self.compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.m = self.m - self.lr * dm
            self.b = self.b - self.lr * db
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iters} - Loss: {loss:.4f}")
        
        return self  # Moved outside the loop
        
    def get_params(self):
        """
        Get the fitted parameters.
        
        Returns:
        --------
        params : dict
            Dictionary containing 'm' (slope) and 'b' (bias)
        """
        return {'m': self.m, 'b': self.b}