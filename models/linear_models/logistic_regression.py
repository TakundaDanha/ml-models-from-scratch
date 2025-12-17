import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=10000):
        """
        Logistic Regression using Gradient Descent.
        Model: h_θ(x) = σ(θᵀx + b) where σ is the sigmoid function
        Loss: Binary Cross-Entropy (Log Loss)
        """
        
        
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.theta = None
        self.b = None
        self.losses = []
        
    def sigmoid(self, z):
        """
        Sigmoid activation function.
        
        σ(z) = 1 / (1 + e^(-z))
        
        Maps any real number to range (0, 1)
        """
        
        # Need to clip z to prevent overflow in exp
        z = np.clip(z,-500,500)
        return 1/(1+np.exp(-z))
    
    def compute_gradients(self, X, y, h):
        """
        Compute gradients using chain rule:
        
        ∂J/∂θⱼ = (1/m) * Σ[(h - y) * xⱼ]
        ∂J/∂b = (1/m) * Σ(h - y)
        
        Matrix form:
        ∇_θ J(θ) = (1/m) * Xᵀ(h - y)
        ∇_b J(b) = (1/m) * Σ(h - y)
        
        Parameters:
        -----------
        X : array, shape (m, n_features)
            Input features
        y : array, shape (m,)
            True labels (0 or 1)
        h : array, shape (m,)
            Predicted probabilities
        """
        
        m = X.shape[0]
        error = h - y
        error = error.reshape(-1,1) # we are reshaping it for broadcasting
        
        dtheta = (1/m)*np.sum(X*error, axis=0) # Gradient for the weights/parameters
        db = (1/m)*np.sum(error)
        
        return dtheta,db
    
    def compute_loss(self,y,h):
        """
        Compute Binary Cross-Entropy Loss.
        
        J(θ) = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
        
        This is the negative average log-likelihood.
        
        Parameters:
        -----------
        y : array, shape (m,)
            True labels
        h : array, shape (m,)
            Predicted probabilities
        """
        
        m = y.shape[0]
        
        # Wanna add small epsilon to prevent log(0)
        epsilon = 1e-15
        h = np.clip(h,epsilon,1-epsilon)
        
        loss = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1-h))
        
        return loss
    
    def predict_proba(self, X):
        """
        Predict probabilities for input samples.
        
        h_θ(x) = σ(θᵀx + b)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        """
        
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1,1)
            
        # Compute the logits: z = θᵀx + b
        z = np.dot(X,self.theta) + self.b
        
        # Apply sigmoid to get probabilities
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels for input samples.
        
        If h_θ(x) >= threshold: predict 1
        If h_θ(x) < threshold: predict 0
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        threshold : float, default=0.5
            Decision threshold
        """
        
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent.
        
        Algorithm:
        1. Initialize θ and b to zeros
        2. For each iteration:
           a. Compute predictions: h = σ(θᵀX + b)
           b. Compute loss: J(θ)
           c. Compute gradients: ∇_θ J and ∇_b J
           d. Update parameters: θ = θ - α∇_θ J, b = b - α∇_b J
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (must be 0 or 1)
        """
        
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1,1)
            
        m,n_features = X.shape
        
        # Initialize parameters to zeros
        self.theta = np.zeros(n_features)
        self.b = 0
        
        # Gradient Descent
        for i in range(self.n_iters):
            # forward pass: Compute predictions
            h = self.predict_proba(X)
            
            # Compute loss
            loss = self.compute_loss(y,h)
            self.losses.append(loss)
            
            # Compute gradients
            dtheta, db = self.compute_gradients(X,y,h)
            
            # Update parameters (move opposite to gradient)
            self.theta = self.theta - self.lr * dtheta
            self.b = self.b - self.lr *db
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iters} - Loss: {loss:.4f}")
                
        return self
    
    def get_params(self):
        """
        Get the fitted parameters.
        """
        return {'theta': self.theta, 'b': self.b}
    
    def score(self, X, y):
        """
        Calculate accuracy score on given data.
        
        Accuracy = (number of correct predictions) / (total predictions)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        y : array-like, shape (n_samples,)
            True labels
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        
        print("Accuracy score: ", accuracy)
        return accuracy