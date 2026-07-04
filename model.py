import numpy as np
import logging

logger = logging.getLogger(__name__)

class LinearRegressionFromScratch:
    """
    A Linear Regression model trained using Gradient Descent.
    This class is implemented without any ML frameworks to demonstrate the core math.
    Supports tracking training and validation loss history.
    """
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate      # Learning rate (alpha), controls size of gradient updates
        self.epochs = epochs          # Number of passes over the training dataset
        self.weights = None          # Weight vector (w_1, w_2, ...)
        self.bias = None             # Bias scalar (b)
        self.loss_history = []       # Store training loss values
        self.val_loss_history = []   # Store validation loss values
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the model using Gradient Descent.
        
        Parameters:
        X: numpy array of shape (N, M) training features
        y: numpy array of shape (N,) training targets
        X_val: numpy array of shape (N_val, M) validation features
        y_val: numpy array of shape (N_val,) validation targets
        """
        num_samples, num_features = X.shape
        
        # 1. Initialize parameters (weights and bias) to zero
        self.weights = np.zeros(num_features)
        self.bias = 0.0
        self.loss_history = []
        self.val_loss_history = []
        
        logger.info(f"Starting training: learning_rate={self.lr}, epochs={self.epochs}, features={num_features}")
        
        # 2. Gradient Descent Loop
        for epoch in range(self.epochs):
            
            # --- FORWARD PASS (Prediction) ---
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # --- COST/LOSS CALCULATION ---
            # Mean Squared Error (MSE): MSE = (1 / 2N) * sum( (y_predicted - y)^2 )
            error = y_predicted - y
            loss = (1.0 / (2.0 * num_samples)) * np.sum(error ** 2)
            self.loss_history.append(loss)
            
            # --- VALIDATION LOSS CALCULATION ---
            val_loss = None
            if X_val is not None and y_val is not None:
                y_val_predicted = np.dot(X_val, self.weights) + self.bias
                val_error = y_val_predicted - y_val
                val_loss = (1.0 / (2.0 * len(y_val))) * np.sum(val_error ** 2)
                self.val_loss_history.append(val_loss)
            
            # --- BACKWARD PASS (Gradient Calculation) ---
            # dw = (1/N) * (X^T . error)
            # db = (1/N) * sum(error)
            dw = (1.0 / num_samples) * np.dot(X.T, error)
            db = (1.0 / num_samples) * np.sum(error)
            
            # --- PARAMETER UPDATE (Gradient Descent Step) ---
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Print/log progress every 10% of epochs and at the final epoch
            if epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1:
                if val_loss is not None:
                    logger.info(f"Epoch {epoch:4d}/{self.epochs} - Loss (MSE) - Train: {loss:.6f} | Val: {val_loss:.6f}")
                else:
                    logger.info(f"Epoch {epoch:4d}/{self.epochs} - Loss (MSE) - Train: {loss:.6f}")
                    
    def predict(self, X):
        """
        Predict target values for new input data.
        
        Parameters:
        X: numpy array of shape (N, M)
        
        Returns:
        numpy array of shape (N,) containing predictions
        """
        # Ensure model has been trained before predicting
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call .fit() first.")
            
        # y_hat = X . w + b
        return np.dot(X, self.weights) + self.bias
