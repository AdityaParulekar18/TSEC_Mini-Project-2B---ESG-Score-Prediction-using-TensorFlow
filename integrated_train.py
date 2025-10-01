# integrated_train_poly_improved.py
import numpy as np
import json


data = np.load('preprocessed_integrated_train.npz')
X_train = data['X_train']  # shape: (n_samples, 16)
y_train = data['y_train']  # overall ESG score


def polynomial_features(X):
    """
    Expand X (n_samples, n_features) to include:
      - Original features,
      - Squared features,
      - Interaction terms for each pair (i < j).
    """
    n_samples, n_features = X.shape

    X_poly = X.copy()
   
    X_sq = X**2
    X_poly = np.concatenate((X_poly, X_sq), axis=1)
   
    interactions = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
    if interactions:
        X_inter = np.concatenate(interactions, axis=1)
        X_poly = np.concatenate((X_poly, X_inter), axis=1)
    return X_poly

X_train_poly = polynomial_features(X_train)
print("Original feature shape:", X_train.shape)
print("Expanded polynomial feature shape:", X_train_poly.shape)


mu_poly = np.mean(X_train_poly, axis=0)
sigma_poly = np.std(X_train_poly, axis=0)

sigma_poly[sigma_poly == 0] = 1
X_train_poly_norm = (X_train_poly - mu_poly) / sigma_poly


def compute_cost(X, y, W, b, lambda_reg=0.0):
    m = X.shape[0]
    predictions = np.dot(X, W) + b
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    reg_cost = (lambda_reg/(2*m)) * np.sum(W**2)
    return cost + reg_cost

def gradient_descent(X, y, W, b, lr, epochs, lambda_reg=0.0):
    m = X.shape[0]
    cost_history = []
    for epoch in range(epochs):
        predictions = np.dot(X, W) + b
        error = predictions - y
        dW = (1/m) * np.dot(X.T, error) + (lambda_reg/m)*W
        db = (1/m) * np.sum(error)
        W = W - lr * dW
        b = b - lr * db
        cost = compute_cost(X, y, W, b, lambda_reg)
        cost_history.append(cost)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    return W, b, cost_history


n_poly = X_train_poly_norm.shape[1]
W = np.zeros(n_poly)
b = 0

lr = 0.001     # Adjust learning rate as needed
epochs = 10000
lambda_reg = 0.001  # L2 regularization term

print("Training integrated polynomial model...")
W_final, b_final, cost_history = gradient_descent(X_train_poly_norm, y_train, W, b, lr, epochs, lambda_reg)
print("Integrated polynomial model training complete!")


# Save the learned weights and bias along with the polynomial scaler parameters
np.savez('integrated_trained_model_poly_improved.npz', W=W_final, b=b_final, mu_poly=mu_poly, sigma_poly=sigma_poly)
print("Model parameters saved in 'integrated_trained_model_poly_improved.npz'.")
