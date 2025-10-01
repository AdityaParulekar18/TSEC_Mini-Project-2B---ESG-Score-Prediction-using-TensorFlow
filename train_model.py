# hierarchical_train_model.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Step 1: Load Preprocessed Data
# ------------------------------
# The NPZ file should include:
#   X: full feature matrix (n_samples x 16)
#   y_overall: overall ESG score
#   y_E: Environmental sub-target
#   y_S: Social sub-target
#   y_G: Governance sub-target
data = np.load('preprocessed_hierarchical_data.npz')
X = data['X']             # shape: (n_samples, 16)
y_overall = data['y_overall']
y_E = data['y_E']
y_S = data['y_S']
y_G = data['y_G']

# ------------------------------
# Step 2: Train-Test Split (ensuring same indices across arrays)
# ------------------------------
X_train, X_test, y_overall_train, y_overall_test, y_E_train, y_E_test, y_S_train, y_S_test, y_G_train, y_G_test = \
    train_test_split(X, y_overall, y_E, y_S, y_G, test_size=0.2, random_state=42)

# ------------------------------
# Step 3: Split X_train into Sub-Groups Based on Feature Order
# ------------------------------
# Columns 0-4: Environmental (5 features)
# Columns 5-9: Social (5 features)
# Columns 10-15: Governance (6 features)
X_E_train = X_train[:, :5]
X_S_train = X_train[:, 5:10]
X_G_train = X_train[:, 10:16]

# ------------------------------
# Step 4: Define Gradient Descent Functions
# ------------------------------
def compute_cost(X, y, W, b):
    m = X.shape[0]
    predictions = np.dot(X, W) + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, W, b, learning_rate, epochs):
    m = X.shape[0]
    cost_history = []
    for epoch in range(epochs):
        predictions = np.dot(X, W) + b
        error = predictions - y
        dW = (1 / m) * np.dot(X.T, error)
        db = (1 / m) * np.sum(error)
        W = W - learning_rate * dW
        b = b - learning_rate * db
        cost = compute_cost(X, y, W, b)
        cost_history.append(cost)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    return W, b, cost_history

# ------------------------------
# Step 5: Train Sub-Models for E, S, and G
# ------------------------------
# Hyperparameters for sub-models
lr_sub = 0.001
epochs_sub = 10000

print("Training Environmental (E) sub-model...")
W_E_final, b_E_final, cost_E = gradient_descent(X_E_train, y_E_train, np.zeros(X_E_train.shape[1]), 0, lr_sub, epochs_sub)

print("Training Social (S) sub-model...")
W_S_final, b_S_final, cost_S = gradient_descent(X_train[:, 5:10], y_S_train, np.zeros(X_train[:, 5:10].shape[1]), 0, lr_sub, epochs_sub)

print("Training Governance (G) sub-model...")
W_G_final, b_G_final, cost_G = gradient_descent(X_train[:, 10:16], y_G_train, np.zeros(X_train[:, 10:16].shape[1]), 0, lr_sub, epochs_sub)

# ------------------------------
# Step 6: Compute Predicted Sub-Scores on Training Set
# ------------------------------
E_pred_train = np.dot(X_E_train, W_E_final) + b_E_final
S_pred_train = np.dot(X_train[:, 5:10], W_S_final) + b_S_final
G_pred_train = np.dot(X_train[:, 10:16], W_G_final) + b_G_final

# ------------------------------
# Step 7: Prepare Data for the Final Model
# ------------------------------
# Stack the predicted sub-scores into one feature matrix
sub_predictions_train = np.column_stack((E_pred_train, S_pred_train, G_pred_train))

# Normalize the sub-score predictions to ensure a stable scale for final model training
scaler_final = StandardScaler()
X_final_train = scaler_final.fit_transform(sub_predictions_train)

# ------------------------------
# Step 8: Train the Final ESG Model
# ------------------------------
# Hyperparameters for the final model
lr_final = 0.001
epochs_final = 10000

print("Training final ESG model using normalized sub-score predictions...")
W_final, b_final, cost_final = gradient_descent(X_final_train, y_overall_train, np.zeros(3), 0, lr_final, epochs_final)

# ------------------------------
# Step 9: Save the Trained Model Parameters and Scaler
# ------------------------------
np.savez('hierarchical_trained_model.npz', 
         W_E=W_E_final, b_E=b_E_final,
         W_S=W_S_final, b_S=b_S_final,
         W_G=W_G_final, b_G=b_G_final,
         W_final=W_final, b_final=b_final,
         scaler_final=scaler_final)

print("Hierarchical model training complete!")
print("Model parameters saved in 'hierarchical_trained_model.npz'.")