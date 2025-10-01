# hierarchical_evaluate_model.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Step 1: Load the Test Data
# ------------------------------
# Assuming you saved your test split as 'test_data.csv' in your preprocessing step.
df_test = pd.read_csv('test_data.csv')

# Feature columns as defined earlier:
feature_cols = [
    'CO2_Emissions', 'Renewable_Energy', 'Water_Consumption', 'Waste_Management', 'Biodiversity_Impact',
    'Gender_Diversity', 'Employee_Satisfaction', 'Community_Investment', 'Safety_Incidents', 'Labor_Rights',
    'Board_Diversity', 'Executive_Pay_Ratio', 'Transparency', 'Shareholder_Rights', 'Anti_Corruption', 'Political_Donations'
]

# Extract the feature matrix and overall target
X_test_full = df_test[feature_cols].values
y_overall_test = df_test['ESG_Score'].values

# Split test features into sub-groups:
X_E_test = X_test_full[:, :5]       # Environmental features (columns 0-4)
X_S_test = X_test_full[:, 5:10]     # Social features (columns 5-9)
X_G_test = X_test_full[:, 10:16]    # Governance features (columns 10-15)

# ------------------------------
# Step 2: Load the Trained Model Parameters and Scaler
# ------------------------------
model = np.load('hierarchical_trained_model.npz', allow_pickle=True)
W_E = model['W_E']
b_E = model['b_E']
W_S = model['W_S']
b_S = model['b_S']
W_G = model['W_G']
b_G = model['b_G']
W_final = model['W_final']
b_final = model['b_final']
scaler_final = model['scaler_final'].item()  # Load the StandardScaler

# ------------------------------
# Step 3: Compute Sub-Model Predictions on Test Set
# ------------------------------
E_pred_test = np.dot(X_E_test, W_E) + b_E
S_pred_test = np.dot(X_S_test, W_S) + b_S
G_pred_test = np.dot(X_G_test, W_G) + b_G

# Stack sub-model predictions into a matrix
sub_preds_test = np.column_stack((E_pred_test, S_pred_test, G_pred_test))

# ------------------------------
# Step 4: Normalize the Stacked Sub-Model Predictions
# ------------------------------
X_final_test = scaler_final.transform(sub_preds_test)

# ------------------------------
# Step 5: Compute Final ESG Predictions
# ------------------------------
y_pred_test = np.dot(X_final_test, W_final) + b_final

# ------------------------------
# Step 6: Evaluate the Model
# ------------------------------
# Calculate Mean Squared Error (MSE)
mse = np.mean((y_pred_test - y_overall_test) ** 2)
print("Mean Squared Error on test set:", mse)

# Calculate R² Score manually
ss_total = np.sum((y_overall_test - np.mean(y_overall_test)) ** 2)
ss_res = np.sum((y_overall_test - y_pred_test) ** 2)
r2_score = 1 - (ss_res / ss_total)
print("R² Score on test set:", r2_score)

# Display sample predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"Predicted ESG: {y_pred_test[i]:.2f}, Actual ESG: {y_overall_test[i]:.2f}")