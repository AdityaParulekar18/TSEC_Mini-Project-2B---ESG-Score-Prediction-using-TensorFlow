# flask_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from other origins

# Define the 16 raw feature names (order must match your training)
feature_cols = [
    'CO2_Emissions', 'Renewable_Energy', 'Water_Consumption', 'Waste_Management', 'Biodiversity_Impact',
    'Gender_Diversity', 'Employee_Satisfaction', 'Community_Investment', 'Safety_Incidents', 'Labor_Rights',
    'Board_Diversity', 'Executive_Pay_Ratio', 'Transparency', 'Shareholder_Rights', 'Anti_Corruption', 'Political_Donations'
]

# Load raw feature scaler parameters (computed during preprocessing)
with open('feature_scaler.json', 'r') as f:
    raw_scaler = json.load(f)

def scale_raw_features(raw_features):
    """Scale a (1,16) numpy array of raw features using the training scaler parameters."""
    scaled = np.empty_like(raw_features, dtype=float)
    for i, col in enumerate(feature_cols):
        mean_val = raw_scaler[col]['mean']
        std_val = raw_scaler[col]['std']
        scaled[0, i] = (raw_features[0, i] - mean_val) / std_val
    return scaled

def polynomial_features(X):
    """
    Expand X (n_samples, n_features) to degree-2 polynomial features:
      - Original features
      - Squared features
      - All pairwise interactions (for i < j)
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

# Load the trained integrated polynomial model parameters.
# The NPZ file should contain: W, b, mu_poly, sigma_poly (for polynomial features)
model_data = np.load('integrated_trained_model_poly_improved.npz', allow_pickle=True)
W = model_data['W']  # Weight vector
b = float(model_data['b'])  # Bias (scalar)
mu_poly = model_data['mu_poly']  # Mean for polynomial features (from training)
sigma_poly = model_data['sigma_poly']  # Std for polynomial features (from training)
sigma_poly[sigma_poly == 0] = 1  # Avoid division by zero

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input containing the 16 raw feature values.
        data = request.get_json(force=True)
        raw_input = np.array([[float(data.get(col, 0)) for col in feature_cols]])
        
        # Step 1: Scale the raw features.
        X_scaled = scale_raw_features(raw_input)
        
        # Step 2: Expand scaled features to polynomial features.
        X_poly = polynomial_features(X_scaled)
        
        # Step 3: Standardize the polynomial features using training parameters.
        X_poly_norm = (X_poly - mu_poly) / sigma_poly
        
        # Step 4: Compute the final ESG prediction.
        prediction = np.dot(X_poly_norm, W) + b
        
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
