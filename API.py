from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib  # to load the scaler

app = Flask(__name__)
CORS(app)

# Load your trained model parameters as before
model = np.load('hierarchical_trained_model.npz', allow_pickle=True)
W_E = model['W_E']
b_E = float(model['b_E'])
W_S = model['W_S']
b_S = float(model['b_S'])
W_G = model['W_G']
b_G = float(model['b_G'])
W_final = model['W_final']
b_final = float(model['b_final'])
scaler_final = model['scaler_final'].item()

# **Load the feature scaler saved during preprocessing**
feature_scaler = joblib.load('feature_scaler.pkl')

def predict_esg(input_data):
    # First, transform the raw input features using the feature scaler.
    normalized_input = feature_scaler.transform(input_data)
    
    # Split into sub-groups (assuming the feature order is as defined)
    X_E = normalized_input[:, :5]
    X_S = normalized_input[:, 5:10]
    X_G = normalized_input[:, 10:16]
    
    # Compute sub-model predictions:
    E_pred = np.dot(X_E, W_E) + b_E
    S_pred = np.dot(X_S, W_S) + b_S
    G_pred = np.dot(X_G, W_G) + b_G
    
    # Stack sub-scores and normalize using the final scaler:
    sub_features = np.column_stack((E_pred, S_pred, G_pred))
    X_final = scaler_final.transform(sub_features)
    
    # Compute the final ESG prediction:
    esg_pred = np.dot(X_final, W_final) + b_final
    return float(esg_pred[0])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        keys = [
            'CO2_Emissions', 'Renewable_Energy', 'Water_Consumption',
            'Waste_Management', 'Biodiversity_Impact',
            'Gender_Diversity', 'Employee_Satisfaction', 'Community_Investment',
            'Safety_Incidents', 'Labor_Rights',
            'Board_Diversity', 'Executive_Pay_Ratio', 'Transparency',
            'Shareholder_Rights', 'Anti_Corruption', 'Political_Donations'
        ]
        input_values = [float(data.get(key, 0)) for key in keys]
        input_array = np.array([input_values])
        prediction = predict_esg(input_array)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
