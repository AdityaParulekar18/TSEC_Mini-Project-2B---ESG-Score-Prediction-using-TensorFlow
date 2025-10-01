# integrated_preprocess.py
import pandas as pd
import numpy as np
import json

# ------------------------------
# Step 1: Load the Dataset
# ------------------------------
df = pd.read_csv('synthetic_esg_dataset_with_subtargets.csv')

# For the integrated model we use only the 16 raw feature columns and overall target.
feature_cols = [
    'CO2_Emissions', 'Renewable_Energy', 'Water_Consumption', 'Waste_Management', 'Biodiversity_Impact',
    'Gender_Diversity', 'Employee_Satisfaction', 'Community_Investment', 'Safety_Incidents', 'Labor_Rights',
    'Board_Diversity', 'Executive_Pay_Ratio', 'Transparency', 'Shareholder_Rights', 'Anti_Corruption', 'Political_Donations'
]

# ------------------------------
# Step 2: Manual Train–Test Split (Training set only for computing scaling parameters)
# ------------------------------
indices = np.arange(len(df))
np.random.shuffle(indices)
test_size = int(0.2 * len(df))
test_indices = indices[:test_size]
train_indices = indices[test_size:]

train_df = df.iloc[train_indices].reset_index(drop=True)
test_df = df.iloc[test_indices].reset_index(drop=True)

# ------------------------------
# Step 3: Compute Scaling Parameters on Training Data and Scale Training Data
# ------------------------------
scaler_params = {}
for col in feature_cols:
    mean_val = train_df[col].mean()
    std_val = train_df[col].std()
    scaler_params[col] = {'mean': mean_val, 'std': std_val}
    # Scale training data column
    train_df[col] = (train_df[col] - mean_val) / std_val

# Save the training scaler parameters to JSON
with open('feature_scaler.json', 'w') as f:
    json.dump(scaler_params, f)
print("Training scaler parameters saved as 'feature_scaler.json'.")


for col in feature_cols:
    mean_val = scaler_params[col]['mean']
    std_val = scaler_params[col]['std']
    test_df[col] = (test_df[col] - mean_val) / std_val


# Save training and test sets as CSV files (optional)
train_df.to_csv('train_data_integrated.csv', index=False)
test_df.to_csv('test_data_integrated.csv', index=False)
print("Training and test data saved as 'train_data_integrated.csv' and 'test_data_integrated.csv'.")

# Extract training features and overall target
X_train = train_df[feature_cols].values
y_train = train_df['ESG_Score'].values

# Save training data as NPZ for model training
np.savez('preprocessed_integrated_train.npz', X_train=X_train, y_train=y_train)
print("Preprocessed training data saved to 'preprocessed_integrated_train.npz'.")
