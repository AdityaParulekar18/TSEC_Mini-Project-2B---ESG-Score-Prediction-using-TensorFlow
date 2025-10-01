import numpy as np
import pandas as pd

# Load your original synthetic dataset
df = pd.read_csv('synthetic_esg_dataset.csv')

# Compute sub-targets using the formulas
df['E_Target'] = (-0.05 * df['CO2_Emissions'] +
                  0.1 * df['Renewable_Energy'] +
                  -0.002 * df['Water_Consumption'] +
                  0.05 * df['Waste_Management'] +
                  0.3 * df['Biodiversity_Impact'])

df['S_Target'] = (0.05 * df['Gender_Diversity'] +
                  0.5 * df['Employee_Satisfaction'] +
                  0.2 * df['Community_Investment'] +
                  -0.3 * df['Safety_Incidents'] +
                  0.4 * df['Labor_Rights'])

df['G_Target'] = (0.1 * df['Board_Diversity'] +
                  -0.02 * df['Executive_Pay_Ratio'] +
                  0.3 * df['Transparency'] +
                  0.3 * df['Shareholder_Rights'] +
                  0.5 * df['Anti_Corruption'] +
                  -0.00001 * df['Political_Donations'])

# (Assuming your original overall ESG_Score was computed as below)
# You might already have this column. Otherwise, compute as:
noise = np.random.normal(0, 5, len(df))
df['ESG_Score'] = 50 + df['E_Target'] + df['S_Target'] + df['G_Target'] + noise

# Save the new dataset with sub-targets
df.to_csv('synthetic_esg_dataset_with_subtargets.csv', index=False)
