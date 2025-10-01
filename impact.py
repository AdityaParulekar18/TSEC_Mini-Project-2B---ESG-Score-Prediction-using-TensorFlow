import numpy as np
import pandas as pd

# Load the existing synthetic ESG dataset
file_path = "synthetic_esg_dataset_with_subtargets.csv"
df = pd.read_csv(file_path)

# Display the first few rows to analyze the structure
df.head()


# Create a copy of the dataset to modify
df_simulated = df.copy()

# Introduce random impact factors (positive or negative) to simulate policy changes
np.random.seed(42)
impact_factors = np.random.uniform(-10, 10, size=(df.shape[0], df.shape[1] - 1))  # Excluding ESG Score

# Apply impact factors to simulate ESG policy changes
df_simulated.iloc[:, :-1] += impact_factors  

# Ensure values remain within realistic bounds (e.g., no negative emissions)
df_simulated[df_simulated < 0] = 0

# Simulate ESG score changes based on weighted feature impact (simple linear impact model)
weights = np.random.uniform(0.1, 1.0, size=df.shape[1] - 1)  # Generate random weights for each feature
df_simulated["Simulated_ESG_Score"] = np.dot(df_simulated.iloc[:, :-1], weights) / np.sum(weights) * 100

# Normalize simulated ESG score between 0-100
df_simulated["Simulated_ESG_Score"] = np.clip(df_simulated["Simulated_ESG_Score"], 0, 100)

# Display sample of new dataset
df_simulated.head()
import pandas as pd
import numpy as np

# Define the number of companies
num_companies = 1000

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic ESG scores (0-100)
environmental_score = np.random.uniform(20, 100, num_companies)
social_score = np.random.uniform(20, 100, num_companies)
governance_score = np.random.uniform(20, 100, num_companies)

# Compute overall ESG score (weighted sum with some randomness)
overall_esg_score = (
    0.4 * environmental_score + 
    0.35 * social_score + 
    0.25 * governance_score + 
    np.random.uniform(-5, 5, num_companies)  # Adding slight variation
)

# Generate potential impact factors (randomized changes)
carbon_emission_reduction = np.random.uniform(-10, 10, num_companies)  # % reduction
renewable_energy_adoption = np.random.uniform(0, 100, num_companies)  # % energy used
employee_satisfaction = np.random.uniform(0, 100, num_companies)  # % satisfaction
board_diversity = np.random.uniform(0, 100, num_companies)  # % diversity in board
regulatory_compliance = np.random.uniform(0, 100, num_companies)  # % compliance score

# Simulate the impact of these factors on ESG scores
simulated_esg_impact = (
    overall_esg_score +
    0.2 * carbon_emission_reduction + 
    0.3 * renewable_energy_adoption +
    0.2 * employee_satisfaction +
    0.15 * board_diversity +
    0.15 * regulatory_compliance +
    np.random.uniform(-3, 3, num_companies)  # Adding slight noise
)

# Create a DataFrame
esg_impact_data = pd.DataFrame({
    "Company_ID": range(1, num_companies + 1),
    "Environmental_Score": environmental_score,
    "Social_Score": social_score,
    "Governance_Score": governance_score,
    "Overall_ESG_Score": overall_esg_score,
    "Carbon_Emission_Reduction": carbon_emission_reduction,
    "Renewable_Energy_Adoption": renewable_energy_adoption,
    "Employee_Satisfaction": employee_satisfaction,
    "Board_Diversity": board_diversity,
    "Regulatory_Compliance": regulatory_compliance,
    "Simulated_ESG_Impact_Score": simulated_esg_impact
})

# Save to CSV
file_path = "simulated_esg_impact_dataset.csv"
esg_impact_data.to_csv(file_path, index=False)
print(f"Dataset saved as {file_path}")
import pandas as pd
import numpy as np

# Define the number of companies
num_companies = 1000

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic ESG scores (0-100)
environmental_score = np.random.uniform(20, 100, num_companies)
social_score = np.random.uniform(20, 100, num_companies)
governance_score = np.random.uniform(20, 100, num_companies)

# Compute overall ESG score (weighted sum with some randomness)
overall_esg_score = (
    0.4 * environmental_score + 
    0.35 * social_score + 
    0.25 * governance_score + 
    np.random.uniform(-5, 5, num_companies)  # Adding slight variation
)

# Generate potential impact factors
carbon_emission_reduction = np.random.uniform(-10, 10, num_companies)  
renewable_energy_adoption = np.random.uniform(0, 100, num_companies)  
employee_satisfaction = np.random.uniform(0, 100, num_companies)  
board_diversity = np.random.uniform(0, 100, num_companies)  
regulatory_compliance = np.random.uniform(0, 100, num_companies)  

# **Check for missing values in any columns**
impact_factors = [
    carbon_emission_reduction, renewable_energy_adoption, 
    employee_satisfaction, board_diversity, regulatory_compliance
]
if any(np.isnan(factor).any() for factor in impact_factors):
    print("Warning: NaN values detected in impact factor columns!")

# **Calculate Simulated ESG Impact Score correctly**
simulated_esg_impact = (
    overall_esg_score +
    0.2 * carbon_emission_reduction + 
    0.3 * renewable_energy_adoption +
    0.2 * employee_satisfaction +
    0.15 * board_diversity +
    0.15 * regulatory_compliance +
    np.random.uniform(-3, 3, num_companies)  # Adding slight noise
)

# **Check if values are being computed properly**
print("Simulated ESG Impact Score Sample:\n", simulated_esg_impact[:5])

# Create a DataFrame
esg_impact_data = pd.DataFrame({
    "Company_ID": range(1, num_companies + 1),
    "Environmental_Score": environmental_score,
    "Social_Score": social_score,
    "Governance_Score": governance_score,
    "Overall_ESG_Score": overall_esg_score,
    "Carbon_Emission_Reduction": carbon_emission_reduction,
    "Renewable_Energy_Adoption": renewable_energy_adoption,
    "Employee_Satisfaction": employee_satisfaction,
    "Board_Diversity": board_diversity,
    "Regulatory_Compliance": regulatory_compliance,
    "Simulated_ESG_Impact_Score": simulated_esg_impact
})

# Save to CSV
file_path = "simulated_esg_impact_dataset.csv"
esg_impact_data.to_csv(file_path, index=False)
print(f"Dataset saved as {file_path}")

# Verify final dataset
print(esg_impact_data.head())

