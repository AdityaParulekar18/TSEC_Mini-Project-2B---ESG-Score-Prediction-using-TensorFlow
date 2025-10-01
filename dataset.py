import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of companies
n = 50000

# Environmental Parameters
co2_emissions = np.random.uniform(50, 500, n)  # Uniform distribution
renewable_energy = np.random.uniform(0, 100, n)  # %
water_consumption = np.random.uniform(0, 1000, n)  # liters per unit revenue
waste_management = np.random.uniform(0, 100, n)  # %
biodiversity_impact = np.random.uniform(0, 10, n)

# Social Parameters
gender_diversity = np.random.uniform(10, 60, n)  # %
employee_satisfaction = np.random.uniform(1, 10, n)
community_investment = np.random.uniform(0, 10, n)  # % of revenue
safety_incidents = np.random.poisson(3, n)  # average incidents per 1,000 employees
labor_rights = np.random.uniform(0, 10, n)

# Governance Parameters
board_diversity = np.random.uniform(0, 100, n)  # %
executive_pay_ratio = np.random.uniform(50, 500, n)
transparency = np.random.uniform(0, 10, n)
shareholder_rights = np.random.uniform(0, 10, n)
anti_corruption = np.random.choice([0, 1], n, p=[0.3, 0.7])  # 70% have policies
political_donations = np.random.exponential(scale=50000, size=n)  # exponential for skewness

# Define weights for each feature for synthetic ESG score generation
# (Weights are arbitrary and can be adjusted based on domain insights)
weights = {
    'co2_emissions': -0.05,
    'renewable_energy': 0.1,
    'water_consumption': -0.002,
    'waste_management': 0.05,
    'biodiversity_impact': 0.3,
    'gender_diversity': 0.05,
    'employee_satisfaction': 0.5,
    'community_investment': 0.2,
    'safety_incidents': -0.3,
    'labor_rights': 0.4,
    'board_diversity': 0.1,
    'executive_pay_ratio': -0.02,
    'transparency': 0.3,
    'shareholder_rights': 0.3,
    'anti_corruption': 0.5,
    'political_donations': -0.00001
}
intercept = 50  # base score

# Calculate synthetic ESG Score using a linear combination and add some noise
noise = np.random.normal(0, 5, n)  # noise with mean 0 and std deviation 5

esg_score = (
    intercept
    + weights['co2_emissions'] * co2_emissions
    + weights['renewable_energy'] * renewable_energy
    + weights['water_consumption'] * water_consumption
    + weights['waste_management'] * waste_management
    + weights['biodiversity_impact'] * biodiversity_impact
    + weights['gender_diversity'] * gender_diversity
    + weights['employee_satisfaction'] * employee_satisfaction
    + weights['community_investment'] * community_investment
    + weights['safety_incidents'] * safety_incidents
    + weights['labor_rights'] * labor_rights
    + weights['board_diversity'] * board_diversity
    + weights['executive_pay_ratio'] * executive_pay_ratio
    + weights['transparency'] * transparency
    + weights['shareholder_rights'] * shareholder_rights
    + weights['anti_corruption'] * anti_corruption
    + weights['political_donations'] * political_donations
    + noise
)

# Optionally, cap the ESG score within a desired range (e.g., 0 to 100)
esg_score = np.clip(esg_score, 0, 100)

# Create a DataFrame
data = pd.DataFrame({
    'CO2_Emissions': co2_emissions,
    'Renewable_Energy': renewable_energy,
    'Water_Consumption': water_consumption,
    'Waste_Management': waste_management,
    'Biodiversity_Impact': biodiversity_impact,
    'Gender_Diversity': gender_diversity,
    'Employee_Satisfaction': employee_satisfaction,
    'Community_Investment': community_investment,
    'Safety_Incidents': safety_incidents,
    'Labor_Rights': labor_rights,
    'Board_Diversity': board_diversity,
    'Executive_Pay_Ratio': executive_pay_ratio,
    'Transparency': transparency,
    'Shareholder_Rights': shareholder_rights,
    'Anti_Corruption': anti_corruption,
    'Political_Donations': political_donations,
    'ESG_Score': esg_score
})

# Check the first few rows of the dataset
print(data.head())

# Optionally, save the dataset to a CSV file for further use
data.to_csv('synthetic_esg_dataset.csv', index=False)
