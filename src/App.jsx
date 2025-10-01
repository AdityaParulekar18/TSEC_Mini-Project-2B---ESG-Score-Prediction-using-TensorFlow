// src/App.jsx
import { useState } from 'react';
import './App.css';

function App() {
  const [regressionInputs, setRegressionInputs] = useState({
    CO2_Emissions: '',
    Renewable_Energy: '',
    Water_Consumption: '',
    Waste_Management: '',
    Biodiversity_Impact: '',
    Gender_Diversity: '',
    Employee_Satisfaction: '',
    Community_Investment: '',
    Safety_Incidents: '',
    Labor_Rights: '',
    Board_Diversity: '',
    Executive_Pay_Ratio: '',
    Transparency: '',
    Shareholder_Rights: '',
    Anti_Corruption: '',
    Political_Donations: ''
  });

  const [regressionPrediction, setRegressionPrediction] = useState(null);

  // Expected ranges for each parameter
  const parameterRanges = {
    CO2_Emissions: '0 - 100 (Tons per employee)',
    Renewable_Energy: '0 - 100 (%)',
    Water_Consumption: '0 - 1000 (Liters per unit)',
    Waste_Management: '0 - 100 (Score)',
    Biodiversity_Impact: '0 - 10 (Score)',
    Gender_Diversity: '0 - 100 (%)',
    Employee_Satisfaction: '0 - 10 (Rating)',
    Community_Investment: '0 - 10 (Score)',
    Safety_Incidents: '0 - 50(Incidents/year)',
    Labor_Rights: '0 - 10 (Score)',
    Board_Diversity: '0 - 100 (%)',
    Executive_Pay_Ratio: '1 - 500 (Ratio)',
    Transparency: '0 - 10(Score)',
    Shareholder_Rights: '0 - 10 (Score)',
    Anti_Corruption: '0 - 10 (Score)',
    Political_Donations: '0 - 100000 (Score)'
  };

  const handleInputChange = (e) => {
    setRegressionInputs({
      ...regressionInputs,
      [e.target.name]: e.target.value
    });
  };

  const handleRegressionSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:3001/api/predict-regression', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(regressionInputs)
      });
      const data = await response.json();
      setRegressionPrediction(data.prediction);
    } catch (error) {
      console.error('Error fetching regression prediction:', error);
      setRegressionPrediction("Error");
    }
  };

  return (
    <div className="App">
      <h1>ESG Score Prediction</h1>
      <form onSubmit={handleRegressionSubmit}>
        <div className="form-container">
          {Object.keys(regressionInputs).map((key) => (
            <div key={key} className="input-group">
              <label>
                {key.replace(/_/g, ' ')} <br />
                <small>Expected: {parameterRanges[key]}</small>
              </label>
              <input
                type="number"
                step="any"
                name={key}
                value={regressionInputs[key]}
                onChange={handleInputChange}
                required
              />
            </div>
          ))}
        </div>
        <button type="submit">Predict ESG Score</button>
      </form>
      {regressionPrediction !== null && (
        <div className="prediction">
          <h2>Predicted ESG Score: {regressionPrediction}</h2>
        </div>
      )}
      <hr />
    </div>
  );
}

export default App;
