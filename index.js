// index.js
const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(express.json());
app.use(cors());

// Regression prediction endpoint: forwards requests to the Flask API.
app.post('/api/predict-regression', async (req, res) => {
    try {
        const inputData = req.body; // Contains the 16 raw feature values.
        const response = await axios.post('http://localhost:5000/predict', inputData);
        res.json(response.data);
    } catch (error) {
        console.error('Error in /api/predict-regression:', error.message);
        res.status(500).json({ error: 'Failed to get regression prediction' });
    }
});

// Placeholder for future classification endpoint.
app.post('/api/predict-classification', (req, res) => {
    res.json({ message: 'Classification prediction endpoint not yet implemented' });
});

app.listen(PORT, () => {
    console.log(`Node.js backend running on port ${PORT}`);
});
