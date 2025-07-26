require('dotenv').config();
const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// Import routes
const queryRoutes = require('./routes/queryRoutes');
app.use('/query', queryRoutes);

app.get('/', (req, res) => {
    res.send('FastAPI-like backend is running with Express!');
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
