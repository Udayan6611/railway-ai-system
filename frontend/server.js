// Node.js API Gateway for Railway AI System
const express = require('express');
const http = require('http');
const path = require('path');
const cors = require('cors');
const axios = require('axios');
const socketIo = require('socket.io');
const socketClient = require('socket.io-client');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const PYTHON_API_URL = 'http://localhost:5000';
const PORT = 3000;

const pythonSocket = socketClient(PYTHON_API_URL);

pythonSocket.on('connect', () => {
    console.log('Connected to Python backend');
});

pythonSocket.on('system_update', (data) => {
    io.emit('system_update', data);
});

pythonSocket.on('disconnect', () => {
    console.log('Disconnected from Python backend');
});

app.get('/api/status', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_API_URL}/api/status`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching status:', error.message);
        res.status(500).json({ error: 'Failed to fetch system status' });
    }
});

app.post('/api/start', async (req, res) => {
    try {
        const response = await axios.post(`${PYTHON_API_URL}/api/start`);
        res.json(response.data);
    } catch (error) {
        console.error('Error starting system:', error.message);
        res.status(500).json({ error: 'Failed to start system' });
    }
});

app.post('/api/stop', async (req, res) => {
    try {
        const response = await axios.post(`${PYTHON_API_URL}/api/stop`);
        res.json(response.data);
    } catch (error) {
        console.error('Error stopping system:', error.message);
        res.status(500).json({ error: 'Failed to stop system' });
    }
});

app.post('/api/disruption', async (req, res) => {
    try {
        const response = await axios.post(`${PYTHON_API_URL}/api/disruption`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error adding disruption:', error.message);
        res.status(500).json({ error: 'Failed to add disruption' });
    }
});

app.get('/api/status', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_API_URL}/api/status`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching status:', error.message);
        res.status(500).json({ error: 'Failed to fetch system status' });
    }
});

app.post('/api/start', async (req, res) => {
    try {
        const response = await axios.post(`${PYTHON_API_URL}/api/start`);
        res.json(response.data);
    } catch (error) {
        console.error('Error starting system:', error.message);
        res.status(500).json({ error: 'Failed to start system' });
    }
});

app.post('/api/stop', async (req, res) => {
    try {
        const response = await axios.post(`${PYTHON_API_URL}/api/stop`);
        res.json(response.data);
    } catch (error) {
        console.error('Error stopping system:', error.message);
        res.status(500).json({ error: 'Failed to stop system' });
    }
});

app.post('/api/disruption', async (req, res) => {
    try {
        const response = await axios.post(`${PYTHON_API_URL}/api/disruption`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error adding disruption:', error.message);
        res.status(500).json({ error: 'Failed to add disruption' });
    }
});

app.get('/api/trains', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_API_URL}/api/trains`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching trains:', error.message);
        res.status(500).json({ error: 'Failed to fetch trains data' });
    }
});

app.get('/api/conflicts', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_API_URL}/api/conflicts`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching conflicts:', error.message);
        res.status(500).json({ error: 'Failed to fetch conflicts data' });
    }
});

app.get('/api/decisions', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_API_URL}/api/decisions`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching decisions:', error.message);
        res.status(500).json({ error: 'Failed to fetch decisions data' });
    }
});

app.get('/api/metrics', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_API_URL}/api/metrics`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching metrics:', error.message);
        res.status(500).json({ error: 'Failed to fetch metrics data' });
    }
});

app.post('/api/predict', async (req, res) => {
    try {
        const response = await axios.post(`${PYTHON_API_URL}/api/predict`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error getting predictions:', error.message);
        res.status(500).json({ error: 'Failed to get conflict predictions' });
    }
});

app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        services: {
            nodeServer: 'running',
            pythonBackend: pythonSocket.connected ? 'connected' : 'disconnected'
        }
    });
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);
    
    socket.on('request_update', () => {
        pythonSocket.emit('request_update');
    });
    
    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
    });
});

app.use((err, req, res, next) => {
    console.error('Server error:', err.stack);
    res.status(500).json({ error: 'Internal server error' });
});

server.listen(PORT, () => {
    console.log(`Node.js API Gateway running on port ${PORT}`);
    console.log(`Frontend available at http://localhost:${PORT}`);
    console.log(`Connecting to Python backend at ${PYTHON_API_URL}`);
});

process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down gracefully');
    pythonSocket.disconnect();
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});

process.on('SIGINT', () => {
    console.log('SIGINT received, shutting down gracefully');
    pythonSocket.disconnect();
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});

module.exports = app;