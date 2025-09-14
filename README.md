# Railway AI Decision Support System

Railway AI Decision Support System is a smart solution built to assist Indian Railways with efficient train scheduling and conflict resolution. It blends machine learning, operations research, and real-time optimization to intelligently manage train movements and avoid conflicts.

![Railway AI System Demo](https://via.placeholder.com/800x400/667eea/ffffff?text=Railway+AI+Decision+Support+System)

## Features

### Intelligent AI Decision-Making
Our system leverages neural network models powered by TensorFlow to predict and resolve conflicts, using real-time optimization through OR-Tools. With explainable AI features, you get clear reasoning and confidence scores for every decision.

### Smooth Railway Operations
Manage different train types – Express, Passenger, Freight, and Suburban – all in one place. The system simulates real-time train movement and dynamically handles scheduling, automatic conflict resolution, and unexpected disruptions like bad weather or signal failures.

### Interactive Dashboard
Monitor the entire railway network visually. The live dashboard shows real-time train positions, performance metrics such as delays and throughput, and AI decision explanations. It works perfectly across devices, including mobiles.

## System Architecture

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Frontend    │    │ Node.js API   │    │ Python AI    │
│ (HTML/CSS/JS)│ ↔ │ (Express)     │ ↔ │ (Flask)      │
└───────────────┘    └───────────────┘    └───────────────┘
      │                   │                  │
      ▼                   ▼                  ▼
Interactive UI       API Routing      AI/ML Models & Simulations
Real-time Updates   WebSocket Proxy  Decision Engine
Train Visualization Static Files     Optimization Algorithms
```

## Getting Started

### Prerequisites
- Python 3.8 or higher  
- Node.js 14 or higher  
- Git installed on your system

### Steps to Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/railway-ai-system.git
   cd railway-ai-system
   ```

2. Set up the Backend:
   ```bash
   cd backend
   python -m venv railway_env
   source railway_env/bin/activate  # For Windows use railway_env\Scripts\activate
   pip install -r requirements.txt
   python railway_ai_system.py
   ```

3. Set up the Frontend:
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. Open your browser and visit:  
   http://localhost:3000

## How to Use

- Start the simulation with a click on "Start Simulation"  
- Track train movements live on the network map  
- View AI decisions as they happen with detailed explanations  
- Test how the system handles disruptions  
- Analyze key performance metrics over time

## AI Models Explained

- **Decision Neural Network**:  
  Uses a deep neural network to decide between delaying or rerouting trains based on priority, delay, and conflict data.  
  Accuracy: ~85%

- **Conflict Predictor**:  
  Predicts potential conflicts up to two hours in advance using time-series data and LSTM models.

- **Route Optimizer**:  
  Solves complex scheduling problems with OR-Tools CP-SAT solver, ensuring minimal delays and maximizing throughput.

## Project Structure

```
railway-ai-system/
├── railway_ai_system.py      # Core backend logic
├── requirements.txt          # Backend dependencies
├── frontend/
│   ├── server.js             # Node.js API gateway
│   ├── package.json          # Node dependencies
│   └── public/
│       └── index.html        # Frontend interface
└── README.md                  # This documentation
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/status | GET | Get current system status |
| /api/start | POST | Start the simulation |
| /api/stop | POST | Stop the simulation |
| /api/trains | GET | Fetch all trains info |
| /api/conflicts | GET | List active conflicts |
| /api/decisions | GET | Show recent AI decisions |
| /api/metrics | GET | System performance metrics |
| /api/disruption | POST | Add disruption scenario |
| /api/predict | POST | Predict future conflicts |

## Performance Metrics

- On-Time Performance: Percentage of trains running as scheduled  
- Average Delay: Mean delay across all trains (in minutes)  
- Throughput: Number of successfully completed journeys  
- Conflict Resolution Time: Time AI takes to resolve conflicts  
- System Availability: Uptime and reliability

## Smart India Hackathon

This project is developed to solve real challenges faced by Indian Railways in train scheduling and conflict management, delivering:  
- Real-time AI-based decision making  
- Scalable microservices design  
- Clear, explainable AI decisions  
- User-friendly interface  
