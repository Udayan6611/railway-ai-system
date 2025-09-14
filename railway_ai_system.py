from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'railway_ai_secret_key'
CORS(app, origins=["*"])

# Custom JSON encoder for Enums and complex objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

# Data Models (simplified to avoid serialization issues)
TRAIN_TYPES = {
    'EXPRESS': {'priority': 1, 'speed': 80, 'color': '#ef4444'},
    'PASSENGER': {'priority': 2, 'speed': 60, 'color': '#3b82f6'},
    'FREIGHT': {'priority': 3, 'speed': 40, 'color': '#10b981'},
    'SUBURBAN': {'priority': 2, 'speed': 70, 'color': '#f59e0b'}
}

@dataclass
class Station:
    id: str
    name: str
    x: float
    y: float
    capacity: int = 3
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'x': self.x,
            'y': self.y,
            'capacity': self.capacity
        }

@dataclass
class Track:
    from_station: str
    to_station: str
    distance: float
    capacity: int
    current_occupancy: int = 0
    
    def to_dict(self):
        return {
            'from_station': self.from_station,
            'to_station': self.to_station,
            'distance': self.distance,
            'capacity': self.capacity,
            'current_occupancy': self.current_occupancy
        }

@dataclass
class Train:
    id: str
    train_type: str  # Changed from Enum to string
    from_station: str
    to_station: str
    scheduled_departure: float
    current_position: str
    delay: float = 0.0
    status: str = 'scheduled'  # Changed from Enum to string
    route: List[str] = None
    estimated_arrival: float = 0.0
    
    def to_dict(self):
        return {
            'id': self.id,
            'train_type': self.train_type,
            'from_station': self.from_station,
            'to_station': self.to_station,
            'scheduled_departure': self.scheduled_departure,
            'current_position': self.current_position,
            'delay': self.delay,
            'status': self.status,
            'route': self.route or [],
            'estimated_arrival': self.estimated_arrival
        }

@dataclass
class Conflict:
    id: str
    conflict_type: str
    location: str
    trains_involved: List[str]
    severity: str
    timestamp: float
    
    def to_dict(self):
        return {
            'id': self.id,
            'conflict_type': self.conflict_type,
            'location': self.location,
            'trains_involved': self.trains_involved,
            'severity': self.severity,
            'timestamp': self.timestamp
        }

@dataclass
class AIDecision:
    decision_id: str
    action: str
    target_train: str
    reason: str
    confidence: float
    timestamp: float
    impact_score: float
    
    def to_dict(self):
        return {
            'decision_id': self.decision_id,
            'action': self.action,
            'target_train': self.target_train,
            'reason': self.reason,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'impact_score': self.impact_score
        }

class RailwayAISystem:
    def __init__(self):
        self.stations = self._initialize_stations()
        self.tracks = self._initialize_tracks()
        self.trains = {}
        self.conflicts = {}
        self.decisions = []
        self.current_time = 8.0  # 8:00 AM
        self.is_running = False
        self.metrics = {
            'on_time_performance': 100.0,
            'average_delay': 0.0,
            'throughput': 0,
            'active_conflicts': 0,
            'total_decisions': 0
        }
        
        # Initialize sample trains
        self._initialize_sample_trains()

    def _initialize_stations(self) -> Dict[str, Station]:
        stations_data = [
            ("A", "Mumbai Central", 100, 200),
            ("B", "Dadar", 200, 200),
            ("C", "Kurla", 300, 200),
            ("D", "Thane", 400, 200),
            ("E", "Kalyan", 500, 200),
            ("F", "Nashik", 350, 100),
            ("G", "Pune", 350, 300)
        ]
        return {s_id: Station(s_id, name, x, y) for s_id, name, x, y in stations_data}

    def _initialize_tracks(self) -> List[Track]:
        tracks_data = [
            ("A", "B", 10, 2), ("B", "C", 8, 2), ("C", "D", 12, 1),
            ("D", "E", 15, 2), ("C", "F", 20, 1), ("C", "G", 18, 1)
        ]
        return [Track(f, t, d, c) for f, t, d, c in tracks_data]

    def _initialize_sample_trains(self):
        sample_trains = [
            ("T001", "EXPRESS", "A", "E", 8.0),
            ("T002", "PASSENGER", "E", "A", 8.5),
            ("T003", "FREIGHT", "C", "F", 9.0),
            ("T004", "SUBURBAN", "A", "D", 8.2),
            ("T005", "EXPRESS", "G", "B", 8.8),
        ]
        
        for train_id, t_type, from_st, to_st, departure in sample_trains:
            route = self._find_simple_route(from_st, to_st)
            self.trains[train_id] = Train(
                id=train_id,
                train_type=t_type,
                from_station=from_st,
                to_station=to_st,
                scheduled_departure=departure,
                current_position=from_st,
                route=route
            )

    def _find_simple_route(self, from_station: str, to_station: str) -> List[str]:
        """Simple route finding"""
        # Simplified routing logic
        routes = {
            ('A', 'E'): ['A', 'B', 'C', 'D', 'E'],
            ('E', 'A'): ['E', 'D', 'C', 'B', 'A'],
            ('C', 'F'): ['C', 'F'],
            ('A', 'D'): ['A', 'B', 'C', 'D'],
            ('G', 'B'): ['G', 'C', 'B']
        }
        return routes.get((from_station, to_station), [from_station, to_station])

    def start_simulation(self):
        """Start the railway simulation"""
        self.is_running = True
        logger.info("Railway simulation started")
        
        def simulation_loop():
            while self.is_running:
                self.update_simulation()
                time.sleep(1)  # Update every second
        
        simulation_thread = threading.Thread(target=simulation_loop)
        simulation_thread.daemon = True
        simulation_thread.start()

    def stop_simulation(self):
        """Stop the railway simulation"""
        self.is_running = False
        logger.info("Railway simulation stopped")

    def update_simulation(self):
        """Main simulation update loop"""
        self.current_time += 0.1  # Advance time by 6 minutes
        
        # Update train positions and statuses
        self._update_trains()
        
        # Detect conflicts
        conflicts = self._detect_conflicts()
        
        # Make AI decisions for conflicts
        if conflicts:
            decisions = self._make_ai_decisions(conflicts)
            self._apply_decisions(decisions)
        
        # Update metrics
        self._update_metrics()

    def _update_trains(self):
        """Update train positions and statuses"""
        for train in self.trains.values():
            if train.status == 'scheduled':
                if self.current_time >= train.scheduled_departure + train.delay:
                    train.status = 'moving'
                    logger.info(f"Train {train.id} started moving")
            
            elif train.status == 'moving':
                # Simulate movement along route
                if random.random() < 0.1:  # 10% chance to advance
                    if train.route and len(train.route) > 1:
                        current_idx = train.route.index(train.current_position)
                        if current_idx < len(train.route) - 1:
                            train.current_position = train.route[current_idx + 1]
                            if train.current_position == train.to_station:
                                train.status = 'completed'
                                logger.info(f"Train {train.id} completed journey")

    def _detect_conflicts(self) -> List[Conflict]:
        """Detect potential conflicts between trains"""
        conflicts = []
        active_trains = [t for t in self.trains.values() if t.status == 'moving']
        
        # Simple conflict detection
        if len(active_trains) > 1 and random.random() < 0.05:  # 5% chance
            train1, train2 = random.sample(active_trains, 2)
            conflict_id = f"conflict_{train1.id}_{train2.id}_{int(self.current_time)}"
            
            conflict = Conflict(
                id=conflict_id,
                conflict_type="track_capacity",
                location=f"{train1.current_position}-next",
                trains_involved=[train1.id, train2.id],
                severity="medium",
                timestamp=self.current_time
            )
            conflicts.append(conflict)
            self.conflicts[conflict_id] = conflict
        
        return conflicts

    def _make_ai_decisions(self, conflicts: List[Conflict]) -> List[AIDecision]:
        """Make AI-powered decisions for conflict resolution"""
        decisions = []
        
        for conflict in conflicts:
            train1_id = conflict.trains_involved[0]
            train2_id = conflict.trains_involved[1]
            
            train1 = self.trains[train1_id]
            train2 = self.trains[train2_id]
            
            # Simple AI decision: delay the lower priority train
            priority1 = TRAIN_TYPES[train1.train_type]['priority']
            priority2 = TRAIN_TYPES[train2.train_type]['priority']
            
            if priority1 < priority2:  # Lower number = higher priority
                target = train2_id
                reason = f"Delay {train2_id} - {train1_id} has higher priority"
            else:
                target = train1_id
                reason = f"Delay {train1_id} - {train2_id} has higher priority"
            
            decision = AIDecision(
                decision_id=f"decision_{len(self.decisions)}_{int(self.current_time)}",
                action="delay",
                target_train=target,
                reason=reason,
                confidence=0.85,
                timestamp=self.current_time,
                impact_score=0.7
            )
            
            decisions.append(decision)
            self.decisions.append(decision)
        
        return decisions

    def _apply_decisions(self, decisions: List[AIDecision]):
        """Apply AI decisions to trains"""
        for decision in decisions:
            train = self.trains[decision.target_train]
            
            if decision.action == "delay":
                delay_amount = 0.25  # 15 minutes
                train.delay += delay_amount
                train.status = 'waiting'
                logger.info(f"Applied delay of {delay_amount*60} minutes to train {train.id}")

    def _update_metrics(self):
        """Update system performance metrics"""
        total_trains = len(self.trains)
        on_time_trains = sum(1 for t in self.trains.values() if t.delay == 0)
        total_delay = sum(t.delay for t in self.trains.values())
        completed_trains = sum(1 for t in self.trains.values() if t.status == 'completed')
        
        self.metrics = {
            'on_time_performance': (on_time_trains / total_trains * 100) if total_trains > 0 else 100,
            'average_delay': (total_delay / total_trains * 60) if total_trains > 0 else 0,  # in minutes
            'throughput': completed_trains,
            'active_conflicts': len([c for c in self.conflicts.values() if self.current_time - c.timestamp < 1.0]),
            'total_decisions': len(self.decisions)
        }

    def add_disruption(self, disruption_type: str, location: str = None, train_id: str = None):
        """Add external disruption to the system"""
        if train_id and train_id in self.trains:
            train = self.trains[train_id]
            train.delay += random.uniform(0.25, 0.75)
            train.status = 'delayed'
            logger.info(f"Added disruption to train {train_id}: {disruption_type}")

    def get_system_status(self):
        """Get current system status - with proper serialization"""
        return {
            'current_time': self.current_time,
            'is_running': self.is_running,
            'trains': {t_id: train.to_dict() for t_id, train in self.trains.items()},
            'stations': {s_id: station.to_dict() for s_id, station in self.stations.items()},
            'conflicts': {c_id: conflict.to_dict() for c_id, conflict in self.conflicts.items()},
            'recent_decisions': [d.to_dict() for d in self.decisions[-5:]],
            'metrics': self.metrics
        }

# Global railway system instance
railway_system = RailwayAISystem()

# API Routes
@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the railway simulation"""
    try:
        railway_system.start_simulation()
        return jsonify({'status': 'started', 'message': 'Railway simulation started'})
    except Exception as e:
        logger.error(f"Error starting system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop the railway simulation"""
    try:
        railway_system.stop_simulation()
        return jsonify({'status': 'stopped', 'message': 'Railway simulation stopped'})
    except Exception as e:
        logger.error(f"Error stopping system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status"""
    try:
        return jsonify(railway_system.get_system_status())
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/disruption', methods=['POST'])
def add_disruption():
    """Add disruption to the system"""
    try:
        data = request.json or {}
        railway_system.add_disruption(
            disruption_type=data.get('type', 'unknown'),
            location=data.get('location'),
            train_id=data.get('train_id')
        )
        return jsonify({'status': 'success', 'message': 'Disruption added'})
    except Exception as e:
        logger.error(f"Error adding disruption: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trains', methods=['GET'])
def get_trains():
    """Get all trains information"""
    try:
        return jsonify({t_id: train.to_dict() for t_id, train in railway_system.trains.items()})
    except Exception as e:
        logger.error(f"Error getting trains: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conflicts', methods=['GET'])
def get_conflicts():
    """Get active conflicts"""
    try:
        active_conflicts = {
            c_id: conflict.to_dict() 
            for c_id, conflict in railway_system.conflicts.items()
            if railway_system.current_time - conflict.timestamp < 1.0
        }
        return jsonify(active_conflicts)
    except Exception as e:
        logger.error(f"Error getting conflicts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/decisions', methods=['GET'])
def get_decisions():
    """Get recent AI decisions"""
    try:
        return jsonify([d.to_dict() for d in railway_system.decisions[-10:]])
    except Exception as e:
        logger.error(f"Error getting decisions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get system performance metrics"""
    try:
        return jsonify(railway_system.metrics)
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_conflicts():
    """Get conflict predictions"""
    try:
        # Simple prediction simulation
        predictions = []
        for train_id in railway_system.trains.keys():
            if random.random() < 0.3:  # 30% chance
                predictions.append({
                    'train_id': train_id,
                    'probability': round(random.uniform(0.7, 0.9), 2),
                    'predicted_time': railway_system.current_time + random.uniform(0.5, 1.5),
                    'predicted_location': railway_system.trains[train_id].current_position
                })
        return jsonify(predictions)
    except Exception as e:
        logger.error(f"Error predicting conflicts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    global railway_system
    railway_system = RailwayAISystem()  # Reset entire system state
    return jsonify({'message': 'Simulation state reset successfully'}), 200

if __name__ == '__main__':
    logger.info("Starting Fixed Railway AI Decision Support System")
    logger.info("Access the system at http://localhost:5000")
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)