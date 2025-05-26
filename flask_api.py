#!/usr/bin/env python3

from flask import Flask, jsonify, request
import logging
import threading

# Import the shared position tracker
from position_tracker import position_tracker

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

@app.route('/api/position', methods=['GET', 'POST'])
def handle_position():
    """Get or update position"""
    if request.method == 'GET':
        # Return current position
        return jsonify(position_tracker.get_position())
    
    elif request.method == 'POST':
        # Update position with data from request
        try:
            data = request.json or {}
            
            if 'x' not in data or 'y' not in data:
                return jsonify({
                    'success': False, 
                    'message': 'Missing required position data (x, y)'
                }), 400
            
            x = float(data['x'])
            y = float(data['y'])
            heading = float(data.get('heading', position_tracker.heading))
            
            position_tracker.update_position(x, y, heading)
            
            return jsonify({
                'success': True,
                'position': position_tracker.get_position()
            })
            
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'message': f'Invalid position data: {str(e)}'
            }), 400

@app.route('/api/reset', methods=['POST'])
def reset_position():
    """Reset robot position to origin"""
    position_tracker.reset_position()
    
    return jsonify({
        'success': True,
        'message': 'Position reset to origin',
        'position': position_tracker.get_position()
    })

@app.route('/api/start', methods=['GET', 'POST'])
def handle_start_flag():
    """Get or set the start flag"""
    if request.method == 'GET':
        # Return current start flag
        return jsonify({
            'start': position_tracker.get_start_flag()
        })
    
    elif request.method == 'POST':
        # Set start flag with data from request
        try:
            data = request.json or {}
            
            if 'start' not in data:
                return jsonify({
                    'success': False, 
                    'message': 'Missing required start flag'
                }), 400
            
            start_value = bool(data['start'])
            position_tracker.set_start_flag(start_value)
            
            return jsonify({
                'success': True,
                'start': position_tracker.get_start_flag()
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error setting start flag: {str(e)}'
            }), 400

@app.route('/api/initial-position', methods=['GET'])
def get_initial_position():
    """Get the initial position"""
    initial_x, initial_y, initial_heading = position_tracker.get_initial_position()
    return jsonify({
        'x': initial_x,
        'y': initial_y,
        'heading': initial_heading
    })

@app.route('/api/finished', methods=['GET'])
def get_finished_flag():
    """Get the finished flag"""
    return jsonify({
        'finished': position_tracker.get_finished_flag()
    })

@app.route('/api/obstacles', methods=['GET'])
def get_obstacles():
    """Get list of detected obstacles"""
    # This would need to be implemented with actual obstacle tracking
    # For now, return empty list as placeholder
    return jsonify({
        'obstacles': []
    })

@app.route('/api/map', methods=['GET'])
def get_map_data():
    """Get current map data"""
    # This would need to be implemented with actual map data
    # For now, return basic info
    return jsonify({
        'grid_size': 50,
        'bounds': {
            'min_x': -500,
            'max_x': 500,
            'min_y': -500,
            'max_y': 500
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'turtlebot-position-api'
    })

@app.errorhandler(Exception)
def handle_error(e):
    """Global error handler"""
    logger.error(f"Error: {str(e)}")
    return jsonify({
        'success': False,
        'message': f'Error: {str(e)}'
    }), 500


class FlaskAPIServer:
    """Flask API server that can be started and stopped"""
    
    def __init__(self, host='0.0.0.0', port=5001):
        self.host = host
        self.port = port
        self.thread = None
        self.server = None
        
    def start(self):
        """Start the Flask server in a separate thread"""
        def run_server():
            logger.info(f"Starting Flask API server on {self.host}:{self.port}")
            # Use werkzeug server directly for better control
            from werkzeug.serving import make_server
            self.server = make_server(self.host, self.port, app, threaded=True)
            self.server.serve_forever()
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        logger.info(f"Flask API server started on http://{self.host}:{self.port}")
        
    def stop(self):
        """Stop the Flask server"""
        if self.server:
            logger.info("Stopping Flask API server...")
            self.server.shutdown()
            self.server = None
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        logger.info("Flask API server stopped")
        
    def is_running(self):
        """Check if the server is running"""
        return self.thread is not None and self.thread.is_alive()


# Create a global instance that can be imported
flask_api_server = FlaskAPIServer()
