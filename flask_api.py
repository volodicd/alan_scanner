#!/usr/bin/env python3

from flask import Flask, jsonify, request
import logging

# Import the shared position tracker
from position_tracker import position_tracker

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

@app.errorhandler(Exception)
def handle_error(e):
    """Global error handler"""
    logger.error(f"Error: {str(e)}")
    return jsonify({
        'success': False,
        'message': f'Error: {str(e)}'
    }), 500

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=5001)
