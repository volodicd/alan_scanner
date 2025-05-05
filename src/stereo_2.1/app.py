# app.py
import os
# In app.py - after creating socketio
import utils

import socket
from flask import Flask
from flask_socketio import SocketIO
from werkzeug.serving import run_simple
from config import logger

# Import route modules
from routes.main_routes import register_main_routes
from routes.api_routes import register_api_routes
from routes.calibration_routes import register_calibration_routes
from routes.code_routes import register_code_routes
from routes.disparity_routes import register_disparity_routes
from routes.logging_routes import register_logging_routes
from routes.mapping_routes import register_mapping_routes

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'stereo_vision_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
utils.init_socketio(socketio)
# Register all routes
register_main_routes(app)
register_api_routes(app)
register_calibration_routes(app, socketio)
register_code_routes(app)
register_disparity_routes(app)
register_logging_routes(app)
register_mapping_routes(app)

# Run the application
if __name__ == '__main__':
    # Make sure the stereo_vision.py file exists
    if not os.path.exists('stereo_vision.py'):
        logger.error("stereo_vision.py not found")
        print("Error: stereo_vision.py not found. Please create it first.")
        exit(1)

    # Start the Flask-SocketIO server
    hostname = socket.gethostname()
    logger.info("Starting Stereo Vision Web Interface")
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)