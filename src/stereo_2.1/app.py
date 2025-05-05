# app.py
import os
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

from config import logger
from utils import init_socketio
from routes import register_all_routes

# Create directories if they don't exist
os.makedirs('static/captures', exist_ok=True)
os.makedirs('static/calibration', exist_ok=True)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'stereo_vision_secret_key'

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
init_socketio(socketio)

# Register all routes
register_all_routes(app)

# Basic route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Run the application
if __name__ == '__main__':
    logger.info("Starting Stereo Vision Web Interface")
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)