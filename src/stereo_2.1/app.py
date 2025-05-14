import os
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
import logging
from logging.handlers import RotatingFileHandler  # Import the specific handler

from app_context import app_ctx
from utils import ensure_app_directories

# Get configuration from environment with secure defaults
secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())
cors_origins = os.environ.get('CORS_ORIGINS', '*')
debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(  # Use the imported handler directly
            "stereo_vision_app.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reduce verbosity for third-party libraries
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING)
logging.getLogger('socketio').setLevel(logging.WARNING)

# Create necessary directories
ensure_app_directories()

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = secret_key

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins=cors_origins, async_mode='threading')
app_ctx.init_socketio(socketio)

# Register all routes
from routes import register_all_routes
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
    logger.info("Starting Stereo Vision Web Interface in %s mode",
                "debug" if debug_mode else "production")
    socketio.run(app, host='0.0.0.0', port=8080, debug=debug_mode)