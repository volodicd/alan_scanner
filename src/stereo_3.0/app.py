from flask import Flask
import logging
from logging.handlers import RotatingFileHandler
import os
from routes import api
from utils import ensure_directories
from waitress import serve

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            "vision_service.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Register blueprints
app.register_blueprint(api)

# Ensure directories exist
ensure_directories()

# Define a basic health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "healthy"}, 200

if __name__ == '__main__':
    logger.info("Starting Vision Service")
    # Use waitress for production-grade WSGI server
    serve(app, host='0.0.0.0', port=5050)