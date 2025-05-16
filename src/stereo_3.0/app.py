from flask import Flask
import logging
from logging.handlers import RotatingFileHandler
from routes import api
from utils import ensure_directories
from waitress import serve

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            "vision_service.log",
            maxBytes=5*1024*1024,
            backupCount=3
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.register_blueprint(api)

# Ensure directories exist
# ! will be removed in the next versions !
ensure_directories()

# Define a basic health check endpoint
# Dima, use it also in the turtlebot, from this u can know if stereo vision is initialized and workng, for other endponts check routes.py
@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "healthy"}, 200

if __name__ == '__main__':
    logger.info("Starting Vision Service")
    # using waitress for cross-platoform and docker integration
    serve(app, host='0.0.0.0', port=5050)