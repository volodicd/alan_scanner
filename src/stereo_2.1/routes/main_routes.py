# routes/main_routes.py
from flask import render_template, send_from_directory, jsonify


def register_main_routes(app):
    """Register main application routes."""

    # Route for main page
    @app.route('/')
    def index():
        return render_template('index.html')

    # Static file serving
    @app.route('/static/<path:path>')
    def serve_static(path):
        return send_from_directory('static', path)