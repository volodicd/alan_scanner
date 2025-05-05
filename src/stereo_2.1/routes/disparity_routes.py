# routes/disparity_routes.py
from flask import request, jsonify

from config import (
    current_config, dl_model, dl_model_loaded, dl_enabled, logger
)
from services.dl_model_service import init_dl_model, update_dl_params, set_disparity_method


def register_disparity_routes(app):
    """Register disparity-related routes."""

    @app.route('/api/disparity/method', methods=['GET', 'POST'])
    def disparity_method():
        """Get or set the disparity computation method."""

        if request.method == 'GET':
            return jsonify({
                'success': True,
                'current_method': current_config['disparity_method'],
                'available_methods': ['sgbm', 'dl'],
                'dl_available': dl_model_loaded,
                'dl_enabled': dl_enabled,
                'dl_model_name': current_config['dl_model_name'] if dl_model_loaded else None
            })

        elif request.method == 'POST':
            try:
                data = request.json
                method = data.get('method')

                success, message = set_disparity_method(method)

                if not success:
                    return jsonify({
                        'success': False,
                        'message': message
                    }), 400 if "Invalid" in message else 500

                return jsonify({
                    'success': True,
                    'message': f"Disparity method set to {method}",
                    'current_method': method,
                    'dl_enabled': dl_enabled
                })

            except Exception as e:
                logger.error(f"Error setting disparity method: {str(e)}")
                return jsonify({'success': False, 'message': str(e)}), 500

    @app.route('/api/disparity/dl_params', methods=['GET', 'POST'])
    def disparity_dl_params():
        """Get or set the deep learning disparity parameters."""

        if request.method == 'GET':
            return jsonify({
                'success': True,
                'dl_params': current_config['dl_params'],
                'dl_model_name': current_config['dl_model_name']
            })

        elif request.method == 'POST':
            try:
                data = request.json
                success, message = update_dl_params(data)

                if not success:
                    return jsonify({
                        'success': False,
                        'message': message
                    }), 400

                return jsonify({
                    'success': True,
                    'message': "Deep learning parameters updated",
                    'dl_params': current_config['dl_params']
                })

            except Exception as e:
                logger.error(f"Error setting deep learning parameters: {str(e)}")
                return jsonify({'success': False, 'message': str(e)}), 500