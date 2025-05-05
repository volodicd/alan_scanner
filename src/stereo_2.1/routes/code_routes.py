# routes/code_routes.py
from flask import request, jsonify

from config import code_versions, logger
from services.code_service import (
    update_code, rollback_code, get_current_code, get_code_versions
)


def register_code_routes(app):
    """Register code management routes."""

    @app.route('/api/code/update', methods=['POST'])
    def handle_update_code():
        """Update the stereo_vision.py file with new code."""
        try:
            # Get the new code from the request
            new_code = request.json.get('code')
            success, message = update_code(new_code)

            if not success:
                return jsonify({'success': False, 'message': message}), 400 if message == 'No code provided' else 500

            return jsonify({
                'success': True,
                'message': message,
                'backup_versions': code_versions
            })

        except Exception as e:
            logger.error(f"Failed to update code: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500

    @app.route('/api/code/rollback', methods=['POST'])
    def handle_rollback_code():
        """Rollback to a previous version of the code."""
        try:
            # Get the version to rollback to
            version = request.json.get('version')
            success, message = rollback_code(version)

            if not success:
                return jsonify(
                    {'success': False, 'message': message}), 400 if message == 'No version specified' else 404

            return jsonify({
                'success': True,
                'message': message,
                'backup_versions': code_versions
            })

        except Exception as e:
            logger.error(f"Failed to rollback code: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500

    @app.route('/api/code/versions', methods=['GET'])
    def handle_get_code_versions():
        """Get the available code versions for rollback."""

        success, result = get_code_versions()

        if not success:
            return jsonify({'success': False, 'message': result}), 500

        return jsonify({
            'success': True,
            'versions': result
        })

    @app.route('/api/code/current', methods=['GET'])
    def handle_get_current_code():
        """Get the current stereo_vision.py code."""

        success, message, code, last_modified = get_current_code()

        if not success:
            return jsonify(
                {'success': False, 'message': message}), 404 if message == 'stereo_vision.py file not found' else 500

        return jsonify({
            'success': True,
            'code': code,
            'last_modified': last_modified
        })