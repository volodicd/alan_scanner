# routes/logging_routes.py
from flask import request, jsonify
import os
from config import logger


def register_logging_routes(app):
    """Register logging-related routes."""

    @app.route('/api/logs', methods=['GET'])
    def get_logs():
        """Get the application logs with filtering options."""

        try:
            # Parse query parameters for filtering
            log_type = request.args.get('type', 'app')  # 'app' (default) or 'stereo'
            level = request.args.get('level', '').upper()  # Filter by log level
            limit = int(request.args.get('limit', 100))  # Number of lines to return
            search = request.args.get('search', '')  # Text to search for

            # Cap the limit to prevent excessive responses
            if limit > 500:
                limit = 500

            # Determine which log file to read
            log_file = "stereo_vision_app.log" if log_type == 'app' else "stereo_vision.log"

            if not os.path.exists(log_file):
                return jsonify({
                    'success': False,
                    'message': f"Log file {log_file} not found",
                    'available_logs': [f for f in os.listdir('.') if f.endswith('.log')]
                }), 404

            # Read the log file
            log_size = os.path.getsize(log_file)
            logger.debug("Reading log file %s (%.1f KB)", log_file, log_size / 1024)

            all_logs = []
            with open(log_file, 'r') as f:
                all_logs = f.readlines()

            # Apply filters
            filtered_logs = []
            for line in all_logs:
                # Filter by level if specified
                if level and level not in line.upper():
                    continue

                # Filter by search text if specified
                if search and search.lower() not in line.lower():
                    continue

                filtered_logs.append(line)

            # Get the limited number of lines (from the end)
            if limit > 0:
                filtered_logs = filtered_logs[-limit:]

            # Create parsed logs with structured information
            parsed_logs = []
            for line in filtered_logs:
                try:
                    # Parse log line into components (rough approximation)
                    parts = line.split(' - ', 3)
                    if len(parts) >= 4:
                        timestamp = parts[0]
                        module = parts[1]
                        level = parts[2]
                        message = parts[3].strip()

                        parsed_logs.append({
                            'timestamp': timestamp,
                            'module': module,
                            'level': level,
                            'message': message,
                            'raw': line.strip()
                        })
                    else:
                        # If we can't parse it, just include the raw line
                        parsed_logs.append({
                            'raw': line.strip()
                        })
                except Exception:
                    # If parsing fails, include the raw line
                    parsed_logs.append({
                        'raw': line.strip()
                    })

            # Log the number of lines returned
            logger.debug("Returning %d log lines (filtered from %d total lines)",
                         len(parsed_logs), len(all_logs))

            return jsonify({
                'success': True,
                'logs': parsed_logs,
                'total_lines': len(all_logs),
                'returned_lines': len(parsed_logs),
                'log_file': log_file,
                'log_size_kb': round(log_size / 1024, 1),
                'filters': {
                    'type': log_type,
                    'level': level,
                    'limit': limit,
                    'search': search
                }
            })

        except Exception as e:
            logger.error("Failed to get logs: %s", str(e))
            return jsonify({'success': False, 'message': str(e)}), 500