# services/code_service.py
import os
from datetime import datetime
import logging

from config import code_versions, logger
from utils import emit

def backup_current_code():
    """Create a backup of the current stereo_vision.py file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        with open('stereo_vision.py', 'r') as f:
            code = f.read()

        backup_path = f'static/code_backups/stereo_vision_{timestamp}.py'
        with open(backup_path, 'w') as f:
            f.write(code)

        code_versions.append({
            'timestamp': timestamp,
            'filename': backup_path,
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Keep only the last 20 versions
        if len(code_versions) > 20:
            old_version = code_versions.pop(0)
            if os.path.exists(old_version['filename']):
                os.remove(old_version['filename'])

        return True
    except Exception as e:
        logger.error(f"Failed to backup code: {str(e)}")
        return False


def update_code(new_code):
    """Update the stereo_vision.py file with new code."""
    try:
        if not new_code:
            return False, 'No code provided'

        # Backup current code
        if not backup_current_code():
            return False, 'Failed to backup current code'

        # Write new code to file
        with open('stereo_vision.py', 'w') as f:
            f.write(new_code)

        logger.info("Code updated successfully")
        return True, 'Code updated successfully'

    except Exception as e:
        logger.error(f"Failed to update code: {str(e)}")
        return False, str(e)


def rollback_code(version):
    """Rollback to a previous version of the code."""
    try:
        if not version:
            return False, 'No version specified'

        # Find the version in the history
        version_found = False
        for v in code_versions:
            if v['timestamp'] == version:
                version_found = True
                # Read the old code
                with open(v['filename'], 'r') as f:
                    old_code = f.read()

                # Backup current code first
                backup_current_code()

                # Write old code to the main file
                with open('stereo_vision.py', 'w') as f:
                    f.write(old_code)

                break

        if not version_found:
            return False, 'Version not found'

        logger.info(f"Code rolled back to version {version}")
        return True, f'Code rolled back to version {version}'

    except Exception as e:
        logger.error(f"Failed to rollback code: {str(e)}")
        return False, str(e)


def get_current_code():
    """Get the current stereo_vision.py code."""
    try:
        # Check if file exists first
        if not os.path.exists('stereo_vision.py'):
            return False, 'stereo_vision.py file not found', None, None

        # Get file modified time
        last_modified = datetime.fromtimestamp(os.path.getmtime('stereo_vision.py')).strftime("%Y-%m-%d %H:%M:%S")

        with open('stereo_vision.py', 'r') as f:
            code = f.read()

        return True, 'Code retrieved', code, last_modified

    except Exception as e:
        logger.error(f"Failed to get current code: {str(e)}")
        return False, str(e), None, None


def get_code_versions():
    """Get the available code versions for rollback."""
    try:
        # Check if code_versions is empty and if backups exist
        if not code_versions:
            # Try to load backup versions from directory
            backup_dir = 'static/code_backups'
            if os.path.exists(backup_dir):
                backup_files = [f for f in os.listdir(backup_dir) if
                                f.startswith('stereo_vision_') and f.endswith('.py')]

                # Sort by timestamp (newest first)
                backup_files.sort(reverse=True)

                # Create version entries
                for filename in backup_files:
                    # Extract timestamp from filename
                    timestamp = filename.replace('stereo_vision_', '').replace('.py', '')
                    try:
                        # Create datetime object for better display
                        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")

                        # Add to code_versions list
                        code_versions.append({
                            'timestamp': timestamp,
                            'filename': os.path.join(backup_dir, filename),
                            'datetime': formatted_date
                        })
                    except ValueError:
                        # Skip if timestamp format is invalid
                        logger.warning(f"Invalid timestamp format in backup file: {filename}")
                        continue

        return True, code_versions

    except Exception as e:
        logger.error(f"Failed to get code versions: {str(e)}")
        return False, str(e)