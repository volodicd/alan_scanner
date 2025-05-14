# routes/camera_routes.py
from flask import jsonify, request
import cv2
import base64
import threading
import time
import os
from datetime import datetime
from collections import deque

from app_context import app_ctx
import logging

logger = logging.getLogger(__name__)


def register_camera_routes(app):
    """Register camera-related routes."""

    @app.route('/api/stream/start', methods=['POST'])
    def start_stream():
        with app_ctx.lock:
            # Check if already streaming
            if app_ctx.is_streaming:
                return jsonify({'success': False, 'message': 'Stream is already running'})

            # Stop previous thread if it exists
            if app_ctx.stream_thread is not None and app_ctx.stream_thread.is_alive():
                app_ctx.stream_thread_active = False
                app_ctx.stream_thread.join(timeout=2.0)

            # Initialize stereo vision if needed
            if app_ctx.stereo_vision is None:
                from stereo_vision import StereoVision
                app_ctx.stereo_vision = StereoVision(
                    left_cam_idx=app_ctx.config["left_cam_idx"],
                    right_cam_idx=app_ctx.config["right_cam_idx"],
                    width=app_ctx.config["width"],
                    height=app_ctx.config["height"]
                )
                # Set SGBM params
                app_ctx.stereo_vision.set_sgbm_params(app_ctx.config["sgbm_params"])

            # Start streaming thread
            app_ctx.is_streaming = True
            app_ctx.stream_thread_active = True
            app_ctx.stream_thread = threading.Thread(target=process_stream)
            app_ctx.stream_thread.daemon = True
            app_ctx.stream_thread.start()

        return jsonify({'success': True})

    @app.route('/api/stream/stop', methods=['POST'])
    def stop_stream():
        with app_ctx.lock:
            if not app_ctx.is_streaming:
                return jsonify({'success': False, 'message': 'Stream is not running'})

            # Set flags to stop streaming
            app_ctx.is_streaming = False
            app_ctx.stream_thread_active = False

            # Wait for thread to finish
            if app_ctx.stream_thread is not None and app_ctx.stream_thread.is_alive():
                app_ctx.stream_thread.join(timeout=5.0)
                if app_ctx.stream_thread.is_alive():
                    logger.warning("Stream thread did not terminate within timeout")

        return jsonify({'success': True})

    @app.route('/api/capture', methods=['POST'])
    def handle_capture_frame():
        with app_ctx.lock:
            if app_ctx.stereo_vision is None:
                return jsonify({'success': False, 'message': 'Stereo vision not initialized'}), 400

        try:
            # Use context manager to ensure cameras are properly handled
            with app_ctx.stereo_vision:
                left_frame, right_frame = app_ctx.stereo_vision.capture_frames()
                if left_frame is None or right_frame is None:
                    return jsonify({'success': False, 'message': 'Failed to capture frames'}), 500

                # Get the current timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Save raw captures
                left_filepath = f"static/captures/left_{timestamp}.jpg"
                right_filepath = f"static/captures/right_{timestamp}.jpg"
                cv2.imwrite(left_filepath, left_frame)
                cv2.imwrite(right_filepath, right_frame)

                # Process and save disparity if calibration is loaded
                disparity_filepath = None
                has_calibration = False

                try:
                    has_calibration = app_ctx.stereo_vision.load_calibration()
                    if has_calibration:
                        left_rect, right_rect = app_ctx.stereo_vision.get_rectified_images(left_frame, right_frame)
                        _, disp_color = app_ctx.stereo_vision.compute_disparity_map(left_rect, right_rect)
                        disparity_filepath = f"static/captures/disparity_{timestamp}.jpg"
                        cv2.imwrite(disparity_filepath, disp_color)
                except Exception as e:
                    logger.error(f"Error processing disparity: {str(e)}")
                    # Continue with returning the raw frames

                return jsonify({
                    'success': True,
                    'timestamp': timestamp,
                    'left_path': left_filepath,
                    'right_path': right_filepath,
                    'disparity_path': disparity_filepath,
                    'has_calibration': has_calibration
                })

        except cv2.error as e:
            logger.error(f"OpenCV error: {str(e)}")
            return jsonify({'success': False, 'message': 'Camera error: Try restarting cameras'}), 500
        except IOError as e:
            logger.error(f"IO error: {str(e)}")
            return jsonify({'success': False, 'message': 'Camera connection lost'}), 500
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return jsonify({'success': False, 'message': 'Internal server error'}), 500


def process_stream():
    """Process and stream frames from the cameras."""
    # Set maximum size for frame times queue
    max_frame_times = 30
    last_full_frame_time = 0

    try:
        with app_ctx.lock:
            if app_ctx.stereo_vision is None:
                logger.error("Stereo vision not initialized")
                app_ctx.emit('error', {'message': 'Stereo vision not initialized'})
                app_ctx.is_streaming = False
                app_ctx.stream_thread_active = False
                return

        # Use context manager to ensure cameras are properly closed
        with app_ctx.stereo_vision:
            # Check if calibration is available
            has_calibration = app_ctx.stereo_vision.load_calibration()
            if has_calibration:
                logger.info("Calibration loaded successfully")
            else:
                logger.info("No calibration data found, streaming raw frames")

            # Target frame rate control
            target_fps = 30
            frame_time = 1.0 / target_fps
            last_frame_time = time.time()

            # Main streaming loop - check both flags in a thread-safe way
            stream_active = True
            while stream_active:
                with app_ctx.lock:
                    stream_active = app_ctx.is_streaming and app_ctx.stream_thread_active

                if not stream_active:
                    break

                # Maintain frame rate
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

                try:
                    # Record this frame's time for FPS calculation
                    current_time = time.time()
                    with app_ctx.lock:
                        app_ctx.frame_times.append(current_time)
                        if len(app_ctx.frame_times) > max_frame_times:
                            app_ctx.frame_times.pop(0)

                    # Update FPS calculation every second
                    if current_time - app_ctx.last_fps_update >= 1.0:
                        with app_ctx.lock:
                            app_ctx.current_fps = app_ctx.calculate_fps()
                            app_ctx.last_fps_update = current_time

                    # Capture frames
                    left_frame, right_frame = app_ctx.stereo_vision.capture_frames()

                    if left_frame is None or right_frame is None:
                        logger.warning("Failed to capture frames")
                        app_ctx.emit('error', {'message': "Frame capture failed. Check camera connections."})
                        time.sleep(0.5)
                        continue

                    # Check if we should send full quality frames (once per second)
                    should_send_full = (current_time - last_full_frame_time) >= 1.0

                    # Process frames if calibration is available
                    frames_to_send = {}
                    if has_calibration:
                        try:
                            # Rectify images
                            left_rect, right_rect = app_ctx.stereo_vision.get_rectified_images(left_frame, right_frame)

                            # Compute disparity map
                            _, disp_color = app_ctx.stereo_vision.compute_disparity_map(left_rect, right_rect)

                            frames_to_send = {
                                'left': left_rect,
                                'right': right_rect,
                                'disparity': disp_color
                            }
                        except Exception as e:
                            logger.error(f"Error processing stereo images: {str(e)}")
                            # Fall back to raw frames if rectification fails
                            frames_to_send = {
                                'left': left_frame,
                                'right': right_frame
                            }
                    else:
                        # Just use raw frames
                        frames_to_send = {
                            'left': left_frame,
                            'right': right_frame
                        }

                    # Convert frames to JPEG for streaming
                    encoded_frames = {}
                    for key, frame in frames_to_send.items():
                        try:
                            # Optimize quality based on frame type and send frequency
                            quality = 80 if should_send_full or key == 'disparity' else 40
                            scale = 1.0 if should_send_full or key == 'disparity' else 0.5

                            # Resize frame for efficiency
                            if scale < 1.0:
                                new_width = int(frame.shape[1] * scale)
                                new_height = int(frame.shape[0] * scale)
                                frame = cv2.resize(frame, (new_width, new_height))

                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                            encoded_frames[key] = base64.b64encode(buffer).decode('utf-8')
                        except Exception as e:
                            logger.error(f"Error encoding {key} frame: {str(e)}")

                    # Update last full frame time if we sent full quality
                    if should_send_full:
                        last_full_frame_time = current_time

                    # Send frames via websocket if we have any
                    if encoded_frames:
                        # Add FPS information
                        with app_ctx.lock:
                            data_to_send = encoded_frames.copy()
                            data_to_send['fps'] = round(app_ctx.current_fps, 1)
                        app_ctx.emit('frames', data_to_send)

                    # Update last frame time
                    last_frame_time = current_time

                except Exception as e:
                    logger.error(f"Error in streaming thread: {str(e)}")
                    app_ctx.emit('error', {'message': f"Streaming error: {str(e)}"})
                    time.sleep(0.5)

    except Exception as e:
        logger.error(f"Fatal error in streaming thread: {str(e)}")
    finally:
        # Clean up when streaming stops - ensure thread state is reset
        with app_ctx.lock:
            app_ctx.is_streaming = False
            app_ctx.stream_thread_active = False
        logger.info("Streaming stopped")