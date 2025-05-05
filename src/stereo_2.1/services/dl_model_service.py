# services/dl_model_service.py
import os
import logging

from config import dl_model, dl_model_loaded, current_config, logger
from utils import emit


def init_dl_model():
    """Initialize the deep learning model for disparity estimation."""
    global dl_model, dl_model_loaded

    try:
        from models import load_model, get_device_info

        # Log device info
        device_info = get_device_info()
        logger.info("Initializing deep learning model with device information:")
        if device_info['cuda_available']:
            logger.info(f"CUDA available with {device_info['cuda_device_count']} devices")
        elif device_info['mps_available']:
            logger.info("Apple MPS (Metal Performance Shaders) available on M1 Mac")
        else:
            logger.info("Running on CPU only - this may be slow for deep learning inference")

        # Set model_name to raft_stereo if using that option
        model_name = 'raft_stereo' if current_config["disparity_method"] == "dl" else current_config['dl_model_name']
        current_config['dl_model_name'] = model_name

        # Check if model weights exist
        weights_dir = 'models/weights'
        weights_path = os.path.join(weights_dir, f"{model_name}.pth")

        if not os.path.exists(weights_path):
            logger.warning(f"Model weights not found at {weights_path}")

            # Check if alternative names might exist
            alt_weights = None
            if model_name == 'raft_stereo':
                for alt_name in ['raftstereo-middlebury.pth', 'raftstereo-sceneflow.pth']:
                    alt_path = os.path.join(weights_dir, alt_name)
                    if os.path.exists(alt_path):
                        logger.info(f"Found alternative weights at {alt_path}")
                        weights_path = alt_path
                        alt_weights = True
                        break

            # If no alternatives found, download weights
            if not alt_weights:
                from models.utils import download_model_weights
                logger.info("Attempting to download model weights...")
                weights_path = download_model_weights(
                    model_name=model_name,
                    save_dir=weights_dir
                )

        # Load model
        logger.info(f"Loading {model_name} model with weights from {weights_path}...")
        dl_model = load_model(
            model_name=model_name,
            weights_path=weights_path,
            max_disp=current_config['dl_params']['max_disp']
        )

        if dl_model is None:
            logger.error(f"Failed to load {model_name} model")
            return False

        dl_model_loaded = True
        logger.info(f"Deep learning model {model_name} loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize deep learning model: {str(e)}")
        dl_model_loaded = False
        return False


def update_dl_params(params):
    """Update deep learning parameters."""
    try:
        if 'max_disp' in params:
            max_disp = int(params['max_disp'])
            if max_disp < 64 or max_disp > 512:
                return False, "max_disp must be between 64 and 512"
            current_config['dl_params']['max_disp'] = max_disp

        if 'mixed_precision' in params:
            current_config['dl_params']['mixed_precision'] = bool(params['mixed_precision'])

        if 'downscale_factor' in params:
            factor = float(params['downscale_factor'])
            if factor <= 0 or factor > 1.0:
                return False, "downscale_factor must be between 0 and 1.0"
            current_config['dl_params']['downscale_factor'] = factor

        # Model needs to be reinitialized if max_disp changes
        if 'max_disp' in params and dl_model_loaded:
            logger.info("max_disp changed - model will be reinitialized on next use")

        return True, "Deep learning parameters updated"
    except Exception as e:
        logger.error(f"Error updating deep learning parameters: {str(e)}")
        return False, str(e)


def set_disparity_method(method):
    """Set the disparity computation method."""
    global dl_model_loaded, dl_enabled

    try:
        if method not in ['sgbm', 'dl']:
            return False, f"Invalid method: {method}. Use 'sgbm' or 'dl'."

        # Check if DL is available when requested
        if method == 'dl':
            if not dl_model_loaded:
                current_config['dl_model_name'] = 'raft_stereo'
                if not init_dl_model():
                    return False, "Deep learning model initialization failed. Using SGBM instead."

            # Enable DL processing
            dl_enabled = True

        # Update configuration
        current_config['disparity_method'] = method

        return True, f"Disparity method set to {method}"
    except Exception as e:
        logger.error(f"Error setting disparity method: {str(e)}")
        return False, str(e)