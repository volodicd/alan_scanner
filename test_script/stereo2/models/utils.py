import os
import torch
import numpy as np
import logging
import gdown
import cv2

logger = logging.getLogger(__name__)

# Pre-trained model URLs
MODEL_URLS = {
    'crestereo_base': 'https://drive.google.com/uc?id=1IUMZm4tzWBQSONmEqpptWKTQw1LgZi9w',
    'raft_stereo': 'https://drive.google.com/uc?id=1QjIwxhDSXWV31gsdEBI1XjYYD3VQzHdi'
}


def download_model_weights(model_name='raft_stereo', save_dir='models/weights'):
    """
    Download pre-trained model weights

    Args:
        model_name (str): Name of the model to download
        save_dir (str): Directory to save the weights

    Returns:
        str: Path to the downloaded weights file
    """
    if model_name not in MODEL_URLS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_URLS.keys())}")

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get download URL
    url = MODEL_URLS[model_name]

    # Determine save path
    save_path = os.path.join(save_dir, f"{model_name}.pth")

    # Check if file already exists
    if os.path.exists(save_path):
        logger.info(f"Model weights for {model_name} already exist at {save_path}")
        return save_path

    # Download weights
    logger.info(f"Downloading {model_name} weights from {url}")
    try:
        gdown.download(url, save_path, quiet=False)

        # Verify file was downloaded and has content
        if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
            logger.info(f"Downloaded {model_name} weights to {save_path}")
            return save_path
        else:
            if os.path.exists(save_path):
                os.remove(save_path)  # Remove empty or tiny file
            logger.error(
                f"Downloaded file appears invalid. Size: {os.path.getsize(save_path) if os.path.exists(save_path) else 'not found'}")
            raise FileNotFoundError(f"Failed to download valid {model_name} weights")
    except Exception as e:
        logger.error(f"Failed to download {model_name} weights: {str(e)}")
        raise


def load_model(model_name='raft_stereo', weights_path=None, max_disp=256, device=None):
    """
    Load a stereo vision model

    Args:
        model_name (str): Name of the model to load
        weights_path (str): Path to the weights file (if None, uses default)
        max_disp (int): Maximum disparity
        device (torch.device): Device to load the model on

    Returns:
        nn.Module: Loaded model
    """
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    logger.info(f"Loading {model_name} model on {device}")

    # Default weights path if not provided
    if weights_path is None:
        weights_path = os.path.join('models/weights', f"{model_name}.pth")
        if not os.path.exists(weights_path):
            logger.warning(f"No weights found at {weights_path}, downloading...")
            weights_path = download_model_weights(model_name)

    # Create model
    if model_name == 'raft_stereo':
        from .raft_stereo_wrapper import RaftStereoWrapper
        model = RaftStereoWrapper(max_disp=max_disp)

        # Load weights
        if os.path.exists(weights_path):
            try:
                success = model.load_model(weights_path)
                if success:
                    logger.info(f"Successfully loaded RAFT-Stereo weights from {weights_path}")
                else:
                    logger.error(f"Failed to load RAFT-Stereo weights: load_model returned False")
                    return None
            except Exception as e:
                logger.error(f"Failed to load RAFT-Stereo weights: {str(e)}")
                return None
        else:
            logger.error(f"Weights file not found at {weights_path}")
            return None

    elif model_name == 'crestereo_base':
        try:
            from .crestereo import CREStereo
            model = CREStereo(max_disp=max_disp)

            # Load weights if provided
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded CREStereo weights from {weights_path}")
            else:
                logger.error(f"CREStereo weights not found at {weights_path}")
                return None

            # Move to device and set to evaluation mode
            model = model.to(device)
            model.eval()
        except ImportError:
            logger.error("Failed to import CREStereo, it may not be available")
            return None
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def compute_confidence_map(disparity, left_img, right_img, window_size=5):
    """
    Compute confidence map for disparity using left-right consistency check

    Args:
        disparity (np.ndarray): Disparity map from left to right [H, W]
        left_img (np.ndarray): Left image [H, W, 3]
        right_img (np.ndarray): Right image [H, W, 3]
        window_size (int): Window size for consistency check

    Returns:
        np.ndarray: Confidence map [H, W] (0-1 range, higher is better)
    """
    H, W = disparity.shape
    confidence = np.ones((H, W), dtype=np.float32)

    # Convert to grayscale for matching
    if len(left_img.shape) == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_img
        right_gray = right_img

    # Create half window size
    hw = window_size // 2

    # Compute confidence based on matching cost
    for y in range(hw, H - hw):
        for x in range(hw, W - hw):
            d = int(disparity[y, x])

            # Skip invalid disparities
            if d <= 0 or x - d < hw:
                confidence[y, x] = 0
                continue

            # Get patches around the pixel in left and corresponding in right
            left_patch = left_gray[y - hw:y + hw + 1, x - hw:x + hw + 1]
            right_patch = right_gray[y - hw:y + hw + 1, x - d - hw:x - d + hw + 1]

            # Compute normalized cross-correlation
            ncc = cv2.matchTemplate(left_patch, right_patch, cv2.TM_CCORR_NORMED)[0][0]

            # Convert to confidence (higher is better)
            confidence[y, x] = max(0, ncc)

    return confidence


def get_device_info():
    """Get information about available devices"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }

    if info['cuda_available']:
        info['cuda_devices'] = []
        for i in range(info['cuda_device_count']):
            info['cuda_devices'].append({
                'name': torch.cuda.get_device_name(i),
                'capability': torch.cuda.get_device_capability(i),
                'memory': torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
            })

    return info