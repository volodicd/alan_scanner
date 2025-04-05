#!/usr/bin/env python3
import cv2
import os
import time
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our wrapper
from models.raft_stereo_wrapper import RaftStereoWrapper


def test_raft_stereo():
    # Find sample images
    sample_dir = 'static/calibration'
    left_images = sorted([f for f in os.listdir(sample_dir) if f.startswith('left_')])
    right_images = sorted([f for f in os.listdir(sample_dir) if f.startswith('right_')])

    if not left_images or not right_images:
        print("No calibration images found")
        return False

    # Get the latest image pair
    left_path = os.path.join(sample_dir, left_images[-1])
    right_path = os.path.join(sample_dir, right_images[-1])

    print(f"Using images: {left_path}, {right_path}")

    # Load images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    if left_img is None or right_img is None:
        print("Failed to load images")
        return False

    # Initialize model
    print("Initializing RAFT-Stereo...")
    model = RaftStereoWrapper()

    # Run inference
    print("Running inference...")
    try:
        start_time = time.time()
        disparity = model.inference(left_img, right_img)
        inference_time = time.time() - start_time

        print(f"Inference successful in {inference_time:.2f} seconds")
        print(f"Disparity shape: {disparity.shape}, range: {disparity.min():.2f} to {disparity.max():.2f}")

        # Visualize result
        os.makedirs('static/test_results', exist_ok=True)
        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        result_path = os.path.join('static/test_results', 'raft_stereo_result.jpg')
        cv2.imwrite(result_path, disp_color)
        print(f"Result saved to {result_path}")

        return True
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return False


if __name__ == "__main__":
    if test_raft_stereo():
        print("✅ RAFT-Stereo test successful!")
    else:
        print("❌ RAFT-Stereo test failed.")