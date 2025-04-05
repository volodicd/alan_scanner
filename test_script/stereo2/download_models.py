#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_device_compatibility():
    """Check if the device is compatible with PyTorch and the model."""
    info = {}

    # Check CUDA availability
    info['cuda_available'] = torch.cuda.is_available()
    if info['cuda_available']:
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_devices'] = []

        for i in range(info['cuda_device_count']):
            device_name = torch.cuda.get_device_name(i)
            properties = torch.cuda.get_device_properties(i)
            memory_gb = properties.total_memory / (1024 ** 3)

            info['cuda_devices'].append({
                'index': i,
                'name': device_name,
                'compute_capability': f"{properties.major}.{properties.minor}",
                'memory_gb': memory_gb
            })

            logger.info(f"CUDA Device {i}: {device_name} ({memory_gb:.2f} GB)")

        # Check if GPU has enough memory for the model (min 4GB recommended)
        for device in info['cuda_devices']:
            if device['memory_gb'] < 4.0:
                logger.warning(
                    f"Device {device['name']} has less than 4GB VRAM ({device['memory_gb']:.2f} GB), which may cause out-of-memory errors with RAFT-Stereo")

    # Check Apple Silicon (M1/M2) availability
    info['mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if info['mps_available']:
        logger.info("Apple Silicon (M1/M2) MPS acceleration available")

    # Final compatibility assessment
    if info['cuda_available'] or info['mps_available']:
        logger.info("Hardware acceleration is available for DL inference")
        return True, info
    else:
        logger.warning("No hardware acceleration (CUDA/MPS) found. Running on CPU will be very slow for DL inference.")
        return False, info


def download_model(model_name="raft_stereo", output_dir="models/weights"):
    """Download the stereo model."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Check device compatibility
        is_compatible, device_info = check_device_compatibility()

        # Import the model utils for downloading
        try:
            from models.utils import download_model_weights
        except ImportError:
            # If running from a different directory, need to ensure proper imports
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            from models.utils import download_model_weights

        # Download the model
        logger.info(f"Downloading {model_name} model...")
        weights_path = download_model_weights(model_name, output_dir)

        # Verify the file exists and has content
        if os.path.exists(weights_path) and os.path.getsize(weights_path) > 1000:
            logger.info(f"Model downloaded successfully to {weights_path}")
            return True, weights_path
        else:
            logger.error(
                f"Model download may have failed. File size: {os.path.getsize(weights_path) if os.path.exists(weights_path) else 'not found'}")
            return False, "Download verification failed"

    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Download and check compatibility of RAFT-Stereo deep learning model.")
    parser.add_argument('--model', type=str, default='raft_stereo', help='Model name (default: raft_stereo)')
    parser.add_argument('--output-dir', type=str, default='models/weights', help='Output directory for model weights')
    parser.add_argument('--check-only', action='store_true', help='Only check device compatibility without downloading')

    args = parser.parse_args()

    # Check device compatibility
    is_compatible, device_info = check_device_compatibility()

    if args.check_only:
        if is_compatible:
            print("✅ Your device is compatible with RAFT-Stereo deep learning model")
            if device_info.get('cuda_available'):
                print(f"   CUDA available with {device_info['cuda_device_count']} device(s)")
                for i, device in enumerate(device_info['cuda_devices']):
                    print(f"   - GPU {i}: {device['name']} ({device['memory_gb']:.2f} GB)")
            elif device_info.get('mps_available'):
                print("   Apple Silicon (M1/M2) MPS acceleration available")
        else:
            print("⚠️  Your device will use CPU for RAFT-Stereo inference, which will be slow")
            print("   Consider using a machine with CUDA-capable GPU or Apple Silicon")

        sys.exit(0)

    # Download model
    success, result = download_model(args.model, args.output_dir)

    if success:
        print(f"✅ Model downloaded successfully to {result}")
        print("   You can now use the deep learning inference for disparity estimation")
    else:
        print(f"❌ Failed to download model: {result}")
        print("   A manual download approach may be needed.")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())