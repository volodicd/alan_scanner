import os
import sys
import torch
import numpy as np
import cv2
import logging
from argparse import Namespace

# Add RAFT-Stereo to path
current_dir = os.path.dirname(os.path.abspath(__file__))
raft_stereo_path = os.path.join(current_dir, '..', 'RAFT-Stereo')
sys.path.append(raft_stereo_path)

# Import RAFT-Stereo components
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder

logger = logging.getLogger(__name__)


class RaftStereoWrapper:
    def __init__(self, max_disp=256, mixed_precision=True):
        self.max_disp = max_disp
        self.mixed_precision = mixed_precision
        self.device = self._get_device()
        self.model = None
        self.iters = 12

        # Create proper args object based on RAFT-Stereo requirements
        self.args = Namespace(
            mixed_precision=mixed_precision,
            hidden_dims=[128] * 3,
            corr_implementation="reg",
            shared_backbone=False,
            corr_levels=4,
            corr_radius=4,
            n_downsample=2,
            context_norm="batch",
            slow_fast_gru=False,
            n_gru_layers=3,
            valid_iters=12  # Number of iterations for inference
        )
        
        # Note: We don't initialize the model in the constructor
        # to avoid CUDA initialization issues. It will be initialized
        # in the load_model method.

        logger.info(f"Initialized RAFT-Stereo wrapper with device: {self.device}")

    def load_model(self, weights_path=None):
        try:
            # Find the model weights if not provided
            if weights_path is None:
                # Try different possible locations
                possible_paths = [
                    os.path.join(raft_stereo_path, 'models', 'raftstereo-middlebury.pth'),
                    os.path.join('models', 'weights', 'raftstereo-middlebury.pth'),
                    os.path.join('models', 'weights', 'raft_stereo.pth')
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        weights_path = path
                        break

            if weights_path is None or not os.path.exists(weights_path):
                logger.error(f"Failed to find model weights at {weights_path}")
                return False

            logger.info(f"Loading model weights from {weights_path}")
            
            # Use the original demo.py approach - initialize with DataParallel first
            logger.info("Creating RAFT-Stereo model with DataParallel")
            model = torch.nn.DataParallel(RAFTStereo(self.args), device_ids=[0])
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            
            # Get the module to avoid DataParallel wrapper during inference
            self.model = model.module
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("RAFT-Stereo model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading RAFT-Stereo model: {str(e)}")
            return False

    def inference(self, left_img, right_img, num_iters=None):
        """Run inference using the same approach as the original demo.py"""
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load RAFT-Stereo model")

        if num_iters is None:
            num_iters = self.args.valid_iters  # Default to what's in the args

        # Convert to RGB if needed (RAFT expects RGB format)
        if len(left_img.shape) == 3 and left_img.shape[2] == 3:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        # Convert to torch tensors [B,C,H,W]
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float()[None].to(self.device) / 255.0
        right_img = torch.from_numpy(right_img).permute(2, 0, 1).float()[None].to(self.device) / 255.0

        # Pad images to match dimensions
        padder = InputPadder(left_img.shape, divis_by=32)
        left_img, right_img = padder.pad(left_img, right_img)

        # Run inference
        with torch.no_grad():
            try:
                # Execute the model with test_mode=True
                _, flow_up = self.model(left_img, right_img, iters=num_iters, test_mode=True)
                
                # Remove padding
                flow_up = padder.unpad(flow_up)
                
                # The output format is [B, C, H, W] where C=1 for disparity
                # We need to extract the disparity (negative of the flow) and convert to numpy
                disparity = -flow_up[0, 0].cpu().numpy()  # Negative of flow along x-axis = disparity
                
                # Log some stats for debugging
                logger.debug(f"Disparity stats - Shape: {disparity.shape}, Range: {disparity.min():.2f} to {disparity.max():.2f}")
                
            except Exception as e:
                logger.error(f"Error during RAFT-Stereo inference: {str(e)}")
                raise RuntimeError(f"RAFT-Stereo inference failed: {str(e)}")

        return disparity

    def _get_device(self):
        """Get best available device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')