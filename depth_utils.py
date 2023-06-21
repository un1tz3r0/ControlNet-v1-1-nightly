import os

import cv2
import numpy as np


def convert_depth(depth):
    depth_norm = np.copy(depth)
    depth_norm = depth_norm / 1000.0
    vmin = np.percentile(depth_norm, 2)
    vmax = np.percentile(depth_norm, 85)

    depth_norm -= vmin
    depth_norm /= vmax - vmin
    depth_norm = 1.0 - depth_norm
    depth_image = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)

    return depth_image
