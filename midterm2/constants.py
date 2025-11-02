from enum import Enum
import numpy as np

# Load camera calibration from numpy files
CAMERA_MATRIX = np.load('midterm2/camera_matrix.npy')
DIST_COEFFS = np.load('midterm2/dist_coeffs.npy')

# Load calibration points from saved numpy files
CAMERA_PX_POINTS = np.load('camera_points.npy').astype(np.float32)
ROBOT_MM_POINTS = np.load('robot_points.npy').astype(np.float32)

class HomeCoordinates(Enum):
    x = 240
    y = 0
    z = 150

class IntermediatePoint(Enum):
    x = 240
    y=0
    z = 80

Z_Draw = -40
Z_Inspect = 70
