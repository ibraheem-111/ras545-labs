import numpy as np
import cv2
from constants import (
    CAMERA_PX_POINTS, 
    ROBOT_MM_POINTS, 
    CAMERA_MATRIX, 
    DIST_COEFFS
)

def pixel_to_robot(pixel_x: float, pixel_y: float, correct_distortion: bool = True) -> tuple[float, float]:
    """Convert camera pixel coordinates to robot coordinates in mm.
    
    Args:
        pixel_x: x-coordinate in camera pixels
        pixel_y: y-coordinate in camera pixels
        correct_distortion: whether to apply camera distortion correction
    
    Returns:
        tuple[float, float]: (x, y) coordinates in robot space (mm)
    """
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    
    # Step 1: Apply camera distortion correction if needed
    if correct_distortion:
        point = cv2.undistortPoints(point, CAMERA_MATRIX, DIST_COEFFS, P=CAMERA_MATRIX)
        point = point.reshape(1, 1, 2)
    
    # Step 2: Get perspective transform matrix
    M = cv2.getPerspectiveTransform(CAMERA_PX_POINTS, ROBOT_MM_POINTS)
    
    # Step 3: Convert point
    transformed = cv2.perspectiveTransform(point, M)
    
    return tuple(transformed[0][0])

def turning_nodes_from_grid_path(path_rc):
    """
    path_rc: [(r,c), ...] cells from A* (4- or 8-connected).
    Returns only the corners: start, every direction change, and goal.
    """
    if not path_rc or len(path_rc) < 2:
        return path_rc[:]
    nodes = [path_rc[0]]
    pr, pc = path_rc[1][0] - path_rc[0][0], path_rc[1][1] - path_rc[0][1]
    for i in range(2, len(path_rc)):
        dr, dc = path_rc[i][0] - path_rc[i-1][0], path_rc[i][1] - path_rc[i-1][1]
        if (dr, dc) != (pr, pc):          # direction changed ⇒ corner at i-1
            nodes.append(path_rc[i-1])
        pr, pc = dr, dc
    nodes.append(path_rc[-1])
    return nodes

def line_of_sight_free(a_rc, b_rc, grid):
    """Returns True if straight segment a→b crosses only free cells (0)."""
    # Bresenham "supercover" line so we touch all crossed cells
    (r0, c0), (r1, c1) = a_rc, b_rc
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    err = dr - dc
    r, c = r0, c0
    H, W = grid.shape
    while True:
        if not (0 <= r < H and 0 <= c < W) or grid[r, c] == 1:
            return False
        if (r, c) == (r1, c1): 
            return True
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 <  dr: err += dr; c += sc

def compress_by_los(nodes_rc, grid):
    if len(nodes_rc) <= 2:
        return nodes_rc[:]
    out = [nodes_rc[0]]
    i = 0
    while i < len(nodes_rc) - 1:
        j = i + 1
        # extend as far as LOS holds
        while j < len(nodes_rc) and line_of_sight_free(nodes_rc[i], nodes_rc[j], grid):
            j += 1
        out.append(nodes_rc[j-1])
        i = j - 1
    return out

def nodes_to_robot_coords(nodes_rc, out_size, grid_size, pixel_to_robot):
    """
    nodes_rc: [(r,c), ...] in grid indices
    out_size: warp size you used (e.g., 600)
    grid_size: number of grid cells per side (e.g., 50)
    pixel_to_robot: your existing function (x_px, y_px) -> (X,Y)
    """
    cell = out_size / float(grid_size)
    robot_pts = []
    for r, c in nodes_rc:
        x_px = (c + 0.5) * cell
        y_px = (r + 0.5) * cell
        robot_pts.append(pixel_to_robot(x_px, y_px))
    return robot_pts

