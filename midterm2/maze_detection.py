import cv2
import numpy as np
import copy


def _normalize_pts(corners):
    pts = np.asarray(corners, dtype=np.float32)
    pts = np.squeeze(pts)            # handles Nx1x2, Nx2, 1xNx2, etc.
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Expected points shaped like (N,2) or (N,1,2).")
    return pts

def _order_tl_tr_br_bl(pts4):
    # Order 4 points as TL, TR, BR, BL
    s = pts4.sum(axis=1)
    diff = np.diff(pts4, axis=1).ravel()
    tl = pts4[np.argmin(s)]
    br = pts4[np.argmax(s)]
    tr = pts4[np.argmin(diff)]
    bl = pts4[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def four_external_corners(corners):
    """
    Select the 4 external corners from any set of 2D points.
    Returns 4x2 float32 points ordered TL, TR, BR, BL.
    Strategy:
      1) Oriented min-area rectangle over all points (robust to rotation)
      2) If degenerate (rare), fall back to convex hull -> approx to 4
      3) If still not 4, fall back to axis-aligned bounding box
    """
    pts = _normalize_pts(corners)

    # 1) Oriented min-area rect (works even if there are many corners & rotation)
    if len(pts) >= 3:
        rect = cv2.minAreaRect(pts)          # (center), (w,h), angle
        box  = cv2.boxPoints(rect)           # 4x2 float32
        return _order_tl_tr_br_bl(box)

    # 2) If too few for minAreaRect, try hull+approx (edge case)
    hull = cv2.convexHull(pts.reshape(-1,1,2))
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    if len(approx) == 4:
        return _order_tl_tr_br_bl(approx.reshape(4,2).astype(np.float32))

    # 3) Final fallback: axis-aligned bounding box
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    box = np.array([[x, y],
                    [x+w, y],
                    [x+w, y+h],
                    [x, y+h]], dtype=np.float32)
    return _order_tl_tr_br_bl(box)

def warp_from_corners(src_gray, corners_xy, out_size=600):
    """
    src_gray: uint8 grayscale image (same one you passed to isolate_maze_with_reconstruction)
    corners_xy: 4x2 float32 array in (x,y) order, TL,TR,BR,BL (already ordered by your function)
    out_size: output square size
    """
    dst = np.float32([[0,0],[out_size,0],[out_size,out_size],[0,out_size]])
    M = cv2.getPerspectiveTransform(corners_xy.astype(np.float32), dst)
    warped = cv2.warpPerspective(src_gray, M, (out_size, out_size))
    return warped

def preprocess_maze(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5,5), 1)
    # frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return frame
# _, frame = capture_image(cap)

class MazeDetector:
    """
    A class to detect and process mazes from video frames.
    """
    
    def __init__(self):
        pass

    def maze_to_grid(self, maze_thresh, grid_size=20):
        """
        Convert maze binary image to a discrete grid for pathfinding.
        
        The idea is to sample the maze at regular intervals to determine
        which cells are walkable (white/open) vs blocked (black/wall).
        
        Parameters:
        -----------
        maze_thresh : numpy array
            Binary image where WHITE = walls (255), BLACK = paths (0)
        grid_size : int
            Number of cells in each dimension (creates grid_size x grid_size grid)
        
        Returns:
        --------
        maze_grid : numpy array
            2D array where 0 = walkable path, 1 = wall/blocked
        """
        if maze_thresh is None:
            return None
        
        h, w = maze_thresh.shape
        cell_h = h / grid_size
        cell_w = w / grid_size
        
        maze_grid = np.zeros((grid_size, grid_size), dtype=int)
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate center point of this grid cell
                center_y = int((i + 0.5) * cell_h)
                center_x = int((j + 0.5) * cell_w)
                
                # Sample a small region around the center
                sample_size = max(2, int(min(cell_h, cell_w) * 0.3))
                y_start = max(0, center_y - sample_size)
                y_end = min(h, center_y + sample_size)
                x_start = max(0, center_x - sample_size)
                x_end = min(w, center_x + sample_size)
                
                sample_region = maze_thresh[y_start:y_end, x_start:x_end]
                
                # If region is mostly WHITE (wall), mark as blocked
                # In the threshold image: white = wall, black = path
                if sample_region.size > 0:
                    avg_intensity = np.mean(sample_region)
                    
                    # If average is > 127 (more white than black), it's a wall
                    if avg_intensity > 127:
                        maze_grid[i, j] = 1  # Wall/blocked
                    else:
                        maze_grid[i, j] = 0  # Path/walkable
        
        return maze_grid


    def visualize_grid(self, maze_thresh, maze_grid):
        """
        Create a visualization showing the grid overlay on the maze.
        
        Parameters:
        -----------
        maze_thresh : numpy array
            Original binary maze image
        maze_grid : numpy array
            Grid representation (0=path, 1=wall)
        
        Returns:
        --------
        visual : numpy array
            Color image with grid visualization
        """
        if maze_thresh is None or maze_grid is None:
            return None
        
        # Ensure proper type
        if maze_thresh.dtype != np.uint8:
            maze_thresh = maze_thresh.astype(np.uint8)
        
        h, w = maze_thresh.shape
        grid_size = maze_grid.shape[0]
        
        cell_h = h / grid_size
        cell_w = w / grid_size
        
        # Create color version
        visual = cv2.cvtColor(maze_thresh, cv2.COLOR_GRAY2BGR)
        
        # Draw grid lines
        for i in range(grid_size + 1):
            y = int(i * cell_h)
            cv2.line(visual, (0, y), (w, y), (100, 100, 100), 1)
        
        for j in range(grid_size + 1):
            x = int(j * cell_w)
            cv2.line(visual, (x, 0), (x, h), (100, 100, 100), 1)
        
        # Draw cell markers
        for i in range(grid_size):
            for j in range(grid_size):
                center_y = int((i + 0.5) * cell_h)
                center_x = int((j + 0.5) * cell_w)
                
                if maze_grid[i, j] == 1:
                    # Wall - red X
                    size = int(min(cell_h, cell_w) * 0.3)
                    cv2.line(visual, 
                            (center_x - size, center_y - size),
                            (center_x + size, center_y + size),
                            (0, 0, 255), 2)
                    cv2.line(visual,
                            (center_x + size, center_y - size),
                            (center_x - size, center_y + size),
                            (0, 0, 255), 2)
                else:
                    # Path - green circle
                    radius = int(min(cell_h, cell_w) * 0.2)
                    cv2.circle(visual, (center_x, center_y), radius, (0, 255, 0), -1)
        
        return visual

    def apply_perspective_correction(self, maze_thresh, contour):
        pass

    def detect_maze(self, frame):
        plain_frame = copy.deepcopy(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours of the grid / workspace
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours in green
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # Highlight the Maze
        if contours:
            largest = max(contours, key=cv2.contourArea)
            # print("Largest contour area:", cv2.cornerHarris(largest))
            corners = self.detect_corners(largest)
            second_largest = sorted(contours, key=cv2.contourArea)[-2] if len(contours) > 1 else None
            corners2 = self.detect_corners(second_largest) if second_largest is not None else None
            all_corners = np.vstack([corners, corners2]) if corners2 is not None else corners
            cv2.drawContours(frame, [largest, second_largest], -1, (0, 0, 255), 3)
            x, y, w, h = cv2.boundingRect(largest)
            x2, y2, w2, h2 = cv2.boundingRect(second_largest)

            all_maze_points = np.vstack([largest, second_largest])

            min_y = min(y, y2)
            min_x = min(x, x2) 
            y_max = max(y + h, y2 + h2)
            x_max = max(x + w, x2 + w2)
            roi = plain_frame[min_y-10:y_max+10, min_x-10:x_max+10]

            cv2.rectangle(frame, (min_x, min_y), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, "Workspace", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            maze_thresh = closed[min_y:y_max, min_x:x_max]
            
            maze_region = (min_x, min_y, x_max - min_x, y_max - min_y)

        return frame, maze_thresh, maze_region, roi, all_corners
    
    def detect_corners(self, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri,
                                    True)
        corners = approx.reshape(-1, 2)
        return corners
    