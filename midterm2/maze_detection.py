import cv2
import numpy as np
import copy

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
            second_largest = sorted(contours, key=cv2.contourArea)[-2] if len(contours) > 1 else None
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

        return frame, maze_thresh, maze_region, roi
