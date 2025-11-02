import heapq
import cv2
import numpy as np
from collections import deque

def find_start_goal_from_dots(warped_color, grid, start_color="green"):
    """
    warped_color : BGR image AFTER perspective warp (same size as the binary warp you used for the grid)
    grid         : 2D int array (0=path, 1=wall)
    start_color  : "green" or "red"

    Returns: (start_row, start_col), (goal_row, goal_col)
             or (None, None) if detection fails.
    """
    H, W = warped_color.shape[:2]
    R, C = grid.shape

    # --- 1) HSV masks (tune S/V if lighting changes) ---
    hsv = cv2.cvtColor(warped_color, cv2.COLOR_BGR2HSV)
    GREEN_LOW, GREEN_HIGH = (35, 80, 60),  (85, 255, 255)
    RED_LOW1,  RED_HIGH1  = (0, 80, 60),   (10, 255, 255)
    RED_LOW2,  RED_HIGH2  = (170, 80, 60), (180, 255, 255)

    mask_g = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    mask_r = cv2.inRange(hsv, RED_LOW1, RED_HIGH1) | cv2.inRange(hsv, RED_LOW2, RED_HIGH2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, k, iterations=1)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_CLOSE, k, iterations=1)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k, iterations=1)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, k, iterations=1)

    # --- 2) largest round blob helper ---
    def largest_round_blob(mask, min_area=150):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_circ = None, -1.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area: 
                continue
            per = cv2.arcLength(c, True)
            if per == 0: 
                continue
            circ = 4.0 * np.pi * area / (per * per)  # 1.0 = perfect circle
            if circ > best_circ:
                best, best_circ = c, circ
        return best

    def centroid(contour):
        M = cv2.moments(contour)
        if M["m00"] == 0: 
            return None
        return (M["m10"]/M["m00"], M["m01"]/M["m00"])  # (x, y)

    cg = largest_round_blob(mask_g)
    cr = largest_round_blob(mask_r)
    green_px = centroid(cg) if cg is not None else None
    red_px   = centroid(cr) if cr is not None else None

    if green_px is None and red_px is None:
        return None, None

    # --- 3) pixel -> grid cell ---
    def px_to_grid(x, y):
        c = int((x / W) * C); c = max(0, min(C-1, c))
        r = int((y / H) * R); r = max(0, min(R-1, r))
        return (r, c)

    g_cell = px_to_grid(*green_px) if green_px else None
    r_cell = px_to_grid(*red_px)   if red_px   else None

    # --- 4) snap to nearest path if dot fell on a wall cell ---
    def snap_to_path(cell):
        if cell is None: 
            return None
        r0, c0 = cell
        if grid[r0, c0] == 0: 
            return cell
        q = deque([cell]); seen = {cell}
        nbrs = [(1,0), (-1,0), (0,1), (0,-1)]
        while q:
            r, c = q.popleft()
            for dr, dc in nbrs:
                rr, cc = r+dr, c+dc
                if 0 <= rr < R and 0 <= cc < C and (rr, cc) not in seen:
                    if grid[rr, cc] == 0:
                        return (rr, cc)
                    seen.add((rr, cc)); q.append((rr, cc))
        return cell  # fallback

    g_cell = snap_to_path(g_cell)
    r_cell = snap_to_path(r_cell)

    # --- 5) choose start/goal by color ---
    if start_color.lower() == "green":
        return g_cell, r_cell
    else:
        return r_cell, g_cell

def astar_search(maze_grid, start, goal):
    """
    Perform A* search on a binary maze grid.
    maze_grid: 2D numpy array (0=free, 1=wall)
    start, goal: (row, col) tuples
    Returns: list of (row, col) waypoints in shortest path, or [] if no path found.
    """

    # Basic setup
    rows, cols = maze_grid.shape
    start, goal = tuple(start), tuple(goal)

    # Check bounds
    if maze_grid[start] == 1 or maze_grid[goal] == 1:
        print("Start or goal is blocked.")
        return []

    # Heuristic: Manhattan distance
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Priority queue of (f_score, g_score, node, path)
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start, [start]))

    visited = set()

    while open_list:
        f, g, current, path = heapq.heappop(open_list)

        if current == goal:
            return path  # success

        if current in visited:
            continue
        visited.add(current)

        r, c = current

        # Explore 4 neighbors (no diagonals)
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze_grid[nr, nc] == 0:
                if (nr, nc) not in visited:
                    new_g = g + 1
                    f_score = new_g + heuristic((nr, nc), goal)
                    heapq.heappush(open_list, (f_score, new_g, (nr, nc), path + [(nr, nc)]))

    return []  # no path


def find_entry_exit(grid):
    rows, cols = grid.shape
    openings = []

    # Top/bottom edges
    for c in range(cols):
        if grid[0, c] == 0:
            openings.append((0, c))
        if grid[rows-1, c] == 0:
            openings.append((rows-1, c))

    # Left/right edges
    for r in range(rows):
        if grid[r, 0] == 0:
            openings.append((r, 0))
        if grid[r, cols-1] == 0:
            openings.append((r, cols-1))

    if len(openings) < 2:
        raise ValueError("Could not find two openings on maze edges.")
    return openings[0], openings[-1]

def turning_nodes_from_grid_path(path_rc):
    if not path_rc or len(path_rc) < 2:
        return path_rc[:]
    nodes = [path_rc[0]]
    pr, pc = path_rc[1][0] - path_rc[0][0], path_rc[1][1] - path_rc[0][1]
    for i in range(2, len(path_rc)):
        dr, dc = path_rc[i][0] - path_rc[i-1][0], path_rc[i][1] - path_rc[i-1][1]
        if (dr, dc) != (pr, pc):
            nodes.append(path_rc[i-1])  # direction change corner
        pr, pc = dr, dc
    nodes.append(path_rc[-1])
    return nodes