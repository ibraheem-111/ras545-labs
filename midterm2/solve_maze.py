import heapq

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
