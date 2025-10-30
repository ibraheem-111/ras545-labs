import cv2
import time
from maze_detection import MazeDetector, preprocess_maze, warp_from_corners, four_external_corners
import copy
from solve_maze import astar_search, find_start_goal_from_dots, grid_path_to_image_points
import numpy as np
from celery_app import app
from robot import move
import threading
from point2coordinates import pixel_to_robot
from constants import Z_Draw

def display_plain_video():
    prev_t = time.time()
    plain_win = "Camera Stream Only"

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Can't receive frame. Exiting...")
            break

        cv2.imshow(plain_win, frame)

        # ===== FPS COUNTER =====
        now = time.time()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)    



def main(cap, md: MazeDetector):
    win_name = 'Camera Stream + Maze Detection'
    Grid_win_name = 'Grid Visualization'
    executing_path = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Can't receive frame. Exiting...")
            break        

        new_frame, maze_thresh, maze_region, roi, corners = md.detect_maze(frame)

        # --- guard for empty ROI / corners ---
        if roi is None or roi.size == 0 or corners is None or len(corners) < 3:
            cv2.imshow(win_name, new_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 1) preprocess ROI (grayscale + blur)
        processed = preprocess_maze(new_frame)

        # 2) get 4 outer corners from all detected corners (in FULL-IMAGE coords)
        external_corners = four_external_corners(corners)

        # 4) warp to a square view
        warped_maze = warp_from_corners(processed, external_corners, out_size=600)
        warped_color = warp_from_corners(new_frame, external_corners, out_size=600)
        grid_size = 50

        _, bin_warp = cv2.threshold(warped_maze, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        normal_grid = md.maze_to_grid(bin_warp, grid_size)

                          
        cell_px = bin_warp.shape[0] // grid_size
        margin_cells = 1                 # try 1 or 2 to stay further from walls
        ksize = max(3, 2*margin_cells*cell_px + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        safe_bin = cv2.dilate(bin_warp, kernel, iterations=1)  # "fatter" walls


        # 5) grid on the warped image (more stable)
        grid   = md.maze_to_grid(safe_bin, grid_size)
        visual = md.visualize_grid(bin_warp, normal_grid)

        # start, goal = find_entry_exit(grid)
        start, goal = find_start_goal_from_dots(warped_color=warped_color, grid=grid, start_color="green")

        solution_path = astar_search(grid, start, goal)

        robot_coords = []

        # Draw solution path on visual
        if solution_path and len(solution_path) > 1:
            h, w = visual.shape[:2]
            cell_h = h / grid.shape[0]
            cell_w = w / grid.shape[1]

            for idx in range(1, len(solution_path)):
                r1, c1 = solution_path[idx - 1]
                r2, c2 = solution_path[idx]

                center_x1 = int((c1 + 0.5) * cell_w)
                center_y1 = int((r1 + 0.5) * cell_h)
                center_x2 = int((c2 + 0.5) * cell_w)
                center_y2 = int((r2 + 0.5) * cell_h)

                cv2.line(visual, (center_x1, center_y1), (center_x2, center_y2), (255, 0, 0), 2)

                x, y = pixel_to_robot(center_x1, center_y1)
                robot_coords.append((x,y))


        # (optional) draw external corners on the main frame for debug
        for (cx, cy) in external_corners.astype(int):
            cv2.circle(new_frame, (cx, cy), 5, (0, 255, 0), -1)

        # ===== DISPLAY RESULT =====
        cv2.imshow(win_name, new_frame)
        if visual is not None:
            cv2.imshow(Grid_win_name, visual)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    time.sleep(2.0) 

    md = MazeDetector()

    threading.Thread(group=None, target=display_plain_video, args=(cap,))
    main(cap, md)