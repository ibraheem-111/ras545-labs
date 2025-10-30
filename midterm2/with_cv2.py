import cv2
import time
from maze_detection import MazeDetector, preprocess_maze, warp_from_corners, four_external_corners
import copy
from solve_maze import astar_search 

def display_video(cap, md: MazeDetector):
    prev_t = time.time()
    plain_win = "Camera Stream Only"
    win_name = 'Camera Stream + Maze Detection'
    Grid_win_name = 'Grid Visualization'

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Can't receive frame. Exiting...")
            break

        plain_frame = copy.deepcopy(frame)
        cv2.imshow(plain_win, plain_frame)

        new_frame, maze_thresh, maze_region, roi, corners = md.detect_maze(frame)

        # --- guard for empty ROI / corners ---
        if roi is None or roi.size == 0 or corners is None or len(corners) < 3:
            cv2.imshow(win_name, new_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 1) preprocess ROI (grayscale + blur)
        processed = preprocess_maze(roi)

        # 2) get 4 outer corners from all detected corners (in FULL-IMAGE coords)
        external_corners = four_external_corners(corners)

        # 3) translate corners to ROI coords before warping
        x0, y0, _, _ = maze_region
        ext_roi = external_corners.copy()
        ext_roi[:, 0] -= x0
        ext_roi[:, 1] -= y0

        # 4) warp to a square view
        warped_maze = warp_from_corners(processed, ext_roi, out_size=600)

        # 5) grid on the warped image (more stable)
        grid = md.maze_to_grid(warped_maze, 50)
        visual = md.visualize_grid(warped_maze, grid)

        solution_path = astar_search(grid, (0,0), (grid.shape[0]-1, grid.shape[1]-1))

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

        # (optional) draw external corners on the main frame for debug
        for (cx, cy) in external_corners.astype(int):
            cv2.circle(new_frame, (cx, cy), 5, (0, 255, 0), -1)

        # ===== FPS COUNTER =====
        now = time.time()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(new_frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

    display_video(cap, md)