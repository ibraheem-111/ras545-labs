import cv2
import time
from maze_detection import MazeDetector, preprocess_maze, warp_from_corners, four_external_corners
import copy

def display_video(cap, md: MazeDetector):
    prev_t = time.time()
    plain_win = "Camera Stream Only"
    win_name = 'Camera Stream + Maze Detection'
    Grid_win_name = 'Grid Visualization'

    while True:
        ret, frame = cap.read()
        plain_frame = copy.deepcopy(frame)

        if not ret:
            print("Can't receive frame. Exiting...")
            break

        new_frame, maze_thresh, maze_region, roi,corners = md.detect_maze(frame)


        if roi is None or roi.size == 0:
            continue

        processed = preprocess_maze(roi)

        external_corners = four_external_corners(corners)

        warped_maze = warp_from_corners(processed, external_corners, out_size=600)

        grid = md.maze_to_grid(maze_thresh, 50)

        visual = md.visualize_grid(warped_maze, grid)

        # ===== FPS COUNTER =====
        now = time.time()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(new_frame, f"FPS: {fps:.1f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ===== DISPLAY RESULT =====
        cv2.imshow(plain_win, plain_frame)
        cv2.imshow(win_name, new_frame)
        cv2.imshow(Grid_win_name, visual)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    time.sleep(2.0) 

    md = MazeDetector()

    display_video(cap, md)