import cv2
import time
import numpy as np
from ultralytics import YOLO
from media_utils import display_video
from maze_detection import maze_to_grid, detect_maze, visualize_grid

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

def display_video(cap):
    prev_t = time.time()
    win_name = 'Camera Stream + Maze Detection'
    Grid_win_name = 'Grid Visualization'

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        new_frame, maze_thresh, maze_region = detect_maze(frame)

        grid = maze_to_grid(maze_thresh, 50)

        visual = visualize_grid(maze_thresh, grid)

        # ===== FPS COUNTER =====
        now = time.time()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(new_frame, f"FPS: {fps:.1f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ===== DISPLAY RESULT =====
        cv2.imshow(win_name, new_frame)
        cv2.imshow(Grid_win_name, visual)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



display_video(cap)