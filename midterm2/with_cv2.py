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

# def display_video(cap):
#     prev_t = time.time()
#     win_name = 'Camera Stream + Maze Detection'
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Can't receive frame. Exiting...")
#             break

#         # Create a copy for visualization
#         display_frame = frame.copy()
        
#         # Convert to grayscale and apply preprocessing
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
#         # Use adaptive threshold to detect dark maze walls
#         thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                        cv2.THRESH_BINARY_INV, 15, 5)
        
#         # Apply morphological operations to clean up the image
#         kernel = np.ones((5, 5), np.uint8)
#         thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
#         thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
#         # Remove small noise
#         kernel_small = np.ones((3, 3), np.uint8)
#         thresh = cv2.erode(thresh, kernel_small, iterations=1)
#         thresh = cv2.dilate(thresh, kernel_small, iterations=1)

#         # Find contours - get all wall segments
#         contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Filter contours by area and aspect ratio to remove noise
#         min_area = 200
#         max_area = frame.shape[0] * frame.shape[1] * 0.8  # Max 80% of frame
#         valid_contours = []
#         for c in contours:
#             area = cv2.contourArea(c)
#             if min_area < area < max_area:
#                 # Filter by aspect ratio to remove very thin noise
#                 x, y, w, h = cv2.boundingRect(c)
#                 aspect_ratio = max(w, h) / (min(w, h) + 1)
#                 if aspect_ratio < 20:  # Remove very elongated contours
#                     valid_contours.append(c)
        
#         # Draw all valid wall contours
#         cv2.drawContours(display_frame, valid_contours, -1, (0, 255, 255), 2)
        
#         # Try to find the largest rectangular boundary that contains the maze
#         if valid_contours:
#             # Get bounding rectangle of all contours combined
#             all_points = np.vstack(valid_contours)
#             rect = cv2.minAreaRect(all_points)
#             box = cv2.boxPoints(rect)
#             box = box.astype(int)
            
#             # Draw the detected maze boundary
#             cv2.drawContours(display_frame, [box], 0, (0, 255, 0), 3)
            
#             # Draw corner points
#             for point in box:
#                 cv2.circle(display_frame, tuple(point), 8, (0, 255, 255), -1)
            
#             # Add status text
#             cv2.putText(display_frame, "Maze Detected", (10, 30),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(display_frame, f"Walls: {len(valid_contours)}", (10, 70),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
#             # Perform perspective transform on the detected region
#             width = int(rect[1][0])
#             height = int(rect[1][1])
            
#             # Ensure width > height for proper orientation
#             if width < height:
#                 width, height = height, width
            
#             # Sort corners for perspective transform
#             pts = sort_corners(box.astype(np.float32))
#             dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            
#             M = cv2.getPerspectiveTransform(pts, dst)
#             warp = cv2.warpPerspective(thresh, M, (width, height))
            
#             # Resize warped image for consistent processing
#             warp_resized = cv2.resize(warp, (500, 500))
            
#             # Apply additional cleaning to warped image
#             warp_cleaned = cv2.morphologyEx(warp_resized, cv2.MORPH_OPEN, 
#                                            np.ones((3, 3), np.uint8), iterations=1)
            
#             # Convert to grid for pathfinding
#             maze_grid = convert_maze_to_grid(warp_cleaned, cell_size=25)
            
#             # Visualize the grid
#             grid_visual = visualize_grid(maze_grid, warp_cleaned)
            
#             # Display the warped maze and grid
#             cv2.imshow("Warped Maze", warp_cleaned)
#             cv2.imshow("Pathfinding Grid", grid_visual)
        
#         # Calculate and display FPS
#         curr_t = time.time()
#         fps = 1 / (curr_t - prev_t)
#         prev_t = curr_t
#         cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 110),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
#         # Show threshold for debugging
#         cv2.imshow("Threshold", thresh)
        
#         cv2.imshow(win_name, display_frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


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