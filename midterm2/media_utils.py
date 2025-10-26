import cv2
import time
from  maze_detection import detect_maze

def capture_image(cap):
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to capture image from camera.")
    # Convert numpy array to JPEG bytes
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        return None
    
    return encoded_image.tobytes(), frame
    

def display_video(cap):
    prev_t = time.time()
    win_name = 'Camera Stream + Maze Detection'

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        new_frame, _, _ = detect_maze(frame)

        # ===== FPS COUNTER =====
        now = time.time()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(new_frame, f"FPS: {fps:.1f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ===== DISPLAY RESULT =====
        cv2.imshow(win_name, new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()