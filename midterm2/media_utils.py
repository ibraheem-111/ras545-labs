import cv2
import time

def capture_image(cap):
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to capture image from camera.")
    # Convert numpy array to JPEG bytes
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        return None
    
    return encoded_image.tobytes(), frame