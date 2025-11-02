import cv2
import os

def capture_images():
    # Initialize camera
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Counter for image names
    counter = 1

    print("Press SPACE to capture an image")
    print("Press 'q' to quit")

    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # If space is pressed, save the image
        if key == ord(' '):
            filename = f'chess{counter}.png'
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            counter += 1

        # If 'q' is pressed, quit
        elif key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()