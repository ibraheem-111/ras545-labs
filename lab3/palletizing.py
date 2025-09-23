import argparse
import traceback
import sys
import time
from typing import Optional, Tuple

import cv2
from ultralytics import YOLO

from pydobot.dobot import MODE_PTP
import pydobot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 3: YOLO classification + Dobot palletizing using suction"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="OpenCV camera index (try 0/1/2).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 model weights (e.g., yolov8n.pt).",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM1",
        help="Serial port for Dobot (e.g., /dev/ttyACM0 or /dev/ttyUSB0).",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Number of pick-place cycles to run (0 for infinite until 'q').",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show camera window with overlay and FPS.",
    )
    return parser.parse_args()


# Robot poses - UPDATE THESE COORDINATES FOR YOUR SETUP
HOME: Tuple[float, float, float, float] = (233.1162, -1.3118, 150.7647, -0.3224)

# Fixed position where cardboard pieces with object pictures are stacked
CARDBOARD_STACK: Tuple[float, float, float, float] = (297.28350830078125, 51.12328338623047, -45.1676025390625, 9.757606506347656)

# Two pallets for different object categories
PALLET_A: Tuple[float, float, float, float] = (254.5684814453125, -46.615028381347656, -21.938823699951172, -10.376693725585938)  # Food items
PALLET_B: Tuple[float, float, float, float] = (352, -46.615028381347656, -21.938823699951172, -10.376693725585938)  # Vehicle items

# Z heights for suction
SAFE_Z: float = -20.0  # Safe height above objects
Z_SUCK_START: float = -45.0  # Starting suction height
Z_SUCK_INCREMENT: float = -3  # How much to lower each attempt
MAX_Z_SUCK: float = -55.0  # Maximum depth to try


# Category mapping from YOLO labels to pallets
FOOD_LABELS = {
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
    "hot dog", "pizza", "donut", "cake", "fork", "spoon"
}

VEHICLE_LABELS = {
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"
}


def connect_robot(serial_port: str) -> pydobot.Dobot:
    device = pydobot.Dobot(port=serial_port)
    device.speed(1000, 1000)
    return device


def move_linear(device: pydobot.Dobot, x: float, y: float, z: float, r: float, wait: bool = True) -> None:
    device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x, y=y, z=z, r=r)


def home(device: pydobot.Dobot) -> None:
    device.home()

def intermediate(device):
    move_linear(device, *CARDBOARD_STACK)


def detect_object_category(model: YOLO, frame) -> Optional[str]:
    """
    Run YOLO and return the category (A or B) of the detection closest to image center.
    Returns None if no valid detections.
    """
    results = model(frame, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    names = r.names

    h, w = frame.shape[:2]
    cx_img, cy_img = w / 2.0, h / 2.0

    best_idx = -1
    best_dist = 1e9
    for i, ((x1, y1, x2, y2), c, k) in enumerate(zip(xyxy, conf, cls)):
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        dist = (mx - cx_img) ** 2 + (my - cy_img) ** 2
        if dist < best_dist or (dist == best_dist and c > conf[best_idx]):
            best_dist = dist
            best_idx = i

    if best_idx == -1:
        return None

    label_idx = cls[best_idx]
    label_name = names[label_idx].lower()
    
    if label_name in FOOD_LABELS:
        return "A"
    elif label_name in VEHICLE_LABELS:
        return "B"
    else:
        return None
# Try suction at different Z heights
current_z = Z_SUCK_START

def pick_from_stack(device: pydobot.Dobot, model: YOLO, cap) -> bool:
    """
    Pick up an object from the cardboard stack with incremental Z lowering.
    Returns True if successful, False if failed after all attempts.
    """
    x, y, _, r = CARDBOARD_STACK
    
    # Move to safe height above stack
    move_linear(device, x, y, SAFE_Z, r)
    time.sleep(1)  # Wait for movement to complete
    
    # Get initial detection before picking
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame for initial detection")
        return False
    
    initial_category = detect_object_category(model, frame)
    print(f"Initial category before pick: {initial_category}")
    
    global current_z
    while current_z >= MAX_Z_SUCK:
        print(f"Trying suction at Z = {current_z}")
        
        # Move to current Z height
        move_linear(device, x, y, current_z, r)
        time.sleep(1)  # Wait for movement to complete
        
        # Activate suction
        device.suck(True)
        time.sleep(1)  # Give suction time to engage
        
        # Try to lift - if successful, we got something
        move_linear(device, x, y, SAFE_Z, r)
        time.sleep(1)  # Wait for movement to complete
        
        # Check if we successfully picked up the object by detecting again
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame for success check")
            device.suck(False)
            return False
            
        # new_category = detect_object_category(model, frame)
        # print(f"Category after pick attempt: {new_category}")
        
        # Success if: no object detected OR different object detected
        # if new_category != initial_category:
        # print(f"Successfully picked up object at Z = {current_z} (category changed from {initial_category} to {new_category})")
        print(current_z)
        current_z = current_z + Z_SUCK_INCREMENT
        return True
        # else:
        #     print(f"Pick failed at Z = {current_z} (same category still detected)")
        #     # Release suction and try lower Z
        #     device.suck(False)
        #     current_z += Z_SUCK_INCREMENT  # Move to next Z level
        #     time.sleep(0.5)
    
    print("Failed to pick up object after all attempts")
    device.suck(False)
    return False


def place_in_pallet(device: pydobot.Dobot, category: str) -> None:
    """Place the picked object in the appropriate pallet"""
    if category == "A":
        target = PALLET_A
        pallet_name = "A (Food)"
    else:
        target = PALLET_B
        pallet_name = "B (Vehicle)"
    
    tx, ty, tz, tr = target
    print(f"Placing object in pallet {pallet_name}")
    
    # Move to pallet location
    move_linear(device, tx, ty, SAFE_Z, tr)
    time.sleep(1)  # Wait for movement to complete
    move_linear(device, tx, ty, tz, tr)
    time.sleep(1)  # Wait for movement to complete
    
    # Release suction
    device.suck(False)
    time.sleep(0.5)
    
    # Lift up
    move_linear(device, tx, ty, SAFE_Z, tr)
    time.sleep(1)  # Wait for movement to complete


def run(args: argparse.Namespace) -> None:
    # Initialize camera
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check --camera-index and permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Load model
    model = YOLO(args.model)

    # Connect robot
    device = connect_robot(args.port)
    home(device)

    prev_t = time.time()
    cycles_run = 0
    win_name = "Palletizing View"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame. Exiting...")
                break

            # Detect object category
            category = detect_object_category(model, frame)
            
            # Add debugging to see what's being detected
            if category is not None:
                print(f"DEBUG: Detected category {category}")

            if args.show:
                # Show overlay information
                now = time.time()
                fps = 1.0 / max(1e-6, (now - prev_t))
                prev_t = now
                h, w = frame.shape[:2]
                cv2.circle(frame, (w // 2, h // 2), 8, (0, 255, 255), 2)
                
                category_text = f"Category: {category or 'None'}"
                cv2.putText(frame, category_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow(win_name, frame)

            # If a valid category is detected, perform pick-and-place
            if category is not None:
                print(f"Detected category {category} -> executing pick/place")
                
                # Pick from stack
                pick_from_stack(device, model, cap)
                    # Place in appropriate pallet
                place_in_pallet(device, category)
                # home(device)
                intermediate(device)
                cycles_run += 1
                print(f"Completed cycle {cycles_run}")
                # else:
                #     print("Failed to pick up object, skipping this cycle")
                #     home(device)

                if args.loop > 0 and cycles_run >= args.loop:
                    print("Completed requested number of cycles. Exiting.")
                    break

            # UI + quit handling
            if args.show:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Without UI, modest sleep to avoid busy-looping the camera
                time.sleep(0.02)
    finally:
        # Cleanup
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            # Release suction and go home for safety
            device.suck(False)
            home(device)
            device.close()
        except Exception:
            pass


if __name__ == "__main__":
    cli_args = parse_args()
    try:
        run(cli_args)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"Error: {exc}")
        traceback.print_exc()
        sys.exit(1)