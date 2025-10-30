import cv2
import numpy as np
from collections import deque
import time

# --- Robot imports ---
from pydobot import Dobot
import serial.tools.list_ports

# ----------------------------
# Utility: pick Dobot port automatically when possible
# ----------------------------
def find_dobot_port(default='/dev/ttyACM1'):
    try:
        ports = list(serial.tools.list_ports.comports())
    except Exception:
        return default
    for p in ports:
        name = (p.description or '') + ' ' + (p.manufacturer or '')
        if any(k in name.lower() for k in ['dobot', 'usb serial', 'wch', 'ch340', 'arduino']):
            return p.device
    return default

# ----------------------------
# Corner ordering + drawing
# ----------------------------
def order_corners_clockwise(pts4):
    """Return points in TL, TR, BR, BL order."""
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def draw_click_ui(frame, pts, msg_lines):
    vis = frame.copy()
    h, w = vis.shape[:2]
    # banner
    cv2.rectangle(vis, (0, 0), (w, 32), (60, 60, 60), -1)
    cv2.putText(vis, "CLICK MODE: left-add/replace | right-remove | A:auto-order | U:undo | C:clear | SPACE/ENTER: freeze",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    # instructions
    for i, text in enumerate(msg_lines or []):
        cv2.putText(vis, text, (10, 56 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    # draw points
    labels = ['1 TL', '2 TR', '3 BR', '4 BL'] if len(pts) == 4 else [str(i+1) for i in range(len(pts))]
    for i, p in enumerate(pts):
        x, y = int(p[0]), int(p[1])
        cv2.circle(vis, (x, y), 6, (0, 200, 255), -1)
        cv2.putText(vis, labels[i if i < len(labels) else -1], (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2)
    return vis

def draw_pair_ui(frame, corners, idx, recorded_count):
    vis = frame.copy()
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w, 32), (40, 160, 40), -1)
    cv2.putText(vis, "PAIR MODE: ENTER=record | N/B=next/back | R=reset to clicks | Q=quit",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    order_text = "Order: 1 TL → 2 TR → 3 BR → 4 BL"
    cv2.putText(vis, order_text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)

    colors = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 255, 255)]
    labels = ['1 TL', '2 TR', '3 BR', '4 BL']
    for i, c in enumerate(corners):
        x, y = int(c[0]), int(c[1])
        cv2.circle(vis, (x, y), 6, colors[i], -1)
        cv2.putText(vis, labels[i], (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[i], 2)
    # highlight current
    cx, cy = int(corners[idx][0]), int(corners[idx][1])
    cv2.circle(vis, (cx, cy), 14, (0, 255, 255), 3)
    cv2.putText(vis, f"Jog robot to: {labels[idx]}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
    cv2.putText(vis, f"Recorded: {recorded_count}/4", (w-180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis

# ----------------------------
# Mouse helpers
# ----------------------------
def nearest_idx(pts, x, y):
    if not pts:
        return None
    arr = np.array(pts, dtype=np.float32)
    d = np.linalg.norm(arr - np.array([x, y], dtype=np.float32), axis=1)
    return int(np.argmin(d))

# ----------------------------
# Main
# ----------------------------
def main():
    # Robot
    port = find_dobot_port('/dev/ttyACM1')
    device = Dobot(port)
    print(f"Robot initialized on {port}")
    print("Moving device to home...")
    try:
        device.home()
    except Exception as e:
        print(f"Warning: home() failed or not supported ({e})")

    # Camera
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open camera (index 2). Try 0/1 if needed.")
        device.close()
        return

    # State
    click_points = []        # user clicks (up to 4), any order
    frozen = False
    frozen_frame = None
    ordered_corners = None
    current_idx = 0
    robot_points = []

    # Mouse callback (uses nonlocal state via closure)
    def on_mouse(event, x, y, flags, userdata):
        nonlocal click_points
        if frozen:
            return  # ignore clicks in pairing mode
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(click_points) < 4:
                click_points.append([float(x), float(y)])
            else:
                # replace nearest point
                ni = nearest_idx(click_points, x, y)
                if ni is not None:
                    click_points[ni] = [float(x), float(y)]
        elif event == cv2.EVENT_RBUTTONDOWN:
            ni = nearest_idx(click_points, x, y)
            if ni is not None:
                click_points.pop(ni)

    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', on_mouse)

    print("\nClick 4 corners in the live view.")
    print("Controls:")
    print("  Left-click = add/replace, Right-click = remove nearest")
    print("  A = auto-order (TL,TR,BR,BL), U = undo, C = clear")
    print("  SPACE/ENTER = freeze and start pairing")
    print("Pairing:")
    print("  ENTER = record current, N/B = next/back, R = reset, Q = quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            if not frozen:
                # Live clicking mode
                msg = []
                if len(click_points) < 4:
                    msg.append(f"Click corners: {len(click_points)}/4 selected")
                else:
                    msg.append("Press SPACE/ENTER to freeze, or A to auto-order")

                disp = draw_click_ui(frame, click_points, msg)
                cv2.imshow('Calibration', disp)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('u'):
                    if click_points:
                        click_points.pop()
                elif key == ord('c'):
                    click_points.clear()
                elif key == ord('a'):
                    if len(click_points) == 4:
                        click_points = order_corners_clockwise(click_points).tolist()
                        print("Auto-ordered to TL,TR,BR,BL.")
                    else:
                        print("Need 4 points to auto-order.")
                elif key in (13, 10, ord(' ')):  # ENTER or SPACE
                    if len(click_points) == 4:
                        # Freeze and move to pairing phase
                        frozen = True
                        frozen_frame = frame.copy()
                        # Ensure consistent order
                        ordered_corners = np.float32(order_corners_clockwise(click_points))
                        current_idx = 0
                        robot_points = []
                        print("Frame frozen. Pair corners with robot in order: 1 TL → 2 TR → 3 BR → 4 BL.")
                    else:
                        print("Select 4 points first.")
            else:
                # Pairing mode (frozen frame)
                disp = draw_pair_ui(frozen_frame, ordered_corners, current_idx, len(robot_points))
                cv2.imshow('Calibration', disp)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # back to clicking
                    frozen = False
                    frozen_frame = None
                    ordered_corners = None
                    current_idx = 0
                    robot_points = []
                    print("Reset to clicking mode.")
                elif key == ord('n'):
                    current_idx = min(3, current_idx + 1)
                elif key == ord('b'):
                    current_idx = max(0, current_idx - 1)
                elif key in (13, 10):  # ENTER to record this corner
                    try:
                        pose = device.get_pose()
                        robot_points.append([pose.position.x, pose.position.y])  # x, y
                    except Exception as e:
                        print(f"Warning: could not read robot pose ({e}). Using [0,0].")
                        robot_points.append([0.0, 0.0])

                    print(f"Recorded corner {current_idx+1}: "
                          f"Camera {ordered_corners[current_idx].tolist()}, "
                          f"Robot  {robot_points[-1]}")

                    current_idx += 1
                    if current_idx == 4:
                        camera_points = np.float32(ordered_corners)
                        robot_points_np = np.float32(robot_points)
                        np.save('camera_points.npy', camera_points)
                        np.save('robot_points.npy', robot_points_np)
                        print("\nCalibration complete! Saved:")
                        print(" - camera_points.npy")
                        print(" - robot_points.npy")
                        break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            device.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
