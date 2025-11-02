import cv2
import time
from maze_detection import MazeDetector, preprocess_maze, warp_from_corners, four_external_corners
from solve_maze import astar_search, find_start_goal_from_dots, turning_nodes_from_grid_path
import numpy as np
from robot import move, home, intermediate, app, start_celery_worker_in_bg
import threading
from point2coordinates import pixel_to_robot
from constants import Z_Draw
from midterm2.vlm import RealtimeMazeSentry
from test_clip import OpenClipSentry

def display_plain_video(hub):
    prev_t = time.time()
    plain_win = "Camera Stream Only"

    while True:
        frame = hub.get()
        cv2.imshow(plain_win, frame)

        # ===== FPS COUNTER =====
        now = time.time()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        

        
class FrameHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.ready = threading.Event()

    def update(self, frame):
        with self.lock:
            self.frame = frame  # store latest
            self.ready.set()

    def get(self):
        self.ready.wait()
        with self.lock:
            return None if self.frame is None else self.frame.copy()

def grabber(cap, hub, stop):
    while not stop.is_set():
        ret, f = cap.read()
        if not ret or f is None:
            break
        hub.update(f)

def main(hub, md: MazeDetector):
    win_name = 'Camera Stream + Maze Detection'
    Grid_win_name = 'Grid Visualization'
    executing_path = False
    warp_size = 600  

    while True:
        frame = hub.get()
        # if not ret or frame is None:
        #     print("Can't receive frame. Exiting...")
        #     break

        new_frame, maze_thresh, maze_region, roi, corners = md.detect_maze(frame)

        # --- guard for empty ROI / corners ---
        if roi is None or roi.size == 0 or corners is None or len(corners) < 3:
            cv2.imshow(win_name, new_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 1) preprocess ROI (grayscale + blur)
        processed = preprocess_maze(new_frame)

        # 2) 4 outer corners (FULL-IMAGE coords)
        external_corners = four_external_corners(corners)

        # 4) warp to a square view
        warped_maze  = warp_from_corners(processed,  external_corners, out_size=warp_size)
        warped_color = warp_from_corners(new_frame, external_corners, out_size=warp_size)
        grid_size = 50

        # <-- ADDED: inverse homography (warped -> full image)
        dst = np.float32([[0,0],[warp_size,0],[warp_size,warp_size],[0,warp_size]])  # TL,TR,BR,BL
        Hinv = cv2.getPerspectiveTransform(dst, external_corners.astype(np.float32))

        _, bin_warp = cv2.threshold(warped_maze, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        normal_grid = md.maze_to_grid(bin_warp, grid_size)

        cell_px = bin_warp.shape[0] // grid_size
        margin_cells = 1
        ksize = max(3, 2*margin_cells*cell_px + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        safe_bin = cv2.dilate(bin_warp, kernel, iterations=1)

        # 5) grid on the warped image
        grid   = md.maze_to_grid(safe_bin, grid_size)
        visual = md.visualize_grid(bin_warp, normal_grid)

        start, goal = find_start_goal_from_dots(warped_color=warped_color, grid=grid, start_color="green")
        solution_path = astar_search(grid, start, goal)

        # Draw solution path on visual (unchanged)
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

        # (optional) debug: draw external corners on the main frame
        for (cx, cy) in external_corners.astype(int):
            cv2.circle(new_frame, (cx, cy), 5, (0, 255, 0), -1)

        # ===== DISPLAY RESULT =====
        cv2.imshow(win_name, new_frame)
        if visual is not None:
            cv2.imshow(Grid_win_name, visual)

        cv2.waitKey(2)
        time.sleep(2)

        # <-- ADDED: build sparse nodes, map to image, convert to robot, send to Celery (and block)
        if solution_path and len(solution_path) > 1 and not executing_path:
            executing_path = True  # prevent re-entry (also we block below)

            # 1) keep only turning nodes (start, corners, goal)
            turn_nodes = turning_nodes_from_grid_path(solution_path)

            # 2) centers of those grid cells in WARPED pixels
            cell = warp_size / float(grid_size)
            pts_warped = np.array([[(c + 0.5) * cell, (r + 0.5) * cell] for (r, c) in turn_nodes], np.float32)
            pts_warped = pts_warped.reshape(-1, 1, 2)

            # 3) warped -> IMAGE pixels (batch)
            pts_img = cv2.perspectiveTransform(pts_warped, Hinv).reshape(-1, 2)

            # 4) IMAGE pixels -> robot coordinates
            robot_coords = [pixel_to_robot(float(x), float(y)) for (x, y) in pts_img]

            try:
                follow_path(robot_coords)

            finally:
                executing_path = False  # allow future runs
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def follow_path(robot_coords):
    # 5) send to Celery tasks and BLOCK until done
    # go to an intermediate safe pose first (if your task needs args, add them)
    res = intermediate.delay()
    res.get()  # block

    print(robot_coords)

    # trace nodes
    for (rx, ry) in robot_coords:
        r = move.delay(rx.item(), ry.item(), Z_Draw)  # adjust signature to your task
        r.get()  # block until this segment finishes

    # go home when finished
    res = home.delay()
    res.get()

    time.sleep(10)



def startup():
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    hub = FrameHub()
    stop = threading.Event()
    threading.Thread(target=grabber, args=(cap, hub, stop), daemon=True).start()

    time.sleep(2.0)

    # worker_proc = start_celery_worker_in_bg(app, loglevel="INFO", pool="solo", concurrency=1)
    md = MazeDetector()
    # optional: run a startup task
    intermediate.delay().get()

    # threading.Thread(target=display_plain_video, args=(hub,))
    # print("Waiting for video to start")
    # time.sleep(3)
    main(hub, md)

    cap.release()

def run_sentry_and_live(camera_index=3, model_fps=1, stop_after_trigger=False):
    """
    Opens the camera, starts the FrameHub grabber, starts the RealtimeMazeSentry,
    shows a live preview, and waits until the sentry invokes main(hub, md).
    - camera_index: which camera to open
    - model_fps: frames/second to send to the Realtime model (throttled inside sentry)
    - stop_after_trigger: if True, exit the live loop right after triggering main()
    """
    # --- camera + hub ---
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    if not cap.isOpened():
        print("Camera failed to open.")
        return

    hub = FrameHub()
    stop = threading.Event()
    threading.Thread(target=grabber, args=(cap, hub, stop), daemon=True).start()

    # MazeDetector instance for your main()
    md = MazeDetector()

    # --- sentry hookup ---
    triggered = threading.Event()

    # home.delay().get()
    # intermediate.delay().get()

    def _on_trigger():
        # Debounce + call your main(hub, md) in its own thread
        if triggered.is_set():
            return
        triggered.set()
        threading.Thread(target=main, args=(hub, md), daemon=True).start()
        print("[Sentry] Triggered main(hub, md).")

    # Use your existing class import: from main_midterm2 import RealtimeMazeSentry
    sentry = RealtimeMazeSentry(on_trigger=_on_trigger, fps=model_fps)

    # --- live preview loop ---
    win = "Live"
    prev_t = time.time()
    try:
        while True:
            frame = hub.get()  # latest frame from grabber
            if frame is None:
                continue

            # send to model (throttled inside)
            sentry.maybe_send(frame)

            # overlay status + fps
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            view = frame.copy()
            status = "Waiting for maze…" if not triggered.is_set() else "Maze detected (running main)"
            cv2.putText(view, f"{status} | FPS {fps:.1f}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(win, view)

            if stop_after_trigger and triggered.is_set():
                cv2.waitKey(300)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # cleanup
        try:
            sentry.stop()
        except Exception:
            pass
        stop.set()
        cap.release()
        cv2.destroyAllWindows()



# if __name__ == "__main__":
#     startup()
#     cap = cv2.VideoCapture(1)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

#     hub = FrameHub()
#     stop = threading.Event()
#     threading.Thread(target=grabber, args=(cap, hub, stop), daemon=True).start()

#     time.sleep(2.0)

#     # worker_proc = start_celery_worker_in_bg(app, loglevel="INFO", pool="solo", concurrency=1)
#     md = MazeDetector()
#     # optional: run a startup task
#     intermediate.delay().get()

#     # threading.Thread(target=display_plain_video, args=(hub,))
#     # print("Waiting for video to start")
#     # time.sleep(3)
#     # main(hub, md)

#     display_plain_video(hub)

#     # cap.release()
# if __name__ == "__main__":
#     run_sentry_and_live(camera_index=0, model_fps=1)


if __name__ == "__main__":
    # --- camera settings ---
    CAMERA_INDEX = 0     # change if needed
    MODEL_FPS    = 1.0   # how often to check frames
    THRESHOLD    = 0.80  # p(maze) to trigger main()

    # (optional) sanity print
    try:
        import torch
        print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
    except Exception:
        pass

    # --- open camera + start grabber -> FrameHub ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    if not cap.isOpened():
        print("Camera failed to open."); raise SystemExit(1)

    hub = FrameHub()
    stop = threading.Event()
    threading.Thread(target=grabber, args=(cap, hub, stop), daemon=True).start()

    # --- detector + sentry wiring ---
    md = MazeDetector()
    triggered = threading.Event()

    # OpenCLIP sentry (assumes OpenClipSentry class is defined above in this file)
    def _on_trigger():
        if triggered.is_set():
            return
        triggered.set()
        print("[OpenCLIP] Triggering main(hub, md)…")
        threading.Thread(target=main, args=(hub, md), daemon=True).start()

    sentry = OpenClipSentry(
        on_trigger=_on_trigger,
        threshold=THRESHOLD,
        fps=MODEL_FPS
    )

    # --- live preview loop ---
    win = "Live (OpenCLIP)"
    prev_t = time.time()
    try:
        while True:
            frame = hub.get()
            if frame is None:
                continue

            # evaluate ~MODEL_FPS
            sentry.maybe_send(frame)

            # overlay simple status + FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            view = frame.copy()
            status = "Waiting for maze…" if not triggered.is_set() else "Maze detected (running main)"
            cv2.putText(view, f"{status} | FPS {fps:.1f}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow(win, view)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stop.set()
        cap.release()
        cv2.destroyAllWindows()
