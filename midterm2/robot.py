import pydobot
from pydobot.dobot import MODE_PTP
import threading
import time
from dataclasses import dataclass
from queue import Queue
from enum import Enum

from celery import Celery
import multiprocessing as mp
import time
class MockDobot:
    def __init__(self, port):
        print(f"Mock Dobot initialized (no real device connected)")
        
    def home(self):
        print("Mock: Homing device")
        
    def move_to(self, x, y, z, r=0, mode=None):
        print(f"Mock: Moving to x={x:.2f}, y={y:.2f}, z={z:.2f}")
# device = pydobot.Dobot('/dev/ttyACM0')
device = MockDobot(port="/dev/ttyACM0")

class IntermediatePoint(Enum):
    x = 240
    y=0
    z = 80


def start_celery_worker_in_bg(app, loglevel="INFO", pool="solo", concurrency=1):
    """
    Starts a Celery worker in a separate process and waits until it responds to ping.
    - pool='solo' avoids fork issues in GUI apps.
    - concurrency=1 ensures robot moves are serialized.
    Returns the Process handle.
    """
    def _run():
        app.worker_main([
            "worker",
            f"--loglevel={loglevel}",
            f"--pool={pool}",
            f"--concurrency={concurrency}",
            "-n", "local.%h",
        ])

    p = mp.Process(target=_run, daemon=True)
    p.start()

    # Wait for worker to come up
    for _ in range(40):  # ~20s
        try:
            if app.control.ping(timeout=0.5):
                return p
        except Exception:
            pass
        time.sleep(0.5)

    raise RuntimeError("Celery worker didn't start. Check broker/backend settings.")

app = Celery(
    "robot",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

app.conf.update(
    task_acks_late=True,           # re-queue on worker crash
    worker_prefetch_multiplier=1,  # helps preserve order
    task_routes={"robot.tasks.*": {"queue": "robot1"}},  # per-robot queue
)

def move_robot_to(x, y, z, device: pydobot.Dobot):
    # Replace with your real SDK calls
    print(f"Control: moving to ({x:.2f}, {y:.2f}, {z:.2f}) ")
    device.move_to(x, y, z, 0, mode = MODE_PTP.MOVL_XYZ)


@app.task(name="robot.tasks.move")
def move(x, y, z):
    move_robot_to(x, y, z, device)

@app.task(name="robot.tasks.home")
def home():
    device.home()

@app.task(name = "robot.tasks.intermediate")
def intermediate():
    move_robot_to(IntermediatePoint.x.value, IntermediatePoint.y.value, IntermediatePoint.z.value, device=device)
