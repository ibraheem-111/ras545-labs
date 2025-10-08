class MockDobot:
    def __init__(self, port):
        print(f"Mock Dobot initialized (no real device connected)")
        
    def home(self):
        print("Mock: Homing device")
        
    def move_to(self, x, y, z, r=0, mode=None):
        print(f"Mock: Moving to x={x:.2f}, y={y:.2f}, z={z:.2f}")