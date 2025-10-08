from pydobot.dobot import MODE_PTP
import time
from constants import origin_x, origin_y, safe_height, drawing_height, cell_size

def intermediate(device):
    device.move_to(x=231.0052032470703, y=-9.224393844604492, z=31.286697387695312, r=-4.332102298736572, mode = MODE_PTP.MOVJ_XYZ)

def make_grid(device):
    # Starting position - move down to drawing height
    device.move_to(x=origin_x, y=origin_y, z=safe_height, r=0, mode=MODE_PTP.MOVJ_XYZ)

    # Draw first vertical line (left) at x = origin_x + 33.33
    device.move_to(x=origin_x, y=origin_y-cell_size/2, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=origin_x, y=origin_y-cell_size/2, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    time.sleep(0.5)
    device.move_to(x=origin_x + 3*cell_size, y=origin_y-cell_size/2, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=origin_x + 3*cell_size, y=origin_y-cell_size/2, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)

    # Draw first vertical line (right) at x = origin_x + 33.33
    device.move_to(x=origin_x, y=origin_y+cell_size/2, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=origin_x, y=origin_y+cell_size/2, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    time.sleep(0.5)
    device.move_to(x=origin_x + 3*cell_size, y=origin_y+cell_size/2, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=origin_x + 3*cell_size, y=origin_y+cell_size/2, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)

    # Draw first vertical line (right) at x = origin_x + 33.33
    start_y= origin_y+3*cell_size/2
    end_y = origin_y-3*cell_size/2
    x = origin_x+cell_size
    device.move_to(x=x, y=start_y, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=x, y=start_y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    time.sleep(0.5)
    device.move_to(x=x, y=end_y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=x, y=end_y+cell_size, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)


    # Draw first vertical line (right) at x = origin_x + 33.33
    start_y= origin_y+3*cell_size/2
    end_y = origin_y-3*cell_size/2
    x = origin_x+2*cell_size
    device.move_to(x=x, y=start_y, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=x, y=start_y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    time.sleep(0.5)
    device.move_to(x=x, y=end_y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=x, y=end_y+cell_size, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)

    intermediate(device)


def draw_x(row, col, device):
    """Draw an X in the specified cell (row, col are 0-2)"""
    # Calculate cell center
    center_x = origin_x + col * cell_size + cell_size / 2
    center_y = origin_y + row * cell_size - cell_size 
    
    
    # X size (slightly smaller than cell)
    x_size = cell_size * 0.6
    offset = x_size / 2
    
    # Draw first diagonal (\)
    device.move_to(x=center_x - offset, y=center_y - offset, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=center_x - offset, y=center_y - offset, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=center_x + offset, y=center_y + offset, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    
    # Draw second diagonal (/)
    device.move_to(x=center_x + offset, y=center_y - offset, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=center_x + offset, y=center_y - offset, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=center_x - offset, y=center_y + offset, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    
    # Lift pen
    device.move_to(x=center_x - offset, y=center_y + offset, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
    intermediate(device)

def draw_o(row, col, device):
    """Draw an O in the specified cell (row, col are 0-2)"""
    import math
    
    # Calculate cell center
    center_x = origin_x + col * cell_size + cell_size / 2
    center_y = origin_y + row * cell_size - cell_size 
    
    # Circle radius (slightly smaller than cell)
    radius = cell_size * 0.3
    num_points = 10  # Number of points to approximate circle
    
    drawing_height= 10
    # Move to starting point
    start_x = center_x + radius
    start_y = center_y
    device.move_to(x=start_x, y=start_y, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=start_x, y=start_y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)

    # Draw circle
    for i in range(num_points + 1):
        angle = 2 * math.pi * i / num_points
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        device.move_to(x=x, y=y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    
    # Lift pen
    device.move_to(x=start_x, y=start_y, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
    intermediate(device)