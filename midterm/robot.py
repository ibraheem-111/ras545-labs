from pydobot.dobot import MODE_PTP
import time
from constants import origin_x, origin_y, safe_height, drawing_height, cell_size, grid_size

def intermediate(device):
    device.move_to(x=201.0052032470703, y=-9.224393844604492, z=70.286697387695312, r=-4.332102298736572, mode = MODE_PTP.MOVJ_XYZ)

def intermediate2(device):
    device.move_to(x=240, y=-0, z=100, r=0, mode = MODE_PTP.MOVJ_XYZ)

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
    col, row = 2-row, 2-col
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
    col, row = 2-row, 2-col
    
    # Calculate cell center
    center_x = origin_x + col * cell_size + cell_size / 2
    center_y = origin_y + row * cell_size - cell_size 
    
    # Circle radius (slightly smaller than cell)
    radius = cell_size * 0.3
    num_points = 10  # Number of points to approximate circle
    
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

def draw_x_d(row, col, device, grid_center_x, grid_center_y, ws_w):
    """Draw an X in the specified cell (row, col are 0-2)"""

    origin_x, origin_y = grid_center_x - ws_w/2, grid_center_y + ws_w/2
    cell_size = ws_w / 3

    # Calculate cell center
    col, row = 2-row, 2-col
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

def draw_o_d(row, col, device, center_x, center_y, ws_w):
    """Draw an O in the specified cell (row, col are 0-2)"""
    import math

    origin_x, origin_y = center_x - ws_w/2, center_y + ws_w/2
    cell_size = ws_w / 3
    col, row = 2-row, 2-col
    
    # Calculate cell center
    center_x = origin_x + col * cell_size + cell_size / 2
    center_y = origin_y + row * cell_size - cell_size 
    
    # Circle radius (slightly smaller than cell)
    radius = cell_size * 0.3
    num_points = 10  # Number of points to approximate circle
    
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

def draw_d(device):
    """Draw a 'D' next to the board to indicate a draw (with inverted coordinates)"""
    # Position D to the right of the board
    d_x = origin_x + grid_size + 20  # 20mm to the right of grid
    d_y = origin_y + grid_size / 2   # Centered vertically
    
    # INVERT Y coordinate
    d_y = origin_y + grid_size - (d_y - origin_y)
    
    d_height = 40
    d_width = 25
    
    # Move to starting position (top of D)
    device.move_to(x=d_x, y=d_y - d_height/2, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
    device.move_to(x=d_x, y=d_y - d_height/2, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    
    # Draw left vertical line
    device.move_to(x=d_x, y=d_y + d_height/2, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
    
    # Draw bottom curve (approximate with points)
    num_points = 15
    for i in range(num_points + 1):
        angle = -90 + (180 * i / num_points)  # From -90° to +90°
        angle_rad = angle * 3.14159 / 180
        x = d_x + d_width/2 + (d_width/2) * (1 - abs(2*i/num_points - 1))
        y = d_y + (d_height/2) * (1 - 2*i/num_points)
        device.move_to(x=x, y=y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)

    device.move_to(x=0, y=-10, z=0, mode=MODE_PTP.MOVJ_INC)
    
    
    # Lift pen
    device.move_to(x=d_x, y=d_y - d_height/2, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)


def draw_win_line(device, win_type, win_index):
    """
    Draw a line through the winning combination (with inverted row coordinates)
    
    Args:
        device: robot device
        win_type: 'row', 'col', 'diag_main', or 'diag_anti'
        win_index: for row/col, which row (0-2) or col (0-2)
    """
    line_offset = 5  # mm offset from cell edges
    
    if win_type == 'row':

        # Vertical line through column (columns don't need inversion)
        win_index = 2-win_index
        x = origin_x + win_index * cell_size + cell_size / 2
        start_y = origin_y -grid_size/2
        end_y = origin_y + grid_size/2
        
        device.move_to(x=x, y=start_y, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
        device.move_to(x=x, y=start_y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
        device.move_to(x=x, y=end_y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
        device.move_to(x=x, y=end_y, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
        
    elif win_type == 'col':
        # INVERT ROW: row 0 -> 2, row 1 -> 1, row 2 -> 0
        robot_row = 2 - win_index
        
        # Horizontal line through row
        start_x = origin_x 
        end_x = origin_x + grid_size
        y = (origin_y - 3*cell_size/2) + robot_row * cell_size + cell_size / 2
        
        device.move_to(x=start_x, y=y, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
        device.move_to(x=start_x, y=y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
        device.move_to(x=end_x, y=y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
        device.move_to(x=end_x, y=y, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
    
        
    
    elif win_type == 'diag_main':
        # Diagonal from top-left to bottom-right (INVERTED: bottom-left to top-right in robot coords)
        start_x = origin_x + line_offset
        start_y = origin_y + grid_size/2 - line_offset  # Inverted start
        end_x = origin_x + grid_size - line_offset
        end_y = origin_y -grid_size/2 + line_offset  # Inverted end
        
        device.move_to(x=start_x, y=start_y, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
        device.move_to(x=start_x, y=start_y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
        device.move_to(x=end_x, y=end_y, z=drawing_height, mode=MODE_PTP.MOVL_XYZ)
        device.move_to(x=end_x, y=end_y, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
    
    elif win_type == 'diag_anti':
        # Diagonal from top-right to bottom-left (INVERTED: top-right to bottom-left stays same)
        start_x = origin_x + line_offset
        start_y = origin_y -grid_size/2 + line_offset  # Top in robot coords
        end_x = origin_x + grid_size + line_offset
        end_y = origin_y + grid_size/2 - line_offset  # Bottom in robot coords
        
        device.move_to(x=start_x, y=start_y, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)
        device.move_to(x=start_x, y=start_y, z=drawing_height, mode=MODE_PTP.MOVJ_XYZ)
        device.move_to(x=end_x, y=end_y, z=drawing_height, mode=MODE_PTP.MOVL_XYZ)
        device.move_to(x=end_x, y=end_y, z=safe_height, mode=MODE_PTP.MOVJ_XYZ)

    device.home()