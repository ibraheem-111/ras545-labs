import base64
import cv2
import numpy as np
from ultralytics import YOLO

def check_winner(board):
    """Check if there's a winner. Returns 1 for X, 2 for O, 0 for none"""
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] != 0:
            return row[0]
    
    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != 0:
            return board[0][col]
    
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    
    return 0

def minimax(board, depth, is_maximizing):
    """
    Recursive minimax algorithm for Tic Tac Toe.
    board: 3x3 list of lists (0 = empty, 1 = X, 2 = O)
    depth: recursion depth (for tie-breaking)
    is_maximizing: True if it's the AI's turn
    Returns: numeric score (+10 for AI win, -10 for human win, 0 for draw)
    """

    winner = check_winner(board)
    if winner == 1:
        return 10 - depth  # Robot (X) wins
    elif winner == 2:
        return depth - 10  # Human (O) wins
    elif all(board[i][j] != 0 for i in range(3) for j in range(3)):
        return 0  # Draw

    if is_maximizing:
        best_score = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    board[i][j] = 1  # Try X
                    score = minimax(board, depth + 1, False)
                    board[i][j] = 0
                    best_score = max(best_score, score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    board[i][j] = 2  # Try O
                    score = minimax(board, depth + 1, True)
                    board[i][j] = 0
                    best_score = min(best_score, score)
        return best_score


def get_best_move(board):
    """Returns (row, col) for best move for the AI (X)."""
    best_score = -float('inf')
    move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                board[i][j] = 1
                score = minimax(board, 0, False)
                board[i][j] = 0
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return (2- move[0], 2-move[1])

def detect_board_state_yolo(image_path_or_bytes, model, prev_board):
    # --- Step 1: Load image ---
    if isinstance(image_path_or_bytes, bytes):
        nparr = np.frombuffer(image_path_or_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        frame = cv2.imread(image_path_or_bytes)
    if frame is None:
        raise ValueError("Invalid image input")

    # --- Step 2: Detect grid boundaries ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No grid found")

    # Assume largest contour is the tic-tac-toe board
    largest = max(contours, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(largest)
    roi = frame[gy:gy+gh, gx:gx+gw]

    # --- Step 3: Run YOLO detection ---
    results = model(roi, verbose=False, conf=0.6)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    cls = results.boxes.cls.cpu().numpy().astype(int)

    # --- Step 4: Build empty board ---
    new_board = [[0 for _ in range(3)] for _ in range(3)]

    # --- Step 5: Divide grid into cells ---
    cell_w = gw / 3
    cell_h = gh / 3

    occupied = set()

    # --- Step 6: Assign detections to cells ---
    for (x1, y1, x2, y2), k in zip(boxes, cls):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Compute cell indices
        col = int((cx) // cell_w)
        row = int((cy) // cell_h)

        if 0 <= row < 3 and 0 <= col < 3:

            if (row, col) in occupied:
                continue  # skip duplicate detections
            occupied.add((row, col))

            if k == 0:  # X
                new_board[row][col] = 1
            elif k == 1:  # O
                new_board[row][col] = 2
    
    if prev_board is not None:
        for i in range(3):
            for j in range(3):
                if new_board[i][j] == 0 and prev_board[i][j] != 0:
                    
                    new_board[i][j] = prev_board[i][j]

    return new_board



def detect_board_state(image_path_or_bytes, client):
    """Use Claude to detect X's and O's on the board"""    
    # Read and encode image
    if isinstance(image_path_or_bytes, str):
        with open(image_path_or_bytes, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    else:
        # If bytes/image array, convert to base64
        image_data = base64.standard_b64encode(image_path_or_bytes).decode("utf-8")
    
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": """Analyze this tic-tac-toe board and return the board state as a 3x3 grid.

Use this format - respond ONLY with a Python list, nothing else:
[[row0_col0, row0_col1, row0_col2],
 [row1_col0, row1_col1, row1_col2],
 [row2_col0, row2_col1, row2_col2]]

Where:
- 0 = empty cell
- 1 = X
- 2 = O

The grid positions are:
Row 0 (top): [0,0] [0,1] [0,2]
Row 1 (mid): [1,0] [1,1] [1,2]
Row 2 (bot): [2,0] [2,1] [2,2]

Return ONLY the 2D list array, no explanation."""
                }
            ],
        }],
    )
    
    # Parse Claude's response
    response_text = message.content[0].text.strip()
    # Extract the list from response
    import ast
    detected_board = ast.literal_eval(response_text)
    
    return detected_board

def find_latest_move(old_board, new_board):
    """Compare boards to find where human played"""
    new_moves = []
    for row in range(3):
        for col in range(3):
            if old_board[row][col] != new_board[row][col]:
                val = new_board[row][col]
                new_moves.append((row, col, val))

    print("New Moves Detected:")
    print(new_moves)
    if len(new_moves) ==0:
        raise Exception("No new move detected")
    if len(new_moves)>1:
        raise Exception("Only one move allowed")
    
    return new_moves[0]