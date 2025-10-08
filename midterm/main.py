import pydobot
import cv2
from ultralytics import YOLO
import threading
from robot import make_grid, draw_x, draw_o
from constants import Valid_Players, origin_x, origin_y, safe_height, drawing_height, cell_size, Human, Robot
from video import display_video, capture_board_image
from ai import check_winner, detect_board_state_yolo, get_best_move, detect_board_state, find_human_move
from dotenv import load_dotenv
from mock_device import MockDobot
import anthropic
import os

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
use_mock = os.getenv("USE_MOCK")

anthropic_client =  anthropic.Anthropic(api_key= anthropic_api_key)

if use_mock:
    device = MockDobot("/dev/ttyACM0")
else:
    device = pydobot.Dobot("/dev/ttyACM0")
    
device.home()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Check --camera-index and permissions.")

# Reduce resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

yolo_path = "/home/ibraheem/Downloads/runs/detect/xo_detector/weights/best.pt"
model = YOLO(yolo_path)

# Start video feed in separate thread
threading.Thread(target=display_video, args=(cap, model)).start()

make_grid(device)

# Initialize board state (0 = empty, 1 = X, 2 = O)
board = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

first_move = input(f"Who Makes the first move? Enter {Robot} or {Human} (Robot[{Robot}]/Human[{Human}]): ")
while first_move not in ['R', 'H']:
    input(f"Input must be in {', '.join(Valid_Players)}")

if first_move == Robot:
    draw_x(1, 1, device)  
    board[1][1] = 2

if first_move == Human:
    image = capture_board_image(cap)

    new_board = detect_board_state_yolo(image, model)

    print(new_board)

    human_move = find_human_move

    while True:
        if human_move:
            print(f"Detected human move at row {human_move[0]}, col {human_move[1]}")
            board = new_board
            break
        else:
            input("Could not detect human move. Please try again.")



while True:
    input("Press Enter after you make your move (O)...")
    
    print("Capturing board image...")
    image = capture_board_image(cap)  

    print(board)
    
    print("Detecting board state ...")
    # new_board = detect_board_state(image)
    new_board = detect_board_state_yolo(image, model)

    print(new_board)
    
    # Find human's move
    human_move = find_human_move(board, new_board)
    if human_move:
        print(f"Detected human move at row {human_move[0]}, col {human_move[1]}")
        board = new_board
    else:
        print("Could not detect human move. Please try again.")
        continue
    
    # Check if human won
    winner = check_winner(board)
    if winner == 2:
        print("Human wins!")
        break
    
    # Check for draw
    if all(board[i][j] != 0 for i in range(3) for j in range(3)):
        print("It's a draw!")
        break
    
    # Robot's turn
    print("Robot is thinking...")
    robot_move = get_best_move(board)
    if robot_move:
        row, col = robot_move
        print(f"Robot plays X at row {row}, col {col}")
        draw_x(row, col, device)
        board[row][col] = 1
    
    # Check if robot won
    winner = check_winner(board)
    if winner == 1:
        print("Robot wins!")
        break
