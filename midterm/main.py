import pydobot
import cv2
from ultralytics import YOLO
import threading
from robot import make_grid, draw_x, draw_o, intermediate, draw_d, draw_win_line
from constants import Valid_Players, symbol_map, val_map , Human, Robot
from video import display_video, capture_board_image
from ai import check_winner, detect_board_state_yolo, get_best_move, find_latest_move
from dotenv import load_dotenv
from mock_device import MockDobot
import anthropic
import os
import time

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
use_mock = os.getenv("USE_MOCK")

anthropic_client =  anthropic.Anthropic(api_key= anthropic_api_key)

# if use_mock:
    # device = MockDobot("/dev/ttyACM0")
# else:
#     device = pydobot.Dobot("/dev/ttyACM0")
device = pydobot.Dobot("/dev/ttyACM0")
    
device.home()

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Check --camera-index and permissions.")

# Reduce resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

yolo_path = "/home/ibraheem/Downloads/runs/detect/xo_detector/weights/best.pt"
model = YOLO(yolo_path)

# Start video feed in separate thread
threading.Thread(target=display_video, args=(cap, model)).start()

human_symbol = None
robot_symbol = None

move_map = {
    'o': draw_o,
    'x': draw_x
}

def make_move(device, coordinates, symbol):
    move_map[symbol](*coordinates, device)

def find_robot_sym(human_val):
    if human_val is None or human_val not in [1, 2]:
        raise Exception("Select a valid symbol for human.")

    if human_val == 1:
        return 2
    if human_val == 2:
        return 1

def get_human_move_func(human_symbol, board):
    input(f"Press Enter after you make your move ({human_symbol})...")
    intermediate(device)
    time.sleep(2)
    image = capture_board_image(cap)

    new_board = detect_board_state_yolo(image, model, board)

    print(new_board)

    human_move = find_latest_move(board, new_board)

    while human_move is None:
        input(f"Press Enter after you make your move ({human_symbol})...")
        new_board = detect_board_state_yolo(image, model, board)

        print(new_board)

        human_move = find_latest_move(board, new_board)

    if val_map[human_move[2]] != human_symbol:
        raise Exception(f"Human move is invalid. Expecting an {human_symbol} as human move")

    print(f"Detected human move at row {human_move[0]}, col {human_move[1]}")

    board = new_board

    return human_move, board

def play_game():

    grid_option = input("Do you want to draw a new grid? (y/n):")

    while grid_option not in ['y', 'n']:
        grid_option = input("Try Again, y or n: ")
    if grid_option == 'y':
        make_grid(device)

    # Initialize board state (0 = empty, 1 = X, 2 = O)
    board = [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]

    first_move = input(f"Who Makes the first move? Enter {Robot} or {Human} (Robot[{Robot}]/Human[{Human}]): ")

    while first_move not in ['R', 'H']:
        first_move= input(f"Input must be in {', '.join(Valid_Players)}")

    prev_move = first_move

    if first_move == Robot:
        robot_symbol='x'
        human_symbol='o'
        draw_x(1, 1, device)  
        board[1][1] = symbol_map[robot_symbol]

    if first_move == Human:

        input("Press Enter after you make your move (O)...")
        intermediate(device)
        time.sleep(2)
        image = capture_board_image(cap)

        new_board = detect_board_state_yolo(image, model, board)

        print(new_board)

        human_move = find_latest_move(board, new_board)

        while human_move is None:
            input(f"Press Enter after you make your move ({human_symbol})...")
            new_board = detect_board_state_yolo(image, model, board)

            print(new_board)

            human_move = find_latest_move(board, new_board)

        human_symbol = val_map[human_move[2]]
        robot_symbol = val_map[find_robot_sym(human_move[2])]

        while True:
            if human_move:
                print(f"Detected human move at row {human_move[0]}, col {human_move[1]}")
                board = new_board
                break
            else:
                input("Could not detect human move. Please try again and press enter.")

    while True:

        # Check if human won
        winner, win_index, win_type = check_winner(board)
        
        if winner != 0:
            def find_winner(winner):
                winner_symbol = val_map[winner]
                if winner_symbol == human_symbol:
                    return "Human"

                if winner_symbol == robot_symbol:
                    return "Robot"
                
            print(f"{find_winner(winner)} wins!")
            draw_win_line(device, win_type, win_index)
            break
        
        if winner == 3:
            draw_d(device)
            print("Draw - Inconclusive")

        # Check for draw
        if all(board[i][j] != 0 for i in range(3) for j in range(3)):
            print("It's a draw!")
            break

        if prev_move == Robot:
            print("Board State Before Human Move")
            print(board)
            _, board = get_human_move_func(human_symbol, board)
            prev_move = Human
            continue
        
        if prev_move == Human:
            # Robot's turn
            print("Robot is thinking...")
            robot_move = get_best_move(board, symbol_map[robot_symbol])
            print(robot_move)
            if robot_move:
                row, col = robot_move
                print(f"Robot plays {robot_symbol} at row {row}, col {col}")
                make_move(device, robot_move, robot_symbol)
                board[row][col] = symbol_map[robot_symbol]

            prev_move = Robot
        

if __name__ == "__main__":
    play_game()