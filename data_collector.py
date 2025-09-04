# 1_data_collector.py - Automated Training Image Capture

import pyautogui
import cv2
import numpy as np
import os
import time
import logging

from config import DATASET_PATH, IMAGE_SIZE

logging.basicConfig(level=logging.INFO, format='%(asctime)s -[%(levelname)s]- %(message)s')

def select_board_region():
    """Lets the user select the board region from a full screenshot."""
    logging.info("A window will appear. Please draw a TIGHT rectangle around the 8x8 squares, INSIDE the coordinates.")
    try:
        screenshot = pyautogui.screenshot()
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        roi = cv2.selectROI("Select Chessboard Region", img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        if roi[2] == 0 or roi[3] == 0:
            logging.error("No region selected. Exiting.")
            return None
        logging.info(f"Capture region selected: {roi}")
        return roi
    except Exception as e:
        logging.error(f"Could not select region: {e}")
        return None

def capture_and_save_squares(board_region, piece_code):
    """
    Captures the board, divides it into 64 squares, and saves any squares
    containing pieces into the correct subfolder.
    """
    if not board_region: return 0
    
    # Create the directory for the piece type if it doesn't exist
    piece_dir = os.path.join(DATASET_PATH, piece_code)
    os.makedirs(piece_dir, exist_ok=True)
    
    screenshot = pyautogui.screenshot(region=board_region)
    board_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    square_h, square_w = board_img.shape[0] // 8, board_img.shape[1] // 8
    count = 0

    for r in range(8):
        for c in range(8):
            square = board_img[r*square_h:(r+1)*square_h, c*square_w:(c+1)*square_w]
            
            # A simple check to see if the square is mostly empty or contains a piece.
            # We check the standard deviation of the grayscale pixels. Empty squares
            # have very little variation, while squares with pieces have a lot.
            gray_square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            if np.std(gray_square) > 15: # This threshold distinguishes pieces from empty squares
                # Resize and save the image
                resized_square = cv2.resize(square, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                
                # Generate a unique filename
                filename = f"{piece_code}_{int(time.time() * 1000)}_{r}{c}.png"
                filepath = os.path.join(piece_dir, filename)
                cv2.imwrite(filepath, resized_square)
                count += 1
                
    logging.info(f"Saved {count} images for piece type '{piece_code}'.")
    return count

def capture_empty_squares(board_region):
    """Captures all 64 squares from an empty board."""
    return capture_and_save_squares(board_region, "empty")

def main():
    """Main function to run the data collection process."""
    print("--- ♟️ Chess AI Data Collector ♟️ ---")
    print("This script will help you capture images to train a neural network.")
    
    board_region = select_board_region()
    if not board_region:
        return

    print("\nStep 1: Capturing EMPTY squares.")
    print("Please make sure the on-screen board is completely empty.")
    input("--> Press ENTER when ready...")
    capture_empty_squares(board_region)
    
    # List of all piece types to capture
    piece_types = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']

    for piece in piece_types:
        print("\n" + "="*50)
        print(f"Step 2: Capturing '{piece}' pieces.")
        print(f"Please place one or more '{piece}' pieces on the board.")
        print("Spread them out on both light and dark squares for best results.")
        input(f"--> Press ENTER when the '{piece}' pieces are ready...")
        capture_and_save_squares(board_region, piece)

    print("\n" + "="*50)
    print("✅ Data collection complete!")
    print(f"All images have been saved to the '{DATASET_PATH}' folder.")
    print("You can run this script again to add more images and improve your dataset.")

if __name__ == "__main__":
    main()

# ### How to Use the Data Collector

# 1.  **Set up your files** as described in the new project structure.
# 2.  **Run as Administrator**: This is still the best practice.
#     ```powershell
#     python 1_data_collector.py
    
