# config.py - Configuration for the CNN Chess Assistant Project

# --- Stockfish Configuration ---
STOCKFISH_PATH = r"C:\Users\HARDIK\OneDrive\Desktop\Chess Assistance Advance\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
STOCKFISH_THINK_TIME = 2.0

# --- Data Collection Configuration ---
# The parent folder where all our training images will be saved.
DATASET_PATH = "dataset"

# The size (in pixels) to which we will resize all captured images.
# This ensures all training data is uniform. 64x64 is a good standard.
IMAGE_SIZE = (64, 64)

# --- Model Training Configuration (for the next step) ---
MODEL_PATH = "models/chess_piece_detector.h5"
EPOCHS = 150
BATCH_SIZE = 32
