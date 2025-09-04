# ♟️ Advance Chess Bot

An advanced AI-powered chess assistant that combines **computer vision**, **deep learning**, and the **Stockfish chess engine** to analyze live chessboards, recognize piece positions, and suggest the best possible moves in real time.

---

## 📌 Project Overview
The **Advance Chess Bot** is designed to act as your personal **chess companion**:
- **Perception (Vision):** Recognizes the chessboard and pieces using **OpenCV** and a **CNN classifier**.
- **Cognition (Decision-making):** Validates board state and move legality with the `python-chess` library.
- **Action (Strategy):** Uses **Stockfish** to evaluate the position and suggest the strongest moves.

This project is a complete pipeline:
1. Collect dataset of chess pieces.
2. Train a CNN model to recognize them.
3. Detect moves from the screen in real-time.
4. Suggest the best move using Stockfish.

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```powershell
git clone https://github.com/hardikkothari2023/advance_chess_bot.git
cd advance_chess_bot
```

### 2. Create & activate a virtual environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Download Stockfish
Stockfish is **already included** in the repo under:

```
stockfish-windows-x86-64-avx2/
```

Ensure the path in `config.py` points to:
```python
STOCKFISH_PATH = "stockfish-windows-x86-64-avx2/stockfish-windows-x86-64-avx2.exe"
```

---

## 📂 Folder Structure
```
advance_chess_bot/
│── assistance.py              # Main assistant script (board recognition + Stockfish)
│── config.py                  # Configuration file (paths, thresholds, etc.)
│── data_collector.py          # Collect training images for dataset
│── training_file.py           # Train CNN model for piece recognition
│── requirements.txt           # Python dependencies
│── README.md                  # Documentation
│── .gitignore
│
├── dataset/                   # Training dataset
│   ├── bB/, bK/, bN/, bP/, bQ/, bR/   # Black pieces
│   ├── wB/, wK/, wN/, wP/, wQ/, wR/   # White pieces
│   └── empty/                 # Empty squares
│
├── models/                    # Trained CNN models
│   ├── chess_piece_detector.h5
│   └── label_map.json
│
├── stockfish-windows-x86-64-avx2/     # Stockfish engine
│   └── stockfish-windows-x86-64-avx2.exe
│
├── __pycache__/               # Python cache
└── .venv/                     # Virtual environment
```

---

## ⚙️ How It Works (Pipeline)

### **1. Board Capture (Perception)**
- Captures a screenshot of the chessboard using `pyautogui`.
- Splits it into **64 squares**.
- Runs each square through the CNN model to identify the piece.

### **2. Board State Validation (Cognition)**
- Converts recognition results into a **FEN string**.
- Uses `python-chess` to validate legality and track moves.

### **3. Best Move Suggestion (Action)**
- Passes the current board state to **Stockfish**.
- Retrieves the **best move + evaluation**.
- Prints suggestions in real time.

---

## ▶️ Usage

### 1. Run the Assistant
```powershell
python assistance.py
```

### 2. Steps in Program
- Select the **chessboard region** (drag a rectangle around the board).
- Confirm if you are playing as **White or Black**.
- The assistant will:
  - Recognize the board.
  - Detect moves.
  - Suggest the **best move**.

---

## 📊 Example Flow
```
✅ Initial position recognized.
It's your turn (White). Analyzing...
♟️ Best Move: e2e4
📊 Top Moves: e2e4 (+0.20) | d2d4 (+0.18) | g1f3 (+0.15)

Watching for moves...
Opponent plays: e7e5
Analyzing for your best move...
♟️ Best Move: g1f3
📊 Top Moves: g1f3 (+0.22) | f1c4 (+0.19)
```

---

## 🚀 Performance Optimizations
- **Dirty-check:** Only re-check changed squares (from 768 → ~24 checks per move).
- **CNN Upgrade Path:** Switch from template matching to trained CNN (`training_file.py`).
- **Confidence Tuning:** Adjust `CONFIDENCE_THRESHOLD` in `config.py` for accuracy.
- **Efficient Stockfish Calls:** Use `multipv=3` to get top move suggestions.

---

## 🛠️ Troubleshooting
- **Pieces not detected properly?** → Adjust `CONFIDENCE_THRESHOLD` in `config.py`.
- **Wrong board region?** → Re-run and select ROI carefully (must be 8x8 squares).
- **Low FPS / Lag?** → Increase `CAPTURE_INTERVAL` in `config.py`.
- **FAILSAFE:** If stuck, press **CTRL + C** to stop safely.

---

## 🌟 Future Enhancements
- GUI interface for easier interaction.
- Faster recognition using a lightweight CNN.
- Auto-detection of board region (no manual ROI).
- Support for online chess platforms with direct hooks.
- Move prediction based on opening databases.
