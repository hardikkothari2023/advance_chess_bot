# 2_train_model.py - The AI Training Script

import os
import cv2
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import settings from our config file
from config import DATASET_PATH, IMAGE_SIZE, MODEL_PATH, EPOCHS, BATCH_SIZE

# 2_train_model.py - The AI Training Script

# ... (all your other imports)
from config import DATASET_PATH, IMAGE_SIZE, MODEL_PATH, EPOCHS, BATCH_SIZE

# --- NEW: GPU DETECTION CODE ---
print("--- GPU Detection ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("⚠️ No GPU detected. TensorFlow will run on the CPU.")
print("--------------------")
# -----------------------------
def load_data(dataset_path):
    """Loads all images from the dataset folder and assigns them numerical labels."""
    images = []
    labels = []
    label_map = {}
    current_label_id = 0

    print(f"INFO: Loading images from '{dataset_path}'...")
    if not os.path.exists(dataset_path):
        print(f"ERROR: The dataset path '{dataset_path}' does not exist. Please run the data collector script first.")
        return None, None, None

    for label_name in sorted(os.listdir(dataset_path)):
        if label_name not in label_map:
            label_map[label_name] = current_label_id
            current_label_id += 1
        
        dir_path = os.path.join(dataset_path, label_name)
        if not os.path.isdir(dir_path): continue

        for filename in os.listdir(dir_path):
            img_path = os.path.join(dir_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                labels.append(label_map[label_name])
    
    print(f"INFO: Loaded {len(images)} images.")
    print(f"INFO: Found {len(label_map)} classes: {list(label_map.keys())}")
    
    # Save the label map for the assistant to use later
    os.makedirs("models", exist_ok=True)
    # We save the inverted map (ID -> Name) for easy lookup during prediction
    inverted_label_map = {v: k for k, v in label_map.items()}
    with open('models/label_map.json', 'w') as f:
        json.dump(inverted_label_map, f)
    print("INFO: Label map saved to models/label_map.json")
    
    return np.array(images), np.array(labels), label_map

def build_model(num_classes):
    """Creates and compiles the CNN model architecture."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    """The main function to orchestrate the model training process."""
    print("\n--- ♟️ Step 2: Training the AI Model ♟️ ---")
    
    # 1. Load Data
    images, labels, label_map = load_data(DATASET_PATH)
    if images is None:
        return

    # 2. Preprocess Data
    images = images.astype('float32') / 255.0
    num_classes = len(label_map)
    labels_categorical = to_categorical(labels, num_classes=num_classes)
    
    # 3. Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels_categorical, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\nINFO: Training set size: {len(X_train)} images")
    print(f"INFO: Validation set size: {len(X_val)} images")

    # 4. Data Augmentation
    print("\nINFO: Setting up data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )
    datagen.fit(X_train)

    # 5. Build and Train the Model
    model = build_model(num_classes)
    model.summary()
    
    print("\nINFO: Starting model training... This may take several minutes.")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )

    # 6. Evaluate and Save
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print("\n" + "="*50)
    print(f"✅ Training complete!")
    print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    model.save(MODEL_PATH)
    print(f"✅ Model saved successfully to '{MODEL_PATH}'")
    print("="*50)
    print("\nYou are now ready for the final step: building the main assistant script!")

if __name__ == "__main__":
    main()

    
