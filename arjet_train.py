import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from train_model import create_gesture_model

# 1. Ladda och preprocessa dataset
def load_dataset(path, gestures, max_per_class=1600):
    X, y = [], []
    for idx, gesture in enumerate(gestures):
        gesture_dir = os.path.join(path, gesture)
        files = os.listdir(gesture_dir)[:max_per_class]
        for f in files:
            img_path = os.path.join(gesture_dir, f)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (100, 120))
            X.append(image)
            label = [0] * len(gestures)
            label[idx] = 1
            y.append(label)
    return np.array(X), np.array(y)

# 2. Kör träning
if __name__ == "__main__":
    dataset_path = "my_dataset/data"
    gestures = ['blank', 'fist', 'five', 'ok', 'thumbsdown', 'thumbsup']

    X, y = load_dataset(dataset_path, gestures)
    X = X.reshape(-1, 100, 120, 1).astype("float32") / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_gesture_model()

    model.fit(X_train, y_train,
              batch_size=128,
              epochs=10,
              validation_data=(X_test, y_test),
              verbose=1)

    model.save("hand_gesture_model.h5")
    print("✅ Modell sparad!")
