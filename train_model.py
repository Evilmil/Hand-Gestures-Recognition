import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


loaded_images = []
outputVectors = []

list_of_gestures = ['blank', 'ok', 'thumbsup', 'thumbsdown', 'fist', 'five']
dataset_root = "my_dataset/data"  # <- ändra om din datasetmapp heter något annat

for idx, gesture in enumerate(list_of_gestures):
    path = os.path.join(dataset_root, gesture)
    files = os.listdir(path)[:1600]  # max 1600 per klass
    for file in files:
        image_path = os.path.join(path, file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (100, 120))
        loaded_images.append(gray_image)
        vector = [0] * 6
        vector[idx] = 1
        outputVectors.append(vector)
       


X = np.array(loaded_images).reshape(-1, 100, 120, 1)
y = np.array(outputVectors)

print("Totalt antal bilder:", len(loaded_images))
for i, g in enumerate(list_of_gestures):
    count = sum(1 for v in outputVectors if v[i] == 1)
    print(f"{g}: {count} bilder")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 120, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])


model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, y_test))


model.save("hand_gesture_recog_model.h5")
