import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam

def create_gesture_model():
    # model
    model = Sequential()

    # first conv layer
    # input shape = (img_rows, img_cols, 1)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100,120, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # second conv layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # flatten and put a fully connected layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu')) # fully connected
    model.add(Dropout(0.5))

    # softmax layer
    model.add(Dense(6, activation='softmax'))

    # model summary
    optimiser = Adam()
    model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model
