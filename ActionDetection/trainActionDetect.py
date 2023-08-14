import os

from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.regularizers import l2
import mediapipe as mp
import cv2
import numpy as np


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('AcData')

# Danh sách các action đã tạo
folder_names = [name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))]

# Convert the list of folder names to a NumPy array
actions = np.array(folder_names)

# Videos are going to be 30 frames in length
sequence_length = 10

# Categorize data
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for video in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        frames = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(video), "{}.npy".format(frame_num)))
            # print(str(frame_num) + ' of ' + str(video))
            print(str(len(res)) + ' of ' + str(video) + ' ac ' + str(action))
            frames.append(res)
        sequences.append(frames)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print(X.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)

# Load Model
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 258))) # input_shape=(sequence_length, 1662)
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu')) # , kernel_regularizer=l2(0.01)  # Add L2 regularization
model.add(Dense(32, activation='relu')) # , kernel_regularizer=l2(0.01)  # Add L2 regularization
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=300, callbacks=[tb_callback], validation_data=(X_test, y_test))
model.save('action_7.h5')

# model_path = 'action_1.h5'

# if os.path.exists(model_path):
#     print('Đã tồn tại model train, tiếp tục train mới data mới...')
#
#     # # Load only the weights of the existing model
#     # model.load_weights(model_path)  # This is the correct function for loading weights
#
#     # Load the saved model
#     model = load_model(model_path)
#
#     # Remove the last layer and add a new output layer for the new class
#     model.pop()  # Remove the last layer
#     model.add(Dense(actions.shape[0], activation='softmax'))  # Add new output layer
#
#     # Compile the model
#     model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#
#     # Continue training on new data
#     new_epochs = 500  # Number of additional epochs
#
#     model.fit(X_train, y_train, epochs=new_epochs, callbacks=[tb_callback])
#
#     # Save the model after continuing training
#     updated_model_path = 'action_1_updated.h5'
#     model.save(updated_model_path)
# else:
#     model = Sequential()
#     model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 1662)))
#     model.add(LSTM(128, return_sequences=True, activation='relu'))
#     model.add(LSTM(64, return_sequences=False, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(actions.shape[0], activation='softmax'))
#     model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#     model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])
#     model.save('action_1.h5')