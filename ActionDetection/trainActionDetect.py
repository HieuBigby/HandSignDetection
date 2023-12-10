import os
import pickle

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt


def read_dictionary():
    action_list = []
    with open('dictionary.txt', 'r', encoding='utf-8') as file:
        # Read each line in the file
        for line in file:
            key, value = map(str.strip, line.split(':', 1))
            action_list.append(key)
    return action_list

DATA_PATH = os.path.join('Frames/Processed')
actions = np.array(read_dictionary())

# Videos are going to be 10 frames in length
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
            # print(str(len(res)) + ' of ' + str(video) + ' ac ' + str(action))
            if len(res) != 126: print(action + "-" + str(video))
            frames.append(res)
        sequences.append(frames)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)

# Load Model
log_dir = os.path.join('Logs/No Others Log')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 126))) # input_shape=(sequence_length, 258)
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu')) # , kernel_regularizer=l2(0.01)  # Add L2 regularization
model.add(Dense(32, activation='relu')) # , kernel_regularizer=l2(0.01)  # Add L2 regularization
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))
model.save('Models/action_final_noothers.h5')
model.summary()

print(history.history.keys())
with open('./trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

