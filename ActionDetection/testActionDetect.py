import math
import os

import keras.utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard
from keras.regularizers import l2
import mediapipe as mp
import cv2
import numpy as np
import random
from PIL import ImageFont, ImageDraw, Image


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    lh_list = []
    if results.left_hand_landmarks:
        lh_pos = results.left_hand_landmarks.landmark[0]
        lh_pos = [round(lh_pos.x, 2), round(lh_pos.y, 2), round(lh_pos.z, 2)]
        lh_list.append(lh_pos)
        lh_vectors = normalized_vectors(results.left_hand_landmarks)
        lh_list.extend(lh_vectors)

    rh_list = []
    if results.right_hand_landmarks:
        rh_pos = results.right_hand_landmarks.landmark[0]
        rh_pos = [round(rh_pos.x, 2), round(rh_pos.y, 2), round(rh_pos.z, 2)]
        rh_list.append(rh_pos)
        rh_vectors = normalized_vectors(results.right_hand_landmarks)
        rh_list.extend(rh_vectors)

    lh_array = np.array(lh_list).flatten() if lh_list else np.zeros(20 * 3 + 3)
    rh_array = np.array(rh_list).flatten() if rh_list else np.zeros(20 * 3 + 3)

    return np.concatenate([lh_array, rh_array])

def normalized_vectors(hand_landmarks):
    vectors = []
    # Define the order of points for calculations
    point_order = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    for start_idx, end_idx in point_order:
        start_point = hand_landmarks.landmark[start_idx]
        end_point = hand_landmarks.landmark[end_idx]

        normalized_vector = normalize_vector(end_point, start_point)
        vectors.append(normalized_vector)

    return vectors

def normalize_vector(point1, point2):
    # Calculate the vector between the two NormalizedLandmark points
    vector = (point2.x - point1.x, point2.y - point1.y, point2.z - point1.z)

    # Calculate the length of the vector
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)

    # Normalize the vector and return it
    normalized_vector = (round(vector[0] / length, 2), round(vector[1] / length, 2), round(vector[2] / length, 2))

    return normalized_vector


def apply_replacements(input_array, replacements):
    output_array = []
    i = 0

    while i < len(input_array):
        combined_word = input_array[i]
        found_match = False

        for j in range(i + 1, len(input_array)):
            combined_word += " " + input_array[j]

            if combined_word in replacements:
                output_array.append(replacements[combined_word])
                i = j + 1
                found_match = True
                break

        if not found_match:
            output_array.append(input_array[i])
            i += 1

    return output_array

def read_dictionary():
    action_list = []
    with open('dictionary.txt', 'r', encoding='utf-8') as file:
        # Read each line in the file
        for line in file:
            key, value = map(str.strip, line.split(':', 1))
            action_list.append(key)
    return action_list

def read_structure():
    structure_dict = {}
    with open('structure.txt', 'r', encoding='utf-8') as file:
        # Read each line in the file
        for line in file:
            # Split each line into key and value based on the colon (':') character
            key, value = map(str.strip, line.split(':', 1))

            # Add the key-value pair to the dictionary
            structure_dict[key] = value
    return structure_dict

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('AcData')
# DATA_PATH = os.path.join('Frames/Processed')
fontpath = "./FiraSans-Regular.ttf"
font = ImageFont.truetype(fontpath, 32)

# Danh sách các action đã tạo
folder_names = [name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))]
# Convert the list of folder names to a NumPy array
actions = np.array(read_dictionary())
replacements = read_structure()

# Videos are going to be 30 frames in length
sequence_length = 10

# Load Model
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

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

model.load_weights('Models/action_final_noothers.h5')
# img_file = './model_arch_2.png'
# keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)


# 1. New detection variables
sequences = []
sentences = []
predictions = []
threshold = 0.9

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequences.append(keypoints)
        sequences = sequences[-10:]

        if len(sequences) == sequence_length:
            res = model.predict(np.expand_dims(sequences, axis=0))[0]
            # print(res)
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            predictions = predictions[-15:]
            # print('prediction length: ' + str(len(predictions)))

            # 3. Viz logic
            # Tìm ra hành động được dự đoán nhiều nhất trong 10 video đưa vào
            if res[np.argmax(res)] > threshold:
                # if actions[np.argmax(res)] != 'None':
                if len(sentences) > 0:
                    most_res = np.bincount(predictions).argmax()
                    if actions[most_res] != sentences[-1]:
                        # if(actions[most_res]) != 'None':
                        sentences.append(actions[most_res])
                else:
                    sentences.append(actions[np.argmax(res)])

            # Reset lại sentences
            sentences = sentences[-10:]

        # Loại bỏ hành động 'Break' ra khỏi phần hiển thị
        filtered_sentences = [s for s in sentences if s != 'None']
        filtered_sentences = apply_replacements(filtered_sentences, replacements)
        sum_lengths = sum(len(s) for s in filtered_sentences)
        # print(sum_lengths)
        if sum_lengths > 25:
            filtered_sentences = filtered_sentences[-5:]
        # if len(filtered_sentences) > 4:
        #     filtered_sentences = filtered_sentences[-4:]
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        # cv2.putText(image, ' '.join(filtered_sentences), (3, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 5), ' '.join(filtered_sentences), font=font, fill=(0,255,0,0))
        image = np.array(img_pil)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()