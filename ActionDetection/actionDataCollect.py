import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


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
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh]) # [pose, face, lh, rh]

def check_subfolder_file_count(folder_path):
    for video in os.listdir(folder_path):
        print('video ' + video + ' ' + str(len(os.listdir(os.path.join(folder_path, video)))))
    # for root, dirs, files in os.walk(folder_path):
    #     for dir in dirs:
    #         for sub_root, sub_dirs, sub_files in os.walk(folder_path + '/' + dir):
    #             print(str(len(sub_files)) + ' in ' + dir)
    #             if(len(sub_files) != 20):
    #                 print(dir + ' khong du data')
        # if len(files) != target_file_count:
        #     print(f"Folder '{root}' does not have {target_file_count} files.")

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('AcData')
fontpath = "./FiraSans-Regular.ttf"
font = ImageFont.truetype(fontpath, 32)

# Actions that we try to detect
actions = np.array(['None', 'Xin chao', 'Cam on', 'Hen', 'Gap', 'Lai', 'Toi', 'Ten',
                    'H', 'I', 'E', 'U'])

# Videos are going to be 30 frames in length
sequence_length = 10
video_index = -1
action_index = 11
current_action = 'H'

# Tạo folder Data nếu chưa có
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        os.makedirs(action_path)

# check_subfolder_file_count('AcData/Xin chao')

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        collecting = False
        frame_counter = 0
        countdown_duration = 1  # 3 seconds countdown
        last_time = time.time()

        while cap.isOpened():
            # Read feed
            ret, image = cap.read()

            # Make detections
            image, results = mediapipe_detection(image, holistic)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            if not collecting:
                cv2.putText(image, 'Prepare action: {}, video num: {}'.format(actions[action_index], str(video_index + 1)),
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                img_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(img_pil)
                draw.text((10, 0), "Xin chào!", font=font, fill=(0, 255, 0, 0))
                image = np.array(img_pil)

            # Lưu ảnh khi nhấn 's'
            key = cv2.waitKey(1)
            if key == ord('s'):
                collecting = True
                frame_counter = 0
                video_index += 1
                last_time = time.time()

            if collecting and frame_counter < sequence_length:
                # Check for countdown
                current_time = time.time()
                countdown_remaining = countdown_duration - int(current_time - last_time)
                if countdown_remaining > 0:
                    # Display countdown text
                    cv2.putText(image, 'Recording starts in {}s'.format(countdown_remaining),
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, 'Collecting frame {} for video {} of action {}'
                        .format(frame_counter, video_index, actions[action_index]), (15, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    folder_path = os.path.join(DATA_PATH, actions[action_index], str(video_index))
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    file_path = folder_path + f'/Image_{frame_counter}.jpg'
                    keypoints = extract_keypoints(results)
                    np.save(folder_path + f'/{frame_counter}', keypoints)
                    print("Saving image:", file_path)
                    success = cv2.imwrite(file_path, image)
                    frame_counter += 1
            else:
                collecting = False

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()