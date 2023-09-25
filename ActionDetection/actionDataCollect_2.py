import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from dataAugmentation import get_image_files
from dataAugmentation import clear_folder
import imgaug as ia
from imgaug import augmenters as iaa
import time
import mediapipe as mp

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
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
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def get_random_offset(image, offset_rate):
    # Get the dimensions of the image
    image_height, image_width = image.shape[:2]

    # Calculate the maximum allowable offsets based on the offset rate
    max_offset_x = int(image_width * offset_rate)
    max_offset_y = int(image_height * offset_rate)

    # Generate random offsets within the allowable range
    offset_x = np.random.randint(-max_offset_x, max_offset_x + 1)
    offset_y = np.random.randint(-max_offset_y, max_offset_y + 1)

    return offset_x, offset_y


def crop_image(image, center_pos_x, center_pos_y, target_width, target_height):
    # Get the dimensions of the image
    image_height, image_width = image.shape[:2]

    # Convert normalized nose position to image coordinates
    center_x = int(center_pos_x * image_width)
    center_y = int(center_pos_y * image_height)

    # Calculate cropping coordinates around the nose point
    start_x = center_x - target_width // 2
    start_y = center_y - target_height // 2
    end_x = start_x + target_width
    end_y = start_y + target_height

    # Adjust cropping coordinates if they go beyond image boundaries
    if start_x < 0:
        end_x -= start_x
        start_x = 0
    if start_y < 0:
        end_y -= start_y
        start_y = 0
    if end_x > image_width:
        start_x -= (end_x - image_width)
        end_x = image_width
    if end_y > image_height:
        start_y -= (end_y - image_height)
        end_y = image_height

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

# Chỉ lấy thư mục, không lấy file
def list_directory(directory_path):
    # Get a list of all items in the directory
    items = os.listdir(directory_path)

    # Filter out directories from the list
    directories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

    return directories


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('AcData')
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

current_action = "E"

frame_path = f'Frames/Raw/{current_action}'
processed_path = 'Frames/Processed'

action_path = f'{processed_path}/{current_action}'

if not os.path.exists(action_path):
    os.makedirs(action_path)

segment_size = 10
num_repeats = 50

if __name__ == '__main__':
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        action_dirs = list_directory(frame_path)

        for num in range(num_repeats):
            print(action_dirs)

            # Nếu hành động đó có nhiều batch thì random trong các batch đó
            if action_dirs:
                random_batch = action_dirs[np.random.randint(0, len(action_dirs))]
                random_batch = os.path.join(frame_path, random_batch)
            else:
                random_batch = frame_path

            print("Đang đọc thư mục " + random_batch)

            img_paths = get_image_files(random_batch)
            img_paths = sorted(img_paths)

            num_segments = len(img_paths) - segment_size + 1

            # Chọn điểm bắt đầu cắt video từ các frame
            start_index = np.random.randint(0, num_segments)  # Choose a random starting index
            frame_cuts = img_paths[start_index: start_index + segment_size]

            print("Start Index:", start_index, "Segment:", frame_cuts)

            # Tìm vị trí trung tâm của ảnh để crop
            lead_image = cv2.imread(img_paths[start_index])
            image, results = mediapipe_detection(lead_image, holistic)
            if results.pose_landmarks:
                center_pos = results.pose_landmarks.landmark[0]    # vị trí của mũi
            else:
                print('Video ' + str(num) + " không có center_point")
                continue

            # Thông số augment
            seq = iaa.Sequential([
                iaa.Fliplr(0.3),  # horizontally flip 30% of the images
                iaa.Affine(translate_px={"x": (-10, 10), "y": 0}),
                iaa.Affine(scale=(0.9, 1.1), rotate=(-5, 5)) # , translate_percent=(-0.1, 0.1)
            ])
            augmentation_params = seq.to_deterministic()

            # Tìm folder lưu frames của hành động
            crop_folder = action_path + '/' + str(num)
            if not os.path.exists(crop_folder):
                os.makedirs(crop_folder)
            else:
                clear_folder(crop_folder)

            for index, framePath in enumerate(frame_cuts):
                frame = cv2.imread(framePath)
                # cropped_img = crop_image(frame, center_pos.x, center_pos.y, offset_x, offset_y, 640, 480)
                cropped_img = crop_image(frame, center_pos.x, center_pos.y, 640, 480)
                augmented_img = augmentation_params.augment_image(cropped_img)

                # Make detections
                image, results = mediapipe_detection(augmented_img, holistic)
                draw_styled_landmarks(image, results)

                # save numpy
                keypoints = extract_keypoints(results)
                np.save(crop_folder + f'/{index}', keypoints)

                # save crop image
                file_name = os.path.basename(framePath)
                file_name = os.path.splitext(file_name)[0]
                crop_file_path = f'{crop_folder}/{file_name}.jpg'
                cv2.imwrite(crop_file_path, image)
