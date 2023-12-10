import math

import cv2
import numpy as np
import os
from Legacy.dataAugmentation import get_image_files
from Legacy.dataAugmentation import clear_folder
from imgaug import augmenters as iaa
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


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Actions that we try to detect
actions = np.array(['None', 'A', 'B', 'C', 'D', 'E', 'I', 'H', 'U', 'Xin chao', 'Toi', 'Ten',
                    'Cam on', 'Hen', 'Gap', 'Lai', 'Vui', 'Khoe', 'Xin loi', 'Tam biet', 'Ban'])
with open("dictionary.txt", 'w') as f:
    for index, action in enumerate(actions):
        f.write('%s: %s\n' % (action, index))

processed_path = 'Frames/Processed'
raw_path = 'Frames/Raw'
segment_size = 10
num_repeats = 100
override_exist = False

if __name__ == '__main__':
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            frame_path = f'{raw_path}/{action}'
            if not os.path.exists(frame_path):
                print("Không tồn tại thư mục action raw " + action)
                continue

            action_path = f'{processed_path}/{action}'
            if not os.path.exists(action_path):
                os.makedirs(action_path)
            else:
                if not override_exist:
                    print("Không override: " + action)
                    continue

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

                # Xuất tất cả các ảnh đã xử lý vào thư mục chứa các hành động
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
