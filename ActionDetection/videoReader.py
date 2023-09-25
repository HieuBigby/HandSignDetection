import cv2
import os
import mediapipe as mp

video_path = ''
output_folder = 'Frames/Raw/H/Batch 1'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

video_capture = cv2.VideoCapture(video_path)
frame_count = 0
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

if not os.path.exists(video_path):
    print('Không tồn tại link đến video, bật camera ngoài...')
    video_path = 0
cap = cv2.VideoCapture(video_path)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        cv2.imshow('Frame', frame)

        frame_filename = f'frame_{frame_count:04d}.jpg'
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()