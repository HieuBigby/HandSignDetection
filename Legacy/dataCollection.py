import math
import os
import time
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(thresholded)
    modified_hand = cv2.bitwise_and(image, image, mask=mask)
    return modified_hand


def draw_landmark_lines(image, landmarks):
    # Create a blank image of the same size as the original image
    image_with_lines = np.zeros_like(image)

    # Define the connections between landmarks based on the rule
    connections = [(0, 1, 2, 3, 4),  # Thumb
                   (0, 5, 6, 7, 8),  # Index finger
                   (9, 10, 11, 12),  # Middle finger
                   (13, 14, 15, 16),  # Ring finger
                   (0, 17, 18, 19, 20), # Pinky finger
                   (5, 9, 13, 17)] # Other

    # Loop through the connections and draw lines
    for connection in connections:
        for i in range(len(connection) - 1):
            # Get the index of the current landmark
            index1 = connection[i]
            # Get the index of the next landmark
            index2 = connection[i + 1]

            # Get the coordinates of the current landmark
            x1, y1, _ = landmarks[index1]

            # Get the coordinates of the next landmark
            x2, y2, _ = landmarks[index2]

            # Draw a line between the current and next landmark
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw a circle at each landmark point
        for index in connection:
            # Get the coordinates of the landmark
            x, y, _ = landmarks[index]

            # Draw a circle at the landmark point
            cv2.circle(image_with_lines, (x, y), 3, (0, 0, 255), -1)

    return image_with_lines


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    segmentor = SelfiSegmentation()

    offset = 20
    imgSize = 300

    imgFolder = "Data/B"
    workingFolder = "/home/hieubigby/IdeaProjects/HandSignDetection/"
    counter = 0

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        imgOutput = img.copy()
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            img = draw_landmark_lines(img, hand['lmList'])

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", imgOutput)

        # Save lại ảnh data khi nhấn s
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            file_path = f'{workingFolder}{imgFolder}/Image_{time.time()}.jpg'
            print("Saving image:", file_path)
            success = cv2.imwrite(file_path, imgWhite)
            if not success:
                raise Exception("Could not write image")
            print("Image saved successfully")
            print("Counter:", counter)
