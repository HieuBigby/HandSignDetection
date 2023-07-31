import math
import os
import time
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

import dataAugmentation
from dataCollection import draw_landmark_lines

workingFolder = "/home/hieubigby/IdeaProjects/HandSignDetection/"

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load model và label
classifier = Classifier(f'{workingFolder}Model/keras_model.h5', f'{workingFolder}Model/labels.txt')
offset = 20
imgSize = 300

imgFolder = "Data/C"
counter = 0
labels = ['A', 'B', 'C', 'D', 'Đ', 'E']

while True:
    success, img = cap.read()
    # img = cv2.imread(dataAugmentation.get_image_files(f'{workingFolder}Data/Other')[1])
    # img = cv2.imread(f'{workingFolder}Data/Other/A.png')
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    # Nếu phát hiện được tay thì cắt phần phát hiện đó
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        img = draw_landmark_lines(img, hand['lmList'])
        imgWhite = np.zeros((imgSize, imgSize, 3), np.uint8) # * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        # Xử lý phần ảnh sau khi đã cắt
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

        # Đưa ảnh vào model để đoán
        prediction, index = classifier.getPrediction(imgWhite, draw = False)
        print(prediction, index)
        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

