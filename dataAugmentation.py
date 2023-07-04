import math
import os
import time
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


def get_image_files(folder_path):
    image_files = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is an image
        if os.path.isfile(file_path) and any(
                file_name.endswith(extension) for extension in ['.jpg', '.jpeg', '.png', '.gif']):
            image_files.append(file_path)

    return image_files


def augment_image(image):
    # Define augmentation techniques
    seq = iaa.Sequential([
        iaa.Fliplr(0.3),  # horizontally flip 30% of the images
        iaa.Flipud(0.1),  # vertically flip 10% of the images
        iaa.Rotate((-45, 45)),  # rotate the image between -45 and 45 degrees
        iaa.GaussianBlur(sigma=(0, 0.5)),  # apply Gaussian blur with random sigma
        iaa.LinearContrast((0.95, 1.05)),  # adjust contrast by a random factor
        # iaa.Multiply((0.8, 1.2)),  # multiply the image with random values
        iaa.MultiplyHueAndSaturation((0.9, 1.1)),  # multiply hue and saturation with random values
        iaa.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-20, 20), shear=(-1, 1))
    ])

    # Convert image to imgaug format
    image = ia.imresize_single_image(image, (image.shape[0], image.shape[1]))

    # Ensure the image has three channels (RGB format)
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Augment the image
    augmented_image = seq(image=image)

    # Convert the augmented image back to OpenCV format
    augmented_image = augmented_image.astype('uint8')

    return augmented_image


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                clear_folder(file_path)
                os.rmdir(file_path)
    else:
        print(f"Folder does not exist: {folder_path}")


offset = 20
imgSize = 300
detector = HandDetector(maxHands=1)


def augment_and_save_images(image, output_dir, num_variations):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        clear_folder(output_dir)

    successCount = 0

    # Generate augmented images and save them to the output directory
    while successCount < num_variations:
        # Augment the image
        augmented_image = augment_image(image)

        # Find Hands
        hands, foundImg = detector.findHands(augmented_image)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = foundImg[y - offset:y + h + offset, x - offset:x + w + offset]
            if imgCrop.size <= 0:
                continue

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Generate output file path
            output_path = os.path.join(output_dir, f"augmented_image_{successCount + 1}.png")

            # Save the augmented image in PNG format
            cv2.imwrite(output_path, imgWhite, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            print(f"Saved augmented image {successCount + 1}/{num_variations} at: {output_path}")
            successCount += 1


workingFolder = "/home/hieubigby/IdeaProjects/HandSignDetection/"
other_path = f'{workingFolder}Data/Other'

if __name__ == "__main__":
    # Specify the number of variations
    num_variations = 300  # Replace with the desired number of augmented images
    labels = ['A', 'B', 'C', 'D']

    img_paths = get_image_files(other_path)
    for imgPath in img_paths:

        # Get the file name with extension
        file_name = os.path.basename(imgPath)
        file_name = os.path.splitext(file_name)[0]

        img = cv2.imread(imgPath)

        # Specify the output directory for saving the augmented images
        output_dir = f'{workingFolder}Data/{file_name}'  # Replace with the desired output folder path

        # Call the function to augment and save the images
        augment_and_save_images(img, output_dir, num_variations)


