import math
import os
import time
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

import dataCollection


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
        # iaa.Fliplr(0.3),  # horizontally flip 30% of the images
        # iaa.Flipud(0.1),  # vertically flip 10% of the images
        # iaa.Rotate((-20, 20)),  # rotate the image between -45 and 45 degrees
        # iaa.GaussianBlur(sigma=(0, 0.5)),  # apply Gaussian blur with random sigma
        # iaa.LinearContrast((0.95, 1.05)),  # adjust contrast by a random factor
        # iaa.Multiply((0.8, 1.2)),  # multiply the image with random values
        # iaa.MultiplyHueAndSaturation((0.9, 1.1)),  # multiply hue and saturation with random values
        iaa.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-5, 5))
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


# Tạo các variant của ảnh
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

        # Generate output file path
        output_path = os.path.join(output_dir, f"augmented_image_{successCount + 1}.png")

        # Save the augmented image in PNG format
        cv2.imwrite(output_path, augmented_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        print(f"Saved augmented image {successCount + 1}/{num_variations} at: {output_path}")
        successCount += 1


workingFolder = "Augments"
other_path = 'Other'

if __name__ == "__main__":
    # Specify the number of variations
    num_variations = 10  # Replace with the desired number of augmented images

    img_paths = get_image_files(other_path)
    for imgPath in img_paths:

        print("Reading " + imgPath)
        # Get the file name with extension
        file_name = os.path.basename(imgPath)
        print('Filename ' + file_name)
        if file_name != 'Image_5.jpg':
            continue
        # file_name = os.path.splitext(file_name)[0]

        # if file_name == 'A' or file_name == 'B':
        #     continue

        img = cv2.imread(imgPath)

        # # Specify the output directory for saving the augmented images
        # output_dir = f'{workingFolder}/{file_name}'  # Replace with the desired output folder path

        # Call the function to augment and save the images
        augment_and_save_images(img, workingFolder, num_variations)
