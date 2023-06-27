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
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.Flipud(0.5),  # vertically flip 50% of the images
        iaa.Rotate((-45, 45)),  # rotate the image between -45 and 45 degrees
        iaa.GaussianBlur(sigma=(0, 3.0)),  # apply Gaussian blur with random sigma
        iaa.ContrastNormalization((0.8, 1.2)),  # adjust contrast by a random factor
        iaa.Multiply((0.8, 1.2)),  # multiply the image with random values
        iaa.MultiplyHueAndSaturation((0.8, 1.2)),  # multiply hue and saturation with random values
        iaa.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-45, 45), shear=(-16, 16))
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


def augment_and_save_images(image, output_dir, num_variations):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        clear_folder(output_dir)

    # Generate augmented images and save them to the output directory
    for i in range(num_variations):
        # Augment the image
        augmented_image = augment_image(image)

        # Resize the augmented image to a fixed size of 300x300 pixels
        augmented_image = cv2.resize(augmented_image, (300, 300))

        # Generate output file path
        output_path = os.path.join(output_dir, f"augmented_image_{i + 1}.png")

        # Save the augmented image in PNG format
        cv2.imwrite(output_path, augmented_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        print(f"Saved augmented image {i + 1}/{num_variations} at: {output_path}")


workingFolder = "/home/hieubigby/IdeaProjects/HandSignDetection/"
other_path = f'{workingFolder}Data/Other'
img = cv2.imread(get_image_files(other_path)[0])

# Specify the output directory for saving the augmented images
output_dir = f'{workingFolder}Data/D' # Replace with the desired output folder path

# Specify the number of variations
num_variations = 300  # Replace with the desired number of augmented images

# Call the function to augment and save the images
augment_and_save_images(img, output_dir, num_variations)