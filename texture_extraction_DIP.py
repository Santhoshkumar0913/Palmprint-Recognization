
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from skimage import morphology
import numpy as np
import glob
import os
import cv2
import random

# Class to store valid images for a specific serial number
class Palm_Graph():
    def __init__(self, valid):
        self.valid = valid

# Function to fetch image based on serial number

def get_data(number):
    number = str(number).zfill(3)  # Ensure 3-digit format
    valid_path = os.path.join(valid_data_path, number)  # Path to the valid folder
    valid_files = sorted(glob.glob(os.path.join(valid_path, number + '-*.bmp')))
    
    if not valid_files:
        print(f"No images found for serial number {number} in the 'valid' dataset.")
        return None
    
    chosen_image_path = random.choice(valid_files)  # Select a random image
    print(f"Processing image: {os.path.basename(chosen_image_path)}")
    valid_data = imread(chosen_image_path)
    palm = Palm_Graph(valid_data)
    return palm

        
# Preprocessing using Gaussian + Laplacian
def LOG_preprocess(img, R0=40, ksize=5):
    AfterGaussian = np.uint8(cv2.GaussianBlur(img, (5,5), R0))
    processed = cv2.Laplacian(AfterGaussian, -1, ksize=ksize)
    img = cv2.equalizeHist(img)
    return processed

# Processing the image for texture extraction
def process(img):
    img = LOG_preprocess(img)  # Preprocess image using Gaussian & Laplacian
    
    _, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    processed_img = cv2.erode(thresholded, kernel)
    processed_img = morphology.remove_small_objects(processed_img > 0, min_size=40, connectivity=1)
    return processed_img

# Dataset path
valid_data_path = './Palmprint/valid'

# Input serial number from terminal
serial_number = input("Enter the palmprint serial number (e.g., 1, 2, 3...): ")

try:
    serial_number = int(serial_number)
    palm = get_data(serial_number)
    
    if palm and palm.valid is not None:
        valid_image = palm.valid
        
        # Show the original image
        plt.imshow(valid_image, cmap='gray')
        plt.title(f'Original Palmprint - Serial {serial_number}')
        plt.axis('off')
        plt.show()

        # Show the preprocessed image
        plt.imshow(LOG_preprocess(valid_image), cmap='gray')
        plt.title(f'Preprocessed Palmprint - Serial {serial_number}')
        plt.axis('off')
        plt.show()

        # Perform texture extraction
        res = process(valid_image)
        plt.imshow(res, cmap='gray')
        plt.title(f'Texture Extraction - Serial {serial_number}')
        plt.axis('off')
        plt.show()

except ValueError:
    print("Invalid input. Please enter a valid serial number (integer).")