import numpy as np
import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import os

# Load the dataset
# Dataframes consisting of their respective collections of happy, sad, and surprised faces
# These images must first be mass converted to 2D arrays using OpenCV

trainingSet = pd.DataFrame()
testingSet = pd.DataFrame()

def convert_images_to_arrays(image_folder, label):
    image_list = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')): # Check if the file is an image
            img_path = os.path.join(image_folder, filename)
            img = cv.imread(img_path)
            if img is not None:
                image_list.append(np.array([img,label],dtype=object))
            else:
                print(f"Error reading image: {filename}")
    return np.array(image_list)

# Example usage:
image_folder_happy = 'archiveDataset/train/happy'
image_folder_sad = 'archiveDataset/train/sad'
image_folder_surprise = 'archiveDataset/train/surprise'

image_array_happy = convert_images_to_arrays(image_folder_happy,0)
print(image_array_happy.shape)
image_array_sad = convert_images_to_arrays(image_folder_happy,1)
print(image_array_sad.shape)
image_array_surprise = convert_images_to_arrays(image_folder_happy,2)
print(image_array_surprise.shape)

image_array = np.concatenate([image_array_happy,image_array_sad,image_array_surprise])
print(image_array.shape)

if image_array.size > 0:
    print(f"Successfully converted {len(image_array)} images to arrays.")
    # Further processing with image_arrays (e.g., saving to a file)
else:
     print("No images were converted.")