import math
import random
import numpy as np
import torch
from cv2 import Canny
from torchvision import (
    datasets,
    transforms
)
from torchvision.transforms.functional import pad
import torchvision.transforms as transforms


# Apply canny edge detection to image
def canny_edge_detection(image):
    img_array = np.array(image) # Convert PIL image to numpy

    # Set threshold for weak and strong edges to 1
    threshold_1 = 1
    threshold_2 = 10
    img_edges = Canny(img_array, threshold1=threshold_1, threshold2=threshold_2)
    return img_edges

# Rotate image by a randomly selected angle between MIN_ANGLE and MAX_ANGLE degrees
def rotate_img(image):
    MIN_ANGLE = -15
    MAX_ANGLE = 15
    angle = random.uniform(MIN_ANGLE, MAX_ANGLE)
    return transforms.functional.rotate(img = image, angle = angle, fill=0)

# Resize image to randomly selected height and width between MIN_DIMENSION and MAX_DIMENSION pixels
def resize_img(image):
    MIN_DIMENSION = 28
    MAX_DIMENSION = 100 # Set MAX_DIMENSION to dimension very close to new_size dimensions for use case

    # Resize image to random dimensions in the valid range
    new_height = random.randint(MIN_DIMENSION, MAX_DIMENSION + 1) 
    new_width = random.randint(MIN_DIMENSION, MAX_DIMENSION + 1)
    resize_transform = transforms.Resize((new_height, new_width)) # Define resize transform for image
    return resize_transform(image)

# Pad image with specified new size 
# and shift image center to a random location
def pad_shift(image,new_size_width,new_size_height):
    image_height=image.shape[-2]
    image_width=image.shape[-1]

    # Calculate halved image dimensions
    half_image_height = image_height/2
    half_image_width = image_width/2

    # Generate random integer center in random place in the valid range
    # and subtract the fractional part of half dimension to calculate a valid center
    new_center_y = int(random.uniform(half_image_height, new_size_height - half_image_height)) - math.modf(half_image_height)[0]
    new_center_x = int(random.uniform(half_image_width, new_size_width - half_image_width)) - math.modf(half_image_width)[0]

    # Define padding (left, right, top, bottom)
    padding_left = int(new_center_x - half_image_width)
    padding_right = int(new_size_width - (new_center_x + half_image_width))
    padding_bottom = int(new_center_y - half_image_height)
    padding_top = int(new_size_height - (new_center_y + half_image_height))
    
    PAD_FILL_VALUE=0 # Set the pad fill value to black
    image = pad(image, (padding_left, padding_top, padding_right, padding_bottom), fill = PAD_FILL_VALUE)
    return image

# Set new_size to be 100, 100 for MNIST image
# For now, use small input size for use case
new_size_width = 112
new_size_height = 112

transform = transforms.Compose([
    # transforms.Lambda(lambda img: canny_edge_detection(img)),
    transforms.ToTensor(), # Convert to tensor
    transforms.Normalize(mean=0, std=1), # Normalizer the tensor
    rotate_img, # Rotate image by a random angle in the specified range
    resize_img, # Resize image to random dimensions in the specified range
    transforms.Lambda(lambda img: pad_shift(img, new_size_width, new_size_height)), # Shift center of image to random location in the specified new image dimensions   
    transforms.Lambda(lambda img: img.to(torch.float32))
])

train_dataset = datasets.MNIST("./dataset",train=True,download=False,transform=transform)
test_dataset = datasets.MNIST("./dataset",train=False,download=False,transform=transform)