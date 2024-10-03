import random
import numpy as np
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
    threshold_2 = 1
    img_edges = Canny(img_array, threshold1=threshold_1, threshold2=threshold_2)
    return img_edges

# Rotate image by a randomly selected angle between MIN_ANGLE and MAX_ANGLE degrees
def rotate_img(image):
    MIN_ANGLE = -15
    MAX_ANGLE = 15
    angle=random.uniform(MIN_ANGLE,MAX_ANGLE)
    return transforms.functional.rotate(img = image, angle = angle, fill=1)

# Pad image with specified new size 
# and shift image center to a random location
def pad_shift(image,new_size_width,new_size_height):
    image_height=image.shape[-2]
    image_width=image.shape[-1]

    # Move the center to a random place in the valid range
    half_image_height = image_height//2
    half_image_width = image_width//2

    new_center_y = int(random.uniform(half_image_height, new_size_height - half_image_height))
    new_center_img = int(random.uniform(half_image_width, new_size_width - half_image_width))

    # Define padding (left, right, top, bottom)
    padding_left = new_center_img - half_image_width
    padding_right = new_size_width - (new_center_img + half_image_width)
    padding_bottom = new_center_y - half_image_height
    padding_top = new_size_height - (new_center_y + half_image_height)

    PAD_FILL_VALUE=1 # Set the pad fill value to white
    image = pad(image, (padding_left, padding_top, padding_right, padding_bottom), fill = PAD_FILL_VALUE)
    return image

# Set new_size to be 100,100 for MNIST image
new_size_width=100
new_size_height=100

transform = transforms.Compose([
    transforms.Lambda(lambda img: canny_edge_detection(img)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0, std=1), # Normalizer the tensor
    transforms.Lambda(lambda img: 1-img), # Invert image to black with white background
    rotate_img, # Rotate image by a random angle in the specified interval
    transforms.Lambda(lambda img: pad_shift(img, new_size_width, new_size_height)), # Shift center of image to random location in the specified new image dimensions   
])

train_dataset = datasets.MNIST("./data",train=True,download=False,transform=transform)
test_dataset = datasets.MNIST("./data",train=False,download=False,transform=transform)