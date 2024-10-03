import random
from torchvision import (
    datasets,
    transforms
)
from torchvision.transforms.functional import pad
from cv2 import Canny


#Rotate image by a randomly selected angle between MIN_ANGLE and MAX_ANGLE degrees
def rotate_img(image):
    MIN_ANGLE = -15
    MAX_ANGLE = 15
    angle=random.uniform(MIN_ANGLE,MAX_ANGLE)
    return transforms.functional.rotate(img=image,angle=angle)

#Pad image with specified new size 
#and shift image center to a random location
def pad_shift(image,new_size_width,new_size_height):
    image_height=image.shape[-2]
    image_width=image.shape[-1]

    #Move the center to a random place in the valid range
    half_image_height = image_height//2
    half_image_width = image_width//2

    new_center_y = int(random.uniform(half_image_height, new_size_height + half_image_height))
    new_center_x = int(random.uniform(half_image_width, new_size_width + half_image_width))

    #Define padding (left, right, top, bottom)
    padding_left = new_center_x - half_image_width
    padding_right = new_size_width - (new_center_x + half_image_width)
    padding_bottom = new_center_y - half_image_height
    padding_top = new_size_height - (new_center_y + half_image_height)
    image = pad(image, (padding_left, padding_top, padding_right, padding_bottom))

    return image

#Set new_size to be 500,500 for MNIST image
new_size_width=500
new_size_height=500

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1-x), #Invert image to black with white background
    rotate_img, #Rotate image by a random angle in the specified interval
    transforms.Lambda(lambda x: pad_shift(x, new_size_width, new_size_height)), #Shift center of image to random location in the specified new image dimensions   
])

train_dataset = datasets.MNIST("./data",train=True,download=False,transform=transform)
test_dataset = datasets.MNIST("./data",train=False,download=False,transform=transform)