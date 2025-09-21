import math
import random
import torch
from typing import Tuple
from torch.utils.data import Dataset
from torchvision import (
    datasets,
    transforms
)
from torchvision.transforms.functional import pad
from network.network import GRID_SIZE
import torchvision.transforms as transforms

from torchvision.transforms import ToPILImage
to_pil_image = ToPILImage()

# Set NEW_SIZE dimensions to be 256, 256 for MNIST image
# These are the dimensions for the full image that the digits are placed into
# For now, use small input size for use case
NEW_SIZE_HEIGHT = 64
NEW_SIZE_WIDTH = 64

GRID_INTERVAL_Y_SIZE = NEW_SIZE_HEIGHT/GRID_SIZE
GRID_INTERVAL_X_SIZE = NEW_SIZE_WIDTH/GRID_SIZE

# An empty class is added to the set of
# possible classes for a grid cell
EMPTY_CLASS_INDEX = 10

# Set max number of digits
# in image
MAX_DIGITS = 4

def crop_blank_space(image):
    bbox = image.getbbox() # Get bounding box as (left, upper, right, lower) from PIL
    cropped_image = image.crop(bbox) # Crop image using bounding box
    return cropped_image

# Rotate image by a randomly selected angle between MIN_ANGLE and MAX_ANGLE degrees
def rotate_img(image):
    MIN_ANGLE = -15
    MAX_ANGLE = 15
    angle = random.uniform(MIN_ANGLE, MAX_ANGLE)
    return transforms.functional.rotate(img = image, angle = angle, fill=0)

# Resize image to randomly selected height and width between MIN_DIMENSION and MAX_DIMENSION pixels
def resize_img(image):
    # Here it is assumed that NEW_SIZE_HEIGHT is equal to NEW_SIZE_WIDTH
    MIN_DIMENSION = NEW_SIZE_HEIGHT // (GRID_SIZE * 2)  #Set MIN_DIMENSON to 1/3 of the dimensions of a grid cell
    MAX_DIMENSION = int(NEW_SIZE_HEIGHT / 1.5) # Set MAX_DIMENSION to dimension very close to new_size dimensions for use case

    # Resize image to random dimensions in the valid range with the requirement 
    # that each dimension is at least .5 times and at most 2 times the other
    random_order = random.randint(0,1)
    if(random_order == 0):
        new_height = random.randint(MIN_DIMENSION, MAX_DIMENSION)

        min_new_width = max(MIN_DIMENSION, new_height // 2)
        max_new_width = min(MAX_DIMENSION, new_height * 2)

        new_width = random.randint(min_new_width, max_new_width)
    else:
        new_width = random.randint(MIN_DIMENSION, MAX_DIMENSION)

        min_new_height = max(MIN_DIMENSION, new_width // 2)
        max_new_height = min(MAX_DIMENSION, new_width * 2)

        new_height = random.randint(min_new_height, max_new_height)

    resize_transform = transforms.Resize((new_height, new_width)) # Define resize transform for image
    return resize_transform(image)

# Return randomly selected image center within new image dimensions
def random_img_center(image_dimensions: Tuple[int,int], new_image_dimensions = (NEW_SIZE_HEIGHT, NEW_SIZE_WIDTH)):
    image_height, image_width = image_dimensions
    new_image_height, new_image_width = new_image_dimensions

    # Calculate halved image dimensions
    half_image_height = image_height/2
    half_image_width = image_width/2
    
    # Generate random integer center in random place in the valid range
    
    # Since a center dimension will be an integer when the image dimesion is even
    # and will have a fraction part of .5 when the image dimension is odd,
    # shift the center dimension range so the endpoints are integers and
    # and shift back by the fractional part of the half image dimension

    half_image_height_frac_part = math.modf(half_image_height)[0]
    half_image_width_frac_part = math.modf(half_image_width)[0]

    new_center_y = (
        int(random.uniform((half_image_height + half_image_height_frac_part), ((new_image_height - half_image_height) + half_image_height_frac_part)))
        - half_image_height_frac_part
    ) 

    new_center_x = (
        int(random.uniform((half_image_width + half_image_width_frac_part), ((new_image_width - half_image_width) + half_image_width_frac_part)))
        - half_image_width_frac_part
    ) 

    return (new_center_x, new_center_y)

#  Compute resulting bbox for image from random resize and random center shift
def compute_bbox(image):
    # Image is tensor of shape 1 x H x W
    image_height, image_width = image.shape[1], image.shape[2]
    
    # Get new center for image and store it as center_x, center_y to later shift image in pad_shift_with_bbox
    new_center_x, new_center_y = random_img_center((image_height, image_width))
    
    bbox = torch.tensor((new_center_x, new_center_y, image_height, image_width))
    return bbox

# Define ImageWithBBox class to represent (img, bbox)
# so that after add_bbox, (img, bbox) can move through transforms 
class ImageWithBBox():
    def __init__(self, image, bbox):
        self.image = image
        self.bbox = bbox

    def __call__(self):
        return self.image, self.bbox

# Add bbox from random resize and random center shift to image
# Return result as ImageWithBBox
def add_bbox(image):
    return ImageWithBBox(image, compute_bbox(image))

# Pad image with specified new size 
# and shift image center to a random location
def pad_shift(image, new_center, new_size_width = NEW_SIZE_WIDTH, new_size_height = NEW_SIZE_HEIGHT):
    # Image is tensor of shape 1 x H x W
    image_height = image.shape[1]
    image_width = image.shape[2]

    # Calculate halved image dimensions
    half_image_height = image_height/2
    half_image_width = image_width/2

    new_center_x, new_center_y = new_center

    # Define padding (left, right, top, bottom)
    padding_left= int(new_center_x - half_image_width)
    padding_right = int(new_size_width - (new_center_x + half_image_width))
    padding_top = int(new_center_y - half_image_height) # Center height is the distance to the top of image
    padding_bottom = int(new_size_height - (new_center_y + half_image_height))

    PAD_FILL_VALUE=0 # Set the pad fill value to black
    image = pad(image, (padding_left, padding_top, padding_right, padding_bottom), fill = PAD_FILL_VALUE)
    return image

# Pad image and shift center of image in image_with_bbox using bbox center
def pad_shift_with_bbox(image_with_bbox: ImageWithBBox):
    # Image bbox is tensor of shape (center_x, center_y, H, W)
    new_center = (image_with_bbox.bbox[0], image_with_bbox.bbox[1])

    image_with_bbox.image = pad_shift(image_with_bbox.image, new_center)
    return image_with_bbox

# This transform is going to be moved and the transforms will be moved into the get item
# This is because this transform is supposed to be just for the image, but the transform
# Has to add a bbox in order to construct the image
# This is simpler done and further transforms are simpler done in get item
transform = transforms.Compose([
    crop_blank_space,
    transforms.ToTensor(), # Convert PIL to 1 x H x W tensor
    transforms.Normalize(mean=.1307, std=.3081), # Normalize image tensor
    rotate_img, # Rotate image by a random angle in the specified range
    resize_img, # Resize image to random dimensions in the specified range
    # transforms.Resize((NEW_SIZE_HEIGHT,NEW_SIZE_WIDTH)),
    add_bbox, #Transform image to class representing (image, bbox) where bbox center is randomly selected from the valid range and bbox height, width are the image dimensions
    pad_shift_with_bbox,  # Shift center of image to random location (specified in bbox) in the specified new image dimensions (global constants)
    transforms.Lambda(lambda image_with_bbox: (image_with_bbox.image, image_with_bbox.bbox)) #Return tuple (image, bbox)
])

# Compute grid cell coordinates of image from the image's center
def compute_grid_cell_coordinates(image_center_x : torch.Tensor, image_center_y : torch.Tensor):
    # Tensor batches have dimension B x C x H x W, so indices will correspond to H x W
    grid_cell_coordinates_y = (image_center_y//GRID_INTERVAL_Y_SIZE).to(torch.int)
    grid_cell_coordinates_x = (image_center_x//GRID_INTERVAL_X_SIZE).to(torch.int)
    
    grid_cell_coordinates = torch.tensor([grid_cell_coordinates_y, grid_cell_coordinates_x])
    return grid_cell_coordinates

# Compute target as grid of class indices
# where the class index of the cell that
# contains the image's center is the
# image's class, and the class index of 
# every other cell is the empty class
def compute_target_grid(image_center_grid_cell_coordinates, target):
    grid_of_repeated_targets = torch.full((GRID_SIZE, GRID_SIZE), EMPTY_CLASS_INDEX)
    grid_of_repeated_targets[image_center_grid_cell_coordinates] = target

    target_grid = grid_of_repeated_targets
    return target_grid

# Define augmented MNIST dataset with bboxes to return (image, bboxs, targets)
class AugmentedMNISTWithBBoxes(Dataset):
    def __init__(self, train=True, transform=None):
        self.mnist_dataset = datasets.MNIST("./dataset",train=train, download=False,transform=transform)        

    def __len__(self):
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx):
        # For each of the next MAX_DIGITS number of digits,
        # add the digit to the image if its center's
        # corresponding grid cell does yet not contain 
        # any other image center

        grid_cell_tracker = torch.zeros((GRID_SIZE,GRID_SIZE))

        # Get the images, bboxes, image center grid cell coordinates, and targets 
        # of the subimages
        # (Note that the indices cannot be collected first 
        # because the transform dynamically centers and resizes images)
        
        # Note that bboxes, image center grid cell coordinates, and targets 
        # must be arrays to collect a number of tensors that is only known
        # after iterating
        subimages = []
        bboxes = []
        image_centers_grid_cell_coordinates = []
        targets = []

        for sub_idx in range(idx, idx + MAX_DIGITS):
            ((image,bbox), target) = self.mnist_dataset[sub_idx]
            target = torch.tensor(target)

            image_center_x, image_center_y = bbox[0], bbox[1]

            image_center_grid_cell_coordinates = compute_grid_cell_coordinates(image_center_x, image_center_y)

            grid_cell_coordinates_y, grid_cell_coordinates_x = image_center_grid_cell_coordinates[0].item(), image_center_grid_cell_coordinates[1].item()   
            if(grid_cell_tracker[grid_cell_coordinates_y, grid_cell_coordinates_x] == 0):
                grid_cell_tracker[grid_cell_coordinates_y, grid_cell_coordinates_x] = 1

                subimages.append(image)
                bboxes.append(bbox)
                image_centers_grid_cell_coordinates.append(image_center_grid_cell_coordinates)
                targets.append(target)
        
        # Stack arrays of tensors into tensor
        subimages = torch.stack(subimages)
        bboxes = torch.stack(bboxes)
        image_centers_grid_cell_coordinates = torch.stack(image_centers_grid_cell_coordinates)
        targets =torch.stack(targets)

        # Compute image as the sum of subimages where
        # the max result is clamped at 1
        image = torch.sum(subimages, dim=0).clamp(min=0,max=1)
        image = to_pil_image(image)
        image.show()
        exit()

            

                
        






            # print(grid_cell_tracker[torch.tensor([0,0])])
            # print(grid_cell_tracker[image_center_grid_cell_coordinates])
            
            # print(type(grid_cell_tracker[image_center_grid_cell_coordinates]))
        exit()



        target_grid = compute_target_grid(image_center_grid_cell_coordinates, target)

        return (image, bbox, image_center_grid_cell_coordinates, target_grid, target)
    
train_dataset = AugmentedMNISTWithBBoxes(train=True, transform=transform)
test_dataset = AugmentedMNISTWithBBoxes(train=False, transform=transform)
