# Copyright (C) 2025 Riley K
# License: GNU AGPL v3

from __init__ import device
from datasets.datasets import EMPTY_CLASS_INDEX, NEW_SIZE_HEIGHT, NEW_SIZE_WIDTH
from datasets.dataloaders import (
    test_dataloader
)
import logging
from training.train import accuracy_of_classes_at_subimage_center_cells
from network.network import (
    multi_digit, GRID_SIZE)
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
from torchvision.utils import draw_bounding_boxes
from PIL import Image

to_pil_image = ToPILImage()

BBOXES_COLORS = ["green", "yellow", "cyan", "red"] * 2

MIN_ACCURACY = .9

# Requires batch size of 1
def integer_accuracy_of_classes_at_subimage_center_cells(bboxes_and_predictions_logits, image_centers_grid_cell_coordinates, target_grids, num_subimages):
    num_batches = num_subimages.shape[0]
    if num_batches != 1 or bboxes_and_predictions_logits.shape[0] != 1 or image_centers_grid_cell_coordinates.shape[0] !=1 or target_grids.shape[0] !=1:
        raise ValueError("Batch size must be 1.")
    
    accuracy_classes_at_subimage_center_cells = accuracy_of_classes_at_subimage_center_cells(bboxes_and_predictions_logits, image_centers_grid_cell_coordinates, target_grids, num_subimages).item()
    
    integer_accuracy_of_classes_at_subimage_center_cells = int(num_subimages * accuracy_classes_at_subimage_center_cells)
    return integer_accuracy_of_classes_at_subimage_center_cells

# Requires batch size of 1
def subimage_center_predictions(bboxes_and_predictions_logits):
    num_batches = bboxes_and_predictions_logits.shape[0]
    if num_batches != 1:
        raise ValueError("Batch size must be 1.")
    
    bboxes_and_predictions_logits = bboxes_and_predictions_logits[0]
    
    logits_classes = bboxes_and_predictions_logits[2:, :, :]
    predictions_classes = torch.nn.functional.softmax(logits_classes, dim = 0)
    max_predictions_classes, indices_predictions_classes = torch.max(predictions_classes, dim=0)

    tensor_empty_class_index = torch.tensor(EMPTY_CLASS_INDEX).to(device)
    tensor_min_accuracy = torch.tensor(MIN_ACCURACY).to(device)

    mask = torch.logical_and((indices_predictions_classes != tensor_empty_class_index), (max_predictions_classes >= tensor_min_accuracy))
    
    subimage_center_predictions = torch.nonzero(mask)    
    return subimage_center_predictions

# Requires batch size of 1
def subimage_relative_height_and_width_predictions(bboxes_and_predictions_logits, subimage_center_predictions):
    num_batches = bboxes_and_predictions_logits.shape[0]
    if bboxes_and_predictions_logits.shape[0] != 1:
        raise ValueError("Batch size must be 1.")
    
    bboxes_and_predictions_logits = bboxes_and_predictions_logits[0]
    
    relative_height_and_width_logits = bboxes_and_predictions_logits[0:2, :, :]
    relative_height_and_width_predictions = torch.sigmoid(relative_height_and_width_logits)

    subimage_center_predictions_y = subimage_center_predictions[:, 0]
    subimage_center_predictions_x = subimage_center_predictions[:, 1]

    relative_height_and_width_predictions = relative_height_and_width_predictions[:, subimage_center_predictions_y, subimage_center_predictions_x]
    return relative_height_and_width_predictions

# Returned bbox predictions are (xmin, ymin, xmax, ymax)
def subimage_bbox_predictions(image_center_predictions, image_relative_height_and_width_predictions):
    image_center_predictions_y, image_center_predictions_x = image_center_predictions[:, 0], image_center_predictions[:, 1]
    
    image_coordinates_predictions_y = image_center_predictions_y * (NEW_SIZE_HEIGHT // GRID_SIZE) + ((NEW_SIZE_HEIGHT // GRID_SIZE) // 2)
    image_coordinates_predictions_x = image_center_predictions_x * (NEW_SIZE_WIDTH // GRID_SIZE) + ((NEW_SIZE_WIDTH // GRID_SIZE) // 2)

    image_relative_width_predictions, image_relative_height_predictions = image_relative_height_and_width_predictions[0, :], image_relative_height_and_width_predictions[1, :]
    
    image_height_predictions = (image_relative_height_predictions * NEW_SIZE_HEIGHT).to(dtype = torch.long)
    image_width_predictions = (image_relative_width_predictions * NEW_SIZE_WIDTH).to(dtype = torch.long)

    image_ymin_predictions = torch.clamp(image_coordinates_predictions_y - (image_height_predictions // 2), min=0+1, max = NEW_SIZE_HEIGHT-1)
    image_ymax_predictions = torch.clamp(image_coordinates_predictions_y + (image_height_predictions // 2), min=0+1, max = NEW_SIZE_HEIGHT-1)

    image_xmin_predictions = torch.clamp(image_coordinates_predictions_x - (image_width_predictions // 2), min=0+1, max = NEW_SIZE_WIDTH-1)
    image_xmax_predictions = torch.clamp(image_coordinates_predictions_x + (image_width_predictions // 2), min=0+1, max = NEW_SIZE_WIDTH-1)

    subimage_bbox_predictions = torch.stack([image_xmin_predictions, image_ymin_predictions, image_xmax_predictions, image_ymax_predictions], dim=1)

    return subimage_bbox_predictions
     
def test(network, test_dataloader):
    total_accuracy = 0
    total_images = 0
    for batch_index, ((imgs, bboxes, image_centers_grid_cell_coordinates, target_grids, targets), num_subimages) in enumerate(test_dataloader):  
        imgs, bboxes, image_centers_grid_cell_coordinates, target_grids, targets = [tensor.to(device) for tensor in [imgs, bboxes, image_centers_grid_cell_coordinates, target_grids, targets]]

        logits=network(imgs) 
        bboxes_and_predictions_logits = logits
        
        total_accuracy += integer_accuracy_of_classes_at_subimage_center_cells(bboxes_and_predictions_logits, image_centers_grid_cell_coordinates, target_grids, num_subimages)
        total_images += num_subimages

    accuracy = total_accuracy / total_images

    #Log accuracy
    logging.info(f"accuracy: {accuracy:.2f}")
    
    # Print accuracy
    print(f"accuracy: {accuracy:.2f}")

    return accuracy

# Batch size of tst dataloader is 1
def display(network, test_dataloader, num_images):
    test_dataloader_iter = iter(test_dataloader)
    for i in range(num_images):
        ((imgs, bboxes, image_centers_grid_cell_coordinates, target_grids, targets), num_subimages) = next(test_dataloader_iter)
        imgs, bboxes, target_grids, targets = [tensor.to(device) for tensor in [imgs, bboxes, target_grids, targets]]
        
        logits=network(imgs) 
        bboxes_and_predictions_logits = logits

        image_center_predictions = subimage_center_predictions(bboxes_and_predictions_logits)
        image_relative_height_and_width_predictions = subimage_relative_height_and_width_predictions(bboxes_and_predictions_logits, image_center_predictions)
        image_bbox_predictions = subimage_bbox_predictions(image_center_predictions, image_relative_height_and_width_predictions)
        
        img = imgs[0]
        num_predicted_subimages = image_bbox_predictions.shape[0]

        bbox_colors = BBOXES_COLORS[0 : num_predicted_subimages]
        image_with_bboxes = draw_bounding_boxes(image = img, boxes = image_bbox_predictions, colors = bbox_colors)
    
        img_pil = to_pil_image(image_with_bboxes)
        img_pil.show()
        img_pil.save(f"./assets/images/img_{i}.png")    

if __name__ == "__main__":
    # Instantiate and load network
    multi_digit_net = multi_digit()
    state_dict = torch.load("./network/network_weights/arch_1.pth")
    multi_digit_net.load_state_dict(state_dict)
    multi_digit_net = multi_digit_net.to(device)

    # Create logger
    logging.basicConfig(filename="./validation/logs/log.log", level=logging.INFO, format="%(message)s")

    # Call test
    test(network=multi_digit_net, test_dataloader=test_dataloader)

    # Call display
    display(network=multi_digit_net, test_dataloader=test_dataloader, num_images=5)