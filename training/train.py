import logging
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from __init__ import device
from datasets.dataloaders import (
    train_dataloader
)
from datasets.datasets import NEW_SIZE_HEIGHT, NEW_SIZE_WIDTH, EMPTY_CLASS_INDEX, MAX_DIGITS
from network.network import multi_digit, GRID_SIZE
from torch.utils.flop_counter import FlopCounterMode

from torchvision.transforms import ToPILImage
to_pil_image = ToPILImage()

LOG_INTERVAL = len(train_dataloader.dataset) / 30
NUM_EPOCHS = 10
num_batches = len(train_dataloader)

def loss_from_bbox_and_class_predictions_grids(bboxes_and_predictions_logits, bboxes, image_center_grid_cell_coordinates, target_grids, num_subimages):
    # Get repeated batch indices for subimages
    num_batches = num_subimages.shape[0]
    batch_indices = torch.arange(num_batches, device=device)
    batch_indices_subimages = torch.repeat_interleave(batch_indices, num_subimages)
    
    # Get subimage indices, repeated for batches
    subimage_indices = torch.arange(MAX_DIGITS, device=device)
    subimage_indices_repeated = subimage_indices.repeat(num_batches)
    num_subimages_repeated_interleaved = torch.repeat_interleave(num_subimages, MAX_DIGITS)
    mask = subimage_indices_repeated < num_subimages_repeated_interleaved
    subimage_indices_batches = subimage_indices_repeated[mask]

    # Get relative heights and widths of images
    image_heights, image_widths = bboxes[batch_indices_subimages, subimage_indices_batches, 2], bboxes[batch_indices_subimages,subimage_indices_batches, 3] # Image bbox is tensor of shape (center_x, center_y, H, W)
    image_relative_heights, image_relative_widths = image_heights / NEW_SIZE_HEIGHT, image_widths / NEW_SIZE_WIDTH

    # Get the relative width and height predictions of the image center grid cell    
    image_center_grid_cell_coordinates_y = image_center_grid_cell_coordinates[batch_indices_subimages, subimage_indices_batches, 0]
    image_center_grid_cell_coordinates_x = image_center_grid_cell_coordinates[batch_indices_subimages, subimage_indices_batches, 1]
    
    logits_image_relative_widths =  bboxes_and_predictions_logits[batch_indices_subimages, 0, image_center_grid_cell_coordinates_y, image_center_grid_cell_coordinates_x]
    logits_image_relative_heights =  bboxes_and_predictions_logits[batch_indices_subimages, 1, image_center_grid_cell_coordinates_y, image_center_grid_cell_coordinates_x]

    # Get grid of class logits
    logits_classes = bboxes_and_predictions_logits[:, 2:, :, :]

    #Compute loss
    loss = 0.0

    # Compute BCE losses for relative width and height logits where
    # the losses for each dimension are averaged across each subimage
    # within a batch element and then across batch elements
    loss_relative_widths_logits = loss_binary_cross_entropy(logits_image_relative_widths, image_relative_widths) 
    loss_relative_heights_logits = loss_binary_cross_entropy(logits_image_relative_heights, image_relative_heights)

    # Since relative width and height logits are one-dimensional ordered by
    # subimage within each batch element, across subimages and across 
    # batch elements by taking a dot product with a weight tensor
    element_inverse_num_subimages = 1.0 / num_subimages
    repeated_element_inverse_num_subimages = torch.repeat_interleave(element_inverse_num_subimages, num_subimages)
    weight_tensor_relative_widths_and_heights = repeated_element_inverse_num_subimages / num_batches

    loss_relative_widths_logits = torch.dot(loss_relative_widths_logits, weight_tensor_relative_widths_and_heights)
    loss_relative_heights_logits = torch.dot(loss_relative_heights_logits, weight_tensor_relative_widths_and_heights)

    # Add losses for relative width and height logits
    loss += loss_relative_widths_logits
    loss += loss_relative_heights_logits

    # Compute weighted categorical cross entropy losses for class logits grid cells
    # where the average of the CCE losses of subimage center grid cell class logits 
    # is weighted equally as the average of the CCE losses of all other grid cells
    # in the grid, and the loss is averaged across each image and grid in the batch.

    # Take a dot product with a weight tensor and then average across each
    # image and grid in the batch
    loss_class_logits = loss_categorical_cross_entropy(logits_classes, target_grids)

    weight_non_subimage_center_cells = 0.5 * (1.0 / (GRID_SIZE**2 - num_subimages))
    weight_subimage_center_cells = 0.5 * (1.0 / num_subimages) 
    
    # Expand weight_non_subimage_center_cells to shape (num_batches, GRID_SIZE, GRID_SIZE)
    weight_non_subimage_center_cells_expanded_dims = weight_non_subimage_center_cells.view((num_batches, 1, 1))
    weight_tensor_class_logits = weight_non_subimage_center_cells_expanded_dims.repeat((1, GRID_SIZE, GRID_SIZE))

    # Set the weight tensor at subimage center cells to weight_subimage_centers_cells as 
    # weight_subimage_center_cells that has repeat interleaved num_subimages times
    weight_subimage_centers_cells = torch.repeat_interleave(weight_subimage_center_cells, num_subimages)
    weight_tensor_class_logits[batch_indices_subimages, image_center_grid_cell_coordinates_y, image_center_grid_cell_coordinates_x] = weight_subimage_centers_cells

    # Take dot product and sum across height and width dimensions, and
    # then average across batches
    loss_class_logits = torch.sum(loss_class_logits * weight_tensor_class_logits, dim=(1, 2))
    loss_class_logits = torch.mean(loss_class_logits, dim = 0)

    # Add loss for class logits
    loss += loss_class_logits

    return loss

# Get average accuracy of class predictions at subimage center cells
def accuracy_of_classes_at_subimage_center_cells(bboxes_and_predictions_logits, image_centers_grid_cell_coordinates, target_grids, num_subimages):
    # Get repeated batch indices for subimages
    num_batches = num_subimages.shape[0]
    batch_indices = torch.arange(num_batches, device=device)
    batch_indices_subimages = torch.repeat_interleave(batch_indices, num_subimages)
    

    # Get subimage indices, repeated for batches
    subimage_indices = torch.arange(MAX_DIGITS, device=device)
    subimage_indices_repeated = subimage_indices.repeat(num_batches)
    num_subimages_repeated_interleaved = torch.repeat_interleave(num_subimages, MAX_DIGITS)
    mask = subimage_indices_repeated < num_subimages_repeated_interleaved
    subimage_indices_batches = subimage_indices_repeated[mask]

    # Get image_center_grid_cell_coordinates
    image_center_grid_cell_coordinates_y = image_centers_grid_cell_coordinates[batch_indices_subimages, subimage_indices_batches, 0]
    image_center_grid_cell_coordinates_x = image_centers_grid_cell_coordinates[batch_indices_subimages, subimage_indices_batches, 1]

    # Get grid of class logits
    logits_classes = bboxes_and_predictions_logits[:, 2:, :, :]

    # Get class logits of subimage_center_cells
    logits_classes_subimage_center_cells = logits_classes[batch_indices_subimages, :, image_center_grid_cell_coordinates_y, image_center_grid_cell_coordinates_x]

    # Get class index prections of subimage center cells and 
    # get targets of subimage center cells from target grids
    indices_class_predictions_subimage_center_cells = torch.argmax(logits_classes_subimage_center_cells, dim=1)
    targets_subimages = target_grids[batch_indices_subimages, image_center_grid_cell_coordinates_y, image_center_grid_cell_coordinates_x]

    # Get accuracy of class predictions of of subimage center cells
    # averaged across subimages per batch element and across batches
    mask = (indices_class_predictions_subimage_center_cells == targets_subimages)
    weight_tensor = torch.repeat_interleave((1.0 / num_subimages), num_subimages) / num_batches

    accuracy_of_classes_at_subimage_center_cells = torch.sum(mask * weight_tensor)
    return accuracy_of_classes_at_subimage_center_cells

# Get average accuracy of class predictions at neighboring cells of subimage center cells
# Accuracy is averaged over neighbors cells per subimage, over subimages, and over
# batch elements per batch. 
# Note that overlapping neighbors cells are involved in the average of neighbor cells
# for each subimage center cell they neighbor
def accuracy_of_classes_at_neighboring_cells_of_subimage_center_cells(bboxes_and_predictions_logits, image_centers_grid_cell_coordinates, target_grids, num_subimages):
    # Get repeated batch indices for subimages
    num_batches = num_subimages.shape[0]
    batch_indices = torch.arange(num_batches, device=device)
    batch_indices_subimages = torch.repeat_interleave(batch_indices, num_subimages)
    
    # Get subimage indices, repeated for batches
    subimage_indices = torch.arange(MAX_DIGITS, device=device)
    subimage_indices_repeated = subimage_indices.repeat(num_batches)
    num_subimages_repeated_interleaved = torch.repeat_interleave(num_subimages, MAX_DIGITS)
    mask = subimage_indices_repeated < num_subimages_repeated_interleaved
    subimage_indices_batches = subimage_indices_repeated[mask]

    # Get image_center_grid_cell_coordinates of subimages 
    image_centers_grid_cell_coordinates_subimages = image_centers_grid_cell_coordinates[batch_indices_subimages, subimage_indices_batches, :]

    # Get grid cell coordinates of neighboring cells of subimage center cells
    # by adding offsets from subimage center cells that are repeated by the number 
    # of neighboring cells per subimage center cell times

    # Get offsets
    offsets_indices_y = torch.linspace(start = -1, end = 1, steps = 3, dtype = torch.long, device=device)
    offsets_indices_x = torch.linspace(start = -1, end = 1, steps = 3, dtype = torch.long, device=device)
    offsets_indices = torch.cartesian_prod(offsets_indices_y, offsets_indices_x)
    offsets_center = torch.tensor([0,0], device=device)
    mask = torch.logical_not((offsets_indices == offsets_center).all(dim = 1)) # Get indices where the tensor element along dimension 1 is [0,0], and then invert indices
    offsets_neighboring_cells_from_subimage_center_cells = offsets_indices[mask, :]
    num_subimages_size = torch.sum(num_subimages) # Since num_subimages is a tensor of shape (num_batches,1), get total number of subimages to repeat offsets by
    offsets_neighboring_cells_from_subimage_center_cells_subimages = offsets_neighboring_cells_from_subimage_center_cells.repeat((num_subimages_size, 1))
    
    # Add grid cell coordinates of subimage center cells to offsets
    # and remove grid cell coordinates that are outside of the range
    # for shape (GRID_SIZE, GRID_SIZE)
    num_neighboring_cells_per_subimage_center_cell = offsets_neighboring_cells_from_subimage_center_cells.shape[0]
    image_centers_grid_cell_coordinates_subimages_neighbors = torch.repeat_interleave(image_centers_grid_cell_coordinates_subimages, num_neighboring_cells_per_subimage_center_cell, dim=0)
    neighboring_cells_of_image_centers_grid_cell_coordinates_subimages = image_centers_grid_cell_coordinates_subimages_neighbors + offsets_neighboring_cells_from_subimage_center_cells_subimages
    mask1 = (neighboring_cells_of_image_centers_grid_cell_coordinates_subimages >= 0).all(dim=1)
    mask2 = (neighboring_cells_of_image_centers_grid_cell_coordinates_subimages < GRID_SIZE).all(dim=1)
    mask_in_range_cells = torch.logical_and(mask1, mask2)
    neighboring_cells_of_image_centers_grid_cell_coordinates_subimages = neighboring_cells_of_image_centers_grid_cell_coordinates_subimages[mask_in_range_cells, :]

    # Get grid of class logits
    logits_classes = bboxes_and_predictions_logits[:, 2:, :, :]

    # Get grid cell coordinates of neighboring cells for each of H and W dimensions
    neighboring_cells_of_image_centers_grid_cell_coordinates_subimages_y = neighboring_cells_of_image_centers_grid_cell_coordinates_subimages[:, 0]
    neighboring_cells_of_image_centers_grid_cell_coordinates_subimages_x = neighboring_cells_of_image_centers_grid_cell_coordinates_subimages[:, 1]

    # Get neighboring cell logits
    batch_indices_subimages_neighbors = torch.repeat_interleave(batch_indices_subimages, num_neighboring_cells_per_subimage_center_cell)
    batch_indices_subimages_neighbors = batch_indices_subimages_neighbors[mask_in_range_cells]
    logits_classes_neighboring_cells_of_subimage_center_cells = logits_classes[batch_indices_subimages_neighbors, :, neighboring_cells_of_image_centers_grid_cell_coordinates_subimages_y, neighboring_cells_of_image_centers_grid_cell_coordinates_subimages_x]
    
    # Get class index prections of neighboring cells of subimage center cells 
    # and get targets from target grids
    indices_class_predictions_subimage_center_cells = torch.argmax(logits_classes_neighboring_cells_of_subimage_center_cells, dim=1)
    targets_subimages = target_grids[batch_indices_subimages_neighbors, neighboring_cells_of_image_centers_grid_cell_coordinates_subimages_y, neighboring_cells_of_image_centers_grid_cell_coordinates_subimages_x]

    # Get accuracy of class predictions of neighboring cells of subimage center cells 
    # averaged across subimage, across subimages, and across batches
    mask = (indices_class_predictions_subimage_center_cells == targets_subimages)

    element_inverse_num_subimages = 1.0 / num_subimages
    repeated_element_inverse_num_subimages = torch.repeat_interleave(element_inverse_num_subimages, num_subimages * num_neighboring_cells_per_subimage_center_cell)
    repeated_element_inverse_num_subimages = repeated_element_inverse_num_subimages[mask_in_range_cells]
    # Note that since some neighbor cells may be out of range, dividing by num_neighboring_cells_per_subimage_center_cell is slightly erroneous for these cases
    # Currently it does not seem there is a direct torch way to do iterative averages over a dimension with strides specified by another tensor, so the average assumes
    # each sub image center cell has num_neighboring_cells_per_subimage_center_cell neighbors
    weight_tensor = repeated_element_inverse_num_subimages / (num_batches * num_neighboring_cells_per_subimage_center_cell) 

    accuracy_of_classes_at_neighboring_cells_of_subimage_center_cells = torch.sum(mask * weight_tensor)
    return accuracy_of_classes_at_neighboring_cells_of_subimage_center_cells

def train(network, num_epochs, train_dataloader):
    for epoch in range(num_epochs):
        error = 0.0
        error_of_classes_at_image_center_cells_total = 0.0
        error_of_classes_at_neighboring_cells_of_subimage_center_cells_total = 0.0
        for batch_index, ((imgs, bboxes, image_centers_grid_cell_coordinates, target_grids, targets), num_subimages) in enumerate(train_dataloader):  
            imgs, bboxes, image_centers_grid_cell_coordinates, target_grids, targets = [tensor.to(device) for tensor in [imgs, bboxes, image_centers_grid_cell_coordinates, target_grids, targets]]
            num_subimages = num_subimages.to(device)
            
            optimizer.zero_grad()
            
            logits=network(imgs)
            
            bboxes_and_predictions_logits = logits
            
            loss = loss_from_bbox_and_class_predictions_grids(bboxes_and_predictions_logits, bboxes, image_centers_grid_cell_coordinates, target_grids, num_subimages)
            
            loss.backward()

            optimizer.step()

            # Summing for logging
            error+=loss.item()
            error_of_classes_at_image_center_cells_total += (1.0 - accuracy_of_classes_at_subimage_center_cells(bboxes_and_predictions_logits, image_centers_grid_cell_coordinates, target_grids, num_subimages).item())
            error_of_classes_at_neighboring_cells_of_subimage_center_cells_total += (1.0 - accuracy_of_classes_at_neighboring_cells_of_subimage_center_cells(bboxes_and_predictions_logits, image_centers_grid_cell_coordinates, target_grids, num_subimages).item())

            # Logging
            if((batch_index +1) % LOG_INTERVAL == 0):
                # Compute averages of error and accuracies
                avg_error = error / batch_index
                avg_error_of_classes_at_image_center_cells = error_of_classes_at_image_center_cells_total / batch_index
                avg_error_of_classes_at_neighboring_cells_of_subimage_center_cells = error_of_classes_at_neighboring_cells_of_subimage_center_cells_total / batch_index
                
                # Log averages of error and accuracies
                error_log.append(avg_error)
                error_of_classes_at_image_center_cells_log.append(avg_error_of_classes_at_image_center_cells)
                error_of_classes_at_neighboring_cells_of_subimage_center_cells_log.append(avg_error_of_classes_at_neighboring_cells_of_subimage_center_cells)

                # Compute progress of the current epoch of the current batch_index
                epoch_progress = (epoch + batch_index / num_batches) / num_epochs

                # Print error and accuracies
                logging.info("----------------------------")
                logging.info(f"error: {avg_error:.5f} percent: {epoch_progress:.2f}")
                logging.info(f"avg_class_error_at_img_cell: {avg_error_of_classes_at_image_center_cells:.5f}")
                logging.info(f"avg_class_error_at_neighboring_cells: {avg_error_of_classes_at_neighboring_cells_of_subimage_center_cells:.5f}")

                print("----------------------------")
                print(f"error: {avg_error:.5f} percent: {epoch_progress:.2f}")
                print(f"avg_class_error_at_img_cell: {avg_error_of_classes_at_image_center_cells:.5f}")
                print(f"avg_class_error_at_neighboring_cells: {avg_error_of_classes_at_neighboring_cells_of_subimage_center_cells:.5f}")


if __name__ == "__main__":
    #Instantiate losses
    loss_categorical_cross_entropy= nn.CrossEntropyLoss(reduction= "none")
    loss_binary_cross_entropy = nn.BCEWithLogitsLoss(reduction = "none")

    # Instantiate network
    multi_digit_net = multi_digit()
    multi_digit_net = multi_digit_net.to(device)

    # Instantiate optimizer
    optimizer = optim.SGD(multi_digit_net.parameters(),lr=.01)

    # Create logger
    logging.basicConfig(filename="./training/logs/log.log", level=logging.INFO, format="%(message)s")

    # Create arrays for logging
    error_log = []
    error_of_classes_at_image_center_cells_log = []
    error_of_classes_at_neighboring_cells_of_subimage_center_cells_log = []

    #Call train function
    train(network=multi_digit_net, num_epochs=NUM_EPOCHS, train_dataloader=train_dataloader)

    #Save network weights
    torch.save(multi_digit_net.state_dict(), "./network/network_weights/arch_1.pth")

    # Save logs by converting log arrays
    # to numpy arrays
    error_log_np = np.array(error_log)
    error_of_classes_at_image_center_cells_log_np = np.array(error_of_classes_at_image_center_cells_log)
    error_of_classes_at_neighboring_cells_of_subimage_center_cells_log_np = np.array(error_of_classes_at_neighboring_cells_of_subimage_center_cells_log)

    np.save("./training/logs/train_error.npy", error_log_np)
    np.save("./training/logs/error_of_classes_at_image_center_cells.npy", error_of_classes_at_image_center_cells_log_np)
    np.save("./training/logs/error_of_classes_at_neighboring_cells_of_subimage_center_cells.npy", error_of_classes_at_neighboring_cells_of_subimage_center_cells_log_np)