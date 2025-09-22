import logging
import matplotlib.pyplot as plt
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

PRINT_INTERVAL = 30
NUM_EPOCHS = 1
num_batches = len(train_dataloader)

#Instantiate losses
loss_categorical_cross_entropy= nn.CrossEntropyLoss(reduction="none")
loss_binary_cross_entropy = nn.BCEWithLogitsLoss(reduction = "none")

def get_max_indices_height_width_of_bboxs_and_predictions_logits(logits):
    logits_classes = logits[:, 2:, :, :]
    logits_classes_flattened = logits_classes.flatten(start_dim = 1, end_dim = 3)
    max_indices_logits_classes_flattened = torch.argmax(logits_classes_flattened, dim = 1)
    logits_classes_shape = logits_classes.shape
    max_indices_logits_classes_by_dim = torch.unravel_index(max_indices_logits_classes_flattened, logits_classes_shape[1:])
    max_indices_logits_classes_height, max_indices_logits_classes_width = max_indices_logits_classes_by_dim[1:]

    return (max_indices_logits_classes_height, max_indices_logits_classes_width)

def get_bboxs_and_predictions_logits_for_grid_cell_from_coordinates(logits, grid_cell_coordinates):
    # max_indices_logits_classes_height, max_indices_logits_classes_width = get_indices_height_width_of_bboxs_and_predictions_logits(logits)
    indices_logits_height, indices_logits_width = grid_cell_coordinates[:, 0], grid_cell_coordinates[:, 1]
    logits_batch_size = logits.shape[0]
    indices_batches = torch.arange(logits_batch_size)

    bboxs_and_predictions = logits[indices_batches, :, indices_logits_height, indices_logits_width]
    return bboxs_and_predictions
# def get_bboxs_and_predictions_logits_for_neighboring_grid_cells_of_image_center(logits, grid_cell_coordinates):
#     indices_logits_height, indices_logits_width = grid_cell_coordinates[:, 0], grid_cell_coordinates[:, 1]
#     logits_batch_size = logits.shape[0]
#     indices_batches = torch.arange(logits_batch_size)

#      bboxs_and_predictions = logits[indices_batches, :, indices_logits_height-1:, indices_logits_width]



    

def loss_from_bbox_and_class_predictions_for_image_center_grid_cell(bboxs_and_predictions_logits, bboxes, grid_cell_coordinates, targets):
    logits_image_relative_widths =  bboxs_and_predictions_logits[:, 0]
    logits_image_relative_heights =  bboxs_and_predictions_logits[:, 1]
    logits_classes = bboxs_and_predictions_logits[:, 2:]
    
    
    # Image bbox is tensor of shape (center_x, center_y, H, W)
    image_heights, image_widths = bboxes[:,2], bboxes[:,3]
    image_relative_heights, image_relative_widths = image_heights / NEW_SIZE_HEIGHT, image_widths / NEW_SIZE_WIDTH
    # print(image_relative_heights, image_relative_widths)
    # exit()
    # print(grid_cell_coordinates.squeeze(0))

    # exit()
    loss = 0.0
    loss += loss_binary_cross_entropy(logits_image_relative_widths, image_relative_widths) 
    loss += loss_binary_cross_entropy(logits_image_relative_heights, image_relative_heights) 
    
    # loss += loss_categorical_cross_entropy(bboxs_and_predictions_logits, targets)
    # exit()
    return loss

def loss_from_bbox_and_class_predictions_grids(bboxs_and_predictions_logits, bboxes, image_center_grid_cell_coordinates, target_grids, num_subimages):
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
    
    logits_image_relative_widths =  bboxs_and_predictions_logits[batch_indices_subimages, 0, image_center_grid_cell_coordinates_y, image_center_grid_cell_coordinates_x]
    logits_image_relative_heights =  bboxs_and_predictions_logits[batch_indices_subimages, 1, image_center_grid_cell_coordinates_y, image_center_grid_cell_coordinates_x]

    # Get grid of class logits
    logits_classes = bboxs_and_predictions_logits[:, 2:, :, :]

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
    # in the grid, and then average across each image and grid in the batch,
    # by taking a dot product with a weight tensor
    loss_class_logits = loss_categorical_cross_entropy(logits_classes, target_grids)

    
    weight_tensor_class_logits = torch.fill()

    
    exit()

    exit()
    return loss

def train(network, num_epochs, train_dataloader):
    for epoch in range(num_epochs):
        error = 0.0
        accuracy_of_classes_at_image_center_cell = 0.0
        accuracy_of_classes_at_neighboring_cells = 0.0
        for batch_index, ((imgs, bboxes, image_centers_grid_cell_coordinates, target_grids), num_subimages) in enumerate(train_dataloader):  
            imgs, bboxes, image_centers_grid_cell_coordinates, target_grids = [tensor.to(device) for tensor in [imgs, bboxes, image_centers_grid_cell_coordinates, target_grids]]
            num_subimages = num_subimages.to(device)
            
            optimizer.zero_grad()
            
            logits=network(imgs)
            
            # bboxs_and_predictions_logits_for_image_center_grid_cell = get_bboxs_and_predictions_logits_for_grid_cell_from_coordinates(logits, images_centers_grid_cell_coordinates)
            # class_logits_for_image_center_grid_cell = bboxs_and_predictions_logits_for_image_center_grid_cell[:, 2:]

            # bboxs_and_predictions_logits_for_neighboring_grid_cells_of_image_center = get_bboxs_and_predictions_logits_for_neighboring_grid_cells_of_image_center(logits, image_centers_grid_cell_coordinates)
            # class_logits_for_neighboring_grid_cells_of_image_center = bboxs_and_predictions_logits_for_neighboring_grid_cells_of_image_center[:, 2:]

            bboxs_and_predictions_logits = logits
            
            # if(torch.argmax(class_logits_for_image_center_grid_cell.squeeze(0)) == targets.squeeze(0)):
            #     accuracy_of_classes_at_image_center_cell+=1.0

            # image_centers_grid_cell_coordinates_flattened = image_centers_grid_cell_coordinates.squeeze(0)
            
            # #Since the new image size is at least 32x32, the corresponding image center grid cell will always have neighboring cells
            # for i in range(-1,2):
            #     for j in range (-1,2):
            #         bboxs_and_predictions_logits_for_cell = get_bboxs_and_predictions_logits_for_grid_cell_from_coordinates(logits, torch.tensor([image_centers_grid_cell_coordinates_flattened[0]+i, image_centers_grid_cell_coordinates_flattened[1]+j]).unsqueeze(0))
            #         if(torch.argmax(bboxs_and_predictions_logits_for_cell.squeeze(0)) == EMPTY_CLASS_INDEX):
            #             accuracy_of_classes_at_neighboring_cells+=1.0/8


            # loss = loss_from_bbox_and_class_predictions_for_image_center_grid_cell(bboxs_and_predictions_logits, bboxes, image_centers_grid_cell_coordinates, targets)
            loss = loss_from_bbox_and_class_predictions_grids(bboxs_and_predictions_logits, bboxes, image_centers_grid_cell_coordinates, target_grids, num_subimages)
            
            loss.backward()

            optimizer.step()

            error+=loss.item()

            if((batch_index +1) % PRINT_INTERVAL == 0):
                # print(logits)
                # print(targets)
                # Temporary
                if((batch_index+1) % (PRINT_INTERVAL * 200) == 0):
                    time.sleep(10)
                avg_accuracy_of_classes_at_image_center_cell = accuracy_of_classes_at_image_center_cell / batch_index
                avg_accuracy_of_classes_at_neighboring_cells = accuracy_of_classes_at_neighboring_cells / batch_index
                avg_error = error / batch_index
                error_log.append(avg_error)

                # avg_accuracy = accuracy / batch_index

                epoch_progress = (epoch + batch_index / num_batches) / num_epochs # Calculate progress of the current epoch based on batch_index
                # print(f"error: {avg_error:.5f} avg accuracy: {avg_accuracy:.2f} percent: {epoch_progress:.2f}")
                print(f"error: {avg_error:.5f} percent: {epoch_progress:.2f}")
                print(f"avg_class_accuracy_at_img_cell: {avg_accuracy_of_classes_at_image_center_cell} avg_class_accuracy_at_neighboring_cells: {avg_accuracy_of_classes_at_neighboring_cells}")

                logging.info("") # Log the printed error and epoch progress

def plot_error(error_log, num_epochs):
    num_samples = len(error_log)

    log_intervals = [i * num_epochs/num_samples for i in range(num_samples)] # Calculate intervals based on num_samples and num_epochs
    
    plt.plot(log_intervals, error_log)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()
    
# Create logger to log training errors in specified directory
logging.basicConfig(filename="./training/logs/training_error.log", level=logging.INFO)

# Instantiate network
multi_digit_net = multi_digit()
multi_digit_net = multi_digit_net.to(device)

#Temporary
# state_dict = torch.load("./network/network_weights.pth")
# multi_digit_net.load_state_dict(state_dict)

# Instantiate optimizer
optimizer = optim.SGD(multi_digit_net.parameters(),lr=.01)

error_log = []

#Call train function
train(network=multi_digit_net, num_epochs=NUM_EPOCHS, train_dataloader=train_dataloader)

#Save network weights
torch.save(multi_digit_net.state_dict(), "./network/network_weights/network_weights_arch_1_train_0.pth")

# Save error log
error_log_np = np.array(error_log)
np.save("./training/logs/train_error_1.npy", error_log_np)

#Call plot_error
# plot_error(error_log=error_log, num_epochs=NUM_EPOCHS)
