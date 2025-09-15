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
from datasets.datasets import NEW_SIZE_HEIGHT, NEW_SIZE_WIDTH
from network.network import multi_digit, GRID_SIZE
from torch.utils.flop_counter import FlopCounterMode

PRINT_INTERVAL = 3
NUM_EPOCHS = 1
num_batches = len(train_dataloader)

#Instantiate losses
loss_categorical_cross_entropy = nn.CrossEntropyLoss()
loss_binary_cross_entropy = nn.BCEWithLogitsLoss()

def get_max_indices_height_width_of_bboxs_and_predictions_logits(logits):
    logits_classes = logits[:, 2:, :, :]
    logits_classes_flattened = logits_classes.flatten(start_dim = 1, end_dim = 3)
    max_indices_logits_classes_flattened = torch.argmax(logits_classes_flattened, dim = 1)
    logits_classes_shape = logits_classes.shape
    max_indices_logits_classes_by_dim = torch.unravel_index(max_indices_logits_classes_flattened, logits_classes_shape[1:])
    max_indices_logits_classes_height, max_indices_logits_classes_width = max_indices_logits_classes_by_dim[1:]

    return (max_indices_logits_classes_height, max_indices_logits_classes_width)

def get_bboxs_and_predictions_logits_for_image_center_grid_cell(logits, grid_cell_coordinates):
    # max_indices_logits_classes_height, max_indices_logits_classes_width = get_indices_height_width_of_bboxs_and_predictions_logits(logits)
    indices_logits_height, indices_logits_width = grid_cell_coordinates[:, 0], grid_cell_coordinates[:, 1]
    logits_batch_size = logits.shape[0]
    indices_batches = torch.arange(logits_batch_size)

    bboxs_and_predictions = logits[indices_batches, :, indices_logits_height, indices_logits_width]
    return bboxs_and_predictions

def loss_from_bbox_and_class_predictions_for_image_center_grid_cell(bboxs_and_predictions_logits, bboxes, grid_cell_coordinates, targets):
    logits_image_relative_widths =  bboxs_and_predictions_logits[:, 0]
    logits_image_relative_heights =  bboxs_and_predictions_logits[:, 1]
    logits_classes = bboxs_and_predictions_logits[:, 2:]
    
    
    # Image bbox is tensor of shape (center_x, center_y, H, W)
    image_heights, image_widths = bboxes[:,2], bboxes[:,3]
    image_relative_heights, image_relative_widths = image_heights / NEW_SIZE_HEIGHT, image_widths / NEW_SIZE_WIDTH

    loss = 0.0
    # loss = loss_binary_cross_entropy(logits_image_relative_widths, image_relative_widths) 
    # loss += loss_binary_cross_entropy(logits_image_relative_heights, image_relative_heights) 
    loss += loss_categorical_cross_entropy(logits_classes, targets)
    return loss

def loss_from_bbox_and_class_predictions_grids(bboxs_and_predictions_logits, bboxes, image_center_grid_cell_coordinates, target_grids):
    # Get the relative width and height predictions of the image center grid cell    
    image_center_grid_cell_coordinates_y, image_center_grid_cell_coordinates_x = image_center_grid_cell_coordinates[:, 0], image_center_grid_cell_coordinates[:, 1]
    
    logits_batch_size = bboxs_and_predictions_logits.shape[0]
    indices_batches = torch.arange(logits_batch_size)

    logits_image_relative_widths =  bboxs_and_predictions_logits[indices_batches, 0, image_center_grid_cell_coordinates_y, image_center_grid_cell_coordinates_x]
    logits_image_relative_heights =  bboxs_and_predictions_logits[indices_batches, 1, image_center_grid_cell_coordinates_y, image_center_grid_cell_coordinates_x]

    # Get relative heights and widths of images
    # Image bbox is tensor of shape (center_x, center_y, H, W)
    image_heights, image_widths = bboxes[:,2], bboxes[:,3]
    image_relative_heights, image_relative_widths = image_heights / NEW_SIZE_HEIGHT, image_widths / NEW_SIZE_WIDTH

    # Get grid of class logits
    logits_classes = bboxs_and_predictions_logits[:, 2:, :, :]

    # Compute losses
    loss = loss_binary_cross_entropy(logits_image_relative_widths, image_relative_widths) 
    loss += loss_binary_cross_entropy(logits_image_relative_heights, image_relative_heights)

    regularizaton_constant = 1/(GRID_SIZE**2 - 1)
    loss += 0.01 * regularizaton_constant * loss_categorical_cross_entropy(logits_classes, target_grids)

    calculated_constant = (.99 - .01 * regularizaton_constant)
    logits_classes_at_image_centers = logits_classes[indices_batches, :, image_center_grid_cell_coordinates_y, image_center_grid_cell_coordinates_x]
    targets_at_image_center = target_grids[indices_batches, image_center_grid_cell_coordinates_y, image_center_grid_cell_coordinates_x]
    loss += calculated_constant * loss_categorical_cross_entropy(logits_classes_at_image_centers, targets_at_image_center)

    # loss += loss_categorical_cross_entropy(logits_classes, target_grids)
    return loss

def train(network, num_epochs, train_dataloader):
    for epoch in range(num_epochs):
        error = 0.0
        for batch_index, (imgs, bboxes, image_centers_grid_cell_coordinates, targets) in enumerate(train_dataloader):                        
            imgs, bboxes, image_centers_grid_cell_coordinates, targets = imgs.to(device), bboxes.to(device), image_centers_grid_cell_coordinates.to(device), targets.to(device)

            optimizer.zero_grad()
            
            # with FlopCounterMode(network) as fcm:
            logits=network(imgs)
            
            # print(fcm.get_total_flops)
            # exit()

            bboxs_and_predictions_logits = get_bboxs_and_predictions_logits_for_image_center_grid_cell(logits, image_centers_grid_cell_coordinates)
            # bboxs_and_predictions_logits = logits
            
            loss = loss_from_bbox_and_class_predictions_for_image_center_grid_cell(bboxs_and_predictions_logits, bboxes, image_centers_grid_cell_coordinates, targets)
            # loss = loss_from_bbox_and_class_predictions_grids(bboxs_and_predictions_logits, bboxes, image_centers_grid_cell_coordinates, target_grids)
            # exit()
            loss.backward()

            optimizer.step()

            error+=loss.item()

            if((batch_index +1) % PRINT_INTERVAL == 0):
                # Temporary
                if((batch_index+1) % (PRINT_INTERVAL * 20) == 0):
                    time.sleep(10)

                avg_error = error / batch_index
                error_log.append(avg_error)

                epoch_progress = (epoch + batch_index / num_batches) / num_epochs # Calculate progress of the current epoch based on batch_index
                print(f"error: {avg_error:.5f} percent: {epoch_progress:.2f}")

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
optimizer = optim.SGD(multi_digit_net.parameters(),lr=.1)

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