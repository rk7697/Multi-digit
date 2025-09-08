import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from __init__ import device
from datasets.dataloaders import (
    train_dataloader
)
from datasets.datasets import NEW_SIZE_HEIGHT, NEW_SIZE_WIDTH
from datasets.dataloaders import BATCH_SIZE
from network.network import GRID_SIZE
from network.network import multi_digit
from torch.utils.flop_counter import FlopCounterMode

PRINT_INTERVAL = 30
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

# Compute grid cell coordinates of images from the images' centers
def compute_grid_cell_coordinates(image_centers_x : torch.Tensor, image_centers_y : torch.Tensor):
    # Tensor batches have dimension B x C x H x W, so indices will correspond to H x W
    grid_cell_coordinates_y = (image_centers_y//(NEW_SIZE_HEIGHT/GRID_SIZE)).to(torch.int)
    grid_cell_coordinates_x = (image_centers_x//(NEW_SIZE_WIDTH/GRID_SIZE)).to(torch.int)
    
    grid_cell_coordinates = torch.stack([grid_cell_coordinates_y, grid_cell_coordinates_x], dim=1)
    return grid_cell_coordinates

def get_bboxs_and_predictions_logits(logits, grid_cell_coordinates):
    # max_indices_logits_classes_height, max_indices_logits_classes_width = get_indices_height_width_of_bboxs_and_predictions_logits(logits)
    indices_logits_height, indices_logits_width = grid_cell_coordinates[:, 0], grid_cell_coordinates[:, 1]
    indices_batches = torch.arange(BATCH_SIZE)

    bboxs_and_predictions = logits[indices_batches, :, indices_logits_height, indices_logits_width]
    return bboxs_and_predictions

def loss_from_bbox_and_class_predictions(bboxs_and_predictions_logits, bboxes, grid_cell_coordinates, labels):
    logits_image_relative_widths =  bboxs_and_predictions_logits[:, 0]
    logits_image_relative_heights =  bboxs_and_predictions_logits[:, 1]
    logits_classes = bboxs_and_predictions_logits[:, 2:]
    
    
    # Image bbox is tensor of shape (center_x, center_y, H, W)
    image_heights, image_widths = bboxes[:,2], bboxes[:,3]
    image_relative_heights, image_relative_widths = image_heights / NEW_SIZE_HEIGHT, image_widths / NEW_SIZE_WIDTH

    loss = loss_binary_cross_entropy(logits_image_relative_widths, image_relative_widths) 
    loss += loss_binary_cross_entropy(logits_image_relative_heights, image_relative_heights) 
    loss += loss_categorical_cross_entropy(logits_classes, labels)
    return loss

def train(network, num_epochs, train_dataloader):
    for epoch in range(num_epochs):
        error = 0.0
        for batch_index, (imgs, bboxes, labels) in enumerate(train_dataloader):                        
            imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)

            image_centers_x, image_centers_y = bboxes[:, 0], bboxes[:, 0]
            grid_cell_coordinates = compute_grid_cell_coordinates(image_centers_x, image_centers_y)

            optimizer.zero_grad()
                
            logits=network(imgs)

            bboxs_and_predictions_logits = get_bboxs_and_predictions_logits(logits, grid_cell_coordinates)
            
            loss = loss_from_bbox_and_class_predictions(bboxs_and_predictions_logits, bboxes, grid_cell_coordinates, labels)
            
            loss.backward()

            optimizer.step()

            error+=loss.item()

            if((batch_index +1) % PRINT_INTERVAL == 0):
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
# Instantiate and load network
# multi_digit_net = multi_digit()
# multi_digit_net = multi_digit_net.to(device)
# state_dict = torch.load("./network/network_weights.pth")
# multi_digit_net.load_state_dict(state_dict)

# Instantiate optimizer
optimizer = optim.Adam(multi_digit_net.parameters(),lr=.00001)

error_log = []

#Call train function
train(network=multi_digit_net, num_epochs=NUM_EPOCHS, train_dataloader=train_dataloader)

#Save network weights
torch.save(multi_digit_net.state_dict(), "./network/network_weights_new_arch.pth")

#Call plot_error
plot_error(error_log=error_log, num_epochs=NUM_EPOCHS)