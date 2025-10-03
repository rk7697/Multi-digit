from __init__ import device
from datasets.datasets import EMPTY_CLASS_INDEX
from datasets.dataloaders import (
    test_dataloader
)
import logging
from training.train import accuracy_of_classes_at_subimage_center_cells
from network.network import (
    multi_digit, GRID_SIZE)
import torch
from torchvision.transforms import ToPILImage
to_pil_image = ToPILImage()

# Requires batch size of 1
def integer_accuracy_of_classes_at_subimage_center_cells(bboxs_and_predictions_logits, image_centers_grid_cell_coordinates, target_grids, num_subimages):
    num_batches = num_subimages.shape[0]
    if num_batches != 1 or bboxs_and_predictions_logits.shape[0] != 1 or image_centers_grid_cell_coordinates.shape[0] !=1 or target_grids.shape[0] !=1:
        raise ValueError("Batch size must be 1.")
    
    num_subimages = num_subimages[0]
    accuracy_of_classes_at_subimage_center_cells = accuracy_of_classes_at_subimage_center_cells(bboxs_and_predictions_logits, image_centers_grid_cell_coordinates, target_grids, num_subimages).item()
    
    integer_accuracy_of_classes_at_subimage_center_cells = int(num_subimages * accuracy_of_classes_at_subimage_center_cells)
    return integer_accuracy_of_classes_at_subimage_center_cells

# Requires batch size of 1
# def center_and_bbox_predictions(bboxs_and_predictions_logits):
#     num_batches = bboxs_and_predictions_logits.shape[0]
#     if bboxs_and_predictions_logits.shape[0] != 1:
#         raise ValueError("Batch size must be 1.")
    
#     bboxs_and_predictions_logits = bboxs_and_predictions_logits[0]
    
#     logits_classes = bboxs_and_predictions_logits[2:, :, :]
#     indices_class_predictions = torch.argmax(logits_classes, dim=0)

#     tensor_empty_class_index = torch.tensor(EMPTY_CLASS_INDEX)
#     tensor_empty_class_index_repeated = tensor_empty_class_index.repeat((GRID_SIZE, GRID_SIZE))

#     mask = (indices_class_predictions != tensor_empty_class_index_repeated)
#     centers_predictions = torch.nonzero(mask)

def test(network, test_dataloader):
    total_accuracy = 0
    total_images = 0
    for batch_index, ((imgs, bboxes, image_centers_grid_cell_coordinates, target_grids, targets), num_subimages) in enumerate(test_dataloader):  
        imgs, bboxes, image_centers_grid_cell_coordinates, target_grids, targets = [tensor.to(device) for tensor in [imgs, bboxes, image_centers_grid_cell_coordinates, target_grids, targets]]
        num_subimages = num_subimages[0]

        logits=network(imgs) 
        bboxs_and_predictions_logits = logits

        total_accuracy += integer_accuracy_of_classes_at_subimage_center_cells(bboxs_and_predictions_logits, image_centers_grid_cell_coordinates, target_grids, num_subimages)
        total_images += num_subimages

    accuracy = total_accuracy / total_images

    #Log accuracy
    logging.info(f"accuracy: {accuracy:.2f}")
    
    # Print accuracy
    print(f"accuracy: {accuracy:.2f}")

    return accuracy

def display(network, test_dataloader):
    for batch_index, ((imgs, bboxes, image_centers_grid_cell_coordinates, target_grids, targets), num_subimages) in enumerate(test_dataloader):  
        imgs, bboxes, target_grids, targets = [tensor.to(device) for tensor in [imgs, bboxes, target_grids, targets]]
        num_subimages = num_subimages[0]

        logits=network(imgs) 
        bboxs_and_predictions_logits = logits




# Instantiate and load network
multi_digit_net = multi_digit()
state_dict = torch.load("./network/new_network_weights/arch_1.pth")
multi_digit_net.load_state_dict(state_dict)

# Create logger
logging.basicConfig(filename="./validation/logs/log.log", level=logging.INFO, format="%(message)s")

# Call test
test(network=multi_digit_net, test_dataloader=test_dataloader)

# Call display





