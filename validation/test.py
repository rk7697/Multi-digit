import logging
from datasets.dataloaders import (
    test_dataloader
)
from network.network import multi_digit
import torch

PRINT_INTERVAL = 30

from torchvision.transforms import ToPILImage
to_pil_image = ToPILImage()

# Create logger to log validation accuracy in the specified directory
logging.basicConfig(filename="./validation/testing_accuracy.log", level=logging.INFO)

def test(network, test_dataloader):
    num_samples = len(test_dataloader)
    total_accuracy = 0

    for batch_index, (img, target) in enumerate(test_dataloader):

        # image = img.squeeze(0)
        # image = to_pil_image(image)
        # image.show()

        logits = network(img)
        # print(logits)
        
       

        # image_centers_grid_cell_coordinates_y, image_centers_grid_cell_coordinates_x = image_centers_grid_cell_coordinates[0]

        # logits_image_grid_cell = logits[0, :, image_centers_grid_cell_coordinates_y, image_centers_grid_cell_coordinates_x]
        # logits_image_grid_cell_wide = logits[0, :, image_centers_grid_cell_coordinates_y-1:image_centers_grid_cell_coordinates_y+2, image_centers_grid_cell_coordinates_x-1:image_centers_grid_cell_coordinates_x+2]


        # print(logits_image_grid_cell_wide[2:, :, :])

        # logits_relative_width = logits_image_grid_cell[0]
        # logits_relative_height = logits_image_grid_cell[1]

        # logits_classes_image_grid_cell = logits_image_grid_cell[2:]

        prediction = torch.argmax(logits)

        # bbox = bbox[0]
        
        # print(f"relative width and height: {bbox[3], bbox[2]}")
        # print(f"target: {target}")

        # print(f"prediction: {prediction}")
        
        # exit()
        if(prediction == target):
            total_accuracy +=1
        
        if(batch_index % 100 ==0):
            print(batch_index/len(test_dataloader))
    return total_accuracy/num_samples

# Instantiate and load network
multi_digit_net = multi_digit()
state_dict = torch.load("./network/network_weights/network_weights_arch_1_train_debug.pth")
multi_digit_net.load_state_dict(state_dict)

accuracy = test(network=multi_digit_net, test_dataloader=test_dataloader)
print(f"accuracy: {accuracy:.2f}")

# logging.info("") # Log the accuracy



