import logging
from datasets.dataloaders import (
    test_dataloader
)
from network.network import multi_digit
import torch


# Create logger to log validation accuracy in the specified directory
logging.basicConfig(filename="./validation/testing_accuracy.log", level=logging.INFO)

def test(network, test_dataloader):
    num_samples = len(test_dataloader)
    total_accuracy = 0

    for (img, label) in test_dataloader:
        logits = network(img)
        prediction = torch.argmax(logits)

        if(prediction == label):
            total_accuracy +=1
    return total_accuracy/num_samples

# Instantiate and load network
multi_digit_net = multi_digit()
state_dict = torch.load("./network/network_weights.pth")
multi_digit_net.load_state_dict(state_dict)

accuracy = test(network=multi_digit_net, test_dataloader=test_dataloader)
print(f"accuracy: {accuracy:.2f}")

logging.info("") # Log the accuracy



