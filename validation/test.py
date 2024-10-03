from datasets.dataloaders import (
    test_dataloader
)
from network.network import multi_digit
import torch


def test(network, test_dataloader):
    num_samples = len(test_dataloader)
    total_accuracy = 0

    for (img, label) in test_dataloader:
        logits = network(img)
        prediction = torch.argmax(logits)

        if(prediction == label):
            accuracy +=1
    return total_accuracy/num_samples

# Instantiate and load network
multi_digit_net = multi_digit
multi_digit_net.load_state_dict(torch.load("./network/network_weights.pth"))

accuracy = test(network=multi_digit, test_dataloader=test_dataloader)
print(f"accuracy: {accuracy:.2f}")



