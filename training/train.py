import logging
from datasets.dataloaders import (
    train_dataloader
)
import matplotlib.pyplot as plt
from network.network import multi_digit
import torch
import torch.nn as nn
import torch.optim as optim


# Create logger to log training errors in specified directory
logging.basicConfig(filemode="./training/logs/training_error.log", level=logging.INFO)

# Instantiate network
multi_digit_net = multi_digit()

#Instantiate loss
loss_function = nn.CrossEntropyLoss()

# Instantiate optimizer
optimizer = optim.Adam(multi_digit_net.parameters(),lr=.001)

PRINT_INTERVAL = 10
NUM_EPOCHS = 3
error_log = []
num_batches = len(train_dataloader)

def train(network, num_epochs, train_dataloader):
    for epoch in range(num_epochs):
        error = 0.0
        for batch_index, (imgs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            logits=network.forward(imgs)

            loss=loss_function(logits,labels)

            loss.backward()

            optimizer.step()

            error+=loss.item()

            if((batch_index +1) % PRINT_INTERVAL == 0):
                avg_error = error / batch_index
                error_log.append(avg_error)

                epoch_progress = (epoch + batch_index / num_batches) / num_epochs # Calculate progress of the current epoch based on batch_index
                print(f"error: {avg_error:.5f} percent: {epoch_progress:.2f}")

def plot_error(error_log, num_epochs):
    num_samples = len(error_log)

    log_intervals = [i * num_epochs/num_samples for i in range(num_samples)] # Calculate intervals based on num_samples and num_epochs
    
    plt.plot(log_intervals, error_log)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()

#Call train function
train(network=multi_digit_net, num_epochs=NUM_EPOCHS, train_dataloader=train_dataloader)

#Save network weights
torch.save(multi_digit_net.state_dict(), "./network/network_weights.pth")

#Call plot_error
plot_error(error_log=error_log, num_epochs=NUM_EPOCHS)


