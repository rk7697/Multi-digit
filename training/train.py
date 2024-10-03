from datasets.dataloaders import (
    train_dataloader
)
import matplotlib.pyplot as plt
from network.network import multi_digit
import torch
import torch.nn as nn
import torch.optim as optim

# Instantiate network
multi_digit_net = multi_digit()

#Instantiate loss
loss_function = nn.CrossEntropyLoss()

# Instantiate optimizer
optimizer = optim.Adam(multi_digit_net.parameters(),lr=.001)

error_log = []
num_batches = len(train_dataloader)
PRINT_INTERVAL = 10

def train(network, num_epochs, train_dataloader):
    for epoch in range(num_epochs):
        error = 0.0
        for batch_index, (imgs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            logits=network.forward(imgs)

            loss=loss_function(logits,labels)

            loss.backward()

            optimizer.step()

            error+=loss

            if((batch_index +1) % PRINT_INTERVAL == 0):
                avg_error = error / batch_index
                error_log.append(avg_error)

                epoch_progress = batch_index / (num_batches * num_epochs)
                print(f"error: {avg_error:.5f} percent: {epoch_progress:.2f}")

def plot_error(error_log):
    log_interval = len(error_log)

    plt.plot(x=log_interval, y=error_log)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()

#Call train function
train(network=multi_digit_net, num_epochs=3, train_dataloader=train_dataloader)

#Save network weights
torch.save(multi_digit_net.state_dict(), "./network/network_weights.pth")

#Call plot_error
plot_error(error_log)


