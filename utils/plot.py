import matplotlib.pyplot as plt
import numpy as np
from training.train import NUM_EPOCHS

def plot_accuracies(error_of_classes_at_image_center_cells_log, error_of_classes_at_neighboring_cells_of_subimage_center_cells_log, num_epochs):
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    # This assumes the lengths of the accuracy logs are the same
    num_samples = len(error_of_classes_at_image_center_cells_log)
    epoch_interval = [i * num_epochs/num_samples for i in range(num_samples)] # Calculate intervals based on num_samples and num_epochs
    
    plt.plot(epoch_interval, error_of_classes_at_image_center_cells_log, label = "image center cells")
    plt.plot(epoch_interval, error_of_classes_at_neighboring_cells_of_subimage_center_cells_log, label = "neighboring cells of image centers")
    
    plt.xlabel("epoch")
    plt.ylabel("error")

    plt.legend(loc = "lower left")
    
    plt.show()

# Load errors as np arrays
error_of_classes_at_image_center_cells_log_np = np.load("./training/new_logs/error_of_classes_at_image_center_cells.npy")
error_of_classes_at_neighboring_cells_of_subimage_center_cells_log_np = np.load("./training/logs/error_of_classes_at_neighboring_cells_of_subimage_center_cells.npy")

# Call plot accuracies
plot_accuracies(error_of_classes_at_image_center_cells_log_np, error_of_classes_at_neighboring_cells_of_subimage_center_cells_log_np, NUM_EPOCHS)
