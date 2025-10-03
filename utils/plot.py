import matplotlib.pyplot as plt
import numpy as np
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
    


plot_accuracies([1,2,3,4,5,6], [1,2,3,4,5,6], 1)
