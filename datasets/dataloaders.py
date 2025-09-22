from torch.utils.data import DataLoader
from .datasets import (
    train_dataset,
    test_dataset
)

TRAIN_BATCH_SIZE = 1 #So far it seems this has worked with batch size 1, but not yet sure for larger batch sizes
TEST_BATCH_SIZE = 1
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)