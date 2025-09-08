from torch.utils.data import DataLoader
from .datasets import (
    train_dataset,
    test_dataset
)

BATCH_SIZE = 1
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)