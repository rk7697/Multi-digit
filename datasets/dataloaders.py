from torch.utils.data import DataLoader
from .datasets import (
    train_dataset,
    test_dataset
)


train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


