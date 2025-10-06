# Copyright (C) 2025 Riley K
# License: GNU AGPL v3

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")