"""
Evaluate saved models.
"""


from pathlib import Path

import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Find and load saved evaluation DataLoaders.
# TODO
