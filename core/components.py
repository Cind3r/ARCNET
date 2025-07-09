import torch.nn as nn
import uuid
import copy
import hashlib
from datetime import datetime
from collections import defaultdict



# POTENTIALLY DEPRECATED:
class TrackedLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.id = str(uuid.uuid4())
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

