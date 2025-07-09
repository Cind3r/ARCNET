import torch.nn as nn

# Designed for MNIST Data ONLY
class ImageAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)
