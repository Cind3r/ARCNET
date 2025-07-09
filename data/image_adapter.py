import torch.nn as nn

# Designed for MNIST Data ONLY
class ImageAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)
