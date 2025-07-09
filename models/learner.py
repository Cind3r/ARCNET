import torch.nn as nn

class ModularGeneralLearner(nn.Module):
    def __init__(self, blueprint):
        super().__init__()
        self.layers = nn.ModuleList(blueprint.modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
