import torch.nn as nn
from models.learner import ModularGeneralLearner
from evolution.fusion import CrossModalFusion

class MultimodalLearner(nn.Module):
    def __init__(self, blueprint, adapters, embed_dim=128):
        super().__init__()
        self.adapters = adapters
        self.fusion = CrossModalFusion(embed_dim)
        self.core = ModularGeneralLearner(blueprint)

    def forward(self, inputs):
        latent = [self.adapters[mod](x) for mod, x in inputs.items()]
        fused = self.fusion(latent)
        return self.core(fused)