import random
import torch.nn as nn
from core.components import TrackedLayer


def mutate_blueprint(blueprint, registry, mutation_rate=0.1):
    for module in blueprint.modules:
        if random.random() < mutation_rate:
            # Mutate dimensions
            old_out = module.linear.out_features
            new_out = max(4, old_out + random.randint(-4, 4))
            module.linear = nn.Linear(module.linear.in_features, new_out)

    if random.random() < mutation_rate:
        # Add a new module
        in_dim = random.choice([m.linear.out_features for m in blueprint.modules])
        new_mod = TrackedLayer(in_dim, 16)
        blueprint.add_module(new_mod)
        registry.register(new_mod)
