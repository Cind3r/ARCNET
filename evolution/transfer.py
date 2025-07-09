def transfer_weights(old_model, new_model):
    old_layers = {mod.id: mod for mod in old_model.core.layers}
    new_layers = {mod.id: mod for mod in new_model.core.layers}

    for mod_id, new_mod in new_layers.items():
        if mod_id in old_layers:
            old_mod = old_layers[mod_id]
            if (
                old_mod.linear.weight.shape == new_mod.linear.weight.shape and
                old_mod.linear.bias.shape == new_mod.linear.bias.shape
            ):
                new_mod.linear.weight.data = old_mod.linear.weight.data.clone()
                new_mod.linear.bias.data = old_mod.linear.bias.data.clone()
                print(f"Transferred weights for layer {mod_id}")
            else:
                print(f"Shape mismatch: skipped {mod_id}")
