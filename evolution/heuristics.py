def update_registry_from_performance(registry, blueprint, reward):
    for mod in blueprint.modules:
        registry.update_score(mod.id, reward)
