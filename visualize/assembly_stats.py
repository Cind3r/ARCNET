# Functions purely made to report assembly stats/structure for readability
def AssemblyStats(modules):
    """
    Generate assembly statistics for all models in the provided list.
    This function prints detailed assembly information for the best model
    based on fitness, including assembly complexity, pathway, operations,
    and other relevant metrics.
    """
    if not modules:
        print("No models available for assembly statistics.")
        return

    # Sort models by fitness
    modules.sort(key=lambda m: m.fitness, reverse=True)

    # Print basic information about all models
    print("=== Assembly Statistics ===")
    print(f"Total models: {len(modules)}")
    print(f"Best model fitness: {modules[0].fitness:.4f}")
    print()

    # Print assembly information of best model
    best_model = max(modules, key=lambda m: m.fitness)

    print("=== Assembly Information for Best Model ===")
    print(f"Model ID: {best_model.id}")
    print(f"Fitness: {best_model.fitness:.4f}")
    print(f"Created at step: {best_model.created_at}")
    print(f"Parent ID: {best_model.parent_id}")
    print()

    # Assembly complexity information
    print("=== Assembly Complexity ===")
    assembly_complexity = best_model.get_assembly_complexity()
    print(f"Assembly Complexity: {assembly_complexity}")
    print(f"Assembly Index: {best_model.assembly_index}")
    print(f"Assembly Steps: {best_model.assembly_steps}")
    print(f"Copy Number: {best_model.copy_number}")
    print()

    # Assembly pathway information
    print("=== Assembly Pathway ===")
    if hasattr(best_model, 'assembly_pathway') and best_model.assembly_pathway:
        print(f"Assembly pathway length: {len(best_model.assembly_pathway)}")
        for i, component in enumerate(best_model.assembly_pathway[:5]):  # Show first 5 components
            if hasattr(component, 'id'):
                print(f"  Component {i}: {component.id}")
            else:
                print(f"  Component {i}: {type(component).__name__}")
        if len(best_model.assembly_pathway) > 5:
            print(f"  ... and {len(best_model.assembly_pathway) - 5} more components")
    else:
        print("No assembly pathway recorded")
    print()

    # Assembly operations
    print("=== Assembly Operations ===")
    if hasattr(best_model, 'assembly_operations') and best_model.assembly_operations:
        print(f"Number of assembly operations: {len(best_model.assembly_operations)}")
        for i, op in enumerate(best_model.assembly_operations):
            print(f"  Operation {i}: {op}")
    else:
        print("No assembly operations recorded")
    print()

    # Catalytic information
    print("=== Catalytic Information ===")
    print(f"Is autocatalytic: {best_model.is_autocatalytic}")
    print(f"Catalyzed by: {best_model.catalyzed_by}")
    print(f"Catalyzes: {best_model.catalyzes}")
    print()

    # Layer components assembly complexity
    print("=== Layer Components Assembly Complexity ===")
    if hasattr(best_model, 'layer_components'):
        for layer_name, component in best_model.layer_components.items():
            if hasattr(component, 'get_minimal_assembly_complexity'):
                complexity = component.get_minimal_assembly_complexity()
                print(f"  {layer_name}: {complexity}")
            else:
                print(f"  {layer_name}: No complexity method available")
    print()

    # System-level assembly complexity
    print("=== System Assembly Complexity ===")
    try:
        from models.arcnet import system_assembly_complexity
        sys_complexity = system_assembly_complexity(modules)
        print(f"Total system assembly complexity: {sys_complexity:.4f}")
    except Exception as e:
        print(f"Could not compute system complexity: {e}")
    print()

    # Position and manifold information
    print("=== Manifold Position Information ===")
    print(f"Manifold dimension: {best_model.manifold_dim}")
    print(f"Position: {best_model.position.data[:5].tolist()}...")  # Show first 5 dimensions / Consider updating to a better visualization for large dimensions
    print(f"Curvature: {best_model.curvature}")
    if best_model.position_info:
        print(f"Position info: {best_model.position_info}")
    print()

    # Q-learning information related to assembly
    print("=== Q-Learning Assembly Information ===")
    print(f"Q-learning method: {best_model.q_learning_method}")
    if hasattr(best_model, 'q_function') and best_model.q_function is not None:
        if hasattr(best_model.q_function, 'replay_buffer'):
            print(f"Q-function replay buffer size: {len(best_model.q_function.replay_buffer)}")
        print(f"Q-memory usage: {best_model.get_q_memory_usage():.2f} MB")
    else:
        print("No Q-function available")