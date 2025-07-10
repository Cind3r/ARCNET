import torch
import os
import copy
from copy import deepcopy
from tqdm.notebook import tqdm
from datetime import datetime
import numpy as np
import random
from models.arcnet import ConceptModule
from evolution.fitness import (
    compute_fitness_adaptive_complexity_enhanced
)
from evolution.rewards import (
    compute_manifold_novelty,
    compute_system_assembly_complexity, 
    compute_reward_adaptive_aan_normalized
)
from evolution.loss import compute_loss
from core.message import (
    comprehensive_manifold_q_message_passing
)
from evolution.bias import (
    monitor_prediction_diversity_with_action,
    catalyst_bias_elimination,
    select_bias_resistant_catalysts
)
from data.loader import (
    load_and_seed_population_fixed,
    save_lineage_snapshot_to_file,
    export_best_models
)

# NEW IMPORTS FOR ASSEMBLY AWARE MUTATION TESTING
from evolution.mutation import (
    assembly_aware_mutation_with_tracking
)
from core.message import (
    enhanced_message_passing_with_assembly_tracking
)
from core.registry import (
    AssemblyTrackingRegistry
)

# This function can definitely be simplified, for now it is comprehensive and includes all the features we want to test.
def Trainer(
        X_train, 
        y_train, 
        input_dim, 
        hidden_dim, 
        output_dim, 
        initial_population=80,
        steps=75, 
        epochs=1,
        num_survivors=50, 
        enable_irxn=True, 
        lineage_prune_rate=1200,
        lineage_kept=600,
        q_learning_method='neural',
        training_method='loss',
        enable_model_save=False, 
        enable_bias_elimination=True,
        enable_lineage_snap=True,
        load_path=None, 
        experiment_name="arcnet", 
        track_best_models=False,
        top_k_per_generation=5,
        debug=False
        ):
    
    """
    Comprehensive AAN with Q-learning and bias elimination. This is the main function that runs the entire AAN evolution process. Disable and enable features as needed.
    Args:
        - X_train: Training data features (torch.Tensor)
        - y_train: Training data labels (torch.Tensor)
        - input_dim: Input dimension for the neural network
        - hidden_dim: Hidden layer dimension for the neural network
        - output_dim: Output dimension for the neural network
        - initial_population: Number of initial ConceptModules
        - steps: Number of evolutionary steps to run
        - epochs: Number of training epochs per step
        - num_survivors: Number of survivors to keep after each step
        - enable_irxn: Whether to enable irreverisble reactions (autocatalysis) -> Modules are consumed to create new ones
        - lineage_prune_rate: Rate at which to prune the lineage registry
        - lineage_kept: Number of modules to keep in the lineage registry
        - q_learning_method: Method for Q-learning ('neural' or 'table')
        - training_method: Method for training modules ('loss' or 'fitness')
        - enable_model_save: Whether to save models after each step (for debugging/multiple runs)
        - enable_bias_elimination: Whether to enable bias elimination (adaptive bias elimination based on prediction diversity and steps)
        - enable_lineage_snap: Whether to enable lineage snapshot saving for visualization
        - load_path: Path to load initial population from (if resuming)
        - experiment_name: Name of the experiment for saving/loading models
        - track_best_models: Whether to track and save the best models per generation for quick analysis
        - top_k_per_generation: Number of top models to track per generation
        - debug: Whether to enable debug mode for additional logging and checks
    Returns:
        - population: Final population of ConceptModules after evolution
        - lineage_registry: Final lineage registry mapping module IDs to their instances
        - filename_base: Base filename used for saving lineage snapshots
        - best_models_history: History of best models per generation (if tracking enabled)
        - generation_stats: Statistics for each generation (if tracking enabled)
        - assembly_registry: The complete assembly tracking registry
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{experiment_name}_phase_{timestamp}"
    population = []
    assembly_registry = AssemblyTrackingRegistry()
    
    if track_best_models:
        best_models_history = []
        generation_stats = []

    # Initialize population
    if enable_model_save:
        if load_path and os.path.exists(load_path):
            print(f"RESUMING EVOLUTION: Loading from {load_path}")
            population = load_and_seed_population_fixed(
                load_path=load_path,
                experiment_name=experiment_name,
                total_population=60,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
            print(f"Seeded population fitness: {[f'{m.fitness:.3f}' for m in population[:5]]}")
            
            # Register loaded modules with assembly registry
            for m in population:
                assembly_registry.register_module_initialization(m)
    else:
        print("STARTING FRESH EVOLUTION")
        for _ in range(initial_population):
            m = ConceptModule(input_dim, hidden_dim, output_dim, q_learning_method=q_learning_method)
            # Initialize with consistent state
            m.get_standardized_state(population)  # Initialize state representation
            population.append(m)
            # Register each new module with assembly registry
            assembly_registry.register_module_initialization(m)

    lineage_registry = {m.id: m for m in population}
    
    # Initialize reward tracking 
    for m in population:
        m.reward_history = []
    
    # Ensure data is tensorized and on CPU
    try:
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)
        X_train = X_train.cpu()
        y_train = y_train.cpu()
    except Exception as e:
        print(f"Error converting data to CPU tensors: {e}")
        print(f"Using original data types.")
        pass
        

    for step in tqdm(range(steps), desc="Evolution Progress", unit="step"):
        # ================ 1. LOSS/FITNESS EVALUATION ================            
        assembly_registry.step_forward()  # Update assembly step
        
        for m in tqdm(population, desc=f"Step {step} Fitness", leave=False, disable=len(population)<20):
            
            # grab current fitness if it exists, otherwise initialize to 0.0 for delta
            pre_fitness = m.fitness if hasattr(m, 'fitness') else 0.0

            if training_method == 'fitness':
                if epochs > 1:
                    for _ in range(epochs):
                        m.train(X_train, y_train)
                m.cpu()
                m.eval()
                m.fitness = compute_fitness_adaptive_complexity_enhanced(m, X_train, y_train)

            elif training_method == 'loss':
                m.train()  # set to training mode
                optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
                for _ in range(epochs):  # e.g., local_steps=3
                    optimizer.zero_grad()
                    loss = compute_loss(m, X_train, y_train, population=population)
                    if not isinstance(loss, torch.Tensor): # Ensure loss is a tensor with gradients
                        print(f"Warning: compute_loss returned {type(loss)}, converting to tensor")
                        loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
                    elif not loss.requires_grad:
                        print(f"Warning: loss tensor doesn't require gradients")
                        loss.requires_grad_(True)
                    loss.backward()
                    optimizer.step()
                m.cpu()
                m.eval()
                
                # Compute fitness from the final loss
                with torch.no_grad():
                    final_loss = compute_loss(m, X_train, y_train, population=population)
                    if isinstance(final_loss, torch.Tensor):
                        final_loss = final_loss.item()
                    
                    # Convert loss to fitness (lower loss = higher fitness)
                    # Option 1: Simple inverse relationship
                    m.fitness = 1.0 / (1.0 + final_loss)
                    
            # Track significant fitness changes in assembly registry
            fitness_delta = m.fitness - pre_fitness
            if abs(fitness_delta) > 0.01:
                fitness_event = {
                    'module_id': m.id,
                    'event_type': 'fitness_update',
                    'step': step,
                    'pre_fitness': pre_fitness,
                    'post_fitness': m.fitness,
                    'fitness_delta': fitness_delta
                }
                assembly_registry.global_assembly_events.append({
                    'type': 'fitness_update',
                    'step': step,
                    'data': fitness_event
                })

        # ================ 2. Q-LEARNING UPDATES WITH ASSEMBLY TRACKING ================
        # Update Q-functions immediately after fitness evaluation
        for m in tqdm(population, desc=f"Step {step} Q-Learning", leave=False, disable=len(population)<20):

            if (m.q_learning_method == 'neural' and
                m.q_function is not None and
                m.last_state is not None and
                m.last_action is not None):
                
                # Track Q-learning state before update
                q_experiences_before = len(m.q_function.replay_buffer) if hasattr(m.q_function, 'replay_buffer') else 0
                
                # Compute enhanced reward for Q-learning
                base_reward = compute_reward_adaptive_aan_normalized(m, X_train, y_train, 
                                            list(lineage_registry.values()), 
                                            step=step, max_steps=steps)
                
                total_reward = base_reward
                next_state = [m.fitness, compute_manifold_novelty(m, list(lineage_registry.values()))]
                
                q_values_before = None
                if hasattr(m.q_function, 'get_q_values'):
                    # Get the Q-values and copy them as tensors/lists without deep copying the entire structure
                    original_q_values = m.q_function.get_q_values(m.last_state)
                    if isinstance(original_q_values, torch.Tensor):
                        q_values_before = original_q_values.clone().detach()
                    elif isinstance(original_q_values, (list, tuple)):
                        q_values_before = [float(val) for val in original_q_values]
                    else:
                        q_values_before = original_q_values

                # Update Q-function with fresh experience
                m.update_q(total_reward, next_state)

                # Store Q-values after update
                q_values_after = None
                if hasattr(m.q_function, 'get_q_values'):
                    original_q_values = m.q_function.get_q_values(m.last_state)
                    if isinstance(original_q_values, torch.Tensor):
                        q_values_after = original_q_values.clone().detach()
                    elif isinstance(original_q_values, (list, tuple)):
                        q_values_after = [float(val) for val in original_q_values]
                    else:
                        q_values_after = original_q_values

                # Track Q-value changes
                q_value_changes = []
                if q_values_before is not None and q_values_after is not None:
                    for action in range(len(q_values_before)):
                        if q_values_before[action] != q_values_after[action]:
                            q_value_changes.append({
                                'state': m.last_state,
                                'action': action,
                                'old_q': q_values_before[action],
                                'new_q': q_values_after[action]
                            })

                # Track Q-learning update in assembly registry
                q_experiences_after = len(m.q_function.replay_buffer) if hasattr(m.q_function, 'replay_buffer') else 0
                if q_experiences_after > q_experiences_before or q_value_changes:
                    q_learning_event = {
                        'module_id': m.id,
                        'step': step,
                        'experiences_before': q_experiences_before,
                        'experiences_after': q_experiences_after,
                        'q_value_changes': q_value_changes,
                        'reward': total_reward,
                        'state': m.last_state,
                        'action': m.last_action
                    }
                    assembly_registry.global_assembly_events.append({
                        'type': 'q_learning_update',
                        'step': step,
                        'data': q_learning_event
                    })

                # Track reward history
                if not hasattr(m, 'reward_history'):
                    m.reward_history = []
                m.reward_history.append(total_reward)
                if len(m.reward_history) > 10:
                    m.reward_history.pop(0)
            
        # ================ 3. MESSAGE PASSING WITH ASSEMBLY TRACKING ================
        # Now do message passing with fresh Q-learning updates
        comprehensive_manifold_q_message_passing(population, step)

        # Update generation statistics in assembly registry (lightweight)
        assembly_registry.update_generation_statistics(step, population)

        # ================ 4. BIAS DETECTION & ELIMINATION ================
        bias_report = monitor_prediction_diversity_with_action(population, X_train, y_train, step, max_steps=steps)
        
        if enable_bias_elimination:
            if bias_report['requires_intervention']:
                # Update bias_report after elimination to tag non-biased modules
                population, eliminated = catalyst_bias_elimination(population, bias_report)
                bias_report = monitor_prediction_diversity_with_action(population, X_train, y_train, step, max_steps=steps)
                
                # Track bias elimination in assembly registry
                if eliminated:
                    bias_elimination_event = {
                        'step': step,
                        'eliminated_modules': [m.id for m in eliminated],
                        'population_before': len(population) + len(eliminated),
                        'population_after': len(population)
                    }
                    assembly_registry.global_assembly_events.append({
                        'type': 'bias_elimination',
                        'step': step,
                        'data': bias_elimination_event
                    })
                
                # Update lineage registry
                for elim_mod in eliminated:
                    if elim_mod.id in lineage_registry:
                        lineage_registry[elim_mod.id] = elim_mod

        # ================ TRACKING AND MONITORING ================
        if track_best_models:
            # Sort population by fitness and get top K
            sorted_population = sorted(population, key=lambda m: m.fitness, reverse=True)
            top_models_this_gen = sorted_population[:top_k_per_generation]
            
            # Create lightweight copies of the best models for analysis
            generation_best = []
            for i, model in enumerate(top_models_this_gen):
                model_snapshot = {
                    'generation': step,
                    'rank': i + 1,
                    'fitness': float(model.fitness),
                    'reward': float(getattr(model, 'reward', 0.0)),
                    'assembly_index': int(getattr(model, 'assembly_index', 0)),
                    'id': model.id,
                    'parent_id': getattr(model, 'parent_id', None),
                    'position': model.position.data.detach().cpu().numpy().tolist() if hasattr(model, 'position') else None,
                    'created_at': getattr(model, 'created_at', step),
                    'q_experiences': len(model.q_function.replay_buffer) if model.q_function else 0,
                    'reward_history': list(getattr(model, 'reward_history', [])),
                    
                    # Store model state for potential later analysis
                    'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    
                    # Performance metrics (if you want to compute them here)
                    'timestamp': datetime.now().isoformat()
                }
                generation_best.append(model_snapshot)
            
            best_models_history.extend(generation_best)
            
            # Track generation-level statistics
            gen_stats = {
                'generation': step,
                'population_size': len(population),
                'avg_fitness': sum(m.fitness for m in population) / len(population),
                'best_fitness': max(m.fitness for m in population),
                'worst_fitness': min(m.fitness for m in population),
                'fitness_std': np.std([m.fitness for m in population]),
                'avg_assembly_index': np.mean([getattr(m, 'assembly_index', 0) for m in population]),
                'total_q_experiences': sum(len(m.q_function.replay_buffer) if m.q_function else 0 for m in population),
                'lineage_size': len(lineage_registry)
            }
            generation_stats.append(gen_stats)

        # Q-Memory tracking
        total_q_memory = sum(m.get_q_memory_usage() for m in population)
        total_q_experiences = sum(len(m.q_function.replay_buffer) if m.q_function else 0 for m in population)
        
        # Output step for tracking progress
        avg_fitness = sum(m.fitness for m in population) / len(population)
        best_fitness = max(m.fitness for m in population)
        assembly_stats = assembly_registry.get_assembly_complexity_history()
        current_reuse_rate = assembly_stats[-1]['reuse_rate'] if assembly_stats else 0.0
        
        if debug:
            print(f"Step {step}: Avg={avg_fitness:.3f}, Best={best_fitness:.3f}, Q-Mem={total_q_memory:.1f}MB, Q-Exp={total_q_experiences}, Reuse={current_reuse_rate:.2f}")

        # ================ 5. Q-LEARNING GUIDED SURVIVAL SELECTION ================
        # Use Q-learning influenced fitness for survival selection
        for m in tqdm(population, desc="Survival Selection", leave=False, disable=len(population)<20):
            # Base fitness
            survival_score = m.fitness
            
            # Q-learning bonus for modules with rich experience
            if (m.q_learning_method == 'neural' and 
                m.q_function is not None):
                q_experience_bonus = len(m.q_function.replay_buffer) * 0.01  # Small bonus
                survival_score += q_experience_bonus
            
            # Store enhanced score for sorting
            m._survival_score = survival_score
        
        # Sort by enhanced survival score
        survivors = sorted(population, key=lambda m: -m._survival_score)[:num_survivors]
        
        # ================ 6. Q-LEARNING GUIDED REPRODUCTION WITH ASSEMBLY TRACKING ================
        offspring = []
        consumed_modules = set()
        
        # PRESERVE Q-KNOWLEDGE FROM CONSUMED CATALYSTS
        global_q_knowledge_pool = []
        
        for parent in survivors[:num_survivors//2]:  # Limit reproduction
            # Q-learning guided target selection
            available_targets = [m for m in lineage_registry.values() 
                               if m.id != parent.id and m.id not in consumed_modules]
            if not available_targets:
                continue
            
            # Q-learning action selection for choosing reproduction target
            with torch.no_grad():
                state = [parent.fitness, compute_manifold_novelty(parent, available_targets)]
                target = parent.choose_action(state, available_targets)
            
            # ================ CATALYST SELECTION FOR REPRODUCTION ================
            if random.random() < 0.7 and len(population) > 1:
                available_cats = [m for m in population 
                        if m.id != parent.id and 
                        m.id not in consumed_modules and
                        m.id in [mod['id'] for mod in bias_report['diverse_modules']]]
                if available_cats:
                    catalysts = select_bias_resistant_catalysts(parent, available_cats, bias_report)
                else:
                    catalysts = [parent]
            else:
                catalysts = [parent]
            
            # PRESERVE Q-KNOWLEDGE BEFORE CONSUMPTION
            if enable_irxn:
                for cat in catalysts:
                    if (cat.id != parent.id and 
                        cat.id in [mod['id'] for mod in bias_report['diverse_modules']] and
                        cat.q_learning_method == 'neural' and 
                        cat.q_function is not None):
                        
                        # Add to global knowledge pool
                        global_q_knowledge_pool.extend(cat.q_function.replay_buffer)
                        consumed_modules.add(cat.id)
                        
                        if debug:
                            print(f"  Preserving {len(cat.q_function.replay_buffer)} Q-experiences from catalyst {cat.id}")

            # ================ 7. CATALYST-BASED MUTATION WITH ASSEMBLY TRACKING ================
            # Collect parent modules for assembly tracking
            # all_parent_modules = [parent] + [cat for cat in catalysts if cat.id != parent.id]
            
            # Create child with assembly tracking
            child = parent.mutate(step, catalysts)
            child.parent_id = parent.id
            
            # Register child with assembly registry, providing parent modules for reuse tracking
            assembly_registry.track_mutation_with_assembly_inheritance(parent, child, catalysts)
            
            # Manifold-aware position update
            if hasattr(parent, 'position') and hasattr(target, 'position'):
                new_pos = parent.geodesic_interpolate(target.position.data, alpha=0.5)
                new_pos += 0.05 * torch.randn(parent.manifold_dim)
                child.position.data = new_pos.clamp(0, 1)
            
            child.cpu()
            child.eval()
            
            offspring.append(child)
            lineage_registry[child.id] = child

        # ================ INHERIT Q-EXPERIENCES FROM GLOBAL POOL ================
        if global_q_knowledge_pool and offspring:
            # Sort knowledge by quality
            global_q_knowledge_pool.sort(key=lambda x: x[2], reverse=True)
            experiences_per_child = min(20, len(global_q_knowledge_pool) // len(offspring))
            
            # All children inherit from global pool
            for i, child in tqdm(enumerate(offspring), desc="Q-Experience Inheritance", leave=False, disable=len(offspring)<20):
                if child.q_function is not None and experiences_per_child > 0:
                    start_idx = i * experiences_per_child
                    end_idx = start_idx + experiences_per_child
                    inherited_exp = global_q_knowledge_pool[start_idx:end_idx]
                    
                    child.q_function.replay_buffer.extend(inherited_exp)
                    # Maintain buffer size
                    if len(child.q_function.replay_buffer) > child.q_function.buffer_size:
                        child.q_function.replay_buffer = child.q_function.replay_buffer[-child.q_function.buffer_size:]
                    
                    # Track Q-inheritance in assembly registry
                    q_inheritance_event = {
                        'child_id': child.id,
                        'step': step,
                        'inherited_experiences': len(inherited_exp),
                        'total_buffer_size': len(child.q_function.replay_buffer)
                    }
                    assembly_registry.global_assembly_events.append({
                        'type': 'q_inheritance',
                        'step': step,
                        'data': q_inheritance_event
                    })
                    
                    if debug:
                        print(f"  Child {child.id} inherited {len(inherited_exp)} Q-experiences from global pool")

        # ================ 8. POPULATION UPDATE ================
        if enable_irxn:
            survivors = [s for s in survivors if s.id not in consumed_modules]
        
        population = survivors + offspring

        # ================ SNAPSHOT SAVING ================
        if enable_lineage_snap:
            saved_file = save_lineage_snapshot_to_file(
                list(lineage_registry.values()), 
                step, 
                filename_base,
                format='pickle'
            )

            if debug:
                print(f"[Step {step}] Saved snapshot to {saved_file}")
        
        # ================ 9. LINEAGE PRUNING ================
        if len(lineage_registry) > lineage_prune_rate:
            sorted_modules = sorted(lineage_registry.values(), key=lambda m: -m.fitness)
            keep_modules = sorted_modules[:lineage_kept] + population
            
            unique_modules = []
            seen = set()
            for m in keep_modules:
                if m.id not in seen:
                    seen.add(m.id)
                    unique_modules.append(m)
            
            lineage_registry = {m.id: m for m in unique_modules}

        # Memory management [every 10 steps clean up memory]
        if step % 10 == 0:
            import gc; gc.collect()

    # Final assembly statistics
    final_assembly_stats = assembly_registry.get_assembly_statistics()
    complexity_history = assembly_registry.get_assembly_complexity_history()
    
    print(f"\n=== FINAL ASSEMBLY STATISTICS ===")
    print(f"Total components tracked: {final_assembly_stats['total_components']}")
    print(f"Reused components: {final_assembly_stats['reused_components']}")
    print(f"Average reuse rate: {final_assembly_stats['average_reuse']:.3f}")
    print(f"Component types: {dict(final_assembly_stats['component_types'])}")
    print(f"Total assembly events: {len(assembly_registry.global_assembly_events)}")
    
    # Report theorem compliance
    fitness_values = [m.fitness for m in population]
    avg_fitness = sum(fitness_values) / len(fitness_values)
    assembly_complexity = compute_system_assembly_complexity(population)
    
    print(f"\n=== THEOREM COMPLIANCE ===")
    print(f"Fitness: {avg_fitness:.3f} ∈ [0,1] check.")
    print(f"Assembly complexity: {assembly_complexity:.3f}")
    print(f"Q-learning active: {sum(1 for m in population if m.q_function is not None)}/{len(population)}")
    print(f"Bias elimination: {len(bias_report['biased_modules'])} biased modules")
    
    print(f"Evolution complete. Total Q-experiences preserved: {sum(len(m.q_function.replay_buffer) if m.q_function else 0 for m in population)}")
    
    if enable_model_save:
        export_files, metadata = export_best_models(
            population=population,
            top_k=8,  # Export top 8 models
            save_path=f"exported_models/{experiment_name}",
            experiment_name=experiment_name
        )
        print(f"Exported best models to {metadata['export_files']}")
        return population, lineage_registry, filename_base, export_files, metadata, assembly_registry
    
    return population, lineage_registry, filename_base, best_models_history, generation_stats, assembly_registry


# ====================================================================
# ================ ENHANCED TRAINER STEP FUNCTION ==================
# ====================================================================

def enhanced_trainer_step_with_assembly_tracking(population, step, assembly_registry, X_train, y_train):
    """
    Enhanced trainer step that integrates assembly tracking into the evolutionary process
    """
    
    print(f"\n=== Step {step} with Assembly Tracking ===")
    
    # Update assembly registry step
    assembly_registry.step_forward()
    
    # Get assembly statistics for this step
    assembly_stats = assembly_registry.get_assembly_statistics()
    complexity_history = assembly_registry.get_assembly_complexity_history()
    
    print(f"Assembly components: {assembly_stats['total_components']}")
    print(f"Reused components: {assembly_stats['reused_components']}")
    if complexity_history:
        print(f"Current reuse rate: {complexity_history[-1]['reuse_rate']:.3f}")
    
    # Continue with normal training step processing...
    # (This function can be expanded based on your specific needs)
    
    return population, assembly_registry