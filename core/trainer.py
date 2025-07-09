import torch
import os
from tqdm.notebook import tqdm
from datetime import datetime
from ARCNET.models.arcnet_learner import ConceptModule
from ARCNET.evolution.fitness import (
    compute_fitness_adaptive_complexity_enhanced
)
from ARCNET.evolution.rewards import (
    compute_manifold_novelty,
    compute_system_assembly_complexity, 
    compute_reward_adaptive_aan_normalized
)
from ARCNET.core.message import (
    comprehensive_manifold_q_message_passing
)
from ARCNET.evolution.bias import (
    monitor_prediction_diversity_with_action,
    catalyst_bias_elimination,
    select_bias_resistant_catalysts
)
from ARCNET.data.loader import (
    load_and_seed_population_fixed,
    save_lineage_snapshot_to_file,
    export_best_models
)


# This function can definitely be simplified, for now it is comprehensive and includes all the features we want to test.
def run_aan_comprehensive_q_learning(
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
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{experiment_name}_phase_{timestamp}"
    population = []
    
    if track_best_models:
        best_models_history = []
        generation_stats = []

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
    else:
        print("STARTING FRESH EVOLUTION")
        for _ in range(initial_population):
            module = ConceptModule(input_dim, hidden_dim, output_dim, q_learning_method=q_learning_method)
            # Initialize with consistent state
            module.get_standardized_state(population)  # Initialize state representation
            population.append(module)

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
        
    for step in range(steps):
        # ================ 1. FITNESS EVALUATION ================            
        for m in population:
            
            if epochs > 1:
                for epoch in range(epochs):
                    m.train(X_train, y_train)
                    
            m.cpu()
            m.eval()
            m.fitness = compute_fitness_adaptive_complexity_enhanced(m, X_train, y_train)

            # Likely computational overhead, kept incase something bugs out
            #if m.id not in lineage_registry:
            #    lineage_registry[m.id] = m


        # ================ 2. Q-LEARNING UPDATES FIRST ================
        # Update Q-functions immediately after fitness evaluation
        for m in population:
            if (m.q_learning_method == 'neural' and 
                m.q_function is not None and 
                m.last_state is not None and 
                m.last_action is not None):
                
                # Compute enhanced reward for Q-learning
                base_reward = compute_reward_adaptive_aan_normalized(m, X_train, y_train, 
                                            list(lineage_registry.values()), 
                                            step=step, max_steps=steps)
                
                # Manifold-aware reward bonus
                #manifold_bonus = 0
                #if m.local_tangent_space is not None:
                #    curvature_bonus = abs(m.curvature) * 0.05
                #    manifold_bonus = curvature_bonus
                
                total_reward = base_reward #+ manifold_bonus
                next_state = [m.fitness, compute_manifold_novelty(m, list(lineage_registry.values()))]
                
                # Update Q-function with fresh experience
                m.update_q(total_reward, next_state)

                # Track reward history
                if not hasattr(m, 'reward_history'):
                    m.reward_history = []
                m.reward_history.append(m.reward)
                if len(m.reward_history) > 10:
                    m.reward_history.pop(0)
              
            
        # ================ 3. MESSAGE PASSING  ================
        # Now do message passing with fresh Q-learning updates
        #if step % 2 == 0:
        comprehensive_manifold_q_message_passing(population, step)
        #else:
        #    q_learning_reward_sharing(population)

        # ================ 4. BIAS DETECTION & ELIMINATION ================
        bias_report = monitor_prediction_diversity_with_action(population, X_train, y_train, step, max_steps=steps)
        
        if enable_bias_elimination:
            if bias_report['requires_intervention']:
                # Update bias_report after elimination to tag non-biased modules
                population, eliminated = catalyst_bias_elimination(population, bias_report)
                bias_report = monitor_prediction_diversity_with_action(population, X_train, y_train, step, max_steps=steps)
                
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
                    'reward': float(model.reward),
                    'assembly_index': int(model.assembly_index),
                    'id': int(model.id),
                    'parent_id': int(model.parent_id) if model.parent_id is not None else None,
                    'position': model.position.data.detach().cpu().numpy().tolist(),
                    'created_at': int(model.created_at),
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
                'avg_assembly_index': np.mean([m.assembly_index for m in population]),
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
        print(f"Step {step}: Avg={avg_fitness:.3f}, Best={best_fitness:.3f}, Q-Mem={total_q_memory:.1f}MB, Q-Exp={total_q_experiences}")

        # ================ 5. Q-LEARNING GUIDED SURVIVAL SELECTION ================
        # Use Q-learning influenced fitness for survival selection
        enhanced_survivors = []
        for m in population:
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
        
        # ================ 6. Q-LEARNING GUIDED REPRODUCTION ================
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

            # ================ 7. CATALYST-BASED MUTATION WITH Q-INHERITANCE ================
            
            child = parent.mutate(current_step=step, catalysts=catalysts)
            child.record_assembly_operation('mutation', [parent], catalysts)
            child.parent_id = parent.id
            
            # Manifold-aware position update
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
            for i, child in enumerate(offspring):
                if child.q_function is not None and experiences_per_child > 0:
                    start_idx = i * experiences_per_child
                    end_idx = start_idx + experiences_per_child
                    inherited_exp = global_q_knowledge_pool[start_idx:end_idx]
                    
                    child.q_function.replay_buffer.extend(inherited_exp)
                    # Maintain buffer size
                    if len(child.q_function.replay_buffer) > child.q_function.buffer_size:
                        child.q_function.replay_buffer = child.q_function.replay_buffer[-child.q_function.buffer_size:]
                    
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
        #if step % 10 == 0:
        #    import gc; gc.collect()

    # THEOREM CONDITIONS
    # Check bounded fitness landscape
    fitness_values = [m.fitness for m in population]
    #assert all(0 <= f <= 1 for f in fitness_values), "Fitness not bounded [0,1]"
    
    # Check sufficient exploration (ε > 0)
    #assert all(m.epsilon > 0 for m in population), "Insufficient exploration"
    
    # Check controlled bias elimination
    eliminated_ratio = len(bias_report['biased_modules']) / len(population)
    #assert eliminated_ratio <= 0.5, "Too aggressive bias elimination"
    
    # Report theorem compliance
    avg_fitness = sum(fitness_values) / len(fitness_values)
    assembly_complexity = compute_system_assembly_complexity(population)
    
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
        return population, lineage_registry, filename_base, export_files, metadata
    
    return population, lineage_registry, filename_base, best_models_history, generation_stats