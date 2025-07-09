import torch
import math
from collections import defaultdict
# ===========================================================
# ================ Message Passing ==========================
# ===========================================================

def select_neighbors(module, population, k=3):
    distances = [(other, torch.norm(module.position.data - other.position.data)) 
                 for other in population if other.id != module.id]
    neighbors = sorted(distances, key=lambda x: x[1])[:k]
    return [n[0] for n in neighbors]

def comprehensive_manifold_q_message_passing_old(population, step):
    for module in population:
        neighbors = select_neighbors(module, population, k=6)
        module.update_local_geometry(neighbors)

    for module in population:
        manifold_neighbors = []
        for other in population:
            if other.id != module.id:
                base_dist = module.manifold_distance(other.position.data)
                q_experience_bonus = reward_bonus = assembly_bonus = 0
                if (other.q_learning_method == 'neural' and other.q_function is not None):
                    q_experience_bonus = len(other.q_function.replay_buffer) * 0.005
                reward_bonus = other.fitness * 0.1
                assembly_diff = abs(module.assembly_index - other.assembly_index)
                assembly_bonus = min(assembly_diff * 0.02, 0.1)
                adjusted_dist = base_dist - q_experience_bonus - reward_bonus - assembly_bonus
                manifold_neighbors.append((other, adjusted_dist))

        manifold_neighbors.sort(key=lambda x: x[1])
        close_neighbors = [n[0] for n in manifold_neighbors[:6]]
        medium_neighbors = [n[0] for n in manifold_neighbors[6:12]]
        far_neighbors = [n[0] for n in manifold_neighbors[12:18]]

        all_neighbor_tiers = [
            (close_neighbors, 1.0),
            (medium_neighbors, 0.6),
            (far_neighbors, 0.3)
        ]

        for neighbors, tier_weight in all_neighbor_tiers:
            for neighbor in neighbors:
                manifold_dist = module.manifold_distance(neighbor.position.data)
                fitness_weight = neighbor.fitness
                distance_weight = 1.0 / (1.0 + manifold_dist)
                assembly_similarity = 1.0 / (1.0 + abs(module.assembly_index - neighbor.assembly_index))
                total_weight = tier_weight * fitness_weight * distance_weight * assembly_similarity
                message = neighbor.forward_summary()
                module.receive_message(message * total_weight)

    for module in population:
        module.process_messages()

def comprehensive_manifold_q_message_passing(population, step):
    """Message passing with O(log n) propagation guarantee"""
    # Phase 1: Update local geometry for all modules
    for module in population:
        neighbors = select_neighbors(module, population, k=min(6, len(population)-1))
        module.update_local_geometry(neighbors)
    
    # Phase 2: Multi-hop message propagation (guarantees O(log n) global reach)
    max_hops = max(1, int(math.log2(len(population))))
    
    for hop in range(max_hops):
        # Each hop, messages propagate to neighbors
        hop_messages = {}  # module_id -> list of messages for this hop
        
        for module in population:
            # Send message to neighbors
            neighbors = select_neighbors(module, population, k=6)
            message = module.forward_summary()
            
            for neighbor in neighbors:
                if neighbor.id not in hop_messages:
                    hop_messages[neighbor.id] = []
                hop_messages[neighbor.id].append(message)
        
        # Deliver messages for this hop
        for module in population:
            if module.id in hop_messages:
                for msg in hop_messages[module.id]:
                    module.receive_message(msg)
    
    # Phase 3: Process all received messages
    for module in population:
        module.process_messages()  


def q_learning_reward_sharing(population):
    """Share successful Q-learning strategies across the population"""
    
    # Identify top performers
    top_performers = sorted(population, key=lambda m: m.fitness, reverse=True)[:len(population)//5]
    
    for top_performer in top_performers:
        if (top_performer.q_learning_method == 'neural' and 
            top_performer.q_function is not None and 
            top_performer.q_function.replay_buffer):
            
            # Find neighbors to share knowledge with
            neighbors = select_neighbors(top_performer, population, k=8)
            
            # Share best Q-experiences
            best_experiences = sorted(top_performer.q_function.replay_buffer, 
                                    key=lambda x: x[2], reverse=True)[:3]
            
            for neighbor in neighbors:
                if (neighbor.q_learning_method == 'neural' and 
                    neighbor.q_function is not None):
                    
                    # Transfer knowledge with fitness weighting
                    for state, action_id, target_q in best_experiences:
                        boosted_q = target_q * (top_performer.fitness / max(neighbor.fitness, 0.1))
                        neighbor.q_function.replay_buffer.append((state, action_id, boosted_q))
                        
                        # Maintain buffer size
                        if len(neighbor.q_function.replay_buffer) > neighbor.q_function.buffer_size:
                            neighbor.q_function.replay_buffer.pop(0)


# ===========================================================
# ============== Assembly Aware Messaging (NEW) ============
# ===========================================================

def enhanced_message_passing_with_assembly_tracking(population, step, assembly_registry):

    """
    Enhanced message passing that tracks all parameter changes in the assembly system
    """
    
    print(f"Step {step}: Enhanced message passing with assembly tracking for {len(population)} modules")
    
    # Update assembly registry step
    assembly_registry.step_forward()
    
    # Phase 1: Pre-message snapshots and neighbor selection
    module_neighbors = {}
    message_events = {}
    
    for module in population:
        # Register module if not already tracked
        if module.id not in assembly_registry.parameter_update_history:
            assembly_registry.register_module_initialization(module)
        
        # Select manifold neighbors
        neighbors = select_enhanced_neighbors(module, population, k=6)
        module_neighbors[module.id] = neighbors
        
        # Update local geometry if method exists
        if hasattr(module, 'update_local_geometry'):
            module.update_local_geometry(neighbors)
    
    # Phase 2: Multi-hop message propagation with tracking
    max_hops = max(1, int(torch.log2(torch.tensor(len(population))).item()))
    
    for hop in range(max_hops):
        print(f"  Message hop {hop + 1}/{max_hops}")
        hop_messages = {}
        hop_influences = defaultdict(list)
        
        # Generate and send messages
        for module in population:
            neighbors = module_neighbors[module.id]
            
            # Create enhanced message with assembly context
            base_message = create_assembly_aware_message(module)
            
            # Send to neighbors and track influences
            for neighbor in neighbors:
                if neighbor.id not in hop_messages:
                    hop_messages[neighbor.id] = []
                    hop_influences[neighbor.id] = []
                
                hop_messages[neighbor.id].append(base_message)
                hop_influences[neighbor.id].append(module)
        
        # Deliver messages and track parameter changes
        for module in population:
            if module.id in hop_messages:
                messages = hop_messages[module.id]
                senders = hop_influences[module.id]
                
                # Start tracking message influence
                message_event = assembly_registry.track_message_passing_update(
                    module, senders, messages
                )
                
                # Process messages (this will modify module parameters)
                if hasattr(module, 'receive_message'):
                    for msg in messages:
                        module.receive_message(msg)
                
                # Complete tracking after message processing
                changes = assembly_registry.complete_message_passing_update(message_event, module)
                
                if changes['weights_changed'] or changes['position_changed']:
                    print(f"    Module {module.id}: weights_changed={changes['weights_changed']}, "
                          f"position_changed={changes['position_changed']}, "
                          f"layers_affected={changes['changed_layers']}")
    
    # Phase 3: Process accumulated messages
    for module in population:
        if hasattr(module, 'process_messages'):
            # Track before processing
            pre_snapshot = assembly_registry.create_parameter_snapshot(module, "pre_message_processing")
            
            # Process messages
            module.process_messages()
            
            # Track after processing
            post_snapshot = assembly_registry.create_parameter_snapshot(module, "post_message_processing")
            changes = assembly_registry.detect_parameter_changes(pre_snapshot, post_snapshot)
            
            if changes['weights_changed']:
                # Record message processing changes
                processing_event = {
                    'module_id': module.id,
                    'event_type': 'message_processing',
                    'step': step,
                    'changes': changes,
                    'pre_snapshot': pre_snapshot,
                    'post_snapshot': post_snapshot
                }
                assembly_registry.global_assembly_events.append({
                    'type': 'message_processing',
                    'step': step,
                    'data': processing_event
                })
    
    # Phase 4: Q-learning reward sharing with assembly tracking
    top_performers = sorted(population, key=lambda m: m.fitness, reverse=True)[:len(population)//5]
    
    for top_performer in top_performers:
        if (hasattr(top_performer, 'q_learning_method') and 
            top_performer.q_learning_method == 'neural' and 
            hasattr(top_performer, 'q_function') and
            top_performer.q_function is not None and 
            hasattr(top_performer.q_function, 'replay_buffer') and
            top_performer.q_function.replay_buffer):
            
            neighbors = module_neighbors[top_performer.id]
            best_experiences = sorted(top_performer.q_function.replay_buffer, 
                                    key=lambda x: x[2], reverse=True)[:3]
            
            for neighbor in neighbors:
                if (hasattr(neighbor, 'q_learning_method') and
                    neighbor.q_learning_method == 'neural' and 
                    hasattr(neighbor, 'q_function') and
                    neighbor.q_function is not None):
                    
                    # Track Q-learning knowledge transfer
                    q_update_info = {
                        'type': 'knowledge_transfer',
                        'parent_modules': [top_performer.id],
                        'experiences_added': len(best_experiences),
                        'source_fitness': top_performer.fitness,
                        'assembly_similarity': calculate_assembly_similarity(top_performer, neighbor)
                    }
                    
                    q_event = assembly_registry.track_q_learning_update(neighbor, q_update_info)
                    
                    # Perform Q-function update
                    assembly_similarity = q_update_info['assembly_similarity']
                    
                    for state, action_id, target_q in best_experiences:
                        # Weight transfer by fitness and assembly similarity
                        fitness_ratio = top_performer.fitness / max(neighbor.fitness, 0.1)
                        boosted_q = target_q * fitness_ratio * assembly_similarity
                        
                        neighbor.q_function.replay_buffer.append((state, action_id, boosted_q))
                        
                        # Maintain buffer size
                        if len(neighbor.q_function.replay_buffer) > neighbor.q_function.buffer_size:
                            neighbor.q_function.replay_buffer.pop(0)
                    
                    # Complete Q-learning tracking
                    changes = assembly_registry.complete_q_learning_update(q_event, neighbor)
                    
                    if changes['q_function_changed']:
                        print(f"    Q-transfer: {top_performer.id} -> {neighbor.id}, "
                              f"experiences={len(best_experiences)}, similarity={assembly_similarity:.3f}")
    
    # Phase 5: Report assembly statistics
    if step % 10 == 0:  # Report every 10 steps
        stats = assembly_registry.get_assembly_statistics()
        print(f"Assembly Tracking Stats at Step {step}:")
        print(f"  Components tracked: {stats['total_components']}")
        print(f"  Modules tracked: {stats['total_modules_tracked']}")
        print(f"  Message influences: {stats['message_influences']}")
        print(f"  Q-learning influences: {stats['q_learning_influences']}")
        print(f"  Max assembly depth: {stats['max_assembly_depth']}")
        print(f"  Total assembly events: {stats['total_assembly_events']}")

def select_enhanced_neighbors(module, population, k=6):
    """Enhanced neighbor selection considering assembly similarity"""
    if not hasattr(module, 'position'):
        # Fallback to random selection
        import random
        others = [m for m in population if m.id != module.id]
        return random.sample(others, min(k, len(others)))
    
    candidates = []
    for other in population:
        if other.id != module.id and hasattr(other, 'position'):
            # Base manifold distance
            manifold_dist = torch.norm(module.position.data - other.position.data).item()
            
            # Assembly similarity bonus
            assembly_similarity = calculate_assembly_similarity(module, other)
            
            # Fitness similarity bonus
            fitness_similarity = 1.0 - abs(module.fitness - other.fitness)
            
            # Combined distance (lower is better)
            combined_distance = manifold_dist - (assembly_similarity * 0.3) - (fitness_similarity * 0.2)
            
            candidates.append((other, combined_distance))
    
    # Sort by combined distance and return top k
    candidates.sort(key=lambda x: x[1])
    return [candidate[0] for candidate in candidates[:k]]

def calculate_assembly_similarity(module1, module2):
    """Calculate assembly similarity between two modules"""
    if not (hasattr(module1, 'assembly_index') and hasattr(module2, 'assembly_index')):
        return 0.5  # Default similarity
    
    # Assembly index similarity
    assembly_diff = abs(module1.assembly_index - module2.assembly_index)
    assembly_similarity = 1.0 / (1.0 + assembly_diff * 0.1)
    
    # Layer complexity similarity if available
    complexity_similarity = 0.5
    if (hasattr(module1, 'layer_components') and hasattr(module2, 'layer_components')):
        try:
            comp1_total = sum(comp.get_minimal_assembly_complexity() 
                            for comp in module1.layer_components.values()
                            if hasattr(comp, 'get_minimal_assembly_complexity'))
            comp2_total = sum(comp.get_minimal_assembly_complexity() 
                            for comp in module2.layer_components.values()
                            if hasattr(comp, 'get_minimal_assembly_complexity'))
            
            complexity_diff = abs(comp1_total - comp2_total)
            complexity_similarity = 1.0 / (1.0 + complexity_diff * 0.05)
        except:
            pass
    
    return (assembly_similarity + complexity_similarity) / 2.0

def create_assembly_aware_message(module):
    """Create a message that includes assembly context"""

    # Base message from module
    if hasattr(module, 'forward_summary'):
        base_message = module.forward_summary()
        # If base_message is not a tensor, convert or extract tensor
        if not isinstance(base_message, torch.Tensor):
            # Try to convert to tensor if possible
            try:
                base_message = torch.tensor(base_message, dtype=torch.float32)
            except Exception:
                # If base_message is an object, extract a tensor attribute or fallback
                if hasattr(base_message, 'to_tensor'):
                    base_message = base_message.to_tensor()
                elif hasattr(base_message, 'data'):
                    base_message = torch.tensor(base_message.data, dtype=torch.float32)
                else:
                    # Fallback: use random tensor
                    base_message = torch.randn(10)
    else:
        base_message = torch.randn(10)  # Fallback
    
    # Assembly context
    assembly_context = torch.tensor([
        module.assembly_index if hasattr(module, 'assembly_index') else 0,
        module.fitness,
        len(module.layer_components) if hasattr(module, 'layer_components') else 0,
        sum(comp.get_minimal_assembly_complexity() 
            for comp in module.layer_components.values()
            if hasattr(comp, 'get_minimal_assembly_complexity')) if hasattr(module, 'layer_components') else 0
    ], dtype=torch.float32)
    
    # Q-learning context
    q_context = torch.tensor([
        len(module.q_function.replay_buffer) if (hasattr(module, 'q_function') and 
                                                module.q_function and 
                                                hasattr(module.q_function, 'replay_buffer')) else 0,
        module.get_q_memory_usage() if hasattr(module, 'get_q_memory_usage') else 0
    ], dtype=torch.float32)
    
    # Combine all contexts
    enhanced_message = torch.cat([
        base_message.flatten(), 
        assembly_context, 
        q_context
    ])
    
    return enhanced_message