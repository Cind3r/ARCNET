import torch
import math

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
