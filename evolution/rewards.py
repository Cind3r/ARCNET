import torch
import torch.nn.functional as F
import random
import numpy as np
from core.message import select_neighbors
from evolution.fitness import (
                    compute_fitness_adaptive_complexity_enhanced,
                    compute_fitness_natural_discovery
                )

# =========================================================
# ================ Reward Computation =====================
# =========================================================


def compute_entropy_reward(module, X):
    with torch.no_grad():
        logits = module(X)
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()
    return entropy

def compute_novelty(mod, population):
    dists = [torch.norm(mod.position.data - other.position.data) for other in population if other.id != mod.id]
    novelty_score = sum(dists) / len(dists) if dists else 0.0
    try:
        mod.set_novelty_score(novelty_score)
    except:
        pass
    return novelty_score

def compute_manifold_novelty(mod, population):
    if len(population) < 4:
        return compute_novelty(mod, population)

    neighbors = select_neighbors(mod, population, k=min(6, len(population)-1))
    mod.update_local_geometry(neighbors)

    manifold_dists = []
    for other in population:
        if other.id != mod.id:
            try:
                dist = mod.manifold_distance(other.position.data)
            except:
                dist = torch.norm(mod.position.data - other.position.data).item()
            manifold_dists.append(dist)

    novelty_score = sum(manifold_dists) / len(manifold_dists) if manifold_dists else 0.0
    curvature_novelty = abs(mod.curvature) * 0.1
    total_novelty = novelty_score + curvature_novelty

    try:
        mod.set_novelty_score(total_novelty)
    except:
        pass
    return total_novelty

def compute_reward_anti_convergence(mod, X, y, population, step=0, max_steps=50):
    """Fast reward that PREVENTS trivial solutions while maintaining AAN principles"""
    
    # 1. FAST FITNESS with strong anti-bias penalties
    with torch.no_grad():
        mod.cpu()
        mod.eval()
        y_pred = mod(X)
        pred = y_pred.argmax(dim=1)
        
        # Standard balanced accuracy
        unique_classes = torch.unique(y)
        class_recalls = []
        for class_idx in unique_classes:
            class_mask = (y.cpu() == class_idx)
            if class_mask.sum() > 0:
                class_pred = pred.cpu()[class_mask]
                class_true = y.cpu()[class_mask]
                class_recall = (class_pred == class_true).float().mean().item()
                class_recalls.append(class_recall)
        
        balanced_accuracy = sum(class_recalls) / len(class_recalls) if class_recalls else 0.0
        
        # STRONG ANTI-CONVERGENCE PENALTIES
        pred_distribution = [(pred == i).sum().float() / len(pred) for i in unique_classes]
        
        # Severe penalty for predicting >80% one class
        max_class_ratio = max(pred_distribution)
        convergence_penalty = 2.0 * max(0, max_class_ratio - 0.8)  # Strong penalty
        
        # Reward for using both classes
        min_class_ratio = min(pred_distribution)
        diversity_bonus = 1.5 * min_class_ratio  # Strong bonus
        
        # Prediction entropy bonus
        entropy_bonus = 0.5 * (-sum(p * torch.log(p + 1e-8) for p in pred_distribution if p > 0))
        
        # Combined fitness with strong diversity enforcement
        fitness = balanced_accuracy + diversity_bonus + entropy_bonus - convergence_penalty
        fitness = max(0.05, fitness)  # Minimum fitness
    
    # 2. FAST NOVELTY (sampled)
    novelty_sample_size = min(15, len(population))
    sample_pop = random.sample(population, novelty_sample_size) if len(population) > novelty_sample_size else population
    
    dists = [torch.norm(mod.position.data - other.position.data).item() 
             for other in sample_pop if other.id != mod.id]
    novelty = sum(dists) / len(dists) if dists else 0.0
    
    # 3. FAST ENTROPY REWARD (already computed above)
    entropy_reward = entropy_bonus
    
    # 4. SIMPLIFIED ASSEMBLY REWARD
    if mod.assembly_index <= 3:
        assembly_reward = 1.0 / (1.0 + mod.assembly_index * 0.3)
    else:
        assembly_reward = 0.3
    
    # 5. DYNAMIC WEIGHTING that emphasizes diversity early
    if step < max_steps * 0.4:
        # Early: Heavily reward diversity and novelty
        weights = [1.0, 0.8, 0.7, 0.2]  # [fitness, novelty, entropy, assembly]
    elif step < max_steps * 0.7:
        # Middle: Balanced
        weights = [1.2, 0.5, 0.4, 0.3]
    else:
        # Late: More emphasis on accuracy, but keep diversity
        weights = [1.5, 0.3, 0.3, 0.4]
    
    total_reward = (weights[0] * fitness + 
                   weights[1] * novelty + 
                   weights[2] * entropy_reward + 
                   weights[3] * assembly_reward)
    
    mod.set_reward(total_reward)
    return total_reward

def compute_system_assembly_complexity(population):
    total_pop = len(population)
    if total_pop == 0:
        return 0.0
    complexity = 0.0
    for module in population:
        module.compute_assembly_index()
        complexity += module.get_assembly_complexity_contribution(population)
    return complexity

def compute_assembly_reward(module, population, complexity_penalty=0.1, max_reasonable_assembly=20):
    if module.assembly_index == 0:
        efficiency_reward = 2.0
    elif module.assembly_index <= 5:
        efficiency_reward = 1.5 / (1.0 + module.assembly_index * 0.2)
    elif module.assembly_index <= max_reasonable_assembly:
        efficiency_reward = 1.0 / (1.0 + module.assembly_index * 0.5)
    else:
        excess = module.assembly_index - max_reasonable_assembly
        efficiency_reward = -0.1 * np.exp(excess * 0.1)

    functional_diversity_bonus = 0.0
    if len(population) > 10:
        similar_modules = sum(
            1 for other in population[:20]
            if other.id != module.id and abs(other.fitness - module.fitness) < 0.1
        )
        diversity_ratio = 1.0 - (similar_modules / min(20, len(population)))
        functional_diversity_bonus = diversity_ratio * 0.3

    system_complexity = compute_system_assembly_complexity(population)
    complexity_penalty_value = complexity_penalty * system_complexity
    autocatalytic_bonus = 0.1 * len(module.catalyzes) if hasattr(module, "catalyzes") else 0.0

    return efficiency_reward + functional_diversity_bonus - complexity_penalty_value + autocatalytic_bonus

def compute_reward_natural_discovery(mod, X, y, population, step=0, max_steps=50, 
                                   base_novelty=0.3, base_entropy=0.5, base_assembly=0.25):
    """AAN reward with natural pattern discovery"""
    
    # Use natural discovery fitness instead of supervised learning
    acc = compute_fitness_natural_discovery(mod, X, y)
    
    # Keep all your existing AAN components
    novelty = compute_manifold_novelty(mod, population)
    entropy = compute_entropy_reward(mod, X)
    assembly_reward = compute_assembly_reward(mod, population)
    
    # Your existing dynamic weighting
    if step < max_steps * 0.3:
        novelty_weight = base_novelty * 2.0
        entropy_weight = base_entropy * 1.8
        assembly_weight = base_assembly * 0.6
    elif step < max_steps * 0.7:
        novelty_weight = base_novelty * 1.5
        entropy_weight = base_entropy * 1.3
        assembly_weight = base_assembly
    else:
        novelty_weight = base_novelty 
        entropy_weight = base_entropy 
        assembly_weight = base_assembly * 1.5
    
    reward_value = (acc + 
                   novelty_weight * novelty + 
                   entropy_weight * entropy + 
                   assembly_weight * assembly_reward)
    
    mod.set_reward(reward_value)
    return reward_value

def compute_reward_adaptive_aan(mod, X, y, population, step=0, max_steps=50):
    """AAN reward that adapts to dataset complexity while preserving auto-generative nature"""
    
    # 1. ADAPTIVE FITNESS (core component)
    fitness = compute_fitness_adaptive_complexity_enhanced(mod, X, y, step, max_steps)
    
    # 2. MANIFOLD NOVELTY (preserve spatial exploration)
    novelty_sample_size = min(15, len(population))
    sample_pop = random.sample(population, novelty_sample_size) if len(population) > novelty_sample_size else population
    
    dists = [torch.norm(mod.position.data - other.position.data).item() 
             for other in sample_pop if other.id != mod.id]
    novelty = sum(dists) / len(dists) if dists else 0.0
    
    # 3. SYSTEM ENTROPY (population-level diversity)
    with torch.no_grad():
        mod.eval()
        y_pred = mod(X)
        probs = F.softmax(y_pred, dim=1)
        system_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()
    
    # 4. ASSEMBLY REWARD (simplified for complex datasets)
    if mod.assembly_index <= 3:
        assembly_reward = 1.0 / (1.0 + mod.assembly_index * 0.2)
    else:
        assembly_reward = 0.3
    
    # 5. ADAPTIVE WEIGHTING STRATEGY
    progress = step / max_steps
    
    if progress < 0.3:
        # Early: Heavy exploration
        weights = [1.0, 1.0, 0.8, 0.3]  # [fitness, novelty, entropy, assembly]
    elif progress < 0.7:
        # Middle: Balanced with slight performance bias
        weights = [1.3, 0.6, 0.5, 0.4]
    else:
        # Late: Performance-focused but maintain some exploration
        weights = [1.8, 0.4, 0.3, 0.5]
    
    total_reward = (weights[0] * fitness + 
                   weights[1] * novelty + 
                   weights[2] * system_entropy + 
                   weights[3] * assembly_reward)
    
    mod.set_reward(total_reward)
    return total_reward

def compute_reward_adaptive_aan_normalized(mod, X, y, population, step=0, max_steps=50):
    """Normalized reward that keeps fitness interpretable"""
    
    # 1. NORMALIZED FITNESS (0 to 1)
    fitness = compute_fitness_adaptive_complexity_enhanced(mod, X, y, step, max_steps)
    
    # 2. SCALED COMPONENTS (all normalized to reasonable ranges)
    novelty_sample_size = min(15, len(population))
    sample_pop = random.sample(population, novelty_sample_size) if len(population) > novelty_sample_size else population
    
    dists = [torch.norm(mod.position.data - other.position.data).item() 
             for other in sample_pop if other.id != mod.id]
    novelty_raw = sum(dists) / len(dists) if dists else 0.0
    novelty = min(1.0, novelty_raw / 2.0)  # Normalize assuming max distance ~2.0
    
    # 3. SYSTEM ENTROPY (0 to 1)
    with torch.no_grad():
        mod.eval()
        y_pred = mod(X)
        probs = F.softmax(y_pred, dim=1)
        entropy_raw = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()
        max_entropy = torch.log(torch.tensor(2.0)).item()  # Binary classification
        system_entropy = entropy_raw / max_entropy
    
    # 4. ASSEMBLY REWARD (0 to 1)
    if mod.assembly_index <= 3:
        assembly_reward = 1.0 / (1.0 + mod.assembly_index * 0.2)
    else:
        assembly_reward = 0.3
    assembly_reward = min(1.0, assembly_reward)
    
    # 5. WEIGHTED COMBINATION (interpretable weights)
    progress = step / max_steps
    
    if progress < 0.3:
        weights = [0.4, 0.3, 0.2, 0.1]  # [fitness, novelty, entropy, assembly] - sum to 1.0
    elif progress < 0.7:
        weights = [0.5, 0.25, 0.15, 0.1]
    else:
        weights = [0.65, 0.15, 0.1, 0.1]
    
    total_reward = (weights[0] * fitness + 
                   weights[1] * novelty + 
                   weights[2] * system_entropy + 
                   weights[3] * assembly_reward)
    
    # GUARANTEE total_reward ∈ [0,1]
    total_reward = max(0.0, min(1.0, total_reward))
    
    mod.set_reward(total_reward)
    return total_reward
