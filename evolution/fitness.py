import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import random
import numpy as np


# ===========================================================
# ========= TESTING NEW FITNESS FUNCTIONS ===========
# ===========================================================

# general fitness computation function
def compute_fitness(model, X, y):
    """F1-based fitness (works well for imbalanced data)"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        pred = y_pred.argmax(dim=1)
        
        # Use macro F1 score (averages F1 across classes)
        f1_macro = f1_score(y.cpu(), pred.cpu(), average='macro', zero_division=0)
        
        # Add small diversity bonus
        unique_classes = torch.unique(y)
        pred_distribution = [(pred == i).sum().float() / len(pred) for i in unique_classes]
        entropy_bonus = -sum(p * torch.log(p + 1e-8) for p in pred_distribution if p > 0) * 0.1
        
        return f1_macro + entropy_bonus

def compute_fitness_balanced(model, X, y):
    """ENHANCED fitness that preserves AAN multi-objective principles"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        pred = y_pred.argmax(dim=1)
        
        # 1. BALANCED ACCURACY (core performance metric)
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
        
        # 2. DIVERSITY REWARDS (enhanced for AAN)
        pred_distribution = [(pred == i).sum().float() / len(pred) for i in unique_classes]
        
        # Strong entropy bonus (AAN principle: reward exploration)
        entropy_bonus = -0.4 * sum(p * torch.log(p + 1e-8) for p in pred_distribution if p > 0)
        
        # Anti-bias penalty (prevents convergence to trivial solutions)
        max_class_ratio = max(pred_distribution)
        bias_penalty = 0.6 * max(0, max_class_ratio - 0.75)  # Penalty if >75% one class
        
        # Minority class exploration bonus (AAN principle: novelty seeking)
        min_class_ratio = min(pred_distribution)
        exploration_bonus = 0.3 * min_class_ratio
        
        # 3. COMBINED FITNESS (preserves AAN multi-objective nature)
        total_fitness = balanced_accuracy + entropy_bonus + exploration_bonus - bias_penalty
        
        return max(0.1, total_fitness)

def compute_mutual_information_fitness(model, X, y):
    """Reward models that discover high mutual information patterns"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        pred = y_pred.argmax(dim=1)
        
        # 1. INPUT-OUTPUT MUTUAL INFORMATION
        # Reward models that find strong input-output relationships
        input_clusters = []
        pred_patterns = []
        
        # Simple clustering of inputs based on model's internal representations
        hidden_reps = []
        for i in range(0, len(X), 10):  # Sample every 10th for efficiency
            batch = X[i:i+1]
            # Get internal representation (you'd need to modify forward() to return hidden)
            with torch.no_grad():
                hidden = model.net[0](batch)
                hidden_reps.append(hidden.mean().item())
                pred_patterns.append(pred[i].item())
        
        # Calculate pattern consistency
        pattern_consistency = 0.0
        for i in range(len(hidden_reps)-1):
            for j in range(i+1, min(i+20, len(hidden_reps))):
                hidden_sim = 1.0 / (1.0 + abs(hidden_reps[i] - hidden_reps[j]))
                pred_sim = 1.0 if pred_patterns[i] == pred_patterns[j] else 0.0
                pattern_consistency += hidden_sim * pred_sim
        
        pattern_consistency /= max(1, len(hidden_reps) * 10)
        
        # 2. PREDICTION DIVERSITY (avoid trivial solutions)
        pred_entropy = 0.0
        unique_preds, counts = torch.unique(pred, return_counts=True)
        if len(unique_preds) > 1:
            probs = counts.float() / len(pred)
            pred_entropy = -sum(p * torch.log(p + 1e-8) for p in probs)
        
        # 3. DECISION BOUNDARY SHARPNESS
        # Reward models with confident, sharp decision boundaries
        probs = F.softmax(y_pred, dim=1)
        confidence = probs.max(dim=1)[0].mean()
        
        return pattern_consistency + 0.3 * pred_entropy + 0.2 * confidence

def compute_fitness_pure_emergent(model, X, y):
    """Ultra-pure AAN fitness with NO prior knowledge"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        pred = y_pred.argmax(dim=1)
        
        # 1. PREDICTION CONSISTENCY (internal coherence)
        # Reward models that make consistent predictions for similar inputs
        consistency_score = 0.0
        for i in range(min(100, len(X))):  # Sample to avoid O(n²)
            for j in range(i+1, min(i+20, len(X))):
                input_similarity = F.cosine_similarity(X[i], X[j], dim=0)
                pred_agreement = (pred[i] == pred[j]).float()
                consistency_score += input_similarity * pred_agreement
        
        consistency_score /= 100  # Normalize
        
        # 2. PREDICTION ENTROPY (AAN principle: avoid trivial solutions)
        pred_distribution = torch.bincount(pred).float() / len(pred)
        entropy_bonus = -sum(p * torch.log(p + 1e-8) for p in pred_distribution if p > 0)
        
        # 3. OUTPUT CONFIDENCE (reward decisive predictions)
        confidence = F.softmax(y_pred, dim=1).max(dim=1)[0].mean()
        
        return consistency_score + 0.4 * entropy_bonus + 0.3 * confidence

def compute_mutual_information_fitness(model, X, y):
    """Reward models that discover high mutual information patterns"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        pred = y_pred.argmax(dim=1)
        
        # 1. INPUT-OUTPUT MUTUAL INFORMATION
        # Reward models that find strong input-output relationships
        input_clusters = []
        pred_patterns = []
        
        # Simple clustering of inputs based on model's internal representations
        hidden_reps = []
        for i in range(0, len(X), 10):  # Sample every 10th for efficiency
            batch = X[i:i+1]
            # Get internal representation (you'd need to modify forward() to return hidden)
            with torch.no_grad():
                hidden = model.net[0](batch)
                hidden_reps.append(hidden.mean().item())
                pred_patterns.append(pred[i].item())
        
        # Calculate pattern consistency
        pattern_consistency = 0.0
        for i in range(len(hidden_reps)-1):
            for j in range(i+1, min(i+20, len(hidden_reps))):
                hidden_sim = 1.0 / (1.0 + abs(hidden_reps[i] - hidden_reps[j]))
                pred_sim = 1.0 if pred_patterns[i] == pred_patterns[j] else 0.0
                pattern_consistency += hidden_sim * pred_sim
        
        pattern_consistency /= max(1, len(hidden_reps) * 10)
        
        # 2. PREDICTION DIVERSITY (avoid trivial solutions)
        pred_entropy = 0.0
        unique_preds, counts = torch.unique(pred, return_counts=True)
        if len(unique_preds) > 1:
            probs = counts.float() / len(pred)
            pred_entropy = -sum(p * torch.log(p + 1e-8) for p in probs)
        
        # 3. DECISION BOUNDARY SHARPNESS
        # Reward models with confident, sharp decision boundaries
        probs = F.softmax(y_pred, dim=1)
        confidence = probs.max(dim=1)[0].mean()
        
        return pattern_consistency + 0.3 * pred_entropy + 0.2 * confidence
    
def compute_contrastive_fitness(model, X, y):
    """Discover patterns by maximizing contrast between different regions"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        pred = y_pred.argmax(dim=1)
        
        # 1. SPATIAL CONTRAST in input space
        contrast_score = 0.0
        n_samples = min(100, len(X))
        
        for i in range(n_samples):
            # Find neighbors and distant points
            current_input = X[i]
            current_pred = pred[i]
            
            # Calculate distances to other inputs
            distances = [F.cosine_similarity(current_input, X[j], dim=0).item() 
                        for j in range(len(X)) if j != i]
            
            # Sort by distance
            sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
            
            # Check if similar inputs get similar predictions (local consistency)
            close_indices = sorted_indices[-5:]  # Most similar
            far_indices = sorted_indices[:5]     # Most different
            
            # Local consistency bonus
            close_consistency = sum(1 for idx in close_indices 
                                  if pred[idx] == current_pred) / len(close_indices)
            
            # Global contrast bonus  
            far_contrast = sum(1 for idx in far_indices 
                             if pred[idx] != current_pred) / len(far_indices)
            
            contrast_score += close_consistency + far_contrast
        
        contrast_score /= n_samples
        
        # 2. INFORMATION BOTTLENECK
        # Reward models that compress inputs while preserving discriminative info
        hidden_outputs = []
        for i in range(0, len(X), 10):
            hidden = model.net[0](X[i:i+1])
            hidden_outputs.append(hidden.mean().item())
        
        # Compression: low variance in hidden representations is good
        compression = 1.0 / (1.0 + np.var(hidden_outputs))
        
        # But must preserve discriminative power
        discrimination = len(torch.unique(pred)) / model.output_dim
        
        return contrast_score + 0.2 * compression + 0.3 * discrimination
    
def compute_self_organizing_fitness(model, X, y):
    """Reward emergent clustering and organization patterns"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        pred = y_pred.argmax(dim=1)
        
        # 1. EMERGENT CLUSTERING QUALITY
        # Measure how well the model naturally clusters the data
        cluster_quality = 0.0
        
        unique_preds = torch.unique(pred)
        for class_id in unique_preds:
            class_mask = (pred == class_id)
            class_inputs = X[class_mask]
            
            if len(class_inputs) > 1:
                # Intra-cluster similarity (should be high)
                intra_similarity = 0.0
                n_pairs = 0
                for i in range(len(class_inputs)):
                    for j in range(i+1, min(i+10, len(class_inputs))):
                        sim = F.cosine_similarity(class_inputs[i], class_inputs[j], dim=0)
                        intra_similarity += sim.item()
                        n_pairs += 1
                
                if n_pairs > 0:
                    intra_similarity /= n_pairs
                    cluster_quality += intra_similarity
        
        cluster_quality /= max(1, len(unique_preds))
        
        # 2. NATURAL BOUNDARIES
        # Reward models that find natural decision boundaries
        boundary_sharpness = 0.0
        n_boundary_samples = 50
        
        for _ in range(n_boundary_samples):
            # Sample two random points
            i, j = random.sample(range(len(X)), 2)
            
            # Interpolate between them
            alpha = random.uniform(0.3, 0.7)
            interpolated = alpha * X[i] + (1 - alpha) * X[j]
            
            # Check prediction on interpolated point
            interp_pred = model(interpolated.unsqueeze(0)).argmax(dim=1).item()
            
            # If prediction changes, we found a boundary
            if pred[i].item() != pred[j].item():
                # Sharp boundary is good - prediction should be decisive
                interp_logits = model(interpolated.unsqueeze(0))
                interp_confidence = F.softmax(interp_logits, dim=1).max().item()
                boundary_sharpness += interp_confidence
        
        boundary_sharpness /= n_boundary_samples
        
        # 3. HIERARCHICAL ORGANIZATION
        # Reward models that show hierarchical pattern organization
        hierarchy_score = 0.0
        if len(unique_preds) > 1:
            # Check if the model creates meaningful sub-patterns
            for class_id in unique_preds:
                class_outputs = y_pred[pred == class_id]
                if len(class_outputs) > 5:
                    # Measure internal diversity within each predicted class
                    class_entropy = 0.0
                    for dim in range(model.output_dim):
                        dim_values = class_outputs[:, dim]
                        dim_var = torch.var(dim_values).item()
                        class_entropy += dim_var
                    
                    hierarchy_score += class_entropy / model.output_dim
        
        return cluster_quality + 0.4 * boundary_sharpness + 0.2 * hierarchy_score

def compute_temporal_pattern_fitness(model, X, y):
    """Discover patterns in how the model's behavior evolves"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        pred = y_pred.argmax(dim=1)
        
        # 1. CONSISTENCY OVER SIMILAR INPUTS
        consistency_score = 0.0
        
        # Group inputs by similarity
        similarity_groups = {}
        for i in range(len(X)):
            # Simple hash of input features for grouping
            input_hash = hash(tuple(X[i].round(decimals=2).tolist()))
            if input_hash not in similarity_groups:
                similarity_groups[input_hash] = []
            similarity_groups[input_hash].append((i, pred[i].item()))
        
        # Measure consistency within groups
        for group in similarity_groups.values():
            if len(group) > 1:
                predictions = [item[1] for item in group]
                most_common = max(set(predictions), key=predictions.count)
                consistency = predictions.count(most_common) / len(predictions)
                consistency_score += consistency
        
        consistency_score /= max(1, len(similarity_groups))
        
        # 2. PROGRESSIVE REFINEMENT
        # Reward models that show evidence of progressive pattern refinement
        refinement_score = 0.0
        
        # Sample random subsets and check if larger subsets give more confident predictions
        subset_sizes = [len(X)//4, len(X)//2, 3*len(X)//4, len(X)]
        confidences = []
        
        for size in subset_sizes:
            indices = random.sample(range(len(X)), min(size, len(X)))
            subset_pred = y_pred[indices]
            subset_confidence = F.softmax(subset_pred, dim=1).max(dim=1)[0].mean().item()
            confidences.append(subset_confidence)
        
        # Check if confidence generally increases with more data
        for i in range(len(confidences)-1):
            if confidences[i+1] >= confidences[i]:
                refinement_score += 1.0
        
        refinement_score /= max(1, len(confidences)-1)
        
        return consistency_score + 0.3 * refinement_score

def compute_fitness_natural_discovery(model, X, y):
    """Combined natural pattern discovery without explicit class knowledge"""
    
    # Combine multiple discovery mechanisms
    mi_score = compute_mutual_information_fitness(model, X, y)
    contrast_score = compute_contrastive_fitness(model, X, y)
    organization_score = compute_self_organizing_fitness(model, X, y)
    temporal_score = compute_temporal_pattern_fitness(model, X, y)
    
    # Weight different discovery mechanisms
    total_score = (0.3 * mi_score + 
                   0.25 * contrast_score + 
                   0.3 * organization_score + 
                   0.15 * temporal_score)
    
    return total_score

# Replace compute_fitness_balanced with this STRONGER version:
def compute_fitness_balanced_strong(model, X, y):
    """STRONGER balanced fitness with severe anti-convergence penalties"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
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
        
        # STRONG diversity enforcement
        pred_distribution = [(pred == i).sum().float() / len(pred) for i in unique_classes]
        
        # SEVERE penalty for convergence
        max_class_ratio = max(pred_distribution)
        if max_class_ratio > 0.85:  # If >85% one class
            convergence_penalty = 3.0 * (max_class_ratio - 0.85)
        else:
            convergence_penalty = 0.0
        
        # STRONG minority class bonus
        min_class_ratio = min(pred_distribution)
        diversity_bonus = 2.0 * min_class_ratio
        
        # Entropy bonus
        entropy_bonus = -0.6 * sum(p * torch.log(p + 1e-8) for p in pred_distribution if p > 0)
        
        total_fitness = balanced_accuracy + diversity_bonus + entropy_bonus - convergence_penalty
        
        return max(0.1, total_fitness)

# Add this NEW adaptive fitness function:

    """Adaptive fitness that scales complexity tolerance based on evolution stage"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        pred = y_pred.argmax(dim=1)
        
        # 1. CORE PERFORMANCE (balanced accuracy)
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
        
        # 2. ADAPTIVE DIVERSITY ENFORCEMENT (key innovation)
        pred_distribution = [(pred == i).sum().float() / len(pred) for i in unique_classes]
        max_class_ratio = max(pred_distribution)
        min_class_ratio = min(pred_distribution)
        
        # PROGRESSIVE SPECIALIZATION ALLOWANCE
        # Early: Force exploration, Late: Allow specialization if performance is good
        progress = step / max_steps
        
        if progress < 0.3:
            # EXPLORATION PHASE: Strong diversity enforcement
            diversity_threshold = 0.75  # Max 75% one class
            convergence_penalty = 3.0 * max(0, max_class_ratio - diversity_threshold)
            diversity_bonus = 2.0 * min_class_ratio
            
        elif progress < 0.7:
            # TRANSITION PHASE: Moderate enforcement with performance consideration
            if balanced_accuracy > 0.6:  # If learning something useful
                diversity_threshold = 0.85  # Allow more specialization
                convergence_penalty = 1.5 * max(0, max_class_ratio - diversity_threshold)
            else:
                diversity_threshold = 0.8
                convergence_penalty = 2.0 * max(0, max_class_ratio - diversity_threshold)
            
            diversity_bonus = 1.5 * min_class_ratio
            
        else:
            # REFINEMENT PHASE: Allow specialization if performance justifies it
            if balanced_accuracy > 0.7:  # High performance allows specialization
                diversity_threshold = 0.95
                convergence_penalty = 0.5 * max(0, max_class_ratio - diversity_threshold)
            elif balanced_accuracy > 0.6:  # Moderate performance
                diversity_threshold = 0.9
                convergence_penalty = 1.0 * max(0, max_class_ratio - diversity_threshold)
            else:  # Poor performance still needs diversity
                diversity_threshold = 0.8
                convergence_penalty = 2.0 * max(0, max_class_ratio - diversity_threshold)
            
            diversity_bonus = 1.0 * min_class_ratio
        
        # 3. PATTERN DISCOVERY BONUS (for complex datasets)
        # Reward models that find non-trivial patterns
        pattern_complexity = 0.0
        if balanced_accuracy > 0.55:  # Better than random
            # Calculate prediction confidence variance (sign of learning)
            probs = F.softmax(y_pred, dim=1)
            confidence_variance = torch.var(probs.max(dim=1)[0]).item()
            pattern_complexity = confidence_variance * 0.5
        
        # 4. ENTROPY BONUS (scaled by progress)
        entropy_bonus = -0.4 * sum(p * torch.log(p + 1e-8) for p in pred_distribution if p > 0)
        entropy_weight = 1.0 - (0.5 * progress)  # Reduce entropy importance over time
        
        # 5. COMBINED FITNESS
        total_fitness = (balanced_accuracy + 
                        diversity_bonus + 
                        entropy_weight * entropy_bonus + 
                        pattern_complexity - 
                        convergence_penalty)
        
        return max(0.1, total_fitness)

# Add this function for better complex pattern detection:
def compute_complex_pattern_bonus(model, X, y, sample_size=100):
    """Reward discovery of complex patterns in high-dimensional data"""
    model.eval()
    with torch.no_grad():
        # Sample for efficiency
        indices = torch.randperm(len(X))[:sample_size]
        X_sample = X[indices]
        y_sample = y[indices]
        
        y_pred = model(X_sample)
        pred = y_pred.argmax(dim=1)
        
        # 1. LOCAL CONSISTENCY DISCOVERY
        # Reward models that find consistent local patterns
        consistency_score = 0.0
        for i in range(min(20, len(X_sample))):
            # Find k nearest neighbors in input space
            current_x = X_sample[i]
            distances = [F.cosine_similarity(current_x, X_sample[j], dim=0).item() 
                        for j in range(len(X_sample)) if j != i]
            
            # Get indices of 3 nearest neighbors
            nearest_indices = sorted(range(len(distances)), key=lambda k: distances[k], reverse=True)[:3]
            
            # Check prediction consistency among neighbors
            current_pred = pred[i]
            neighbor_preds = [pred[idx] for idx in nearest_indices]
            consistency = sum(1 for p in neighbor_preds if p == current_pred) / len(neighbor_preds)
            consistency_score += consistency
        
        consistency_score /= min(20, len(X_sample))
        
        # 2. DISCRIMINATIVE POWER
        # Reward models that can distinguish between classes
        if len(torch.unique(pred)) > 1:
            # Calculate inter-class vs intra-class distances in prediction space
            probs = F.softmax(y_pred, dim=1)
            
            intra_class_variance = 0.0
            inter_class_distance = 0.0
            unique_preds = torch.unique(pred)
            
            for class_id in unique_preds:
                class_mask = (pred == class_id)
                class_probs = probs[class_mask]
                
                if len(class_probs) > 1:
                    # Intra-class variance (should be low)
                    class_var = torch.var(class_probs, dim=0).mean().item()
                    intra_class_variance += class_var
            
            # Inter-class distance (should be high)
            if len(unique_preds) > 1:
                class_centers = []
                for class_id in unique_preds:
                    class_mask = (pred == class_id)
                    if class_mask.sum() > 0:
                        class_center = probs[class_mask].mean(dim=0)
                        class_centers.append(class_center)
                
                if len(class_centers) > 1:
                    for i in range(len(class_centers)):
                        for j in range(i+1, len(class_centers)):
                            dist = F.cosine_similarity(class_centers[i], class_centers[j], dim=0).item()
                            inter_class_distance += (1.0 - dist)  # Higher distance is better
            
            discriminative_power = inter_class_distance - intra_class_variance
        else:
            discriminative_power = 0.0
        
        return max(0, consistency_score + 0.3 * discriminative_power)

# Then modify compute_fitness_adaptive_complexity to include this:
def compute_fitness_adaptive_complexity_enhanced(model, X, y, step=0, max_steps=50):
    """Enhanced version with PERFORMANCE-ADAPTIVE transitions (not just time-based)"""
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        pred = y_pred.argmax(dim=1)
        
        # 1. CORE PERFORMANCE
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
        
        # 2. DUAL-CRITERIA PHASE DETECTION (time AND performance)
        pred_distribution = [(pred == i).sum().float() / len(pred) for i in unique_classes]
        max_class_ratio = max(pred_distribution)
        min_class_ratio = min(pred_distribution)
        
        progress = step / max_steps
        
        # PERFORMANCE-BASED PHASE OVERRIDE
        if balanced_accuracy > 0.65:
            # HIGH PERFORMANCE: Allow specialization regardless of time
            phase = "specialization"
        elif balanced_accuracy > 0.55 and progress > 0.4:
            # MODERATE PERFORMANCE + LATE TIME: Transition to learning
            phase = "learning" 
        elif progress < 0.25:
            # EARLY TIME: Always explore
            phase = "exploration"
        elif balanced_accuracy < 0.48 and progress > 0.6:
            # POOR PERFORMANCE + LATE TIME: Force more exploration
            phase = "forced_exploration"
        else:
            # DEFAULT: Balanced transition
            phase = "transition"
        
        # PHASE-SPECIFIC PENALTIES AND BONUSES
        if phase == "exploration":
            diversity_threshold = 0.75
            convergence_penalty = 3.0 * max(0, max_class_ratio - diversity_threshold)
            diversity_bonus = 2.0 * min_class_ratio
            
        elif phase == "forced_exploration":
            # Poor performance late in evolution - force more exploration
            diversity_threshold = 0.7  # Even stricter
            convergence_penalty = 4.0 * max(0, max_class_ratio - diversity_threshold)
            diversity_bonus = 2.5 * min_class_ratio
            
        elif phase == "transition":
            diversity_threshold = 0.82
            convergence_penalty = 1.5 * max(0, max_class_ratio - diversity_threshold)
            diversity_bonus = 1.5 * min_class_ratio
            
        elif phase == "learning":
            diversity_threshold = 0.88
            convergence_penalty = 0.8 * max(0, max_class_ratio - diversity_threshold)
            diversity_bonus = 1.0 * min_class_ratio
            
        else:  # specialization
            diversity_threshold = 0.95
            convergence_penalty = 0.3 * max(0, max_class_ratio - diversity_threshold)
            diversity_bonus = 0.5 * min_class_ratio
        
        # 3. ENTROPY AND PATTERN DETECTION (same as before)
        entropy_raw = -sum(p * torch.log(p + 1e-8) for p in pred_distribution if p > 0)
        max_entropy = torch.log(torch.tensor(len(unique_classes)))
        entropy_score = (entropy_raw / max_entropy).item() if max_entropy > 0 else 0
        entropy_weight = 1.0 - (0.5 * progress)
        
        pattern_score = 0.0
        if balanced_accuracy > 0.52:  # LOWERED threshold
            probs = F.softmax(y_pred, dim=1)
            confidence_variance = torch.var(probs.max(dim=1)[0]).item()
            pattern_score = min(1.0, confidence_variance * 2.0)
        
        # 4. WEIGHTED COMBINATION
        components = [
            0.50 * balanced_accuracy,
            0.20 * diversity_bonus,
            0.15 * entropy_weight * entropy_score,
            0.15 * pattern_score
        ]
        
        total_fitness = sum(components) - (0.3 * convergence_penalty)
        normalized_fitness = max(0.0, min(1.0, total_fitness))
        
        return normalized_fitness
