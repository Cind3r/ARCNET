import torch

# ==============================================================
# ========== Bias Monitoring and Elimination Functions ============
# ==============================================================

# This module provides functions to monitor prediction diversity
# and eliminate biased modules in a population of neural networks.
# Ideally, the evolutionary process should intentionally seek diversity 
# in predictions to avoid bias convergence, but these functions
# can be used to enforce diversity and mitigate bias when it occurs. (in complex n-dimensional data spaces)


def monitor_prediction_diversity(population, X, y, step, debug=False):
    """Monitor if modules are converging to trivial solutions"""

    # quick method to supress printing without having to change all the function
    if debug == False:
        org_print = print  # Save original print function
        def suppress_printing(*args, **kwargs):
            pass
        print = suppress_printing  # Override print function

    if step % 3 == 0:
        print(f"\n=== Step {step} Diversity Check ===")
        for i, mod in enumerate(population[:3]):  # Check first 3
            with torch.no_grad():
                mod.eval()
                pred = mod(X).argmax(dim=1)
                class_0_pct = (pred == 0).float().mean().item() * 100
                class_1_pct = (pred == 1).float().mean().item() * 100
                print(f"Module {mod.id}: Class 0: {class_0_pct:.1f}%, Class 1: {class_1_pct:.1f}%")
        print("="*40)
        

def monitor_prediction_diversity_with_action(population, X, y, step, max_steps=75, bias_threshold=None, debug=False):  # LOWERED from 0.85
    """Enhanced diversity monitoring that returns bias information and can trigger elimination"""
    
    # quick method to supress printing without having to change all the function
    if debug == False:
        org_print = print  # Save original print function
        def suppress_printing(*args, **kwargs):
            pass
        print = suppress_printing  # Override print function
    
    bias_report = {
        'step': step,
        'biased_modules': [],
        'diverse_modules': [],
        'population_bias_risk': 0.0,
        'requires_intervention': False
    }
    
    # Check if bias threshold is provided, otherwise adaptively set it
    if bias_threshold is None:
        # threshold: Start lenient, become STRICT
        progress = min(1.0, step / max_steps)
        # 0.90 (lenient) -> 0.70 (strict) - DECREASING tolerance
        bias_threshold = 0.90 - 0.20 * progress  


    if step % 3 == 0:  # CHECK MORE FREQUENTLY
        print(f"\n=== Step {step} Diversity Check ===")
        
    total_bias_risk = 0.0
    
    for i, mod in enumerate(population):
        with torch.no_grad():
            mod.eval()
            pred = mod(X).argmax(dim=1)
            class_0_pct = (pred == 0).float().mean().item()
            class_1_pct = (pred == 1).float().mean().item()
            
            max_class_ratio = max(class_0_pct, class_1_pct)
            is_biased = max_class_ratio > bias_threshold
            
            module_info = {
                'id': mod.id,
                'fitness': mod.fitness,
                'class_0_pct': class_0_pct * 100,
                'class_1_pct': class_1_pct * 100,
                'bias_ratio': max_class_ratio,
                'is_biased': is_biased,
                'assembly_index': mod.assembly_index
            }
            
            if is_biased:
                bias_report['biased_modules'].append(module_info)
                total_bias_risk += max_class_ratio
            else:
                bias_report['diverse_modules'].append(module_info)
            
            if step % 3 == 0 and i < 8:  # SHOW MORE MODULES
                status = "BIASED" if is_biased else "DIVERSE"
                print(f"Module {mod.id}: {status} - Class 0: {class_0_pct*100:.1f}%, Class 1: {class_1_pct*100:.1f}%, Fitness: {mod.fitness:.3f}")
    
    bias_report['population_bias_risk'] = total_bias_risk / len(population)
    #print(f"bias_report type: {type(bias_report)}")
    #print(f"bias_report keys: {bias_report.keys() if isinstance(bias_report, dict) else 'Not a dict'}")

    # MORE AGGRESSIVE intervention
    biased_ratio = len(bias_report['biased_modules']) / len(population)
    bias_report['requires_intervention'] = (
        biased_ratio > 0.15 or  # LOWERED from 0.3 to 0.15
        bias_report['population_bias_risk'] > 0.6 or  # LOWERED from 0.75
        step > 10  # EARLIER intervention
    )
    
    if step % 3 == 0:
        print(f"Population Bias Risk: {bias_report['population_bias_risk']:.3f}")
        print(f"Biased Modules: {len(bias_report['biased_modules'])}/{len(population)}")
        if bias_report['requires_intervention']:
            print(" INTERVENTION REQUIRED: High bias detected")
        print("="*40)
    
    return bias_report

def catalyst_bias_elimination(population, bias_report, elimination_rate=0.25, debug=False):
    
    """Use catalyst system to eliminate biased modules"""
    
    # quick method to supress printing without having to change all the function
    if debug == False:
        org_print = print  # Save original print function
        def suppress_printing(*args, **kwargs):
            pass
        print = suppress_printing  # Override print function

    if not bias_report['biased_modules'] or not bias_report['requires_intervention']:
        return population, []
    
    print(f"\nCATALYST BIAS ELIMINATION ACTIVATED")
    print(f"Target: {len(bias_report['biased_modules'])} biased modules")
    
    # Select elimination catalysts (diverse, high-fitness modules)
    diverse_modules = []
    for mod in population:
        for diverse_info in bias_report['diverse_modules']:
            if mod.id == diverse_info['id']:
                diverse_modules.append(mod)
                break
    
    if not diverse_modules:
        print("No diverse modules available as catalysts")
        return population, []
    
    # Sort diverse modules by elimination potential
    catalyst_candidates = []
    for mod in diverse_modules:
        q_experience_bonus = len(mod.q_function.replay_buffer) if mod.q_function else 0
        elimination_score = mod.fitness + 0.01 * q_experience_bonus
        catalyst_candidates.append((mod, elimination_score))
    
    catalyst_candidates.sort(key=lambda x: x[1], reverse=True)
    catalysts = [mod for mod, score in catalyst_candidates[:3]]  # Top 3 catalysts
    
    # Target biased modules for elimination
    biased_module_ids = [mod['id'] for mod in bias_report['biased_modules']]
    elimination_targets = []
    
    for mod in population:
        if mod.id in biased_module_ids:
            elimination_targets.append(mod)
    
    # Sort targets by bias severity (worst first)
    elimination_targets.sort(
        key=lambda m: max([d['bias_ratio'] for d in bias_report['biased_modules'] 
                          if d['id'] == m.id]), 
        reverse=True
    )
    
    # Calculate elimination count
    max_eliminations = min(
        len(elimination_targets),
        int(len(population) * elimination_rate),
        len(elimination_targets) // 2
    )
    
    eliminated_modules = elimination_targets[:max_eliminations]
    eliminated_ids = [mod.id for mod in eliminated_modules]
    
    # Record catalysis relationships (Assembly Theory)
    for catalyst in catalysts:
        if not hasattr(catalyst, 'catalyzes'):
            catalyst.catalyzes = []
        catalyst.catalyzes.extend(eliminated_ids)
        catalyst.fitness += 0.1  # Reward successful elimination
        
        if not hasattr(catalyst, 'eliminations_performed'):
            catalyst.eliminations_performed = 0
        catalyst.eliminations_performed += len(eliminated_ids)
    
    for eliminated in eliminated_modules:
        eliminated.eliminated_by = [cat.id for cat in catalysts]
        eliminated.elimination_reason = "bias_convergence"
    
    # Remove biased modules from population
    surviving_population = [mod for mod in population if mod.id not in eliminated_ids]
    
    print(f"Eliminated {len(eliminated_modules)} biased modules using {len(catalysts)} catalysts")
    print(f"Catalysts: {[f'ID {cat.id} (fit:{cat.fitness:.3f})' for cat in catalysts]}")
    print(f"Population size: {len(population)} -> {len(surviving_population)}")
    
    return surviving_population, eliminated_modules

def select_bias_resistant_catalysts(parent, available_cats, bias_report, max_catalysts=2):
    """Select catalysts that are specifically bias-resistant"""
    
    verified_catalysts = [parent]
    
    # Prefer diverse modules from bias report
    diverse_ids = [mod['id'] for mod in bias_report['diverse_modules']]
    bias_resistant_cats = [cat for cat in available_cats if cat.id in diverse_ids]
    
    if not bias_resistant_cats:
        bias_resistant_cats = available_cats
    
    catalyst_candidates = []
    
    for cat in bias_resistant_cats:
        diversity_bonus = 0.3 if cat.id in diverse_ids else 0.0
        elimination_penalty = getattr(cat, 'eliminations_performed', 0) * 0.05
        q_richness = len(cat.q_function.replay_buffer) if cat.q_function else 0
        
        score = (cat.fitness + diversity_bonus + 0.05 * (q_richness / 100) - elimination_penalty)
        
        catalyst_candidates.append({
            'module': cat,
            'score': score,
            'is_bias_resistant': cat.id in diverse_ids
        })
    
    catalyst_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    for candidate in catalyst_candidates:
        if len(verified_catalysts) >= max_catalysts:
            break
        verified_catalysts.append(candidate['module'])
    
    return verified_catalysts