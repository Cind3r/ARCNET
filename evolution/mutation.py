import random
import torch
import torch.nn as nn
from core.components import TrackedLayer

# ===========================================
# == Mutation Functions for ARCNET Models ===
# ===========================================

def assembly_aware_mutation_with_tracking(parent, catalysts, current_step, assembly_registry):
    """
    Enhanced mutation that tracks all assembly influences from messages and Q-learning
    """
    
    # Gather message influences on parent
    parent_history = assembly_registry.get_module_assembly_history(parent.id)
    recent_message_influences = [
        event for event in parent_history 
        if event.get('event_type') == 'post_message' and 
           event.get('step', 0) > current_step - 10  # Recent influences
    ]
    
    recent_q_influences = [
        event for event in parent_history 
        if event.get('update_type') == 'q_learning' and 
           event.get('step', 0) > current_step - 10
    ]
    
    message_influence_summary = {
        'recent_message_count': len(recent_message_influences),
        'recent_q_updates': len(recent_q_influences),
        'message_senders': list(set([
            sender_id for event in recent_message_influences 
            for sender_id in event.get('sender_ids', [])
        ])),
        'q_inheritance_sources': list(set([
            source for event in recent_q_influences 
            for source in event.get('q_update_info', {}).get('parent_modules', [])
        ]))
    }
    
    print(f"  Mutation with assembly influences: parent={parent.id}, "
          f"msg_influences={message_influence_summary['recent_message_count']}, "
          f"q_influences={message_influence_summary['recent_q_updates']}")
    
    # Perform standard mutation
    if hasattr(parent, 'mutate'):
        child = parent.mutate(current_step=current_step, catalysts=catalysts)
    else:
        # Fallback mutation if method doesn't exist
        child = create_mutated_child(parent, catalysts, current_step)
    
    # Track the mutation with assembly influences
    mutation_event = assembly_registry.track_mutation_with_assembly_influence(
        parent, child, catalysts, message_influence_summary
    )
    
    # Apply assembly-aware enhancements to child based on recent influences
    apply_assembly_influences_to_child(child, recent_message_influences, recent_q_influences, assembly_registry)
    
    # Register child in assembly system
    assembly_registry.register_module_initialization(child)
    
    # Update component relationships based on influences
    update_child_components_with_influences(parent, child, message_influence_summary, assembly_registry)
    
    return child

def create_mutated_child(parent, catalysts, current_step):
    """Fallback mutation function if parent.mutate doesn't exist"""
    from models.arcnet import ConceptModule
    
    # Create new module with same architecture
    child = ConceptModule(
        parent.input_dim, parent.hidden_dim, parent.output_dim,
        created_at=current_step, q_learning_method=parent.q_learning_method
    )
    
    # Copy parent state and add noise
    child.load_state_dict(parent.state_dict())
    child.id = random.randint(0, int(1e6))
    child.parent_id = parent.id
    
    # Mutate parameters
    with torch.no_grad():
        for param in child.parameters():
            if param.requires_grad:
                param.add_(0.01 * torch.randn_like(param))
    
    # Inherit some Q-learning experience from catalysts
    if catalysts and hasattr(child, 'q_function') and child.q_function is not None:
        for catalyst in catalysts:
            if (hasattr(catalyst, 'q_function') and 
                catalyst.q_function is not None and 
                hasattr(catalyst.q_function, 'replay_buffer')):
                
                # Copy some experiences
                catalyst_experiences = catalyst.q_function.replay_buffer[-5:]  # Last 5
                for exp in catalyst_experiences:
                    child.q_function.replay_buffer.append(exp)
    
    return child

def apply_assembly_influences_to_child(child, message_influences, q_influences, assembly_registry):
    """Apply assembly-aware modifications to child based on parent's recent influences"""
    
    # Track pre-influence state
    pre_snapshot = assembly_registry.create_parameter_snapshot(child, "pre_influence_application")
    
    influence_applied = False
    
    # Apply message-based influences
    if message_influences:
        print(f"    Applying {len(message_influences)} message influences to child {child.id}")
        
        # Extract successful message patterns
        successful_influences = [
            event for event in message_influences 
            if event.get('changes_detected', {}).get('fitness_delta', 0) > 0
        ]
        
        if successful_influences:
            # Apply small parameter adjustments based on successful message patterns
            with torch.no_grad():
                for param in child.parameters():
                    if param.requires_grad:
                        # Small influence from successful message patterns
                        influence_factor = len(successful_influences) * 0.005
                        param.add_(influence_factor * torch.randn_like(param))
                        influence_applied = True
    
    # Apply Q-learning influences
    if q_influences:
        print(f"    Applying {len(q_influences)} Q-learning influences to child {child.id}")
        
        # Enhance Q-function with inherited knowledge
        if hasattr(child, 'q_function') and child.q_function is not None:
            total_inherited_experiences = sum(
                event.get('experiences_added', 0) for event in q_influences
            )
            
            if total_inherited_experiences > 0:
                # Boost exploration based on inherited Q-knowledge
                if hasattr(child.q_function, 'epsilon'):
                    child.q_function.epsilon *= 0.9  # Reduce exploration since we have inherited knowledge
                influence_applied = True
    
    # Track post-influence state
    if influence_applied:
        post_snapshot = assembly_registry.create_parameter_snapshot(child, "post_influence_application")
        changes = assembly_registry.detect_parameter_changes(pre_snapshot, post_snapshot)
        
        # Record influence application
        influence_event = {
            'child_id': child.id,
            'event_type': 'assembly_influence_application',
            'step': assembly_registry.step_counter,
            'message_influences_applied': len(message_influences),
            'q_influences_applied': len(q_influences),
            'changes': changes,
            'pre_snapshot': pre_snapshot,
            'post_snapshot': post_snapshot
        }
        
        assembly_registry.global_assembly_events.append({
            'type': 'influence_application',
            'step': assembly_registry.step_counter,
            'data': influence_event
        })

def update_child_components_with_influences(parent, child, message_influences, assembly_registry):
    """Update child component relationships to reflect assembly influences"""
    
    if not (hasattr(parent, 'layer_components') and hasattr(child, 'layer_components')):
        return
    
    # Update component registry with influence information
    for layer_name in child.layer_components.keys():
        child_comp_id = f"{child.id}_{layer_name}"
        
        if child_comp_id in assembly_registry.component_registry:
            comp_data = assembly_registry.component_registry[child_comp_id]
            
            # Add message influence information
            if message_influences['recent_message_count'] > 0:
                influence_record = {
                    'type': 'inherited_message_influence',
                    'step': assembly_registry.step_counter,
                    'source_senders': message_influences['message_senders'],
                    'influence_count': message_influences['recent_message_count']
                }
                comp_data['message_influences'].append(influence_record)
            
            # Add Q-learning influence information  
            if message_influences['recent_q_updates'] > 0:
                influence_record = {
                    'type': 'inherited_q_influence',
                    'step': assembly_registry.step_counter,
                    'source_modules': message_influences['q_inheritance_sources'],
                    'influence_count': message_influences['recent_q_updates']
                }
                comp_data['q_influences'].append(influence_record)
            
            # Create assembly dependencies to influence sources
            for sender_id in message_influences['message_senders']:
                for sender_layer in ['fc1', 'fc2', 'fc3']:
                    sender_comp_id = f"{sender_id}_{sender_layer}"
                    if sender_comp_id in assembly_registry.component_registry:
                        assembly_registry.assembly_dependency_graph[child_comp_id].add(sender_comp_id)
            
            for source_id in message_influences['q_inheritance_sources']:
                for source_layer in ['fc1', 'fc2', 'fc3']:
                    source_comp_id = f"{source_id}_{source_layer}"
                    if source_comp_id in assembly_registry.component_registry:
                        assembly_registry.assembly_dependency_graph[child_comp_id].add(source_comp_id)


# POTENTIALLY DEPRECATED:
def mutate_blueprint(blueprint, registry, mutation_rate=0.1):
    for module in blueprint.modules:
        if random.random() < mutation_rate:
            # Mutate dimensions
            old_out = module.linear.out_features
            new_out = max(4, old_out + random.randint(-4, 4))
            module.linear = nn.Linear(module.linear.in_features, new_out)

    if random.random() < mutation_rate:
        # Add a new module
        in_dim = random.choice([m.linear.out_features for m in blueprint.modules])
        new_mod = TrackedLayer(in_dim, 16)
        blueprint.add_module(new_mod)
        registry.register(new_mod)
