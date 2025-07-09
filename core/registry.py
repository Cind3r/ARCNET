import uuid
import torch
import copy
import hashlib
from datetime import datetime
from collections import defaultdict

class AssemblyTrackingRegistry:
    """
    Comprehensive assembly tracking that monitors all parameter updates
    from message passing, Q-learning, and layer weight changes
    """
    
    def __init__(self):
        self.component_registry = {}  # component_id -> component_data
        self.parameter_update_history = {}  # module_id -> list of updates
        self.message_influence_graph = defaultdict(list)  # sender_id -> list of influenced modules
        self.q_learning_inheritance_graph = defaultdict(list)  # parent_q -> list of children
        self.assembly_dependency_graph = defaultdict(set)  # component_id -> set of dependent components
        self.step_counter = 0
        self.global_assembly_events = []
        
    def create_parameter_snapshot(self, module, event_type="initialization"):
        """Create a snapshot of all module parameters for tracking"""
        snapshot = {
            'timestamp': datetime.now(),
            'step': self.step_counter,
            'event_type': event_type,
            'module_id': module.id,
            'fitness': getattr(module, 'fitness', 0.0),
            'assembly_index': getattr(module, 'assembly_index', 0),
            'position': getattr(module, 'position', None)
        }
        
        # Capture layer weights
        layer_hashes = {}
        for name, param in module.named_parameters():
            if 'weight' in name or 'bias' in name:
                param_hash = hashlib.md5(param.data.cpu().numpy().tobytes()).hexdigest()[:8]
                layer_hashes[name] = param_hash
        snapshot['layer_hashes'] = layer_hashes
        
        # Capture Q-function state if exists
        if hasattr(module, 'q_function') and module.q_function is not None:
            q_hash = "no_buffer"
            if hasattr(module.q_function, 'replay_buffer') and module.q_function.replay_buffer:
                # Hash the Q-function's experience buffer
                buffer_str = str(len(module.q_function.replay_buffer))
                for exp in module.q_function.replay_buffer[-5:]:  # Last 5 experiences
                    buffer_str += str(exp)
                q_hash = hashlib.md5(buffer_str.encode()).hexdigest()[:8]
            snapshot['q_function_hash'] = q_hash
            snapshot['q_buffer_size'] = len(module.q_function.replay_buffer) if hasattr(module.q_function, 'replay_buffer') else 0
        
        # Capture position/manifold state
        if hasattr(module, 'position'):
            position_hash = hashlib.md5(module.position.data.cpu().numpy().tobytes()).hexdigest()[:8]
            snapshot['position_hash'] = position_hash
            
        return snapshot
    
    def register_module_initialization(self, module):
        """Register initial state of a module"""
        snapshot = self.create_parameter_snapshot(module, "initialization")
        
        if module.id not in self.parameter_update_history:
            self.parameter_update_history[module.id] = []
        self.parameter_update_history[module.id].append(snapshot)
        
        # Register initial components
        if hasattr(module, 'layer_components'):
            for layer_name, component in module.layer_components.items():
                comp_id = f"{module.id}_{layer_name}"
                self.component_registry[comp_id] = {
                    'component': component,
                    'module_id': module.id,
                    'layer_name': layer_name,
                    'creation_step': self.step_counter,
                    'creation_event': 'initialization',
                    'parent_components': [],
                    'derived_components': [],
                    'message_influences': [],
                    'q_influences': []
                }
    
    def track_message_passing_update(self, receiver_module, sender_modules, message_content):
        """Track parameter updates from message passing"""
        # Create snapshot before message processing
        pre_snapshot = self.create_parameter_snapshot(receiver_module, "pre_message")
        
        # Store message influence information
        message_event = {
            'receiver_id': receiver_module.id,
            'sender_ids': [s.id for s in sender_modules],
            'message_type': 'manifold_message',
            'step': self.step_counter,
            'pre_snapshot': pre_snapshot,
            'message_content_hash': hashlib.md5(str(message_content).encode()).hexdigest()[:8] if message_content is not None else "none"
        }
        
        # This will be completed after message processing
        return message_event
    
    def complete_message_passing_update(self, message_event, receiver_module):
        """Complete tracking after message has been processed"""
        # Create snapshot after message processing
        post_snapshot = self.create_parameter_snapshot(receiver_module, "post_message")
        message_event['post_snapshot'] = post_snapshot
        
        # Check what changed
        changes_detected = self.detect_parameter_changes(
            message_event['pre_snapshot'], 
            post_snapshot
        )
        message_event['changes_detected'] = changes_detected
        
        # Add to history
        if receiver_module.id not in self.parameter_update_history:
            self.parameter_update_history[receiver_module.id] = []
        self.parameter_update_history[receiver_module.id].append(message_event)
        
        # Track influence relationships
        for sender_id in message_event['sender_ids']:
            self.message_influence_graph[sender_id].append({
                'influenced_module': receiver_module.id,
                'step': self.step_counter,
                'changes': changes_detected
            })
        
        # Update component dependencies if parameters changed
        if changes_detected['weights_changed'] or changes_detected['position_changed']:
            self.update_component_dependencies_from_message(receiver_module, message_event)
        
        # Add to global events
        self.global_assembly_events.append({
            'type': 'message_passing',
            'step': self.step_counter,
            'data': message_event
        })
        
        return changes_detected
    
    def track_q_learning_update(self, module, q_update_info):
        """Track Q-learning parameter updates"""
        pre_snapshot = self.create_parameter_snapshot(module, "pre_q_update")
        
        q_event = {
            'module_id': module.id,
            'update_type': 'q_learning',
            'step': self.step_counter,
            'pre_snapshot': pre_snapshot,
            'q_update_info': q_update_info,
            'experiences_added': q_update_info.get('experiences_added', 0),
            'q_value_changes': q_update_info.get('q_value_changes', [])
        }
        
        return q_event
    
    def complete_q_learning_update(self, q_event, module):
        """Complete Q-learning update tracking"""
        post_snapshot = self.create_parameter_snapshot(module, "post_q_update")
        q_event['post_snapshot'] = post_snapshot
        
        changes_detected = self.detect_parameter_changes(
            q_event['pre_snapshot'], 
            post_snapshot
        )
        q_event['changes_detected'] = changes_detected
        
        # Add to history
        if module.id not in self.parameter_update_history:
            self.parameter_update_history[module.id] = []
        self.parameter_update_history[module.id].append(q_event)
        
        # Track Q-learning inheritance if this was from parent experiences
        if 'parent_modules' in q_event['q_update_info']:
            for parent_id in q_event['q_update_info']['parent_modules']:
                self.q_learning_inheritance_graph[parent_id].append({
                    'child_module': module.id,
                    'step': self.step_counter,
                    'experiences_inherited': q_event['experiences_added']
                })
        
        # Update component assembly if Q-function influenced weights
        if changes_detected['q_function_changed']:
            self.update_component_dependencies_from_q_learning(module, q_event)
        
        self.global_assembly_events.append({
            'type': 'q_learning_update',
            'step': self.step_counter,
            'data': q_event
        })
        
        return changes_detected
    
    def track_mutation_with_assembly_influence(self, parent_module, child_module, catalysts, message_influences):
        """Track mutation that incorporates message passing and Q-learning influences"""
        mutation_event = {
            'parent_id': parent_module.id,
            'child_id': child_module.id,
            'catalyst_ids': [c.id for c in catalysts],
            'step': self.step_counter,
            'parent_snapshot': self.create_parameter_snapshot(parent_module, "mutation_parent"),
            'child_snapshot': self.create_parameter_snapshot(child_module, "mutation_child"),
            'message_influences': message_influences,
            'assembly_inheritance': []
        }
        
        # Track component inheritance and creation
        if hasattr(parent_module, 'layer_components') and hasattr(child_module, 'layer_components'):
            for layer_name in child_module.layer_components.keys():
                parent_comp_id = f"{parent_module.id}_{layer_name}"
                child_comp_id = f"{child_module.id}_{layer_name}"
                
                # Register new child component
                child_component = child_module.layer_components[layer_name]
                self.component_registry[child_comp_id] = {
                    'component': child_component,
                    'module_id': child_module.id,
                    'layer_name': layer_name,
                    'creation_step': self.step_counter,
                    'creation_event': 'mutation',
                    'parent_components': [parent_comp_id],
                    'derived_components': [],
                    'message_influences': [],
                    'q_influences': []
                }
                
                # Update parent component's derived list
                if parent_comp_id in self.component_registry:
                    self.component_registry[parent_comp_id]['derived_components'].append(child_comp_id)
                
                # Track assembly inheritance
                mutation_event['assembly_inheritance'].append({
                    'parent_component': parent_comp_id,
                    'child_component': child_comp_id,
                    'layer': layer_name
                })
                
                # Add assembly dependencies
                self.assembly_dependency_graph[child_comp_id].add(parent_comp_id)
        
        # Add to global events
        self.global_assembly_events.append({
            'type': 'mutation_with_influences',
            'step': self.step_counter,
            'data': mutation_event
        })
        
        return mutation_event
    
    def detect_parameter_changes(self, pre_snapshot, post_snapshot):
        """Detect what parameters changed between snapshots"""
        changes = {
            'weights_changed': False,
            'position_changed': False,
            'q_function_changed': False,
            'changed_layers': [],
            'fitness_delta': post_snapshot['fitness'] - pre_snapshot['fitness']
        }
        
        # Check layer weight changes
        pre_hashes = pre_snapshot.get('layer_hashes', {})
        post_hashes = post_snapshot.get('layer_hashes', {})
        
        for layer_name in set(pre_hashes.keys()) | set(post_hashes.keys()):
            if pre_hashes.get(layer_name) != post_hashes.get(layer_name):
                changes['weights_changed'] = True
                changes['changed_layers'].append(layer_name)
        
        # Check position changes
        if pre_snapshot.get('position_hash') != post_snapshot.get('position_hash'):
            changes['position_changed'] = True
        
        # Check Q-function changes
        if pre_snapshot.get('q_function_hash') != post_snapshot.get('q_function_hash'):
            changes['q_function_changed'] = True
        
        return changes
    
    def update_component_dependencies_from_message(self, module, message_event):
        """Update component dependencies based on message passing influences"""
        if not hasattr(module, 'layer_components'):
            return
        
        for layer_name, component in module.layer_components.items():
            comp_id = f"{module.id}_{layer_name}"
            
            if comp_id in self.component_registry:
                # Add message influence to component
                influence_record = {
                    'type': 'message_passing',
                    'step': self.step_counter,
                    'sender_ids': message_event['sender_ids'],
                    'changes': message_event['changes_detected']
                }
                self.component_registry[comp_id]['message_influences'].append(influence_record)
                
                # Create dependency edges to sender components
                for sender_id in message_event['sender_ids']:
                    for sender_layer in ['fc1', 'fc2', 'fc3']:  # Known layer names
                        sender_comp_id = f"{sender_id}_{sender_layer}"
                        if sender_comp_id in self.component_registry:
                            self.assembly_dependency_graph[comp_id].add(sender_comp_id)
    
    def update_component_dependencies_from_q_learning(self, module, q_event):
        """Update component dependencies based on Q-learning influences"""
        if not hasattr(module, 'layer_components'):
            return
        
        for layer_name, component in module.layer_components.items():
            comp_id = f"{module.id}_{layer_name}"
            
            if comp_id in self.component_registry:
                # Add Q-learning influence to component
                influence_record = {
                    'type': 'q_learning',
                    'step': self.step_counter,
                    'q_update_info': q_event['q_update_info'],
                    'changes': q_event['changes_detected']
                }
                self.component_registry[comp_id]['q_influences'].append(influence_record)
    
    def get_component_assembly_graph(self):
        """Generate a NetworkX graph of component assembly dependencies"""
        import networkx as nx
        
        G = nx.DiGraph()
        
        # Add all components as nodes
        for comp_id, comp_data in self.component_registry.items():
            G.add_node(comp_id, **comp_data)
        
        # Add assembly dependency edges
        for child_comp, parent_comps in self.assembly_dependency_graph.items():
            for parent_comp in parent_comps:
                if parent_comp in self.component_registry:
                    G.add_edge(parent_comp, child_comp, relationship='assembly_dependency')
        
        # Add message influence edges
        for comp_id, comp_data in self.component_registry.items():
            for msg_influence in comp_data['message_influences']:
                for sender_id in msg_influence['sender_ids']:
                    # Add edges from sender components
                    for layer in ['fc1', 'fc2', 'fc3']:
                        sender_comp_id = f"{sender_id}_{layer}"
                        if sender_comp_id in self.component_registry:
                            G.add_edge(sender_comp_id, comp_id, 
                                     relationship='message_influence',
                                     step=msg_influence['step'])
        
        return G
    
    def get_assembly_statistics(self):
        """Get comprehensive assembly statistics"""
        total_components = len(self.component_registry)
        total_modules = len(self.parameter_update_history)
        
        # Count different types of influences
        message_influences = sum(len(comp['message_influences']) for comp in self.component_registry.values())
        q_influences = sum(len(comp['q_influences']) for comp in self.component_registry.values())
        
        # Count assembly events
        event_counts = defaultdict(int)
        for event in self.global_assembly_events:
            event_counts[event['type']] += 1
        
        # Calculate assembly dependency depth
        G = self.get_component_assembly_graph()
        max_depth = 0
        if G.number_of_nodes() > 0:
            try:
                import networkx as nx
                # Find longest path (assembly chain)
                for node in G.nodes():
                    if G.in_degree(node) == 0:  # Root node
                        try:
                            paths = nx.single_source_shortest_path_length(G, node)
                            max_depth = max(max_depth, max(paths.values()) if paths else 0)
                        except:
                            pass
            except:
                pass
        
        return {
            'total_components': total_components,
            'total_modules_tracked': total_modules,
            'message_influences': message_influences,
            'q_learning_influences': q_influences,
            'max_assembly_depth': max_depth,
            'event_counts': dict(event_counts),
            'total_assembly_events': len(self.global_assembly_events),
            'current_step': self.step_counter
        }
    
    def step_forward(self):
        """Advance the step counter"""
        self.step_counter += 1
    
    def get_module_assembly_history(self, module_id):
        """Get complete assembly history for a specific module"""
        return self.parameter_update_history.get(module_id, [])
    
    def get_component_lineage(self, component_id):
        """Get the complete lineage of a component including all influences"""
        if component_id not in self.component_registry:
            return None
        
        comp_data = self.component_registry[component_id]
        lineage = {
            'component_id': component_id,
            'component_data': comp_data,
            'parent_lineage': [],
            'influence_lineage': []
        }
        
        # Trace parent components
        for parent_id in comp_data['parent_components']:
            parent_lineage = self.get_component_lineage(parent_id)
            if parent_lineage:
                lineage['parent_lineage'].append(parent_lineage)
        
        # Add influence information
        lineage['influence_lineage'].extend(comp_data['message_influences'])
        lineage['influence_lineage'].extend(comp_data['q_influences'])
        
        return lineage



class ComponentRegistry:
    def __init__(self):
        self.data = {}

    def register(self, module):
        self.data[module.id] = {
            'type': type(module).__name__,
            'in_dim': getattr(module, 'linear', None).in_features if hasattr(module, 'linear') else None,
            'out_dim': getattr(module, 'linear', None).out_features if hasattr(module, 'linear') else None,
            'score': 0.0,
            'uses': 0,
            'assembly_complexity': module.get_assembly_complexity() if hasattr(module, 'get_assembly_complexity') else None
        }

    def update_score(self, module_id, delta):
        if module_id in self.data:
            self.data[module_id]['score'] += delta
            self.data[module_id]['uses'] += 1

    def get_assembly_complexity(self, module_id):
        if module_id in self.data:
            return self.data[module_id].get('assembly_complexity', None)
        return None


class ModuleComponent:
    def __init__(self, data, parents=None, operation=None, assembly_pathway=None):
        self.id = uuid.uuid4().hex  # Unique identifier
        self.data = data  # e.g., weight tensor
        self.parents = parents or []  # List of parent component IDs
        self.operation = operation  # 'mutation', 'crossover', etc.
        self.assembly_pathway = assembly_pathway or [self.id]

    def copy(self):
        # Create a new component with the same data and lineage
        return ModuleComponent(self.data.clone(), parents=self.parents, operation=self.operation, assembly_pathway=list(self.assembly_pathway))

    def get_minimal_assembly_complexity(self, memo=None):
        """
        Returns the minimal number of unique construction steps (assembly complexity)
        for this component, using memoization to avoid recomputation.
        """
        if memo is None:
            memo = {}
        if self.id in memo:
            return memo[self.id]
        if not self.parents:
            memo[self.id] = 1
            return 1
        # Complexity is 1 + sum of unique parent complexities (reuse subcomponents)
        parent_complexities = []
        for parent in self.parents:
            parent_complexity = parent.get_minimal_assembly_complexity(memo)
            parent_complexities.append(parent_complexity)
        
        # Assembly theory: complexity is 1 + max of parent complexities (not sum)
        # This represents the number of assembly steps needed
        memo[self.id] = 1 + max(parent_complexities)
        return memo[self.id]