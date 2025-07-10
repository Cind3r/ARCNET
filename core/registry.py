# CHANGED:
# - changed random assigned uuid to content based hashes
# - removed copy.deepcopy() in favor of clone() for tensors (to preserve memory IDs)
# - added classes to handle the above and separate operations
import hashlib
import torch
from collections import defaultdict
from datetime import datetime
import uuid
import copy

class AssemblyComponent:
    """
    Represents an atomic assembly component 
        Components are defined by their content/structure, not random IDs
        Components can be weights, Q-values, biases, positions, etc.
        Components can be assembled from other components, following specific rules
 
    Attributes:
    - content: The actual data (weights, q-values, etc.)
    - component_type: Type of component (e.g., 'layer_weight', 'q_state', 'bias', 'position', 'q_experience')
    - source_module_id: ID of the module this component came from
    - layer_name: Name of the layer this component belongs to
    - id: Unique identifier for the component, generated from content
    - assembly_step: When this component was first assembled
    - parents: Components this was assembled from
    - reuse_count: How many times this component has been reused
    - assembly_complexity: Base complexity of this component
    
    Methods:
    - _generate_content_id: Generate a unique ID based on content
    - can_be_assembled_from: Check if this component can be assembled from given components
    - get_assembly_complexity: Calculate minimal assembly complexity
    - get_minimal_assembly_complexity: Get the minimal assembly complexity recursively

    """

    def __init__(self, content, component_type="generic", source_module_id=None, layer_name=None):
        self.content = content  # The actual data (weights, q-values, etc.)
        self.component_type = component_type  # 'layer_weight', 'q_state', 'bias', 'position', 'q_experience'
        self.source_module_id = source_module_id  # Which module this came from
        self.layer_name = layer_name  # Which layer/component this represents
        self.id = self._generate_content_id()
        self.assembly_step = None  # When this component was first assembled
        self.parents = []  # Components this was assembled from
        self.reuse_count = 0  # How many times this component has been reused
        self.assembly_complexity = 1  # Base complexity
    
    def _generate_content_id(self):
        """Generate ID based on content, not random UUID"""
        if isinstance(self.content, torch.Tensor):
            # Hash the tensor content
            content_hash = hashlib.md5(self.content.detach().cpu().numpy().tobytes()).hexdigest()[:12]
        elif isinstance(self.content, (list, tuple)):
            # Hash the list/tuple content
            content_str = str(self.content)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()[:12]
        else:
            content_hash = hashlib.md5(str(self.content).encode()).hexdigest()[:12]
        
        return f"{self.component_type}_{content_hash}"
    
    def can_be_assembled_from(self, components):
        """Check if this component can be assembled from given components"""
        # Assembly rules based on component type
        if self.component_type.startswith('weight_'):
            # Weight components can be assembled from smaller weight components
            return any(comp.component_type.startswith('weight_') for comp in components)
        elif self.component_type == 'q_experience':
            # Q-experiences can be assembled from state-action pairs
            return len(components) >= 2  # Need at least state and action
        elif self.component_type == 'position':
            # Position can be assembled from previous positions
            return any(comp.component_type == 'position' for comp in components)
        return len(components) >= 1
    
    def get_assembly_complexity(self, memo=None):
        """Calculate minimal assembly complexity (like abracadabra example)"""
        if memo is None:
            memo = {}
        if self.id in memo:
            return memo[self.id]
        
        if not self.parents:
            # Base component - complexity 1
            memo[self.id] = 1
            return 1
        
        # Assembly complexity is 1 + max parent complexity
        parent_complexities = [parent.get_assembly_complexity(memo) for parent in self.parents]
        memo[self.id] = 1 + max(parent_complexities)
        return memo[self.id]


class AssemblyTracker:

    """
    Tracks assembly sequences and computes assembly indices for ConceptModule components

    Attributes:
    - component_library: Dictionary of all components by ID
    - assembly_sequences: Dictionary of assembly sequences by module ID
    - global_assembly_step: Global step counter for assembly
    - reuse_threshold: Threshold for considering components equivalent (default 1e-6)
    Methods:
    - extract_components_from_module: Extracts all atomic components from a ConceptModule
    - find_assembly_sequence: Finds the minimal assembly sequence to build target components from available components
    - compute_assembly_index: Computes assembly index for a module based on its components and parent modules
    - get_assembly_statistics: Returns comprehensive assembly statistics
    - _components_equivalent: Checks if two components are equivalent (can be reused)
    - get_assembly_complexity: Returns the minimal assembly complexity of a component

    """
    
    def __init__(self):
        self.component_library = {}  # id -> AssemblyComponent
        self.assembly_sequences = {}  # module_id -> list of assembly steps
        self.global_assembly_step = 0
        self.reuse_threshold = 1e-6  # Threshold for considering components equivalent
    
    def extract_components_from_module(self, module):
        """Extract all atomic components from a ConceptModule"""
        components = []
        
        # Extract neural network layer weights as components
        for name, param in module.named_parameters():
            if 'weight' in name:
                # Split large weight matrices into row components (like letters in abracadabra)
                if param.dim() == 2:
                    for i, row in enumerate(param):
                        comp = AssemblyComponent(
                            row.clone(), 
                            f"weight_{name}_row{i}", 
                            source_module_id=module.id,
                            layer_name=name
                        )
                        components.append(comp)
                else:
                    comp = AssemblyComponent(
                        param.clone(), 
                        f"weight_{name}", 
                        source_module_id=module.id,
                        layer_name=name
                    )
                    components.append(comp)
            elif 'bias' in name:
                comp = AssemblyComponent(
                    param.clone(), 
                    f"bias_{name}", 
                    source_module_id=module.id,
                    layer_name=name
                )
                components.append(comp)
        
        # Extract Q-function components if present (CompressedQModule)
        if hasattr(module, 'q_function') and module.q_function is not None:
            # Extract Q-network weights
            for name, param in module.q_function.named_parameters():
                if 'weight' in name:
                    comp = AssemblyComponent(
                        param.clone(), 
                        f"q_weight_{name}", 
                        source_module_id=module.id,
                        layer_name=f"q_function.{name}"
                    )
                    components.append(comp)
            
            # Extract Q-experiences from replay buffer
            if hasattr(module.q_function, 'replay_buffer') and module.q_function.replay_buffer:
                for i, experience in enumerate(module.q_function.replay_buffer[-5:]):  # Last 5 experiences
                    comp = AssemblyComponent(
                        experience, 
                        f"q_experience_{i}", 
                        source_module_id=module.id,
                        layer_name="q_function.replay_buffer"
                    )
                    components.append(comp)
        
        # Extract position/manifold components
        if hasattr(module, 'position'):
            comp = AssemblyComponent(
                module.position.clone(), 
                "position", 
                source_module_id=module.id,
                layer_name="position"
            )
            components.append(comp)
        
        # Extract manifold encoder components
        if hasattr(module, 'manifold_encoder'):
            for name, param in module.manifold_encoder.named_parameters():
                comp = AssemblyComponent(
                    param.clone(), 
                    f"manifold_{name}", 
                    source_module_id=module.id,
                    layer_name=f"manifold_encoder.{name}"
                )
                components.append(comp)
        
        # Extract curvature predictor components
        if hasattr(module, 'curvature_predictor'):
            for name, param in module.curvature_predictor.named_parameters():
                comp = AssemblyComponent(
                    param.clone(), 
                    f"curvature_{name}", 
                    source_module_id=module.id,
                    layer_name=f"curvature_predictor.{name}"
                )
                components.append(comp)
        
        # Extract layer_components (ModuleComponent instances)
        if hasattr(module, 'layer_components'):
            for layer_name, layer_comp in module.layer_components.items():
                if hasattr(layer_comp, 'data'):
                    comp = AssemblyComponent(
                        layer_comp.data.clone(), 
                        f"layer_component_{layer_name}", 
                        source_module_id=module.id,
                        layer_name=layer_name
                    )
                    components.append(comp)
        
        # Extract assembly pathway components
        if hasattr(module, 'assembly_pathway'):
            for i, pathway_comp in enumerate(module.assembly_pathway):
                if hasattr(pathway_comp, 'data'):
                    comp = AssemblyComponent(
                        pathway_comp.data.clone(), 
                        f"pathway_component_{i}", 
                        source_module_id=module.id,
                        layer_name=f"assembly_pathway[{i}]"
                    )
                    components.append(comp)
        
        return components
    
    def find_assembly_sequence(self, target_components, available_components):
        """
        Find the minimal assembly sequence to build target_components from available_components
        This is like finding how to build 'abracadabra' from letters
        """
        assembly_sequence = []
        built_components = set()
        component_library = {comp.id: comp for comp in available_components}
        
        # Add target components to library
        for comp in target_components:
            component_library[comp.id] = comp
        
        # Track reused components (like 'abra' in 'abracadabra')
        reused_components = {}
        
        def can_build_component(comp_id, step):
            """Check if we can build this component at this step"""
            if comp_id in built_components:
                return True
            
            comp = component_library[comp_id]
            if not comp.parents:
                # Base component - can always build
                return True
            
            # Check if all parents are already built
            return all(parent.id in built_components for parent in comp.parents)
        
        def find_reusable_component(target_comp):
            """Find if target component can reuse an existing component"""
            for available_comp in available_components:
                if (available_comp.id != target_comp.id and 
                    self._components_equivalent(available_comp, target_comp)):
                    return available_comp
            return None
        
        # First pass: identify reusable components
        for target_comp in target_components:
            reusable = find_reusable_component(target_comp)
            if reusable:
                reused_components[target_comp.id] = reusable
                target_comp.parents = [reusable]
                reusable.reuse_count += 1
        
        # Simulate assembly process
        for step in range(len(target_components) * 2):  # Max possible steps
            components_built_this_step = []
            reused_this_step = []
            
            for comp in target_components:
                if comp.id not in built_components and can_build_component(comp.id, step):
                    components_built_this_step.append(comp)
                    built_components.add(comp.id)
                    
                    # Check if this component reuses another
                    if comp.id in reused_components:
                        reused_this_step.append(reused_components[comp.id])
            
            if components_built_this_step:
                assembly_sequence.append({
                    'step': step,
                    'components_built': components_built_this_step,
                    'reused_components': reused_this_step
                })
            
            # Check if we've built all target components
            if all(comp.id in built_components for comp in target_components):
                break
        
        return assembly_sequence
    
    def compute_assembly_index(self, module, parent_modules=None):
        
        """
        Compute assembly index for a module 
        - e.g., a + b --> ab + r --> abr + a --> abra + c --> abrac + a --> abraca + d --> abracad + abra --> abracadabra
        is index of 7 (number of steps to assemble 'abracadabra' from 'a', 'b', 'r', 'c', 'd')
        """
        
        # Extract components from this module
        current_components = self.extract_components_from_module(module)
        
        # Extract components from parent modules (available for reuse)
        available_components = []
        if parent_modules:
            for parent in parent_modules:
                if hasattr(parent, 'id'):  # Make sure it's a valid module
                    available_components.extend(self.extract_components_from_module(parent))
        
        # Find assembly sequence
        assembly_sequence = self.find_assembly_sequence(current_components, available_components)
        
        # Store the assembly sequence
        self.assembly_sequences[module.id] = assembly_sequence
        
        # Assembly index is the number of steps needed
        assembly_index = len(assembly_sequence)
        
        # Update component library
        for comp in current_components:
            self.component_library[comp.id] = comp
        
        return assembly_index, assembly_sequence
    
    def _components_equivalent(self, comp1, comp2):
        """Check if two components are equivalent (can be reused)"""
        if comp1.component_type != comp2.component_type:
            return False
        
        if isinstance(comp1.content, torch.Tensor) and isinstance(comp2.content, torch.Tensor):
            # Check if tensors are similar (within tolerance)
            return torch.allclose(comp1.content, comp2.content, atol=self.reuse_threshold)
        
        return comp1.content == comp2.content
    
    def get_assembly_statistics(self):
        """Get comprehensive assembly statistics"""
        if not self.component_library:
            return {
                'total_components': 0,
                'reused_components': 0,
                'average_reuse': 0.0,
                'component_types': {}
            }
        
        stats = {
            'total_components': len(self.component_library),
            'reused_components': sum(1 for comp in self.component_library.values() if comp.reuse_count > 0),
            'average_reuse': sum(comp.reuse_count for comp in self.component_library.values()) / len(self.component_library),
            'component_types': defaultdict(int)
        }
        
        for comp in self.component_library.values():
            stats['component_types'][comp.component_type] += 1
        
        return stats

class AssemblyTrackingRegistry:
    """
    Tracking registry for ConceptModule components

    Attributes:
    - assembly_tracker: Tracks assembly information for components
    - parameter_update_history: History of parameter updates for modules
    - message_influence_graph: Graph of message influences between components
    - q_learning_inheritance_graph: Graph of Q-learning inheritance relationships
    - step_counter: Global step counter for assembly events
    - global_assembly_events: List of all assembly events
    - complexity_history: History of assembly complexity over time
    - component_registry: Registry for tracking legacy components (for backward compatibility)
    Methods:
    - create_parameter_snapshot: Creates a snapshot of all module parameters for tracking
    - register_module_initialization: Registers initial state of a module with proper assembly tracking
    - track_mutation_with_assembly_inheritance: Tracks mutation with proper assembly inheritance
    - step_forward: Advances the step counter and updates complexity history
    - get_assembly_complexity_history: Gets the history of assembly complexity over time
    - get_component_lineage: Gets the complete lineage of a component

    """
    
    def __init__(self):
        self.assembly_tracker = AssemblyTracker()
        self.parameter_update_history = {}
        self.message_influence_graph = defaultdict(list)
        self.q_learning_inheritance_graph = defaultdict(list)
        self.step_counter = 0
        self.global_assembly_events = []
        self.complexity_history = []
        self.component_registry = {}  
    
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
                buffer_str = str(len(module.q_function.replay_buffer))
                for exp in module.q_function.replay_buffer[-5:]:
                    buffer_str += str(exp)
                q_hash = hashlib.md5(buffer_str.encode()).hexdigest()[:8]
            snapshot['q_function_hash'] = q_hash
            snapshot['q_buffer_size'] = len(module.q_function.replay_buffer) if hasattr(module.q_function, 'replay_buffer') else 0
        
        # Capture position/manifold state
        if hasattr(module, 'position'):
            position_hash = hashlib.md5(module.position.data.cpu().numpy().tobytes()).hexdigest()[:8]
            snapshot['position_hash'] = position_hash
        
        # Capture manifold encoder state
        if hasattr(module, 'manifold_encoder'):
            manifold_hashes = {}
            for name, param in module.manifold_encoder.named_parameters():
                param_hash = hashlib.md5(param.data.cpu().numpy().tobytes()).hexdigest()[:8]
                manifold_hashes[name] = param_hash
            snapshot['manifold_hashes'] = manifold_hashes
        
        # Capture curvature
        if hasattr(module, 'curvature'):
            snapshot['curvature'] = float(module.curvature)
        
        return snapshot
    
    def register_module_initialization(self, module, parent_modules=None):
        """Register initial state of a module with proper assembly tracking"""
        # Compute assembly index using the new tracker
        assembly_index, assembly_sequence = self.assembly_tracker.compute_assembly_index(
            module, parent_modules
        )
        
        # Store assembly index on the module
        module.assembly_index = assembly_index
        module.assembly_sequence = assembly_sequence
        
        # Create snapshot for tracking
        snapshot = self.create_parameter_snapshot(module, "initialization")
        
        if module.id not in self.parameter_update_history:
            self.parameter_update_history[module.id] = []
        self.parameter_update_history[module.id].append(snapshot)
        
        # Register components in the component registry (for backward compatibility)
        if hasattr(module, 'layer_components'):
            for layer_name, component in module.layer_components.items():
                comp_id = getattr(component, 'id', f"{module.id}_{layer_name}")
                
                self.component_registry[comp_id] = {
                    'component': component,
                    'module_id': module.id,
                    'layer_name': layer_name,
                    'creation_step': self.step_counter,
                    'creation_event': 'initialization',
                    'parent_components': [],
                    'derived_components': [],
                    'assembly_complexity': getattr(component, 'assembly_complexity', 1),
                    'original_id': comp_id,
                    'message_influences': [],
                    'q_influences': []
                }
        
        # Track the assembly event
        self.global_assembly_events.append({
            'type': 'module_initialization',
            'step': self.step_counter,
            'module_id': module.id,
            'assembly_index': assembly_index,
            'assembly_sequence': assembly_sequence,
            'components_count': len(self.assembly_tracker.extract_components_from_module(module))
        })
        
        return assembly_index, assembly_sequence
    
    def track_mutation_with_assembly_inheritance(self, parent_module, child_module, catalysts=None):
        """Track mutation with proper assembly inheritance"""
        # Collect all parent modules (including catalysts)
        all_parents = [parent_module]
        if catalysts:
            all_parents.extend(catalysts)
        
        # Register child module with parents available for reuse
        assembly_index, assembly_sequence = self.assembly_tracker.compute_assembly_index(
            child_module, all_parents
        )
        
        child_module.assembly_index = assembly_index
        child_module.assembly_sequence = assembly_sequence
        
        # Track component inheritance
        inheritance_info = []
        if hasattr(child_module, 'layer_components'):
            for layer_name, child_comp in child_module.layer_components.items():
                # Check if this component was inherited/reused
                for parent in all_parents:
                    if hasattr(parent, 'layer_components') and layer_name in parent.layer_components:
                        parent_comp = parent.layer_components[layer_name]
                        if hasattr(parent_comp, 'data') and hasattr(child_comp, 'data'):
                            if torch.allclose(parent_comp.data, child_comp.data, atol=1e-6):
                                inheritance_info.append({
                                    'layer': layer_name,
                                    'parent_module': parent.id,
                                    'reused': True
                                })
                                break
                else:
                    inheritance_info.append({
                        'layer': layer_name,
                        'parent_module': None,
                        'reused': False
                    })
        
        # Track the mutation event
        mutation_event = {
            'parent_id': parent_module.id,
            'child_id': child_module.id,
            'catalyst_ids': [c.id for c in catalysts] if catalysts else [],
            'parent_assembly_index': getattr(parent_module, 'assembly_index', 0),
            'child_assembly_index': assembly_index,
            'step': self.step_counter,
            'assembly_sequence': assembly_sequence,
            'inheritance_info': inheritance_info
        }
        
        self.global_assembly_events.append({
            'type': 'mutation_with_assembly',
            'step': self.step_counter,
            'data': mutation_event
        })
        
        return mutation_event
    
    def step_forward(self):
        """Advance the step counter and update complexity history"""
        self.step_counter += 1
        
        # Update complexity history
        stats = self.assembly_tracker.get_assembly_statistics()
        self.complexity_history.append({
            'step': self.step_counter,
            'total_components': stats['total_components'],
            'reused_components': stats['reused_components'],
            'reuse_rate': stats['reused_components'] / max(1, stats['total_components'])
        })
    
    def get_assembly_complexity_history(self):
        """Get the history of assembly complexity over time"""
        return self.complexity_history
    
    def get_component_lineage(self, component_id):
        """Get the complete lineage of a component"""
        if component_id not in self.assembly_tracker.component_library:
            return None
        
        component = self.assembly_tracker.component_library[component_id]
        
        lineage = {
            'component_id': component_id,
            'component_type': component.component_type,
            'source_module_id': component.source_module_id,
            'layer_name': component.layer_name,
            'assembly_complexity': component.get_assembly_complexity(),
            'reuse_count': component.reuse_count,
            'parents': [parent.id for parent in component.parents],
            'assembly_step': component.assembly_step
        }
        
        return lineage
    
    def get_assembly_statistics(self):
        """Get assembly statistics from the tracker"""
        return self.assembly_tracker.get_assembly_statistics()

# Keep the original ModuleComponent for backward compatibility
class ModuleComponent:
    def __init__(self, data, parents=None, operation=None, assembly_pathway=None):
        self.id = uuid.uuid4().hex
        self.data = data
        self.parents = parents or []
        self.operation = operation
        self.assembly_pathway = assembly_pathway or [self.id]
        self.assembly_complexity = 1

    def copy(self, preserve_id=False):
        new_component = ModuleComponent(self.data.clone(), parents=self.parents, operation=self.operation, assembly_pathway=list(self.assembly_pathway))
        if preserve_id:
            new_component.id = self.id
        return new_component

    def get_minimal_assembly_complexity(self, memo=None):
        if memo is None:
            memo = {}
        if self.id in memo:
            return memo[self.id]
        if not self.parents:
            memo[self.id] = 1
            return 1
        parent_complexities = [parent.get_minimal_assembly_complexity(memo) for parent in self.parents]
        memo[self.id] = 1 + max(parent_complexities)
        return memo[self.id]

# Component registry for backward compatibility
class ComponentRegistry:
    def __init__(self):
        self.data = {}
    
    def register(self, module_id, **kwargs):
        self.data[module_id] = kwargs
    
    def get_assembly_complexity(self, module_id):
        if module_id in self.data:
            return self.data[module_id].get('assembly_complexity', None)
        return None