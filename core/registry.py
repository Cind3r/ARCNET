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


# ====================================================================
# ========== Lightweight Registry (above is heavy tracking) ==============
# ====================================================================

# ...existing imports...

class LightweightComponent:
    """
    Lightweight component that tracks assembly without storing full tensor copies
    Uses content hashes and references instead of cloning data
    """
    
    def __init__(self, content_hash, component_type, source_module_id, layer_name, shape=None):
        self.content_hash = content_hash  # Hash instead of full data
        self.component_type = component_type
        self.source_module_id = source_module_id
        self.layer_name = layer_name
        self.shape = shape  # Store shape for compatibility checks
        self.id = f"{component_type}_{content_hash}"
        self.assembly_step = None
        self.parents = []  # Store parent hashes, not full components
        self.reuse_count = 0
        self.assembly_complexity = 1
    
    def can_be_assembled_from(self, components):
        """Simplified assembly rules based on component types"""
        if self.component_type.startswith('weight'):
            return any(comp.component_type.startswith('weight') for comp in components)
        elif self.component_type == 'q_state':
            return len(components) >= 1
        return len(components) >= 1
    
    def get_assembly_complexity(self, memo=None):
        """Calculate minimal assembly complexity"""
        if memo is None:
            memo = {}
        if self.id in memo:
            return memo[self.id]
        
        if not self.parents:
            memo[self.id] = 1
            return 1
        
        # Use parent complexity without loading full parent objects
        parent_complexity = max([memo.get(parent_id, 1) for parent_id in self.parents])
        memo[self.id] = 1 + parent_complexity
        return memo[self.id]

class CompactAssemblyTracker:
    """
    Memory-efficient assembly tracker that uses hashes and aggregated representations
    """
    
    def __init__(self):
        self.component_hashes = {}  # hash -> metadata
        self.assembly_sequences = {}
        self.global_assembly_step = 0
        self.reuse_threshold = 1e-6
    
    def _generate_layer_hash(self, param):
        """Generate hash for a layer parameter without cloning"""
        if isinstance(param, torch.Tensor):
            # Use tensor's data_ptr and shape for lightweight hashing
            return hashlib.md5(f"{param.data_ptr()}_{param.shape}_{param.dtype}".encode()).hexdigest()[:12]
        return hashlib.md5(str(param).encode()).hexdigest()[:12]
    
    def extract_components_from_module(self, module):
        """Extract lightweight components without cloning tensors"""
        components = []
        
        # Extract layer weights as aggregated components (not row-wise)
        for name, param in module.named_parameters():
            if 'weight' in name or 'bias' in name:
                content_hash = self._generate_layer_hash(param)
                comp = LightweightComponent(
                    content_hash=content_hash,
                    component_type=f"{'weight' if 'weight' in name else 'bias'}_{name}",
                    source_module_id=module.id,
                    layer_name=name,
                    shape=param.shape
                )
                components.append(comp)
        
        # Extract Q-function state as single component (not individual experiences)
        if hasattr(module, 'q_function') and module.q_function is not None:
            if hasattr(module.q_function, 'replay_buffer') and module.q_function.replay_buffer:
                # Hash the buffer size and last few experiences as aggregate
                buffer_summary = f"buffer_size_{len(module.q_function.replay_buffer)}"
                if len(module.q_function.replay_buffer) >= 3:
                    buffer_summary += f"_last_{str(module.q_function.replay_buffer[-3:])}"
                q_hash = hashlib.md5(buffer_summary.encode()).hexdigest()[:12]
                
                comp = LightweightComponent(
                    content_hash=q_hash,
                    component_type="q_state",
                    source_module_id=module.id,
                    layer_name="q_function",
                    shape=(len(module.q_function.replay_buffer),)
                )
                components.append(comp)
        
        # Extract position as single component
        if hasattr(module, 'position'):
            pos_hash = self._generate_layer_hash(module.position)
            comp = LightweightComponent(
                content_hash=pos_hash,
                component_type="position",
                source_module_id=module.id,
                layer_name="position",
                shape=module.position.shape
            )
            components.append(comp)
        
        # Extract manifold encoder as single aggregated component
        if hasattr(module, 'manifold_encoder'):
            manifold_hash = self._generate_module_hash(module.manifold_encoder)
            comp = LightweightComponent(
                content_hash=manifold_hash,
                component_type="manifold_encoder",
                source_module_id=module.id,
                layer_name="manifold_encoder"
            )
            components.append(comp)
        
        return components
    
    def _generate_module_hash(self, module):
        """Generate hash for entire module without parameter cloning"""
        param_info = []
        for name, param in module.named_parameters():
            param_info.append(f"{name}_{param.shape}_{param.data_ptr()}")
        return hashlib.md5("_".join(param_info).encode()).hexdigest()[:12]
    
    def compute_assembly_index(self, module, parent_modules=None):
        """Compute assembly index using lightweight components"""
        current_components = self.extract_components_from_module(module)
        
        # Extract components from parents without deep copying
        available_components = []
        if parent_modules:
            for parent in parent_modules:
                if hasattr(parent, 'id'):
                    parent_components = self.extract_components_from_module(parent)
                    available_components.extend(parent_components)
        
        # Find assembly sequence using hash-based matching
        assembly_sequence = self._find_lightweight_assembly_sequence(current_components, available_components)
        
        # Store sequence
        self.assembly_sequences[module.id] = assembly_sequence
        
        # Update component registry
        for comp in current_components:
            self.component_hashes[comp.id] = {
                'component_type': comp.component_type,
                'source_module_id': comp.source_module_id,
                'layer_name': comp.layer_name,
                'shape': comp.shape,
                'reuse_count': comp.reuse_count
            }
        
        return len(assembly_sequence), assembly_sequence
    
    def _find_lightweight_assembly_sequence(self, target_components, available_components):
        """Find assembly sequence using hash-based component matching"""
        assembly_sequence = []
        reused_components = {}
        
        # Create hash lookup for available components
        available_hash_map = {comp.content_hash: comp for comp in available_components}
        
        for target_comp in target_components:
            # Check if component can be reused
            if target_comp.content_hash in available_hash_map:
                available_comp = available_hash_map[target_comp.content_hash]
                if self._components_equivalent_lightweight(target_comp, available_comp):
                    reused_components[target_comp.id] = available_comp.id
                    target_comp.parents = [available_comp.id]
                    available_comp.reuse_count += 1
        
        # Create assembly sequence
        for i, comp in enumerate(target_components):
            step_info = {
                'step': i,
                'component_built': comp.id,
                'reused_from': reused_components.get(comp.id, None)
            }
            assembly_sequence.append(step_info)
        
        return assembly_sequence
    
    def _components_equivalent_lightweight(self, comp1, comp2):
        """Check component equivalence using metadata"""
        return (comp1.component_type == comp2.component_type and 
                comp1.shape == comp2.shape and
                comp1.content_hash == comp2.content_hash)
    
    def get_assembly_statistics(self):
        """Get assembly statistics"""
        if not self.component_hashes:
            return {
                'total_components': 0,
                'reused_components': 0,
                'average_reuse': 0.0,
                'component_types': {}
            }
        
        total_components = len(self.component_hashes)
        reused_components = sum(1 for comp in self.component_hashes.values() if comp['reuse_count'] > 0)
        
        component_types = {}
        for comp in self.component_hashes.values():
            comp_type = comp['component_type']
            component_types[comp_type] = component_types.get(comp_type, 0) + 1
        
        return {
            'total_components': total_components,
            'reused_components': reused_components,
            'average_reuse': sum(comp['reuse_count'] for comp in self.component_hashes.values()) / total_components,
            'component_types': component_types
        }

class MemoryEfficientRegistry:
    """
    Memory-efficient registry that uses the compact assembly tracker
    """
    
    def __init__(self):
        self.assembly_tracker = CompactAssemblyTracker()
        self.parameter_snapshots = {}  # Store only essential snapshots
        self.step_counter = 0
        self.assembly_events = []
        self.max_snapshots_per_module = 5  # Limit snapshot history
    
    def create_lightweight_snapshot(self, module, event_type="initialization"):
        """Create lightweight snapshot using hashes instead of full data"""
        snapshot = {
            'timestamp': datetime.now(),
            'step': self.step_counter,
            'event_type': event_type,
            'module_id': module.id,
            'fitness': getattr(module, 'fitness', 0.0),
            'assembly_index': getattr(module, 'assembly_index', 0)
        }
        
        # Store only essential parameter hashes
        layer_hashes = {}
        for name, param in module.named_parameters():
            if 'weight' in name or 'bias' in name:
                layer_hashes[name] = self.assembly_tracker._generate_layer_hash(param)
        snapshot['layer_hashes'] = layer_hashes
        
        # Q-function summary
        if hasattr(module, 'q_function') and module.q_function is not None:
            if hasattr(module.q_function, 'replay_buffer'):
                snapshot['q_buffer_size'] = len(module.q_function.replay_buffer)
            else:
                snapshot['q_buffer_size'] = 0
        
        # Position hash
        if hasattr(module, 'position'):
            snapshot['position_hash'] = self.assembly_tracker._generate_layer_hash(module.position)
        
        return snapshot
    
    def register_module_initialization(self, module, parent_modules=None):
        """Register module with lightweight tracking"""
        # Compute assembly index
        assembly_index, assembly_sequence = self.assembly_tracker.compute_assembly_index(
            module, parent_modules
        )
        
        # Store on module
        module.assembly_index = assembly_index
        module.assembly_sequence = assembly_sequence
        
        # Create lightweight snapshot
        snapshot = self.create_lightweight_snapshot(module, "initialization")
        
        # Maintain limited snapshot history
        if module.id not in self.parameter_snapshots:
            self.parameter_snapshots[module.id] = []
        
        self.parameter_snapshots[module.id].append(snapshot)
        
        # Keep only recent snapshots
        if len(self.parameter_snapshots[module.id]) > self.max_snapshots_per_module:
            self.parameter_snapshots[module.id].pop(0)
        
        # Track assembly event
        self.assembly_events.append({
            'type': 'module_initialization',
            'step': self.step_counter,
            'module_id': module.id,
            'assembly_index': assembly_index,
            'components_count': len(self.assembly_tracker.extract_components_from_module(module))
        })
        
        return assembly_index, assembly_sequence
    
    def track_mutation_with_lightweight_inheritance(self, parent_module, child_module, catalysts=None):
        """Track mutation with lightweight inheritance tracking"""
        all_parents = [parent_module]
        if catalysts:
            all_parents.extend(catalysts)
        
        # Register child with lightweight tracking
        assembly_index, assembly_sequence = self.assembly_tracker.compute_assembly_index(
            child_module, all_parents
        )
        
        child_module.assembly_index = assembly_index
        child_module.assembly_sequence = assembly_sequence
        
        # Create lightweight mutation record
        mutation_event = {
            'parent_id': parent_module.id,
            'child_id': child_module.id,
            'catalyst_ids': [c.id for c in catalysts] if catalysts else [],
            'parent_assembly_index': getattr(parent_module, 'assembly_index', 0),
            'child_assembly_index': assembly_index,
            'step': self.step_counter
        }
        
        self.assembly_events.append({
            'type': 'mutation_with_assembly',
            'step': self.step_counter,
            'data': mutation_event
        })
        
        return mutation_event
    
    def step_forward(self):
        """Advance step counter"""
        self.step_counter += 1
        
        # Periodic cleanup to prevent memory growth
        if self.step_counter % 100 == 0:
            self._cleanup_old_data()
    
    def _cleanup_old_data(self):
        """Clean up old tracking data to prevent memory growth"""
        # Remove old assembly events (keep last 1000)
        if len(self.assembly_events) > 1000:
            self.assembly_events = self.assembly_events[-1000:]
        
        # Clean up old snapshots
        for module_id in list(self.parameter_snapshots.keys()):
            if len(self.parameter_snapshots[module_id]) > self.max_snapshots_per_module:
                self.parameter_snapshots[module_id] = self.parameter_snapshots[module_id][-self.max_snapshots_per_module:]
    
    def get_assembly_statistics(self):
        """Get assembly statistics"""
        return self.assembly_tracker.get_assembly_statistics()
    
    def get_memory_usage_estimate(self):
        """Estimate memory usage of the registry"""
        import sys
        
        total_size = 0
        total_size += sys.getsizeof(self.assembly_tracker.component_hashes)
        total_size += sys.getsizeof(self.parameter_snapshots)
        total_size += sys.getsizeof(self.assembly_events)
        
        return {
            'total_bytes': total_size,
            'total_mb': total_size / (1024 * 1024),
            'components_tracked': len(self.assembly_tracker.component_hashes),
            'snapshots_stored': sum(len(snapshots) for snapshots in self.parameter_snapshots.values()),
            'assembly_events': len(self.assembly_events)
        }


# Update the existing registry to use the memory-efficient version
class AssemblyTrackingRegistry(MemoryEfficientRegistry):
    """
    Drop-in replacement for the existing registry with memory optimizations
    Includes global tracking system using hashes
    """
    
    def __init__(self):
        super().__init__()
        # Keep some backward compatibility
        self.component_registry = {}
        self.parameter_update_history = self.parameter_snapshots  # Alias for compatibility
        
        # Add global tracking attributes for compatibility
        self.global_assembly_events = self.assembly_events  # Alias for compatibility
        self.message_influence_graph = {}  # Lightweight message tracking
        self.q_learning_inheritance_graph = {}  # Lightweight Q-learning inheritance
        self.complexity_history = []  # Track system complexity over time
        
        # Global hash-based tracking
        self.global_component_hashes = set()  # All unique component hashes seen
        self.global_reuse_graph = {}  # hash -> list of modules that use it
        self.step_wise_complexity = []  # Complexity at each step
        self.generation_statistics = {}  # Per-generation assembly stats
    
    def create_parameter_snapshot(self, module, event_type="initialization"):
        """Backward compatibility method"""
        return self.create_lightweight_snapshot(module, event_type)
    
    def track_mutation_with_assembly_inheritance(self, parent_module, child_module, catalysts=None):
        """Backward compatibility method with enhanced global tracking"""
        result = self.track_mutation_with_lightweight_inheritance(parent_module, child_module, catalysts)
        
        # Add global tracking for message and Q-learning inheritance
        if catalysts:
            for catalyst in catalysts:
                # Track message influence (lightweight)
                catalyst_hash = self._generate_module_summary_hash(catalyst)
                child_hash = self._generate_module_summary_hash(child_module)
                
                if catalyst_hash not in self.message_influence_graph:
                    self.message_influence_graph[catalyst_hash] = []
                self.message_influence_graph[catalyst_hash].append(child_hash)
                
                # Track Q-learning inheritance (lightweight)
                if hasattr(catalyst, 'q_function') and hasattr(child_module, 'q_function'):
                    if catalyst_hash not in self.q_learning_inheritance_graph:
                        self.q_learning_inheritance_graph[catalyst_hash] = []
                    self.q_learning_inheritance_graph[catalyst_hash].append(child_hash)
        
        return result
    
    def _generate_module_summary_hash(self, module):
        """Generate a lightweight hash for a module based on key properties"""
        summary_parts = [
            str(getattr(module, 'id', 'unknown')),
            str(getattr(module, 'fitness', 0.0))[:8],  # Truncate for stability
            str(getattr(module, 'assembly_index', 0)),
            str(getattr(module, 'created_at', 0))
        ]
        
        # Add position hash if available
        if hasattr(module, 'position'):
            pos_hash = self.assembly_tracker._generate_layer_hash(module.position)
            summary_parts.append(pos_hash)
        
        return hashlib.md5("_".join(summary_parts).encode()).hexdigest()[:12]
    
    def track_global_component_reuse(self, module):
        """Track component reuse at the global level using hashes"""
        module_components = self.assembly_tracker.extract_components_from_module(module)
        
        for comp in module_components:
            comp_hash = comp.content_hash
            
            # Add to global hash registry
            self.global_component_hashes.add(comp_hash)
            
            # Track which modules use this component
            if comp_hash not in self.global_reuse_graph:
                self.global_reuse_graph[comp_hash] = []
            
            module_summary = {
                'module_id': module.id,
                'generation': getattr(module, 'created_at', 0),
                'component_type': comp.component_type,
                'layer_name': comp.layer_name
            }
            self.global_reuse_graph[comp_hash].append(module_summary)
    
    def step_forward(self):
        """Advance step counter with enhanced global tracking"""
        super().step_forward()
        
        # Track complexity progression
        stats = self.get_assembly_statistics()
        self.complexity_history.append({
            'step': self.step_counter,
            'total_components': stats['total_components'],
            'reused_components': stats['reused_components'],
            'reuse_rate': stats['reused_components'] / max(1, stats['total_components']),
            'unique_hashes': len(self.global_component_hashes),
            'global_reuse_instances': sum(len(instances) for instances in self.global_reuse_graph.values())
        })
        
        self.step_wise_complexity.append(stats['total_components'])
    
    def update_generation_statistics(self, generation, population):
        """Update statistics for a specific generation"""
        generation_components = set()
        generation_modules = []
        
        for module in population:
            if getattr(module, 'created_at', 0) == generation:
                generation_modules.append(module)
                # Track global component reuse for this module
                self.track_global_component_reuse(module)
                
                # Extract component hashes for this generation
                module_components = self.assembly_tracker.extract_components_from_module(module)
                for comp in module_components:
                    generation_components.add(comp.content_hash)
        
        # Calculate generation-specific statistics
        total_population = len(generation_modules)
        avg_fitness = sum(getattr(m, 'fitness', 0.0) for m in generation_modules) / max(1, total_population)
        avg_assembly_index = sum(getattr(m, 'assembly_index', 0) for m in generation_modules) / max(1, total_population)
        
        # Count reused components in this generation
        reused_count = 0
        for comp_hash in generation_components:
            if len(self.global_reuse_graph.get(comp_hash, [])) > 1:
                reused_count += 1
        
        self.generation_statistics[generation] = {
            'population_size': total_population,
            'unique_components': len(generation_components),
            'reused_components': reused_count,
            'reuse_rate': reused_count / max(1, len(generation_components)),
            'avg_fitness': avg_fitness,
            'avg_assembly_index': avg_assembly_index,
            'total_global_components': len(self.global_component_hashes)
        }
    
    def get_assembly_complexity_history(self):
        """Get the history of assembly complexity over time (backward compatibility)"""
        return self.complexity_history
    
    def get_global_assembly_statistics(self):
        """Get comprehensive global assembly statistics"""
        total_reuse_instances = sum(len(instances) for instances in self.global_reuse_graph.values())
        highly_reused_components = [
            (comp_hash, len(instances)) 
            for comp_hash, instances in self.global_reuse_graph.items() 
            if len(instances) > 2
        ]
        
        return {
            'total_unique_components': len(self.global_component_hashes),
            'total_reuse_instances': total_reuse_instances,
            'average_reuse_per_component': total_reuse_instances / max(1, len(self.global_component_hashes)),
            'highly_reused_components': len(highly_reused_components),
            'top_reused_components': sorted(highly_reused_components, key=lambda x: x[1], reverse=True)[:10],
            'message_influence_connections': len(self.message_influence_graph),
            'q_learning_inheritance_connections': len(self.q_learning_inheritance_graph),
            'generations_tracked': len(self.generation_statistics)
        }
    
    def get_component_lineage(self, component_hash):
        """Get the complete lineage of a component using hash-based tracking"""
        if component_hash not in self.global_reuse_graph:
            return None
        
        instances = self.global_reuse_graph[component_hash]
        
        lineage = {
            'component_hash': component_hash,
            'total_instances': len(instances),
            'first_appearance': min(inst['generation'] for inst in instances),
            'last_appearance': max(inst['generation'] for inst in instances),
            'generations_spanned': max(inst['generation'] for inst in instances) - min(inst['generation'] for inst in instances) + 1,
            'modules_using': [inst['module_id'] for inst in instances],
            'component_types': list(set(inst['component_type'] for inst in instances)),
            'layer_names': list(set(inst['layer_name'] for inst in instances))
        }
        
        return lineage
    
    def get_memory_usage_estimate(self):
        """Enhanced memory usage estimation including global tracking"""
        base_usage = super().get_memory_usage_estimate()
        
        import sys
        
        global_tracking_size = 0
        global_tracking_size += sys.getsizeof(self.global_component_hashes)
        global_tracking_size += sys.getsizeof(self.global_reuse_graph)
        global_tracking_size += sys.getsizeof(self.message_influence_graph)
        global_tracking_size += sys.getsizeof(self.q_learning_inheritance_graph)
        global_tracking_size += sys.getsizeof(self.generation_statistics)
        
        base_usage.update({
            'global_tracking_bytes': global_tracking_size,
            'global_tracking_mb': global_tracking_size / (1024 * 1024),
            'unique_component_hashes': len(self.global_component_hashes),
            'global_reuse_entries': len(self.global_reuse_graph),
            'message_influence_entries': len(self.message_influence_graph),
            'q_inheritance_entries': len(self.q_learning_inheritance_graph)
        })
        
        return base_usage
    
    def track_message_passing_update(self, module, senders, messages):
        """
        Track the start of a message passing update event.
        Returns a message event object for tracking parameter changes.
        """
        # Create a pre-update snapshot
        pre_update_snapshot = self.create_lightweight_snapshot(module, "pre_message_passing")
        
        # Create message event tracking object
        message_event = {
            'event_id': f"msg_{self.step_counter}_{module.id}_{len(self.assembly_events)}",
            'module_id': module.id,
            'step': self.step_counter,
            'senders': [getattr(sender, 'id', str(sender)) for sender in senders],
            'message_count': len(messages),
            'pre_update_snapshot': pre_update_snapshot,
            'post_update_snapshot': None,
            'parameter_changes': {}
        }
        
        # Store sender hashes for lightweight tracking
        sender_hashes = []
        for sender in senders:
            sender_hash = self._generate_module_summary_hash(sender)
            sender_hashes.append(sender_hash)
            
            # Track message influence in global graph
            if sender_hash not in self.message_influence_graph:
                self.message_influence_graph[sender_hash] = []
            
            target_hash = self._generate_module_summary_hash(module)
            if target_hash not in self.message_influence_graph[sender_hash]:
                self.message_influence_graph[sender_hash].append(target_hash)
        
        message_event['sender_hashes'] = sender_hashes
        
        return message_event
    
    def complete_message_passing_update(self, message_event, module):
        """
        Complete tracking of a message passing update event.
        Compares pre and post update states and records changes.
        """
        # Create post-update snapshot
        post_update_snapshot = self.create_lightweight_snapshot(module, "post_message_passing")
        message_event['post_update_snapshot'] = post_update_snapshot
        
        # Compare snapshots to detect changes
        changes = {}
        pre_hashes = message_event['pre_update_snapshot'].get('layer_hashes', {})
        post_hashes = post_update_snapshot.get('layer_hashes', {})
        
        # Detect parameter changes
        for layer_name in pre_hashes:
            if layer_name in post_hashes:
                if pre_hashes[layer_name] != post_hashes[layer_name]:
                    changes[layer_name] = {
                        'pre_hash': pre_hashes[layer_name],
                        'post_hash': post_hashes[layer_name],
                        'changed': True
                    }
                else:
                    changes[layer_name] = {
                        'pre_hash': pre_hashes[layer_name],
                        'post_hash': post_hashes[layer_name],
                        'changed': False
                    }
        
        # Check for position changes
        pre_pos_hash = message_event['pre_update_snapshot'].get('position_hash')
        post_pos_hash = post_update_snapshot.get('position_hash')
        if pre_pos_hash and post_pos_hash:
            changes['position'] = {
                'pre_hash': pre_pos_hash,
                'post_hash': post_pos_hash,
                'changed': pre_pos_hash != post_pos_hash
            }
        
        # Check for Q-function changes
        pre_q_size = message_event['pre_update_snapshot'].get('q_buffer_size', 0)
        post_q_size = post_update_snapshot.get('q_buffer_size', 0)
        if pre_q_size != post_q_size:
            changes['q_function'] = {
                'pre_buffer_size': pre_q_size,
                'post_buffer_size': post_q_size,
                'changed': True
            }
        
        message_event['parameter_changes'] = changes
        
        # Record the complete event
        self.assembly_events.append({
            'type': 'message_passing_update',
            'step': self.step_counter,
            'data': message_event
        })
        
        # Update global tracking if components were reused through message passing
        if any(change.get('changed', False) for change in changes.values()):
            self.track_global_component_reuse(module)
        
        return changes
    
# ====================================================================
# ========= Legacy ModuleComponent and ComponentRegistry =============
# ====================================================================
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