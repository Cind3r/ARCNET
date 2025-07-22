import torch.nn as nn
import torch
import random
from sklearn.decomposition import PCA
import math
import numpy as np
from evolution.rewards import compute_manifold_novelty
from core.registry import ModuleComponent
from core.QModule import CompressedQModule
from core.registry import ComponentRegistry
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import hashlib
import matplotlib.pyplot as plt

# Assembly Theory Components
class AssemblyComponent:
    """A reusable component in assembly theory - can be a weight pattern, Q-experience, or operation"""
    def __init__(self, component_id: str, component_type: str, data, assembly_steps: int = 1, 
                 parents: List['AssemblyComponent'] = None, operation: str = 'atomic'):
        self.component_id = component_id
        self.component_type = component_type  # 'weight', 'q_experience', 'operation', 'molecular'
        self.data = data
        self.assembly_steps = assembly_steps
        self.parents = parents or []
        self.operation = operation  # 'atomic', 'mutation', 'catalysis', 'q_transfer', 'combination'
        self.created_at = 0
        self.used_count = 0
        
    def __hash__(self):
        return hash(self.component_id)
    
    def __eq__(self, other):
        return self.component_id == other.component_id
    
    def get_assembly_pathway(self): # -> List[str]:
        """Get the assembly pathway for this component"""
        if not self.parents:
            return [f"atomic({self.component_id})"]
        
        pathway = []
        for parent in self.parents:
            pathway.extend(parent.get_assembly_pathway())
        pathway.append(f"{self.operation}({self.component_id})")
        return pathway

class WeightMolecule:
    """A discrete molecular unit representing a spatial region of weights"""
    def __init__(self, atomic_weight: float, atomic_symbol: str, 
                 position: Tuple[int, int], size: Tuple[int, int]):
        self.atomic_weight = atomic_weight
        self.atomic_symbol = atomic_symbol
        self.position = position
        self.size = size
    
    def __str__(self):
        return f"{self.atomic_symbol}_{self.atomic_weight:.2f}"
    
    def __hash__(self):
        return hash((self.atomic_symbol, round(self.atomic_weight, 2)))
    
    def __eq__(self, other):
        return (self.atomic_symbol == other.atomic_symbol and 
                abs(self.atomic_weight - other.atomic_weight) < 0.01)

class MolecularLattice:
    """A 2D lattice structure of weight molecules"""
    def __init__(self, molecules: List[List[WeightMolecule]], 
                 layer_name: str, epoch: int, lattice_id: str = None):
        self.molecules = molecules
        self.layer_name = layer_name
        self.epoch = epoch
        self.lattice_id = lattice_id or self._generate_lattice_id()
    
    def _generate_lattice_id(self):
        """Generate unique ID based on molecular composition"""
        molecule_string = ""
        for row in self.molecules:
            for mol in row:
                molecule_string += str(mol)
        return hashlib.md5(molecule_string.encode()).hexdigest()[:8]
    
    def get_molecular_formula(self):
        """Get chemical-like formula for the lattice"""
        molecule_counts = defaultdict(int)
        for row in self.molecules:
            for mol in row:
                molecule_counts[mol.atomic_symbol] += 1
        
        formula = ""
        for symbol, count in sorted(molecule_counts.items()):
            if count > 1:
                formula += f"{symbol}{count}"
            else:
                formula += symbol
        return formula

class GenerationalAssemblyTracker:
    """Tracks assembly components across generations for proper Assembly Theory implementation"""
    _global_component_library = {}  # Class variable to store all components across all modules
    _global_assembly_pathways = {}  # Global pathway tracking
    _generation_components = defaultdict(set)  # Components available per generation
    
    @classmethod
    def register_component(cls, component: AssemblyComponent, generation: int):
        """Register a component in the global library"""
        cls._global_component_library[component.component_id] = component
        cls._generation_components[generation].add(component.component_id)
        component.created_at = generation
    
    @classmethod
    def get_available_components(cls, up_to_generation: int) -> Dict[str, AssemblyComponent]:
        """Get all components available up to a specific generation"""
        available = {}
        for gen in range(up_to_generation + 1):
            for comp_id in cls._generation_components[gen]:
                if comp_id in cls._global_component_library:
                    available[comp_id] = cls._global_component_library[comp_id]
        return available
    
    @classmethod
    def find_minimal_assembly_path(cls, target_components: List[AssemblyComponent], 
                                   available_components: Dict[str, AssemblyComponent]) -> Tuple[int, List[str]]:
        """Find minimal assembly pathway using dynamic programming approach"""
        if not target_components:
            return 0, []
        
        # Create target component set
        target_ids = {comp.component_id for comp in target_components}
        
        # DP approach: for each subset of targets, find minimal assembly steps
        memo = {}
        
        def min_steps(remaining_targets: frozenset) -> Tuple[int, List[str]]:
            if not remaining_targets:
                return 0, []
            
            if remaining_targets in memo:
                return memo[remaining_targets]
            
            min_cost = float('inf')
            best_path = []
            
            # Try using each available component
            for comp_id, component in available_components.items():
                if comp_id in remaining_targets:
                    # Can directly use this component
                    new_remaining = remaining_targets - {comp_id}
                    sub_cost, sub_path = min_steps(new_remaining)
                    total_cost = component.assembly_steps + sub_cost
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_path = [f"reuse({comp_id})"] + sub_path
            
            # Try atomic construction for each target
            for target_id in remaining_targets:
                new_remaining = remaining_targets - {target_id}
                sub_cost, sub_path = min_steps(new_remaining)
                total_cost = 1 + sub_cost  # 1 step for atomic construction
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_path = [f"atomic({target_id})"] + sub_path
            
            memo[remaining_targets] = (min_cost, best_path)
            return min_cost, best_path
        
        return min_steps(frozenset(target_ids))
    
    @classmethod
    def get_assembly_statistics(cls): # -> Dict:
        """Get global assembly statistics"""
        return {
            'total_components': len(cls._global_component_library),
            'generations': len(cls._generation_components),
            'components_by_type': defaultdict(int),
            'reuse_statistics': {}
        }

class ConceptModule(nn.Module):
    
    """
    ConceptModule: A neural module implementing proper Assembly Theory across generations.
    
    This module tracks:
    - Weight components and their assembly pathways
    - Q-learning experiences as reusable components  
    - Catalytic operations and their assembly contributions
    - Cross-generational component reuse
    - Minimal assembly indices using dynamic programming
    """
    def __init__(self, input_dim, hidden_dim, output_dim=2, created_at=0, increase_spread=False,
                  q_learning_method='neural', manifold_dim=None, molecule_size=2, weight_precision=3,
                  generation=0, parent_modules=None):
        super().__init__()
        
        # ==========================================================
        # ================ Neural Network Layers ===================
        # ==========================================================
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        # ==========================================================
        # ================ Basic Properties ========================
        # ==========================================================

        # Basic properties
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.id = random.randint(0, int(1e6))
        self.parent_id = None
        self.created_at = created_at
        self.generation = generation
        self.parent_modules = parent_modules or []

        # Fitness and reward
        self.fitness = 0.0
        self.reward = 0.0
        self.best_reward = 0.0
        
        # Position info
        self.position_info = {
            'step': self.created_at,
            'novelty_score': 0.0,
            'reward_value': 0.0,
        }

        # ==========================================================
        # ================ Assembly Theory Implementation ===========
        # ==========================================================
        
        # Assembly tracking with proper component library
        self.assembly_tracker = GenerationalAssemblyTracker()
        self.my_components = {}  # Components owned by this module
        self.assembly_operations = []  # Actual operations performed to create this module
        self.true_assembly_index = 0  # Calculated using Assembly Theory
        
        # Track construction pathway
        self.construction_pathway = []
        self.minimal_assembly_steps = 0
        
        # Component types we track
        self.component_types = ['weight', 'q_experience', 'operation', 'molecular', 'catalytic']

        # ==========================================================
        # ================ Molecular Assembly Tracking =============
        # ==========================================================
        
        # Assembly tracking parameters
        self.molecule_size = molecule_size
        self.weight_precision = weight_precision
        
        # Assembly structures
        self.atomic_library = set()  # All unique molecules discovered
        self.lattice_library = {}    # All lattice structures by ID
        self.assembly_pathways = {}  # How lattices are assembled
        self.assembly_indices = {}   # Assembly complexity of each lattice
        
        # Tracking
        self.layer_lattices = defaultdict(list)  # Lattices per layer over time
        self.epoch_data = []
        
        # Reuse tracking
        self.molecule_reuse = defaultdict(set)  # Which lattices use each molecule
        self.lattice_reuse = defaultdict(list)  # Temporal reuse of lattices
        
        # Define atomic symbols based on weight magnitudes
        self.atomic_symbols = {
            'Ze': (0.0, 0.01),      # Zero-like
            'Sm': (0.01, 0.1),      # Small
            'Md': (0.1, 0.5),       # Medium
            'Lg': (0.5, 1.0),       # Large
            'Xl': (1.0, 2.0),       # Extra Large
            'Xx': (2.0, float('inf'))  # Extreme
        }
        
        # Molecular statistics tracking
        self.molecular_evolution_stats = {
            'total_molecules_discovered': 0,
            'unique_lattices_created': 0,
            'assembly_complexity_evolution': [],
            'molecular_reuse_patterns': {},
            'complexity_rewards': []
        }

        # ==========================================================
        # ================ Q-Learning System =======================
        # ==========================================================

        # STANDARDIZED state representation
        self.state_dim = 4  # [fitness, novelty, assembly_complexity, manifold_curvature]
        self.action_space_size = 100000  # Consistent action space
        self.last_state = None
        self.last_action = None
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.15

        # ENHANCED Q-LEARNING SYSTEM
        self.q_learning_method = q_learning_method
        if self.q_learning_method == 'neural':
            self.q_function = CompressedQModule(
                state_dim=self.state_dim, 
                action_embedding_dim=8, 
                hidden_dim=16
            )
        else: # Use traditional Q-table (fallback)
            self.q_table = {}
            self.q_function = None
        
        # Messaging
        self.message_buffer = []
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.last_input = None
        self.last_hidden = None

        # ==========================================================
        # ================ Assembly Properties =====================
        # ==========================================================

        # Assembly properties
        self.is_autocatalytic = False
        self.assembly_steps = 0
        self.catalyzed_by = []
        self.catalyzes = []
        self.assembly_index = 0
        self.copy_number = 1
        
        # Initialize with proper assembly components
        self._initialize_assembly_components()

        # ==========================================================
        # ================ Manifold Components =====================
        # ==========================================================

        # QISRL Manifold Components
        if manifold_dim is None:
            manifold_dim = min(8, input_dim // 4) if input_dim > 8 else 3
        self.manifold_dim = manifold_dim
        self.manifold_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.manifold_dim)
        )
        self.curvature_predictor = nn.Linear(self.manifold_dim, 1)
        
        # Manifold learning parameters
        self.local_tangent_space = None
        self.curvature = 0.0
        self.position_initialized = False
        
        # Initialize position (will be updated with first data)
        self.position = torch.rand(self.manifold_dim)

        # Make position learnable
        self.position = nn.Parameter(self.position)
        self.manifold_optimizer = torch.optim.Adam(
            list(self.manifold_encoder.parameters()) + list(self.curvature_predictor.parameters()),
            lr=0.01,
            weight_decay=1e-5
        )

    # ==========================================================
    # ================ Assembly Theory Implementation ===========
    # ==========================================================
    
    def _initialize_assembly_components(self):
        """Initialize assembly components for this module"""
        # Create weight components for each layer
        for layer_name, layer in [('fc1', self.fc1), ('fc2', self.fc2), ('fc3', self.fc3)]:
            weight_hash = hashlib.md5(layer.weight.data.cpu().numpy().tobytes()).hexdigest()[:8]
            component_id = f"{self.id}_{layer_name}_{weight_hash}"
            
            # Check if we can reuse components from parent modules
            available_components = self.assembly_tracker.get_available_components(self.generation - 1)
            parent_components = []
            
            if self.parent_modules:
                for parent in self.parent_modules:
                    if hasattr(parent, 'my_components'):
                        for comp_id, comp in parent.my_components.items():
                            if comp.component_type == 'weight' and layer_name in comp_id:
                                parent_components.append(comp)
            
            # Create new component
            operation = 'mutation' if parent_components else 'atomic'
            assembly_steps = 1 if not parent_components else min(p.assembly_steps for p in parent_components) + 1
            
            component = AssemblyComponent(
                component_id=component_id,
                component_type='weight',
                data=layer.weight.data.clone(),
                assembly_steps=assembly_steps,
                parents=parent_components,
                operation=operation
            )
            
            self.my_components[component_id] = component
            self.assembly_tracker.register_component(component, self.generation)
    
    def record_assembly_operation(self, operation_type: str, components_used: List[str], 
                                  result_component: str, catalysts: List = None):
        """Record an assembly operation for proper tracking"""
        operation = {
            'type': operation_type,
            'inputs': components_used,
            'output': result_component,
            'catalysts': [c.id for c in (catalysts or [])],
            'generation': self.generation,
            'step': self.created_at,
            'module_id': self.id
        }
        self.assembly_operations.append(operation)
        
        # Create component for this operation
        op_id = f"op_{self.id}_{operation_type}_{len(self.assembly_operations)}"
        op_component = AssemblyComponent(
            component_id=op_id,
            component_type='operation',
            data=operation,
            assembly_steps=len(components_used) + 1,
            parents=[],
            operation=operation_type
        )
        
        self.my_components[op_id] = op_component
        self.assembly_tracker.register_component(op_component, self.generation)
    
    def record_q_learning_transfer(self, source_module, experiences_transferred: int):
        """Record Q-learning knowledge transfer as assembly operation"""
        q_transfer_id = f"q_transfer_{self.id}_{source_module.id}"
        
        # Create Q-learning component
        q_component = AssemblyComponent(
            component_id=q_transfer_id,
            component_type='q_experience',
            data={'experiences': experiences_transferred, 'source': source_module.id},
            assembly_steps=1,  # Q-learning transfer is considered atomic
            parents=[],
            operation='q_transfer'
        )
        
        self.my_components[q_transfer_id] = q_component
        self.assembly_tracker.register_component(q_component, self.generation)
        
        # Record the operation
        self.record_assembly_operation(
            'q_learning_transfer',
            [f"q_experiences_{source_module.id}"],
            q_transfer_id,
            catalysts=[source_module]
        )
    
    def record_catalytic_operation(self, catalyst_modules: List, operation_type: str):
        """Record catalytic operations (mutations, crossovers, etc.)"""
        catalyst_ids = [str(c.id) for c in catalyst_modules]
        catalysis_id = f"catalysis_{self.id}_{operation_type}_{hash(tuple(catalyst_ids)) % 10000}"
        
        # Create catalytic component
        catalytic_component = AssemblyComponent(
            component_id=catalysis_id,
            component_type='catalytic',
            data={'catalysts': catalyst_ids, 'operation': operation_type},
            assembly_steps=len(catalyst_modules),
            parents=[],
            operation='catalysis'
        )
        
        self.my_components[catalysis_id] = catalytic_component
        self.assembly_tracker.register_component(catalytic_component, self.generation)
        
        # Record the operation
        self.record_assembly_operation(
            'catalytic_' + operation_type,
            catalyst_ids,
            catalysis_id,
            catalysts=catalyst_modules
        )
        
        # Update catalytic relationships
        self.catalyzed_by.extend(catalyst_ids)
        for catalyst in catalyst_modules:
            if hasattr(catalyst, 'catalyzes'):
                catalyst.catalyzes.append(str(self.id))
    
    def compute_true_assembly_index(self): # -> int:
        """Compute the true assembly index using Assembly Theory principles"""
        # Get all components this module requires
        my_target_components = list(self.my_components.values())
        
        # Get available components from previous generations
        available_components = self.assembly_tracker.get_available_components(self.generation - 1)
        
        # Find minimal assembly pathway
        min_steps, pathway = self.assembly_tracker.find_minimal_assembly_path(
            my_target_components, available_components
        )
        
        self.true_assembly_index = min_steps
        self.construction_pathway = pathway
        self.minimal_assembly_steps = min_steps
        
        return self.true_assembly_index

    def get_assembly_complexity_breakdown(self): # -> Dict:
        """Get detailed breakdown of assembly complexity"""
        breakdown = {
            'true_assembly_index': self.true_assembly_index,
            'component_count': len(self.my_components),
            'reused_components': 0,
            'atomic_components': 0,
            'operations_count': len(self.assembly_operations),
            'construction_pathway': self.construction_pathway,
            'components_by_type': defaultdict(int)
        }
        
        available_components = self.assembly_tracker.get_available_components(self.generation - 1)
        
        for comp in self.my_components.values():
            breakdown['components_by_type'][comp.component_type] += 1
            if comp.component_id in available_components:
                breakdown['reused_components'] += 1
            else:
                breakdown['atomic_components'] += 1
        
        return breakdown

    # ==========================================================
    # ================ Molecular Assembly Methods ==============
    # ==========================================================

    def weight_to_atomic_symbol(self, weight: float): # -> str:
        """Convert weight magnitude to atomic symbol"""
        abs_weight = abs(weight)
        
        for symbol, (min_val, max_val) in self.atomic_symbols.items():
            if min_val <= abs_weight < max_val:
                return symbol + ('+' if weight >= 0 else '-')
        
        return 'Xx' + ('+' if weight >= 0 else '-')
    
    def tensor_to_molecular_lattice(self, weight_tensor: torch.Tensor, layer_name: str, epoch: int):
        """Convert weight tensor to molecular lattice structure"""
        if len(weight_tensor.shape) != 2:
            raise ValueError("Only 2D weight tensors supported")
        
        weights = weight_tensor.detach().cpu().numpy()
        rows, cols = weights.shape
        
        # Calculate lattice dimensions
        lattice_rows = rows // self.molecule_size
        lattice_cols = cols // self.molecule_size
        
        molecules = []
        
        for i in range(lattice_rows):
            molecule_row = []
            for j in range(lattice_cols):
                # Extract nxn region
                start_row = i * self.molecule_size
                end_row = start_row + self.molecule_size
                start_col = j * self.molecule_size
                end_col = start_col + self.molecule_size
                
                region = weights[start_row:end_row, start_col:end_col]
                
                # Calculate molecular properties
                atomic_weight = np.mean(region)
                atomic_symbol = self.weight_to_atomic_symbol(atomic_weight)
                
                molecule = WeightMolecule(
                    atomic_weight=round(atomic_weight, self.weight_precision),
                    atomic_symbol=atomic_symbol,
                    position=(i, j),
                    size=(self.molecule_size, self.molecule_size)
                )
                
                molecule_row.append(molecule)
                self.atomic_library.add(molecule)
            
            molecules.append(molecule_row)
        
        # Create molecular component
        lattice = MolecularLattice(molecules, layer_name, epoch)
        molecular_component = AssemblyComponent(
            component_id=f"molecular_{lattice.lattice_id}",
            component_type='molecular',
            data=lattice,
            assembly_steps=len(molecules) * len(molecules[0]) if molecules else 1,
            parents=[],
            operation='molecular_assembly'
        )
        
        self.my_components[molecular_component.component_id] = molecular_component
        self.assembly_tracker.register_component(molecular_component, self.generation)
        
        return lattice
    
    def track_molecular_epoch(self, epoch: int):
        """Track molecular lattice structures with proper assembly tracking"""
        epoch_data = {
            'epoch': epoch,
            'layer_lattices': {},
            'new_lattices': [],
            'assembly_stats': {}
        }
        
        assembly_indices = []
        
        # Process each layer's weights
        for name, param in [('fc1', self.fc1.weight), ('fc2', self.fc2.weight), ('fc3', self.fc3.weight)]:
            if len(param.shape) == 2:
                # Convert to molecular lattice
                lattice = self.tensor_to_molecular_lattice(param, name, epoch)
                
                # Store in library
                self.lattice_library[lattice.lattice_id] = lattice
                
                # Calculate assembly index using available components
                available_components = self.assembly_tracker.get_available_components(self.generation)
                molecular_components = [comp for comp in available_components.values() 
                                      if comp.component_type == 'molecular']
                
                assembly_index = self._calculate_molecular_assembly_index(lattice, molecular_components)
                assembly_indices.append(assembly_index)
                
                # Track reuse
                self.track_molecule_reuse(lattice)
                self.layer_lattices[name].append(lattice.lattice_id)
                self.lattice_reuse[lattice.lattice_id].append(epoch)
                
                epoch_data['layer_lattices'][name] = {
                    'lattice_id': lattice.lattice_id,
                    'molecular_formula': lattice.get_molecular_formula(),
                    'assembly_index': assembly_index,
                }
                
                epoch_data['new_lattices'].append({
                    'layer': name,
                    'lattice_id': lattice.lattice_id,
                    'molecular_formula': lattice.get_molecular_formula(),
                    'assembly_index': assembly_index
                })
        
        # Calculate epoch statistics
        epoch_data['assembly_stats'] = {
            'total_lattices': len(epoch_data['new_lattices']),
            'avg_assembly_index': np.mean(assembly_indices) if assembly_indices else 0,
            'max_assembly_index': max(assembly_indices) if assembly_indices else 0,
            'total_molecules': len(self.atomic_library),
            'total_lattices_library': len(self.lattice_library),
            'true_assembly_index': self.compute_true_assembly_index()
        }
        
        self.epoch_data.append(epoch_data)
        return epoch_data

    def _calculate_molecular_assembly_index(self, lattice: MolecularLattice, 
                                          available_molecular_components: List[AssemblyComponent]): # -> int:
        """Calculate assembly index for molecular lattice using available components"""
        target_molecules = set()
        for row in lattice.molecules:
            for mol in row:
                target_molecules.add(mol)
        
        # Check reusable molecular patterns
        reusable_molecules = set()
        for comp in available_molecular_components:
            if hasattr(comp.data, 'molecules'):
                for row in comp.data.molecules:
                    for mol in row:
                        if mol in target_molecules:
                            reusable_molecules.add(mol)
        
        # Assembly index = unique molecules needed + reused patterns
        unique_needed = len(target_molecules - reusable_molecules)
        reused_patterns = len(reusable_molecules)
        
        return unique_needed + (reused_patterns // 2)  # Reused patterns cost less

    def track_molecule_reuse(self, lattice: MolecularLattice):
        """Track which molecules are reused across lattices"""
        for row in lattice.molecules:
            for mol in row:
                self.molecule_reuse[mol].add(lattice.lattice_id)

    # ==========================================================
    # ================ Enhanced Methods ========================
    # ==========================================================

    def geodesic_interpolate(self, target_pos, alpha=0.5):
        """Interpolate along geodesic path"""
        if self.local_tangent_space is None:
            # Fallback to linear interpolation
            return alpha * self.position.data + (1 - alpha) * target_pos
        
        # Simple geodesic approximation
        diff = target_pos - self.position.data
        tangent_proj = torch.matmul(diff, self.local_tangent_space)
        
        # Exponential map approximation
        geodesic_step = alpha * tangent_proj
        new_pos = self.position.data + torch.matmul(geodesic_step, self.local_tangent_space.T)
        return new_pos.clamp(0, 1)

    def manifold_distance(self, other_pos):
        # Always compute Euclidean as baseline
        euclidean_dist = torch.norm(self.position.data - other_pos).item()
        
        if self.local_tangent_space is None:
            return euclidean_dist
        
        try:
            # Project to tangent space
            diff = other_pos - self.position.data
            tangent_proj = torch.matmul(diff, self.local_tangent_space)
            
            # Add curvature correction (theorem formula)
            tangent_norm = torch.norm(tangent_proj)
            curvature_factor = 1.0 + 0.1 * abs(self.curvature) * tangent_norm
            geodesic_dist = (tangent_norm * curvature_factor).item()
            
            # Sanity check: geodesic shouldn't be wildly different from Euclidean
            if geodesic_dist > 3 * euclidean_dist:
                print(f"Warning: Geodesic distance {geodesic_dist:.4f} is unusually high compared to Euclidean {euclidean_dist:.4f}. Using Euclidean instead.")
                return euclidean_dist
            return geodesic_dist
        except Exception:
            return euclidean_dist

    def update_local_geometry(self, neighbors):
        """Estimate local tangent space and curvature"""
        if len(neighbors) < 3:
            self.local_tangent_space = None
            self.curvature = 0.0
            return
        
        try:
            positions = torch.stack([n.position.data for n in neighbors])
            centered = positions - positions.mean(dim=0)
            
            # ROBUST SVD with regularization
            U, S, V = torch.svd(centered + 1e-6 * torch.eye(centered.shape[1]))
            
            # Only use SVD result if singular values are well-conditioned
            if len(S) >= 2 and S[1] / S[0] > 0.1:  # Condition number check
                self.local_tangent_space = V[:, :2]
                # Estimate curvature
                self.curvature = self.curvature_predictor(self.position.data).item()

            else:
                # Fallback: use PCA on positions directly
                positions_np = positions.detach().cpu().numpy()
                pca = PCA(n_components=2)
                pca.fit(positions_np)
                self.local_tangent_space = torch.tensor(pca.components_.T, dtype=torch.float32)
                self.curvature = 0.1  # Small default curvature

        except Exception:
            # Final fallback: no local geometry
            self.local_tangent_space = None
            self.curvature = 0.0

    def update_manifold_position(self, x):
        if not self.position_initialized:
            with torch.no_grad():
                manifold_pos = self.manifold_encoder(x.mean(dim=0).unsqueeze(0)).squeeze()
                self.position.data = torch.sigmoid(manifold_pos)
            self.position_initialized = True
        else:
            if hasattr(self, 'manifold_optimizer') and random.random() < 0.3:
                try:
                    with torch.no_grad():
                        target_pos = torch.sigmoid(self.manifold_encoder(x.mean(dim=0).unsqueeze(0)).squeeze())
                        alpha = 0.1
                        self.position.data = (1 - alpha) * self.position.data + alpha * target_pos
                except Exception:
                    pass 

    def forward(self, x):
        h1 = self.act1(self.fc1(x))
        h1 = self.dropout(h1)
        h2 = self.act2(self.fc2(h1))
        
        # Update manifold position based on input data
        self.update_manifold_position(x)
        
        self.last_input = x
        self.last_hidden = h1.detach()

        # Add processed messages
        messages = self.process_messages()
        if messages is not None:
            if messages.shape != h2.shape:
                messages = messages.expand_as(h2)
            h2 = h2 + messages

        out = self.fc3(h2)
        return out


    def forward_summary(self):
        """ENHANCED forward summary with Q-learning, molecular assembly, AND reward data"""
        base_summary = (self.last_hidden.mean(dim=0) if self.last_hidden is not None 
                       else torch.zeros(self.hidden_dim))
        
        # CREATE COMPREHENSIVE MESSAGE WITH MOLECULAR DATA
        if (self.q_learning_method == 'neural' and 
            self.q_function is not None):
            
            class ComprehensiveMessage:
                def __init__(self, summary, q_experiences, reward_history, fitness, 
                           manifold_position, assembly_index, molecular_summary):
                    self.data = summary
                    self.content = summary  # For backward compatibility
                    self.q_experiences = q_experiences
                    self.reward_history = reward_history
                    self.fitness = fitness
                    self.manifold_position = manifold_position
                    self.assembly_index = assembly_index
                    self.molecular_summary = molecular_summary
                    
                def __mul__(self, other):
                    return ComprehensiveMessage(
                        self.data * other, 
                        self.q_experiences, 
                        self.reward_history,
                        self.fitness,
                        self.manifold_position,
                        self.assembly_index,
                        self.molecular_summary
                    )
                
                @property
                def shape(self):
                    return self.data.shape
                
                def expand_as(self, other):
                    return ComprehensiveMessage(
                        self.data.expand_as(other), 
                        self.q_experiences,
                        self.reward_history,
                        self.fitness,
                        self.manifold_position,
                        self.assembly_index,
                        self.molecular_summary
                    )
            
            # Collect comprehensive data for message
            recent_q_exp = (self.q_function.replay_buffer[-5:] 
                          if len(self.q_function.replay_buffer) >= 5 
                          else self.q_function.replay_buffer)
            
            reward_hist = getattr(self, 'reward_history', [self.reward])
            molecular_summary = self.get_molecular_assembly_summary()
            
            return ComprehensiveMessage(
                base_summary, 
                recent_q_exp, 
                reward_hist,
                self.fitness,
                self.position.data.detach().cpu().numpy().tolist(),
                self.assembly_index,
                molecular_summary
            )
        
        return base_summary

    def receive_message(self, message):
        """Enhanced message receiving with assembly tracking"""
        self.message_buffer.append(message)
        
        # Track message passing as assembly operation
        if hasattr(message, 'sender_id'):
            self.record_assembly_operation(
                'message_passing',
                [f"message_{message.sender_id}"],
                f"received_message_{self.id}_{len(self.message_buffer)}"
            )
        
        # Process Q-learning transfers
        if hasattr(message, 'q_experiences') and hasattr(message, 'sender_module'):
            self.record_q_learning_transfer(message.sender_module, len(message.q_experiences))
            
            if (self.q_learning_method == 'neural' and self.q_function is not None):
                for exp in message.q_experiences[:3]:
                    self.q_function.replay_buffer.append(exp)
                    if len(self.q_function.replay_buffer) > self.q_function.buffer_size:
                        self.q_function.replay_buffer.pop(0)

    def process_messages(self):
        """Process messages with assembly tracking"""
        if not self.message_buffer:
            return None
        
        processed_messages = []
        
        for msg in self.message_buffer:
            try:
                if hasattr(msg, 'content'):
                    processed_messages.append(msg.content)
                elif isinstance(msg, torch.Tensor):
                    processed_messages.append(msg)
                else:
                    tensor_msg = torch.tensor(msg, dtype=torch.float32)
                    if tensor_msg.numel() > 0:
                        processed_messages.append(tensor_msg)
            except Exception:
                dummy_msg = torch.zeros(self.hidden_dim // 2)
                processed_messages.append(dummy_msg)
        
        if not processed_messages:
            self.message_buffer = []
            return torch.zeros(self.hidden_dim // 2)
        
        try:
            target_shape = processed_messages[0].shape
            reshaped_messages = []
            
            for msg in processed_messages:
                if msg.shape == target_shape:
                    reshaped_messages.append(msg)
                else:
                    if msg.numel() >= target_shape.numel():
                        reshaped = msg.view(-1)[:target_shape.numel()].view(target_shape)
                    else:
                        padded = torch.zeros(target_shape)
                        padded.view(-1)[:msg.numel()] = msg.view(-1)
                        reshaped = padded
                    reshaped_messages.append(reshaped)
            
            if len(reshaped_messages) > 1:
                combined = torch.stack(reshaped_messages).mean(dim=0)
            else:
                combined = reshaped_messages[0]
            
            self.message_buffer = []
            return torch.sigmoid(self.gate) * combined
            
        except Exception:
            self.message_buffer = []
            return torch.zeros(self.hidden_dim // 2)

    def get_standardized_state(self, population):
        """State vector including true assembly complexity"""
        try:
            fitness = float(self.fitness)
            novelty = float(compute_manifold_novelty(self, population))
            assembly_complexity = float(self.true_assembly_index) / 10.0
            manifold_curvature = float(self.curvature)
            
            return [fitness, novelty, assembly_complexity, manifold_curvature]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]

    def choose_action(self, population, available_targets, epsilon=0.1):
        """Enhanced action selection"""
        if epsilon is None:
            epsilon = self.epsilon
        
        state = self.get_standardized_state(population)
        
        if random.random() < epsilon:
            action = random.choice(available_targets)
        else:
            if self.q_learning_method == 'neural' and self.q_function is not None:
                action_ids = [t.id % self.action_space_size for t in available_targets]
                q_values = self.q_function.get_q_values_batch(state, action_ids)
                best_idx = int(np.argmax(q_values))
                action = available_targets[best_idx]
            else:
                action = random.choice(available_targets)
        
        self.last_state = state
        self.last_action = action.id % self.action_space_size
        return action
        
    def update_q(self, reward, population, alpha=None, gamma=None):
        """Q-update with assembly complexity reward"""
        if alpha is None: alpha = self.alpha
        if gamma is None: gamma = self.gamma
        
        if self.last_state is None or self.last_action is None:
            return
        
        # Add assembly complexity consideration to reward
        complexity_factor = max(0.1, 1.0 - (self.true_assembly_index / 20.0))
        combined_reward = reward * complexity_factor
        
        next_state = self.get_standardized_state(population)
        
        if self.q_learning_method == 'neural' and self.q_function is not None:
            try:
                sample_actions = list(range(0, self.action_space_size, self.action_space_size // 10))
                next_q_values = self.q_function.get_q_values_batch(next_state, sample_actions)
                next_max_q = max(next_q_values) if next_q_values else 0.0
                
                target_q = combined_reward + gamma * next_max_q
                self.q_function.update_q_network(self.last_state, self.last_action, target_q)
                
            except Exception:
                self.q_function.update_q_network(self.last_state, self.last_action, combined_reward)

    def mutate(self, current_step=0, catalysts=None):
        """Enhanced mutation with proper assembly tracking"""
        if catalysts is None:
            catalysts = [self]

        new_mod = ConceptModule(
            self.input_dim, self.hidden_dim, self.output_dim,
            created_at=current_step, q_learning_method=self.q_learning_method,
            molecule_size=self.molecule_size, weight_precision=self.weight_precision,
            generation=self.generation + 1, parent_modules=[self] + catalysts
        )   
        
        new_mod.id = random.randint(0, int(1e6))
        new_mod.parent_id = self.id
        new_mod.load_state_dict(self.state_dict())

        # Record the catalytic mutation operation
        new_mod.record_catalytic_operation(catalysts, 'mutation')
        
        # Transfer Q-learning knowledge
        if self.q_learning_method == 'neural' and self.q_function is not None:
            new_mod.q_function = CompressedQModule(
                state_dim=self.q_function.state_dim,
                action_embedding_dim=self.q_function.action_embedding_dim,
                hidden_dim=self.q_function.hidden_dim
            )
            
            try:
                new_mod.q_function.load_state_dict(self.q_function.state_dict())
            except Exception as e:
                print(f"Warning: Could not copy Q-function weights: {e}")
            
            # Aggregate Q-knowledge from catalysts
            all_experiences = []
            for catalyst in catalysts:
                if (catalyst.q_learning_method == 'neural' and 
                    catalyst.q_function is not None):
                    catalyst_exp = catalyst.q_function.replay_buffer
                    if catalyst_exp:
                        # Record Q-learning transfer
                        new_mod.record_q_learning_transfer(catalyst, len(catalyst_exp))
                        all_experiences.extend(catalyst_exp)
            
            if all_experiences:
                all_experiences.sort(key=lambda x: x[2], reverse=True)
                selected_exp = all_experiences[:new_mod.q_function.buffer_size//2]
                new_mod.q_function.replay_buffer = selected_exp

        # Position and weight mutations
        with torch.no_grad():
            new_mod.position.data = self.position.data + 0.1 * torch.randn(self.manifold_dim)
            new_mod.position.data = new_mod.position.data.clamp(0, 1)
            
            for param in new_mod.parameters():
                if param.requires_grad and 'q_function' not in str(param):
                    param.add_(0.01 * torch.randn_like(param))

        # Update assembly tracking
        new_mod.assembly_steps = max([c.assembly_steps for c in catalysts]) + 1
        new_mod.compute_true_assembly_index()
        
        # Track molecular evolution
        try:
            new_mod.track_molecular_epoch(current_step)
        except Exception as e:
            print(f"Warning: Could not track molecular evolution: {e}")

        return new_mod

    def _calculate_complexity_reward(self, assembly_complexity: float): # -> float:
        """Calculate reward based on assembly complexity (can be positive or negative)"""
        # Reward moderate complexity, penalize excessive complexity
        optimal_complexity = 5.0
        if assembly_complexity <= optimal_complexity:
            return assembly_complexity / optimal_complexity  # 0 to 1
        else:
            # Penalize excessive complexity
            excess = assembly_complexity - optimal_complexity
            return max(0.1, 1.0 - (excess * 0.1))  # Decrease reward for high complexity

    def compute_assembly_gradient_modifier(self, lattice: MolecularLattice, original_grad: torch.Tensor):# -> torch.Tensor:
        """Modify gradients based on molecular assembly complexity"""
        modifier = torch.ones_like(original_grad)
        
        # Get assembly properties
        assembly_index = self.calculate_assembly_index(lattice)
        
        # Convert lattice back to tensor positions
        for i, molecule_row in enumerate(lattice.molecules):
            for j, molecule in enumerate(molecule_row):
                # Calculate position in original tensor
                start_row = i * self.molecule_size
                end_row = start_row + self.molecule_size
                start_col = j * self.molecule_size
                end_col = start_col + self.molecule_size
                
                # Assembly-based modification
                if len(self.molecule_reuse[molecule]) > 1:
                    # Reduce updates for reused molecules (preserve learned patterns)
                    modifier[start_row:end_row, start_col:end_col] *= 0.5
                elif molecule.atomic_symbol.startswith('Ze'):
                    # Encourage sparsity for zero-like weights
                    modifier[start_row:end_row, start_col:end_col] *= 1.2
                elif assembly_index > 5:
                    # Reduce learning rate for complex assemblies
                    modifier[start_row:end_row, start_col:end_col] *= 0.8
        
        return modifier

    def get_molecular_assembly_summary(self):# -> Dict:
        """Get comprehensive summary of molecular assembly state"""
        return {
            'total_molecules': len(self.atomic_library),
            'total_lattices': len(self.lattice_library),
            'assembly_pathways': len(self.assembly_pathways),
            'molecular_reuse_count': len([mol for mol, lattices in self.molecule_reuse.items() if len(lattices) > 1]),
            'average_assembly_complexity': np.mean(list(self.assembly_indices.values())) if self.assembly_indices else 0,
            'evolution_stats': self.molecular_evolution_stats
        }



    # ==========================================================
    # ================ Analysis Methods ========================
    # ==========================================================

    def print_assembly_analysis(self):
        """Print comprehensive assembly analysis with proper Assembly Theory"""
        print("=== ASSEMBLY THEORY ANALYSIS ===")
        print(f"Module ID: {self.id}, Generation: {self.generation}")
        print(f"True Assembly Index: {self.true_assembly_index}")
        print(f"Assembly Steps: {self.minimal_assembly_steps}")
        
        breakdown = self.get_assembly_complexity_breakdown()
        print(f"Total Components: {breakdown['component_count']}")
        print(f"Reused Components: {breakdown['reused_components']}")
        print(f"Atomic Components: {breakdown['atomic_components']}")
        print(f"Operations Count: {breakdown['operations_count']}")
        
        print("\nComponents by Type:")
        for comp_type, count in breakdown['components_by_type'].items():
            print(f"  {comp_type}: {count}")
        
        print(f"\nConstruction Pathway ({len(self.construction_pathway)} steps):")
        for i, step in enumerate(self.construction_pathway[:10]):  # Show first 10 steps
            print(f"  {i+1}. {step}")
        if len(self.construction_pathway) > 10:
            print(f"  ... and {len(self.construction_pathway) - 10} more steps")
        
        print("\nAssembly Operations:")
        for op in self.assembly_operations[-5:]:  # Show last 5 operations
            print(f"  {op['type']}: {op['inputs']} -> {op['output']}")

    def get_assembly_statistics(self): # -> Dict:
        """Get comprehensive assembly statistics"""
        return {
            'true_assembly_index': self.true_assembly_index,
            'generation': self.generation,
            'component_breakdown': self.get_assembly_complexity_breakdown(),
            'global_stats': self.assembly_tracker.get_assembly_statistics(),
            'molecular_stats': self.molecular_evolution_stats,
            'construction_pathway_length': len(self.construction_pathway),
            'operations_performed': len(self.assembly_operations)
        }

        # ==========================================================
    # ================ Missing Methods - Updated for Assembly Theory ========================
    # ==========================================================
    
    def _boost_q_values(self, boost_factor=1.1):
        """Boost Q-values based on successful neighbors - Updated for Assembly Theory"""
        if self.q_learning_method == 'neural' and self.q_function is not None:
            try:
                # Boost recent successful experiences in replay buffer
                for i in range(-min(5, len(self.q_function.replay_buffer)), 0):
                    state, action_id, target_q = self.q_function.replay_buffer[i]
                    if target_q > 0:  # Only boost positive experiences
                        boosted_q = target_q * boost_factor
                        self.q_function.replay_buffer[i] = (state, action_id, boosted_q)
                
                # Record this as an assembly operation
                self.record_assembly_operation(
                    'q_value_boost',
                    [f"q_experiences_{self.id}"],
                    f"boosted_q_{self.id}_{len(self.assembly_operations)}"
                )
                
                print(f"Module {self.id}: Boosted Q-values by factor {boost_factor}")
            except Exception as e:
                print(f"Warning: Q-value boost failed for module {self.id}: {e}")
    
    def get_q_memory_usage(self):
        
        if self.q_learning_method == 'neural' and self.q_function is not None:
            # Calculate actual memory usage in MB
            q_network_params = sum(p.numel() for p in self.q_function.parameters())
            param_memory_mb = (q_network_params * 4) / (1024 * 1024)  # 4 bytes per float32, convert to MB
            
            # Estimate replay buffer memory (approximate)
            buffer_memory_mb = len(self.q_function.replay_buffer) * 0.001  # Rough estimate
            
            total_memory_mb = param_memory_mb + buffer_memory_mb
            
            memory_stats = {
                'replay_buffer_size': len(self.q_function.replay_buffer),
                'buffer_capacity': self.q_function.buffer_size,
                'memory_utilization': len(self.q_function.replay_buffer) / self.q_function.buffer_size,
                'q_network_parameters': q_network_params,
                'total_q_experiences_stored': len(self.q_function.replay_buffer),
                'assembly_related_experiences': 0,  # Count assembly-related experiences
                'total_memory': total_memory_mb  # Add this for trainer compatibility
            }
            
            # Count experiences from assembly operations
            for exp in self.q_function.replay_buffer:
                state, action_id, target_q = exp
                # Check if this experience relates to assembly operations
                if len(state) >= 3 and state[2] > 0:  # Assembly complexity in state
                    memory_stats['assembly_related_experiences'] += 1
            
            return memory_stats
        else:
            return {
                'replay_buffer_size': 0,
                'buffer_capacity': 0,
                'memory_utilization': 0.0,
                'q_network_parameters': 0,
                'total_q_experiences_stored': 0,
                'assembly_related_experiences': 0,
                'total_memory': 0.0  # Add this for trainer compatibility
            }
    
    def set_reward(self, reward):
        """Set reward value with Assembly Theory complexity consideration"""
        self.reward = reward
        self.position_info['reward_value'] = reward
        
        # Update best reward tracking
        if reward > self.best_reward:
            self.best_reward = reward
        
        # Initialize reward history if not exists
        if not hasattr(self, 'reward_history'):
            self.reward_history = []
        
        # Add assembly complexity factor to reward
        complexity_factor = self._calculate_complexity_reward(self.true_assembly_index)
        adjusted_reward = reward * complexity_factor
        
        self.reward_history.append({
            'raw_reward': reward,
            'adjusted_reward': adjusted_reward,
            'complexity_factor': complexity_factor,
            'assembly_index': self.true_assembly_index,
            'step': self.created_at
        })
        
        # Keep only recent reward history
        if len(self.reward_history) > 20:
            self.reward_history = self.reward_history[-20:]
        
        # Record reward setting as assembly operation
        reward_component_id = f"reward_{self.id}_{len(self.reward_history)}"
        reward_component = AssemblyComponent(
            component_id=reward_component_id,
            component_type='operation',
            data={'reward': reward, 'adjusted_reward': adjusted_reward, 'complexity_factor': complexity_factor},
            assembly_steps=1,
            parents=[],
            operation='reward_assignment'
        )
        
        self.my_components[reward_component_id] = reward_component
        self.assembly_tracker.register_component(reward_component, self.generation)
    
    def set_novelty_score(self, novelty_score):
        """Set novelty score with Assembly Theory integration"""
        self.position_info['novelty_score'] = novelty_score
        
        # Initialize novelty history if not exists
        if not hasattr(self, 'novelty_history'):
            self.novelty_history = []
        
        # Assembly-aware novelty calculation
        # Higher assembly complexity can contribute to novelty
        assembly_novelty_bonus = min(0.2, self.true_assembly_index / 50.0)
        enhanced_novelty = novelty_score + assembly_novelty_bonus
        
        self.novelty_history.append({
            'raw_novelty': novelty_score,
            'enhanced_novelty': enhanced_novelty,
            'assembly_bonus': assembly_novelty_bonus,
            'assembly_index': self.true_assembly_index,
            'step': self.created_at
        })
        
        # Keep only recent novelty history
        if len(self.novelty_history) > 15:
            self.novelty_history = self.novelty_history[-15:]
        
        # Record novelty as assembly operation
        novelty_component_id = f"novelty_{self.id}_{len(self.novelty_history)}"
        novelty_component = AssemblyComponent(
            component_id=novelty_component_id,
            component_type='operation',
            data={'novelty': novelty_score, 'enhanced_novelty': enhanced_novelty, 'assembly_bonus': assembly_novelty_bonus},
            assembly_steps=1,
            parents=[],
            operation='novelty_assignment'
        )
        
        self.my_components[novelty_component_id] = novelty_component
        self.assembly_tracker.register_component(novelty_component, self.generation)
    
    def get_best_reward(self):
        """Get best reward achieved with Assembly Theory context"""
        if hasattr(self, 'reward_history') and self.reward_history:
            # Find best reward considering both raw and adjusted rewards
            best_raw = max(entry['raw_reward'] for entry in self.reward_history)
            best_adjusted = max(entry['adjusted_reward'] for entry in self.reward_history)
            best_entry = max(self.reward_history, key=lambda x: x['adjusted_reward'])
            
            return {
                'best_raw_reward': best_raw,
                'best_adjusted_reward': best_adjusted,
                'best_reward_context': best_entry,
                'total_rewards_received': len(self.reward_history),
                'current_assembly_index': self.true_assembly_index
            }
        else:
            return {
                'best_raw_reward': self.best_reward,
                'best_adjusted_reward': self.best_reward,
                'best_reward_context': None,
                'total_rewards_received': 0,
                'current_assembly_index': self.true_assembly_index
            }
    
    def hashable_op(self, op):
        """Convert operations to hashable format for Assembly Theory tracking"""
        if isinstance(op, dict):
            # Handle assembly operations
            if 'type' in op and 'inputs' in op:
                op_type = op['type']
                inputs = tuple(sorted(op['inputs'])) if isinstance(op['inputs'], list) else op['inputs']
                output = op.get('output', 'unknown')
                catalysts = tuple(sorted(op.get('catalysts', [])))
                return f"{op_type}:{inputs}→{output}:{catalysts}"
            else:
                # General dict hashing
                items = tuple(sorted(op.items()))
                return str(hash(items))
        elif isinstance(op, (list, tuple)):
            return str(hash(tuple(str(item) for item in op)))
        elif hasattr(op, 'component_id'):
            # AssemblyComponent
            return f"comp:{op.component_id}:{op.component_type}:{op.operation}"
        elif hasattr(op, 'lattice_id'):
            # MolecularLattice
            return f"lattice:{op.lattice_id}:{op.layer_name}"
        else:
            return str(hash(str(op)))
    
    def get_position(self):
        """Get position in data space with Assembly Theory context"""
        position_data = self.position.data.detach().cpu().numpy()
        
        return {
            'manifold_position': position_data.tolist(),
            'manifold_dimension': self.manifold_dim,
            'position_initialized': self.position_initialized,
            'curvature': self.curvature,
            'generation': self.generation,
            'assembly_index': self.true_assembly_index,
            'position_history': getattr(self, 'position_history', []),
            'local_geometry_available': self.local_tangent_space is not None
        }
    
    def get_assembly_complexity_contribution(self, population):
        """Calculate module's contribution to system complexity using Assembly Theory"""
        if not population:
            return 0.0
        
        # Calculate this module's relative contribution to system assembly complexity
        total_system_assembly = sum(getattr(mod, 'true_assembly_index', 1) for mod in population)
        my_contribution = self.true_assembly_index / total_system_assembly if total_system_assembly > 0 else 0.0
        
        # Factor in reusable components
        available_components = self.assembly_tracker.get_available_components(self.generation)
        reusable_count = sum(1 for comp in self.my_components.values() 
                           if comp.component_id in available_components)
        
        # Calculate reusability factor
        total_components = len(self.my_components)
        reusability_factor = reusable_count / total_components if total_components > 0 else 0.0
        
        # Enhanced contribution considering Assembly Theory
        complexity_contribution = {
            'raw_contribution': my_contribution,
            'assembly_index': self.true_assembly_index,
            'total_system_assembly': total_system_assembly,
            'reusable_components': reusable_count,
            'total_components': total_components,
            'reusability_factor': reusability_factor,
            'effective_contribution': my_contribution * (1 + reusability_factor),
            'complexity_efficiency': self.true_assembly_index / max(1, total_components),
            'generation_factor': 1.0 / max(1, self.generation),  # Earlier generations are more fundamental
            'weighted_contribution': my_contribution * (1 + reusability_factor) * (1.0 / max(1, self.generation))
        }
        
        # Record complexity analysis as assembly operation
        analysis_id = f"complexity_analysis_{self.id}_{len(self.assembly_operations)}"
        analysis_component = AssemblyComponent(
            component_id=analysis_id,
            component_type='operation',
            data=complexity_contribution,
            assembly_steps=1,
            parents=[],
            operation='complexity_analysis'
        )
        
        self.my_components[analysis_id] = analysis_component
        self.assembly_tracker.register_component(analysis_component, self.generation)
        
        return complexity_contribution
    
    def build_lineage_graph(self):
        """Build lineage graph with Assembly Theory component tracking"""
        lineage_data = {
            'module_id': self.id,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'parent_modules': [p.id for p in self.parent_modules] if self.parent_modules else [],
            'created_at': self.created_at,
            'assembly_lineage': [],
            'component_inheritance': {},
            'catalytic_relationships': {
                'catalyzed_by': self.catalyzed_by,
                'catalyzes': self.catalyzes
            }
        }
        
        # Track assembly component lineage
        for comp_id, component in self.my_components.items():
            component_lineage = {
                'component_id': comp_id,
                'component_type': component.component_type,
                'assembly_steps': component.assembly_steps,
                'operation': component.operation,
                'parents': [p.component_id for p in component.parents],
                'assembly_pathway': component.get_assembly_pathway()
            }
            lineage_data['assembly_lineage'].append(component_lineage)
        
        # Track component inheritance from parents
        if self.parent_modules:
            for parent in self.parent_modules:
                if hasattr(parent, 'my_components'):
                    inherited_components = []
                    for comp_id, comp in parent.my_components.items():
                        # Check if we inherited or reused this component
                        if any(my_comp.component_id == comp_id or comp in my_comp.parents 
                              for my_comp in self.my_components.values()):
                            inherited_components.append({
                                'component_id': comp_id,
                                'component_type': comp.component_type,
                                'inherited_from': parent.id
                            })
                    
                    if inherited_components:
                        lineage_data['component_inheritance'][parent.id] = inherited_components
        
        # Add construction pathway
        lineage_data['construction_pathway'] = self.construction_pathway
        lineage_data['minimal_assembly_steps'] = self.minimal_assembly_steps
        lineage_data['true_assembly_index'] = self.true_assembly_index
        
        return lineage_data
    
    def visualize_lattice_evolution(self, save_path=None, show_plot=True):
        """Visualize molecular lattice evolution over time"""
        if not self.epoch_data:
            print("No epoch data available for visualization")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Extract data for visualization
            epochs = [data['epoch'] for data in self.epoch_data]
            assembly_indices = [data['assembly_stats']['avg_assembly_index'] for data in self.epoch_data]
            total_molecules = [data['assembly_stats']['total_molecules'] for data in self.epoch_data]
            total_lattices = [data['assembly_stats']['total_lattices_library'] for data in self.epoch_data]
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Molecular Lattice Evolution - Module {self.id}', fontsize=16)
            
            # Plot 1: Assembly Index Evolution
            ax1.plot(epochs, assembly_indices, 'b-o', linewidth=2, markersize=4)
            ax1.set_title('Average Assembly Index Over Time')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Assembly Index')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Molecular Discovery
            ax2.plot(epochs, total_molecules, 'g-s', linewidth=2, markersize=4)
            ax2.set_title('Total Unique Molecules Discovered')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Molecule Count')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Lattice Library Growth
            ax3.plot(epochs, total_lattices, 'r-^', linewidth=2, markersize=4)
            ax3.set_title('Lattice Library Growth')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Total Lattices')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Assembly Efficiency (molecules reused)
            if len(epochs) > 1:
                reuse_efficiency = []
                for data in self.epoch_data:
                    total_mols = data['assembly_stats']['total_molecules']
                    total_lats = data['assembly_stats']['total_lattices_library']
                    efficiency = total_mols / max(1, total_lats)  # Molecules per lattice
                    reuse_efficiency.append(efficiency)
                
                ax4.plot(epochs, reuse_efficiency, 'm-d', linewidth=2, markersize=4)
                ax4.set_title('Molecular Reuse Efficiency')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Molecules per Lattice')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Insufficient data\nfor efficiency plot', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Molecular Reuse Efficiency')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Lattice evolution plot saved to {save_path}")
            
            if show_plot:
                plt.show()
            
            return fig
            
        except ImportError:
            print("Matplotlib not available. Cannot create visualization.")
            return None
        except Exception as e:
            print(f"Error creating lattice evolution visualization: {e}")
            return None
    
    def print_molecular_analysis(self):
        """Print comprehensive molecular assembly analysis"""
        print("=== MOLECULAR ASSEMBLY ANALYSIS ===")
        print(f"Module ID: {self.id}, Generation: {self.generation}")
        print(f"Molecule Size: {self.molecule_size}x{self.molecule_size}")
        print(f"Weight Precision: {self.weight_precision}")
        
        print(f"\nMolecular Library:")
        print(f"  Total Unique Molecules: {len(self.atomic_library)}")
        print(f"  Total Lattice Structures: {len(self.lattice_library)}")
        
        if self.atomic_library:
            # Analyze molecular composition
            symbol_counts = defaultdict(int)
            for molecule in self.atomic_library:
                base_symbol = molecule.atomic_symbol[:-1]  # Remove +/- suffix
                symbol_counts[base_symbol] += 1
            
            print(f"  Molecular Composition:")
            for symbol, count in sorted(symbol_counts.items()):
                percentage = (count / len(self.atomic_library)) * 100
                print(f"    {symbol}: {count} ({percentage:.1f}%)")
        
        print(f"\nMolecular Reuse Patterns:")
        reused_molecules = [mol for mol, lattices in self.molecule_reuse.items() if len(lattices) > 1]
        print(f"  Reused Molecules: {len(reused_molecules)}")
        print(f"  Reuse Efficiency: {len(reused_molecules) / max(1, len(self.atomic_library)) * 100:.1f}%")
        
        if reused_molecules:
            print(f"  Top Reused Molecules:")
            sorted_reuse = sorted([(mol, len(lattices)) for mol, lattices in self.molecule_reuse.items()], 
                                key=lambda x: x[1], reverse=True)
            for mol, reuse_count in sorted_reuse[:5]:
                print(f"    {mol}: used in {reuse_count} lattices")
        
        print(f"\nAssembly Evolution:")
        if self.epoch_data:
            latest_stats = self.epoch_data[-1]['assembly_stats']
            print(f"  Latest Average Assembly Index: {latest_stats['avg_assembly_index']:.2f}")
            print(f"  Latest Max Assembly Index: {latest_stats['max_assembly_index']}")
            print(f"  True Assembly Index: {latest_stats['true_assembly_index']}")
            
            if len(self.epoch_data) > 1:
                improvement = (self.epoch_data[-1]['assembly_stats']['avg_assembly_index'] - 
                             self.epoch_data[0]['assembly_stats']['avg_assembly_index'])
                print(f"  Assembly Index Improvement: {improvement:.2f}")
        
        print(f"\nMolecular Evolution Stats:")
        stats = self.molecular_evolution_stats
        print(f"  Total Molecules Discovered: {stats['total_molecules_discovered']}")
        print(f"  Unique Lattices Created: {stats['unique_lattices_created']}")
        if stats['complexity_rewards']:
            print(f"  Average Complexity Reward: {np.mean(stats['complexity_rewards']):.3f}")
        
        print(f"\nLayer-wise Lattice Distribution:")
        for layer_name, lattice_ids in self.layer_lattices.items():
            print(f"  {layer_name}: {len(lattice_ids)} lattices")
            if lattice_ids:
                latest_lattice = self.lattice_library.get(lattice_ids[-1])
                if latest_lattice:
                    print(f"    Latest Formula: {latest_lattice.get_molecular_formula()}")

    # ==========================================================
    # ================ Enhanced Missing Method Integration =====
    # ==========================================================
    
    def update_local_geometry_with_assembly(self, neighbors):
        """Update local geometry considering assembly relationships"""
        # Call original geometry update
        self.update_local_geometry(neighbors)
        
        # Add assembly-based geometric adjustments
        if self.local_tangent_space is not None:
            # Adjust curvature based on assembly complexity
            assembly_curvature_factor = min(0.1, self.true_assembly_index / 100.0)
            self.curvature += assembly_curvature_factor
            
            # Record geometry update as assembly operation
            geo_id = f"geometry_update_{self.id}_{len(self.assembly_operations)}"
            geo_component = AssemblyComponent(
                component_id=geo_id,
                component_type='operation',
                data={'curvature': self.curvature, 'assembly_factor': assembly_curvature_factor},
                assembly_steps=1,
                parents=[],
                operation='geometry_update'
            )
            
            self.my_components[geo_id] = geo_component
            self.assembly_tracker.register_component(geo_component, self.generation)


    # Legacy compatibility methods
    def compute_assembly_index(self):
        """Legacy method - now calls true assembly index calculation"""
        return self.compute_true_assembly_index()
    
    def get_assembly_complexity(self):
        """Legacy method - returns breakdown"""
        return self.get_assembly_complexity_breakdown()

    # Other legacy methods...
    def set_reward(self, reward):
        self.reward = reward
        self.position_info['reward_value'] = reward
    
    def set_novelty_score(self, novelty_score):
        self.position_info['novelty_score'] = novelty_score

    def get_best_reward(self):
        return getattr(self, 'best_reward', None)

    def get_position(self):
        return self.position.data.detach().numpy()



def system_assembly_complexity(population):
    """
    Computes system assembly complexity using proper Assembly Theory.
    Now uses true assembly indices.
    """
    if not population:
        return 0.0
    
    from collections import Counter
    
    # Calculate true assembly complexities
    complexities = []
    for module in population:
        if hasattr(module, 'compute_true_assembly_index'):
            complexity = module.compute_true_assembly_index()
        else:
            complexity = getattr(module, 'true_assembly_index', 1)
        complexities.append(complexity)
    
    # Count module types (simplified by ID for now)
    ids = [m.id for m in population]
    counts = Counter(ids)
    N = len(population)
    
    # System complexity formula from Assembly Theory
    system_complexity = 0.0
    for i, module in enumerate(population):
        a_i = complexities[i]
        n_i = counts[module.id]
        contribution = math.exp(a_i) * (n_i - 1) / N
        system_complexity += contribution
    
    return system_complexity / N