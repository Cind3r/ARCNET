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

# Import assembly tracking components from CASTLE.py
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

class ConceptModule(nn.Module):
    
    """
    ConceptModule: A neural module for concept learning with enhanced Q-learning, manifold-aware components,
    and molecular assembly tracking capabilities.
    
    This module includes: 
    - A neural network for concept representation
    - An enhanced Q-learning system with neural Q-function
    - Manifold learning components for geometric understanding
    - Messaging system for inter-module communication
    - Assembly properties for autocatalytic behavior
    - Molecular assembly tracking from CASTLE.py
    - Assembly statistics and analysis capabilities
    - Fitness and reward tracking
    - Mutation capabilities for evolutionary adaptation
    - Comprehensive message passing with Q-learning inheritance

    Args:
    - input_dim (int): Dimension of input features.
    - hidden_dim (int): Dimension of hidden layers.
    - output_dim (int): Dimension of output layer (default is 2 for binary classification).
    - created_at (int): Step at which this module was created (default is 0).
    - increase_spread (bool): Whether to increase the spread of the module's position in the manifold (default is False).
    - q_learning_method (str): Method for Q-learning ('neural' for neural Q-function, 'table' for traditional Q-table).
    - manifold_dim (int): Dimension of the manifold representation (default is None, which will be set based on input_dim).
    - molecule_size (int): Size of molecular units for assembly tracking (default is 2).
    - weight_precision (int): Precision for weight molecule tracking (default is 3).
    """
    def __init__(self, input_dim, hidden_dim, output_dim=2, created_at=0, increase_spread=False,
                  q_learning_method='neural', manifold_dim=None, molecule_size=2, weight_precision=3):
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
        self.layer_components = {
            'fc1': ModuleComponent(self.fc1.weight.data.clone()),
            'fc2': ModuleComponent(self.fc2.weight.data.clone()),
            'fc3': ModuleComponent(self.fc3.weight.data.clone())
        }
        self.assembly_pathway = [self.layer_components['fc1'], self.layer_components['fc2'], self.layer_components['fc3']]
        self.assembly_operations = [] # Track actual operations to construct this module
        self.minimal_construction_path = []  # Shortest path to construct

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
    # ================ Molecular Assembly Methods ==============
    # ==========================================================

    def weight_to_atomic_symbol(self, weight: float) -> str:
        """Convert weight magnitude to atomic symbol"""
        abs_weight = abs(weight)
        
        for symbol, (min_val, max_val) in self.atomic_symbols.items():
            if min_val <= abs_weight < max_val:
                return symbol + ('+' if weight >= 0 else '-')
        
        return 'Xx' + ('+' if weight >= 0 else '-')
    
    def tensor_to_molecular_lattice(self, weight_tensor: torch.Tensor, layer_name: str, epoch: int):# -> MolecularLattice:
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
        
        return MolecularLattice(
            molecules=molecules,
            layer_name=layer_name,
            epoch=epoch,
            lattice_id=None
        )
    
    def find_assembly_pathway(self, target_lattice: MolecularLattice):# -> Optional[List[str]]:
        """Find how target lattice can be assembled from existing components"""
        target_molecules = set()
        for row in target_lattice.molecules:
            for mol in row:
                target_molecules.add(mol)
        
        # Find existing lattices that could contribute molecules
        pathway = []
        remaining_molecules = target_molecules.copy()
        
        # Sort available lattices by how many molecules they can contribute
        available_lattices = list(self.lattice_library.values())
        lattice_contributions = []
        
        for lattice in available_lattices:
            if lattice.lattice_id == target_lattice.lattice_id:
                continue
            
            lattice_molecules = set()
            for row in lattice.molecules:
                for mol in row:
                    lattice_molecules.add(mol)
            
            contribution = len(lattice_molecules & remaining_molecules)
            if contribution > 0:
                lattice_contributions.append((lattice, contribution, lattice_molecules))
        
        # Greedily select lattices that contribute most molecules
        lattice_contributions.sort(key=lambda x: x[1], reverse=True)
        
        for lattice, contribution, lattice_molecules in lattice_contributions:
            if remaining_molecules & lattice_molecules:
                pathway.append(f"reuse_lattice({lattice.lattice_id})")
                remaining_molecules -= lattice_molecules
            
            if not remaining_molecules:
                break
        
        # Add any remaining molecules as atomic components
        for mol in remaining_molecules:
            pathway.append(f"atomic_molecule({mol.atomic_symbol})")
        
        return pathway if pathway else None

    def calculate_assembly_index(self, lattice: MolecularLattice):# -> int:
        """Calculate assembly complexity of a lattice"""
        if lattice.lattice_id in self.assembly_indices:
            return self.assembly_indices[lattice.lattice_id]
        
        # Get assembly pathway
        pathway = self.find_assembly_pathway(lattice)
        
        if not pathway:
            # New lattice with no reusable components
            unique_molecules = set()
            for row in lattice.molecules:
                for mol in row:
                    unique_molecules.add(mol)
            assembly_index = len(unique_molecules)
        else:
            # Count reused components and atomic additions
            reused_lattices = sum(1 for step in pathway if step.startswith('reuse_lattice'))
            atomic_additions = sum(1 for step in pathway if step.startswith('atomic_molecule'))
            
            # Assembly index = number of assembly steps
            assembly_index = reused_lattices + atomic_additions
        
        self.assembly_indices[lattice.lattice_id] = assembly_index
        return assembly_index
    
    def track_molecule_reuse(self, lattice: MolecularLattice):
        """Track which molecules are reused across lattices"""
        for row in lattice.molecules:
            for mol in row:
                self.molecule_reuse[mol].add(lattice.lattice_id)
    
    def track_molecular_epoch(self, epoch: int):
        """Track molecular lattice structures for this module in an epoch"""
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
                
                # Find assembly pathway
                pathway = self.find_assembly_pathway(lattice)
                if pathway:
                    self.assembly_pathways[lattice.lattice_id] = pathway
                
                # Calculate assembly index
                assembly_index = self.calculate_assembly_index(lattice)
                assembly_indices.append(assembly_index)
                
                # Track reuse
                self.track_molecule_reuse(lattice)
                self.layer_lattices[name].append(lattice.lattice_id)
                self.lattice_reuse[lattice.lattice_id].append(epoch)
                
                # Store epoch data
                epoch_data['layer_lattices'][name] = {
                    'lattice_id': lattice.lattice_id,
                    'molecular_formula': lattice.get_molecular_formula(),
                    'assembly_index': assembly_index,
                    'pathway': pathway
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
            'total_lattices_library': len(self.lattice_library)
        }
        
        self.epoch_data.append(epoch_data)
        
        # Update molecular evolution stats
        self.molecular_evolution_stats['total_molecules_discovered'] = len(self.atomic_library)
        self.molecular_evolution_stats['unique_lattices_created'] = len(self.lattice_library)
        self.molecular_evolution_stats['assembly_complexity_evolution'].append(
            epoch_data['assembly_stats']['avg_assembly_index']
        )
        
        # Calculate complexity reward
        complexity_reward = self._calculate_complexity_reward(epoch_data['assembly_stats']['avg_assembly_index'])
        self.molecular_evolution_stats['complexity_rewards'].append(complexity_reward)
        
        return epoch_data

    def _calculate_complexity_reward(self, assembly_complexity: float):# -> float:
        """Calculate reward based on assembly complexity (can be positive or negative)"""
        # Reward moderate complexity, penalize excessive complexity
        optimal_complexity = 5.0
        if assembly_complexity <= optimal_complexity:
            return assembly_complexity / optimal_complexity  # 0 to 1
        else:
            # Penalize excessive complexity
            excess = assembly_complexity - optimal_complexity
            return max(0.1, 1.0 - (excess * 0.1))  # Decrease reward for high complexity

    def compute_assembly_gradient_modifier(self, lattice: MolecularLattice, original_grad: torch.Tensor) -> torch.Tensor:
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
    # ================ Original Methods (Modified) =============
    # ==========================================================

    def update_manifold_position(self, x):
        if not self.position_initialized:
            with torch.no_grad():
                manifold_pos = self.manifold_encoder(x.mean(dim=0).unsqueeze(0)).squeeze()
                self.position.data = torch.sigmoid(manifold_pos)
            self.position_initialized = True
        else:
            #  manifold learning with fallback
            if hasattr(self, 'manifold_optimizer') and random.random() < 0.3:
                try:
                    # Use EMA instead of direct optimization to avoid instability
                    with torch.no_grad():
                        target_pos = torch.sigmoid(self.manifold_encoder(x.mean(dim=0).unsqueeze(0)).squeeze())
                        # Exponential moving average update
                        alpha = 0.1
                        self.position.data = (1 - alpha) * self.position.data + alpha * target_pos
                except Exception:
                    pass 

    def forward(self, x):
        h1 = self.act1(self.fc1(x))
        h1 = self.dropout(h1)
        h2 = self.act2(self.fc2(h1))
        out = self.fc3(h2)
        
        # Update manifold position based on input data
        self.update_manifold_position(x)
        
        self.last_input = x
        self.last_hidden = h1.detach()  # or h2.detach(), depending on what you want

        # Add processed messages
        messages = self.process_messages()
        if messages is not None:
            if messages.shape != h2.shape:
                messages = messages.expand_as(h2)
            h2 = h2 + messages

        out = self.fc3(h2)
        return out

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

    # ==========================================================
    # ================ Messaging and Q-learning =================
    # ==========================================================
    def receive_message(self, message):
        """ENHANCED message receiving with comprehensive Q-learning transfer"""
        self.message_buffer.append(message)
        
        # CRITICAL FIX: Process enhanced messages with Q-learning data
        if hasattr(message, 'q_experiences') and hasattr(message, 'reward_history'):
            if (self.q_learning_method == 'neural' and 
                self.q_function is not None):
                
                # Transfer Q-learning experiences
                for exp in message.q_experiences[:3]:  # Limit transfer
                    self.q_function.replay_buffer.append(exp)
                    if len(self.q_function.replay_buffer) > self.q_function.buffer_size:
                        self.q_function.replay_buffer.pop(0)
                
                # Learn from sender's reward patterns
                if message.reward_history:
                    avg_sender_reward = sum(message.reward_history) / len(message.reward_history)
                    # Boost own Q-values based on successful neighbor
                    if avg_sender_reward > 0.7:  # High-performing neighbor
                        self._boost_q_values(boost_factor=1.1)

    def process_messages(self):
        """Process all received messages - for ComprehensiveMessage objects"""
        if not self.message_buffer:
            return None
        
        # ROBUST message processing with multiple fallback strategies
        processed_messages = []
        
        for msg in self.message_buffer:
            try:
                if hasattr(msg, 'content'):
                    processed_messages.append(msg.content)
                elif isinstance(msg, torch.Tensor):
                    processed_messages.append(msg)
                else:
                    # Convert to tensor with standardized shape
                    tensor_msg = torch.tensor(msg, dtype=torch.float32)
                    if tensor_msg.numel() > 0:
                        processed_messages.append(tensor_msg)
            except Exception:
                # Create dummy message rather than dropping
                dummy_msg = torch.zeros(self.hidden_dim // 2)
                processed_messages.append(dummy_msg)
        
        if not processed_messages:
            self.message_buffer = []
            return torch.zeros(self.hidden_dim // 2)  # Return zeros, don't return None
        
        try:
            # ROBUST tensor combination
            target_shape = processed_messages[0].shape
            
            # Ensure all messages have compatible shapes
            reshaped_messages = []
            for msg in processed_messages:
                if msg.shape == target_shape:
                    reshaped_messages.append(msg)
                else:
                    # Reshape to target shape
                    if msg.numel() >= target_shape.numel():
                        # Truncate
                        reshaped = msg.view(-1)[:target_shape.numel()].view(target_shape)
                    else:
                        # Pad
                        padded = torch.zeros(target_shape)
                        padded.view(-1)[:msg.numel()] = msg.view(-1)
                        reshaped = padded
                    reshaped_messages.append(reshaped)
            
            # Combine messages
            if len(reshaped_messages) > 1:
                combined = torch.stack(reshaped_messages).mean(dim=0)
            else:
                combined = reshaped_messages[0]
            
            self.message_buffer = []
            return torch.sigmoid(self.gate) * combined
            
        except Exception:
            # Final fallback: return zeros
            self.message_buffer = []
            return torch.zeros(self.hidden_dim // 2)

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

    def get_standardized_state(self, population):
        """ALWAYS return consistent 4D state vector including molecular complexity"""
        try:
            fitness = float(self.fitness)
            novelty = float(compute_manifold_novelty(self, population))
            # Include molecular assembly complexity in state
            molecular_complexity = len(self.atomic_library) / 100.0  # Normalized molecular count
            assembly_complexity = float(self.assembly_index) / 10.0  # Normalized assembly index
            combined_complexity = (molecular_complexity + assembly_complexity) / 2.0
            manifold_curvature = float(self.curvature)
            
            return [fitness, novelty, combined_complexity, manifold_curvature]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]  # Fallback state

    def choose_action(self, population, available_targets, epsilon=0.1):
        """Enhanced action selection with neural Q-function"""
        if epsilon is None:
            epsilon = self.epsilon
        
        # CONSISTENT state representation
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
                # Fallback to random if Q-function fails
                action = random.choice(available_targets)
        
        self.last_state = state  # Store standardized state
        self.last_action = action.id % self.action_space_size
        return action
        
    def update_q(self, reward, population, alpha=None, gamma=None):
        """Enhanced Q-update with consistent state transitions and molecular complexity"""
        if alpha is None: alpha = self.alpha
        if gamma is None: gamma = self.gamma
        
        if self.last_state is None or self.last_action is None:
            return
        
        # Add molecular complexity reward to base reward
        complexity_reward = self._calculate_complexity_reward(
            len(self.atomic_library) + self.assembly_index
        )
        combined_reward = reward + 0.1 * complexity_reward  # Weight molecular contribution
        
        # Consistent next state
        next_state = self.get_standardized_state(population)
        
        if self.q_learning_method == 'neural' and self.q_function is not None:
            # Compute target with next state max Q-value (proper Bellman equation)
            try:
                # Sample available actions for next state
                sample_actions = list(range(0, self.action_space_size, self.action_space_size // 10))
                next_q_values = self.q_function.get_q_values_batch(next_state, sample_actions)
                next_max_q = max(next_q_values) if next_q_values else 0.0
                
                target_q = combined_reward + gamma * next_max_q
                self.q_function.update_q_network(self.last_state, self.last_action, target_q)
                
            except Exception:
                # Fallback: simple target
                print("Warning: Q-function update failed, using fallback.")
                self.q_function.update_q_network(self.last_state, self.last_action, combined_reward)
    
    def _boost_q_values(self, boost_factor=1.1):
        """Boost Q-values based on successful neighbors"""
        if (self.q_learning_method == 'neural' and 
            self.q_function is not None and 
            self.q_function.replay_buffer):
            
            # Boost recent experiences
            for i in range(len(self.q_function.replay_buffer)):
                state, action_id, target_q = self.q_function.replay_buffer[i]
                boosted_q = target_q * boost_factor
                self.q_function.replay_buffer[i] = (state, action_id, boosted_q)

    def get_q_memory_usage(self):
        """Get Q-learning memory usage"""
        if self.q_learning_method == 'neural' and self.q_function is not None:
            return self.q_function.get_memory_usage()
        else:
            # Estimate traditional Q-table memory
            return len(self.q_table) * 200 / (1024 * 1024)  # Rough estimate in MB

    # ==========================================================
    # ================ Enhanced Mutation with Molecular Tracking
    # ==========================================================

    def mutate(self, current_step=0, catalysts=None):
        """ENHANCED mutation with Q-learning inheritance and molecular assembly tracking"""
        if catalysts is None:
            catalysts = [self]

        new_mod = ConceptModule(
            self.input_dim, self.hidden_dim, self.output_dim,
            created_at=current_step, q_learning_method=self.q_learning_method,
            molecule_size=self.molecule_size, weight_precision=self.weight_precision
        )   
        
        new_mod.id = random.randint(0, int(1e6))
        new_mod.parent_id = self.id
        new_mod.load_state_dict(self.state_dict())

        # ENSURE NEW MODULE HAS ALL REQUIRED ATTRIBUTES
        if not hasattr(new_mod, 'class_predictions'):
            new_mod.class_predictions = {'0': 0, '1': 0}
        if not hasattr(new_mod, 'reward_history'):
            new_mod.reward_history = []

        # INHERIT MOLECULAR ASSEMBLY KNOWLEDGE
        new_mod.atomic_library = self.atomic_library.copy()
        new_mod.lattice_library = self.lattice_library.copy()
        new_mod.assembly_pathways = self.assembly_pathways.copy()
        new_mod.assembly_indices = self.assembly_indices.copy()
        new_mod.molecular_evolution_stats = self.molecular_evolution_stats.copy()

        # COMPREHENSIVE Q-LEARNING INHERITANCE
        if self.q_learning_method == 'neural' and self.q_function is not None:
            new_mod.q_function = CompressedQModule(
                state_dim=self.q_function.state_dim,
                action_embedding_dim=self.q_function.action_embedding_dim,
                hidden_dim=self.q_function.hidden_dim
            )
            
            # Copy parent Q-network weights
            try:
                new_mod.q_function.load_state_dict(self.q_function.state_dict())
            except Exception as e:
                print(f"Warning: Could not copy Q-function weights: {e}")
            
            # AGGREGATE Q-KNOWLEDGE FROM ALL CATALYSTS
            all_experiences = []
            catalyst_rewards = []
            
            for catalyst in catalysts:
                if (catalyst.q_learning_method == 'neural' and 
                    catalyst.q_function is not None):
                    
                    # Collect experiences from each catalyst
                    catalyst_exp = catalyst.q_function.replay_buffer
                    if catalyst_exp:
                        # Weight experiences by catalyst fitness and molecular complexity
                        molecular_bonus = len(catalyst.atomic_library) / 100.0
                        weight_factor = catalyst.fitness + molecular_bonus
                        weighted_exp = [(state, action_id, target_q * weight_factor) 
                                      for state, action_id, target_q in catalyst_exp]
                        all_experiences.extend(weighted_exp)
                    
                    # Collect reward patterns
                    catalyst_rewards.append(catalyst.reward)
            
            # INTELLIGENT EXPERIENCE SELECTION
            if all_experiences:
                # Sort by weighted Q-value and select best
                all_experiences.sort(key=lambda x: x[2], reverse=True)
                selected_exp = all_experiences[:new_mod.q_function.buffer_size//2]
                new_mod.q_function.replay_buffer = selected_exp
            
            # Initialize reward history tracking
            new_mod.reward_history = catalyst_rewards[:10]  # Keep recent rewards

        # Position and weight mutations
        with torch.no_grad():
            new_mod.position.data = self.position.data + 0.1 * torch.randn(self.manifold_dim)
            new_mod.position.data = new_mod.position.data.clamp(0, 1)
            
            for param in new_mod.parameters():
                if param.requires_grad and 'q_function' not in str(param):
                    param.add_(0.01 * torch.randn_like(param))

        # Assembly tracking with molecular components
        new_mod.assembly_steps = max([c.assembly_steps for c in catalysts]) + 1
        new_mod.compute_assembly_index() 
        new_mod.catalyzed_by = [c.id for c in catalysts]
        
        # Create new ModuleComponent for mutated weights with molecular tracking
        for name, layer in [('fc1', new_mod.fc1), ('fc2', new_mod.fc2), ('fc3', new_mod.fc3)]:
            mutated_weight = layer.weight.data.clone()
            parent_component = self.layer_components[name]
            new_component = ModuleComponent(mutated_weight, parents=[parent_component], operation='mutation')
            new_mod.layer_components[name] = new_component
        
        # Update assembly pathway for new module
        new_mod.assembly_pathway = [new_mod.layer_components['fc1'], new_mod.layer_components['fc2'], new_mod.layer_components['fc3']]

        # Track molecular evolution in the new module
        try:
            new_mod.track_molecular_epoch(current_step)
        except Exception as e:
            print(f"Warning: Could not track molecular evolution for new module: {e}")

        for c in catalysts:
            c.catalyzes.append(new_mod.id)
        
        return new_mod

    # ==========================================================
    # ================ Assembly Tracking Methods ===============
    # ==========================================================

    def record_assembly_operation(self, operation_type, parent_modules, catalysts):
        """Record operations for true Assembly Theory with molecular tracking"""
        operation = {
            'type': operation_type,  # 'mutation', 'crossover', 'catalysis'
            'inputs': [m.id for m in parent_modules],
            'catalysts': [c.id for c in catalysts],
            'step': getattr(self, 'created_at', 0),
            'molecular_state': self.get_molecular_assembly_summary()
        }
        self.assembly_operations.append(operation)
        
        # Update assembly index
        self.compute_assembly_index()

    def get_assembly_complexity_contribution(self, population):
        """Corrected system complexity following theorem with molecular components"""
        if not population:
            return 0.0
        
        total_complexity = 0.0
        total_population = len(population)
        
        for module in population:
            # A(S) = Σ e^(a_i) * (n_i - 1) / N_T with molecular complexity
            a_i = module.compute_assembly_index()  # True assembly index
            molecular_complexity = len(module.atomic_library) / 10.0  # Normalized molecular contribution
            combined_complexity = a_i + molecular_complexity
            n_i = getattr(module, 'copy_number', 1)  # Number of copies
            
            contribution = math.exp(combined_complexity) * (n_i - 1) / total_population
            total_complexity += contribution
        
        return total_complexity

    def get_assembly_complexity(self):
        """Returns a dict with per-layer, molecular, and total assembly complexity."""
        complexities = {}
        total = 0
        
        # Layer-wise complexity
        for name, comp in self.layer_components.items():
            complexity = comp.get_minimal_assembly_complexity()
            complexities[name] = complexity
            total += complexity
        
        # Molecular complexity
        molecular_complexity = len(self.atomic_library)
        lattice_complexity = len(self.lattice_library)
        
        complexities['molecular'] = molecular_complexity
        complexities['lattice'] = lattice_complexity
        complexities['total'] = total + molecular_complexity + lattice_complexity
        
        return complexities

    def compute_assembly_index(self):
        """Computes the assembly index including molecular assembly complexity"""
        total_complexity = 0
        
        # Layer component complexity
        for name, component in self.layer_components.items():
            total_complexity += component.get_minimal_assembly_complexity()
        
        # Molecular assembly complexity
        total_complexity += len(self.atomic_library) * 0.1  # Weight molecular contribution
        total_complexity += len(self.lattice_library) * 0.2  # Weight lattice contribution
        
        self.assembly_index = total_complexity
        return self.assembly_index

    # ==========================================================
    # ================ Analysis Methods ========================
    # ==========================================================

    def print_molecular_analysis(self):
        """Print comprehensive molecular assembly analysis"""
        print("=== MOLECULAR LATTICE ASSEMBLY ANALYSIS ===")
        print(f"Module ID: {self.id}")
        print(f"Total unique molecules discovered: {len(self.atomic_library)}")
        print(f"Total lattice structures: {len(self.lattice_library)}")
        print(f"Lattices with assembly pathways: {len(self.assembly_pathways)}")
        
        # Show molecular library
        molecules_by_symbol = defaultdict(list)
        for mol in self.atomic_library:
            molecules_by_symbol[mol.atomic_symbol].append(mol)
        
        print(f"\nMolecular library (by atomic symbol):")
        for symbol, molecules in sorted(molecules_by_symbol.items()):
            print(f"  {symbol}: {len(molecules)} variants")
        
        # Most complex lattices
        if self.assembly_indices:
            print(f"\n=== MOST COMPLEX LATTICES ===")
            complex_lattices = sorted(self.assembly_indices.items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
            
            for lattice_id, assembly_index in complex_lattices:
                if lattice_id in self.lattice_library:
                    lattice = self.lattice_library[lattice_id]
                    print(f"Lattice {lattice_id}: Assembly Index {assembly_index}")
                    print(f"  Layer: {lattice.layer_name}, Epoch: {lattice.epoch}")
                    print(f"  Molecular Formula: {lattice.get_molecular_formula()}")
                    
                    if lattice_id in self.assembly_pathways:
                        print(f"  Assembly Pathway: {self.assembly_pathways[lattice_id]}")
                    print()

        # Reuse analysis
        print("=== MOLECULAR REUSE ANALYSIS ===")
        highly_reused_molecules = {mol: lattices for mol, lattices in self.molecule_reuse.items()
                                if len(lattices) > 1}
        print(f"Molecules reused across lattices: {len(highly_reused_molecules)}")
        
        temporally_reused_lattices = {lid: epochs for lid, epochs in self.lattice_reuse.items()
                                    if len(epochs) > 1}
        print(f"Lattices reused across epochs: {len(temporally_reused_lattices)}")

    # ==========================================================
    # ====================== Other Methods =====================
    # ==========================================================

    def set_reward(self, reward):
        self.reward = reward
        self.position_info['reward_value'] = reward
    
    def set_novelty_score(self, novelty_score):
        self.position_info['novelty_score'] = novelty_score

    def get_best_reward(self):
        return getattr(self, 'best_reward', None)

    def hashable_op(self, op):
        # Convert lists in op to tuples for hashing
        return tuple(
            (k, tuple(v) if isinstance(v, list) else v)
            for k, v in sorted(op.items())
        )

    def get_position(self):
        return self.position.data.detach().numpy()    


def system_assembly_complexity(population):
    """
    Computes system assembly complexity including molecular components.
    population: list of ConceptModule instances
    """
    from collections import Counter
    complexities = [m.compute_assembly_index() for m in population]
    ids = [id(m) for m in population]
    counts = Counter(ids)
    N = len(population)
    return sum(math.exp(a_i) * (counts[ids[i]] - 1) / N for i, a_i in enumerate(complexities)) / N