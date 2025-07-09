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

class ConceptModule(nn.Module):
    
    """
    ConceptModule: A neural module for concept learning with enhanced Q-learning and manifold-aware components.
    This module includes: 
    - A neural network for concept representation
    - An enhanced Q-learning system with neural Q-function
    - Manifold learning components for geometric understanding
    - Messaging system for inter-module communication
    - Assembly properties for autocatalytic behavior
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

    Attributes:
    - fc1, fc2, fc3: Fully connected layers for the neural network.
    - act1, act2: Activation functions (ReLU).
    - dropout: Dropout layer for regularization.
    - q_function: Neural Q-function for enhanced Q-learning.
    - q_table: Traditional Q-table for fallback Q-learning.
    - position: Learnable parameter representing the module's position in the manifold.
    - manifold_encoder: Encoder for mapping input features to manifold representation.
    - curvature_predictor: Predictor for estimating curvature in the manifold.
    - local_tangent_space: Estimated local tangent space for geodesic computations.
    - curvature: Estimated curvature of the manifold.
    - message_buffer: Buffer for incoming messages from other modules.
    - gate: Learnable parameter for gating messages.
    - last_input, last_hidden: Store the last input and hidden state for message processing.
    - is_autocatalytic: Flag indicating if the module is autocatalytic.
    - assembly_steps: Number of assembly steps this module has undergone.
    - catalyzed_by: List of module IDs that catalyzed this module.
    - catalyzes: List of module IDs that this module catalyzes.
    - assembly_index: Index of this module in the assembly process.
    - copy_number: Number of copies of this module in the population.
    - assembly_pathway: Pathway of assembly steps leading to this module.
    - position_info: Dictionary containing position-related information (step, novelty score, reward value).

    Methods:
    - forward: Forward pass through the neural network.
    - update_manifold_position: Updates the module's position in the manifold based on input data.
    - geodesic_interpolate: Interpolates along the geodesic path in the manifold.
    - manifold_distance: Computes the geodesic distance to another position in the manifold.
    - receive_message: Receives and processes messages from other modules.
    - process_messages: Processes all received messages and returns a combined tensor.
    - forward_summary: Generates a summary of the module's state for message passing.
    - choose_action: Selects an action based on the current state and available targets using Q-learning.
    - update_q: Updates the Q-values based on the received reward and next state.
    - mutate: Mutates the module by perturbing weights and position, inheriting Q-learning experiences from catalysts.
    - compute_assembly_index: Computes the assembly index for this module.
    - get_assembly_complexity_contribution: Calculates this module's contribution to system complexity.
    - build_lineage_graph: Builds a lineage graph of the module's ancestry.
    - set_reward: Sets the reward value for this module.
    - set_novelty_score: Sets the novelty score for this module.
    - get_best_reward: Returns the best reward achieved by this module.
    - get_q_memory_usage: Returns the memory usage of the Q-learning system (neural or table).

    """
    def __init__(self, input_dim, hidden_dim, output_dim=2, created_at=0, increase_spread=False,
                  q_learning_method='neural', manifold_dim=None):
        super().__init__()
        

        # ==========================================================
        # ================ Neural Network Layers ===================
        # ==========================================================

        # Originally set as nn.Sequential, but now using individual layers for flexibility
        # potentially add more layers or change activation functions later (TanH works well
        # for biological models, Sigmoid for binary classification, etc.) 
        
        # This needs to be made more flexible in the future
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

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

        

    # Initialize the module's position
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

    # NOTE: Consider removal of geodesic as complexity becomes too large and computationally expensive
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

    # =========================================================
    # ================ Messaging and Q-learning =================
    # =========================================================
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
        """
        Process all received messages - for ComprehensiveMessage objects
        """
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
        """ENHANCED forward summary with Q-learning AND reward data"""
        base_summary = (self.last_hidden.mean(dim=0) if self.last_hidden is not None 
                       else torch.zeros(self.hidden_dim))
        
        # CREATE COMPREHENSIVE MESSAGE
        if (self.q_learning_method == 'neural' and 
            self.q_function is not None):
            
            class ComprehensiveMessage:
                def __init__(self, summary, q_experiences, reward_history, fitness, 
                           manifold_position, assembly_index):
                    self.data = summary
                    self.q_experiences = q_experiences
                    self.reward_history = reward_history
                    self.fitness = fitness
                    self.manifold_position = manifold_position
                    self.assembly_index = assembly_index
                    
                def __mul__(self, other):
                    return ComprehensiveMessage(
                        self.data * other, 
                        self.q_experiences, 
                        self.reward_history,
                        self.fitness,
                        self.manifold_position,
                        self.assembly_index
                    )
                
                def shape(self):
                    return self.data.shape
                
                def expand_as(self, other):
                    return ComprehensiveMessage(
                        self.data.expand_as(other), 
                        self.q_experiences,
                        self.reward_history,
                        self.fitness,
                        self.manifold_position,
                        self.assembly_index
                    )
            
            # Collect comprehensive data for message
            recent_q_exp = (self.q_function.replay_buffer[-5:] 
                          if len(self.q_function.replay_buffer) >= 5 
                          else self.q_function.replay_buffer)
            
            reward_hist = getattr(self, 'reward_history', [self.reward])
            
            return ComprehensiveMessage(
                base_summary, 
                recent_q_exp, 
                reward_hist,
                self.fitness,
                self.position.data.detach().cpu().numpy().tolist(),
                self.assembly_index
            )
        
        return base_summary

    def get_standardized_state(self, population):
        """ALWAYS return consistent 4D state vector"""
        try:
            fitness = float(self.fitness)
            novelty = float(compute_manifold_novelty(self, population))
            assembly_complexity = float(self.assembly_index) / 10.0  # Normalized
            manifold_curvature = float(self.curvature)
            
            return [fitness, novelty, assembly_complexity, manifold_curvature]
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
        """Enhanced Q-update with consistent state transitions"""
        if alpha is None: alpha = self.alpha
        if gamma is None: gamma = self.gamma
        
        if self.last_state is None or self.last_action is None:
            return
        
        # Consistent next state
        next_state = self.get_standardized_state(population)
        
        if self.q_learning_method == 'neural' and self.q_function is not None:
            # Compute target with next state max Q-value (proper Bellman equation)
            try:
                # Sample available actions for next state
                sample_actions = list(range(0, self.action_space_size, self.action_space_size // 10))
                next_q_values = self.q_function.get_q_values_batch(next_state, sample_actions)
                next_max_q = max(next_q_values) if next_q_values else 0.0
                
                target_q = reward + gamma * next_max_q
                self.q_function.update_q_network(self.last_state, self.last_action, target_q)
                
            except Exception:
                # Fallback: simple target
                print("Warning: Q-function update failed, using fallback.")
                self.q_function.update_q_network(self.last_state, self.last_action, reward)
    

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
    # ================ Neural Network Methods ==================
    # ==========================================================
    # mutate function to perturb weights and position
    def mutate(self, current_step=0, catalysts=None):
        """ENHANCED mutation with Q-learning inheritance from ALL catalysts"""
        if catalysts is None:
            catalysts = [self]

        new_mod = ConceptModule(
            self.input_dim, self.hidden_dim, self.output_dim,
            created_at=current_step, q_learning_method=self.q_learning_method
        )   
        
        new_mod.id = random.randint(0, int(1e6))
        new_mod.parent_id = self.id
        new_mod.load_state_dict(self.state_dict())

        # ENSURE NEW MODULE HAS ALL REQUIRED ATTRIBUTES
        if not hasattr(new_mod, 'class_predictions'):
            new_mod.class_predictions = {'0': 0, '1': 0}
        if not hasattr(new_mod, 'reward_history'):
            new_mod.reward_history = []

        # COMPREHENSIVE Q-LEARNING INHERITANCE
        if self.q_learning_method == 'neural' and self.q_function is not None:
            new_mod.q_function = CompressedQModule(
                state_dim=self.q_function.state_dim,
                action_embedding_dim=self.q_function.action_embedding_dim,
                hidden_dim=self.q_function.hidden_dim # Same hidden dimension as parent
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
                        # Weight experiences by catalyst fitness
                        weighted_exp = [(state, action_id, target_q * catalyst.fitness) 
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
              
                #print(f"Child {new_mod.id} inherited {len(selected_exp)} weighted Q-experiences from {len(catalysts)} catalysts")
            
            # Initialize reward history tracking
            new_mod.reward_history = catalyst_rewards[:10]  # Keep recent rewards

        # Position and weight mutations
        with torch.no_grad():
            new_mod.position.data = self.position.data + 0.1 * torch.randn(self.manifold_dim)
            new_mod.position.data = new_mod.position.data.clamp(0, 1)
            
            for param in new_mod.parameters():
                if param.requires_grad and 'q_function' not in str(param):
                    param.add_(0.01 * torch.randn_like(param))

        # Assembly tracking
        new_mod.assembly_steps = max([c.assembly_steps for c in catalysts]) + 1
        new_mod.assembly_index = new_mod.assembly_steps
        new_mod.catalyzed_by = [c.id for c in catalysts]
        # When mutating, create new ModuleComponent for mutated weights
        for name, layer in [('fc1', new_mod.fc1), ('fc2', new_mod.fc2), ('fc3', new_mod.fc3)]:
            # Get the mutated weight from the new module (already mutated by load_state_dict + parameter mutation)
            mutated_weight = layer.weight.data.clone()
            parent_component = self.layer_components[name]  # Get parent component from SELF
            new_component = ModuleComponent(mutated_weight, parents=[parent_component], operation='mutation')
            new_mod.layer_components[name] = new_component  # Assign to NEW module
        
        # Update assembly pathway for new module
        new_mod.assembly_pathway = [new_mod.layer_components['fc1'], new_mod.layer_components['fc2'], new_mod.layer_components['fc3']]

        for c in catalysts:
            c.catalyzes.append(new_mod.id)
        
        return new_mod
    
    # ==========================================================
    # ================ Simplified Assembly Tracking ==============
    # ==========================================================

    """
    Something in this section is deprecated, and needs to be removed. Tracking system
    complexity is more difficult than I thought. 
    """
    
    #def compute_assembly_index(self):
    #    """Compute assembly index A(m_i) = a_i"""
    #    self.assembly_index = self.assembly_steps
    #    return self.assembly_index
    
    #def get_assembly_complexity_contribution(self, total_population):
    #    """Calculate this module's contribution to system complexity"""
    #    # A(S) = Σ e^(a_i) * (n_i - 1) / N_T
    #    return math.exp(self.assembly_index) * (self.copy_number - 1) / total_population
    
    
    def record_assembly_operation(self, operation_type, parent_modules, catalysts):
        """Record operations for true Assembly Theory"""
        operation = {
            'type': operation_type,  # 'mutation', 'crossover', 'catalysis'
            'inputs': [m.id for m in parent_modules],
            'catalysts': [c.id for c in catalysts],
            'step': getattr(self, 'created_at', 0)
        }
        self.assembly_operations.append(operation)
        
        # Update assembly index
        self.compute_assembly_index()

    # POTENTIALLY DEPRECATED:
    def get_assembly_complexity_contribution(self, population):
        """Corrected system complexity following theorem"""
        if not population:
            return 0.0
        
        total_complexity = 0.0
        total_population = len(population)
        
        for module in population:
            # A(S) = Σ e^(a_i) * (n_i - 1) / N_T
            a_i = module.compute_assembly_index()  # True assembly index
            n_i = getattr(module, 'copy_number', 1)  # Number of copies
            
            contribution = math.exp(a_i) * (n_i - 1) / total_population
            total_complexity += contribution
        
        return total_complexity

    def get_assembly_complexity(self):
        """
        Returns a dict with per-layer and total assembly complexity.
        """
        complexities = {}
        total = 0
        for name, comp in self.layer_components.items():
            complexity = comp.get_minimal_assembly_complexity()
            complexities[name] = complexity
            total += complexity
        
        complexities['total'] = total
        return complexities

    def compute_assembly_index(self):
        """
        Computes the assembly index for this module as the sum of per-layer complexities.
        This represents the total assembly complexity of all components.
        """
        total_complexity = 0
        for name, component in self.layer_components.items():
            total_complexity += component.get_minimal_assembly_complexity()
        
        self.assembly_index = total_complexity
        return self.assembly_index

    # ==========================================================
    # ====================== Other Methods =====================
    # ==========================================================

    # Reward tracker
    def set_reward(self, reward):
        self.reward = reward
        self.position_info['reward_value'] = reward
    
    # novelty score tracker
    def set_novelty_score(self, novelty_score):
        self.position_info['novelty_score'] = novelty_score

    # best reward
    def get_best_reward(self):
        return getattr(self, 'best_reward', None)

    # I dont remember what this is for tbh
    def hashable_op(self, op):
        # Convert lists in op to tuples for hashing
        return tuple(
            (k, tuple(v) if isinstance(v, list) else v)
            for k, v in sorted(op.items())
        )

    # get position in data space
    def get_position(self):
        return self.position.data.detach().numpy()    
    
    # ==================== END OF CLASS =====================


def system_assembly_complexity(population):
    """
    Computes system assembly complexity as in the notebook formula.
    population: list of ConceptModule instances
    """
    from collections import Counter
    complexities = [m.compute_assembly_index() for m in population]
    ids = [id(m) for m in population]
    counts = Counter(ids)
    N = len(population)
    return sum(math.exp(a_i) * (counts[ids[i]] - 1) / N for i, a_i in enumerate(complexities)) / N