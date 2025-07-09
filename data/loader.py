import pickle
import json
import torch
import os
import glob
from datetime import datetime
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ARCNET.models.arcnet_learner import ConceptModule

# This module provides functions to save and load lineage snapshots,
# and create animations of lineage evolution in 3D. It generally handles
# any type of file generation or loading aside from the main evolution process.

# lineage snapshots for 3d visualization
def save_lineage_snapshot_to_file(modules, step, filename_base, format='pickle'):
    
    """
    Save a single lineage snapshot to file 
    """
    
    # Create directory if it doesn't exist
    Path("ARCNET/experiments/lineage_snaps").mkdir(exist_ok=True)
    
    # Create lightweight snapshot data
    snapshot_data = {
        'step': step,
        'timestamp': datetime.now().isoformat(),
        'modules': []
    }
    
    for m in modules:
        # Extract only essential data for plotting
        try:
            # FIXED: Force conversion to plain Python list
            if hasattr(m.position, 'data'):
                position_data = m.position.data.detach().cpu().numpy()
            elif hasattr(m.position, 'detach'):
                position_data = m.position.detach().cpu().numpy()
            else:
                position_data = np.array(m.position)
            
            # Ensure it's a plain Python list
            position_list = position_data.tolist()
            
            module_data = {
                'id': int(m.id),
                'parent_id': int(m.parent_id) if m.parent_id is not None else None,
                'fitness': float(m.fitness),
                'assembly_index': int(m.assembly_index),
                'created_at': int(m.created_at),
                'position': position_list,  # Plain Python list
                'copy_number': int(getattr(m, 'copy_number', 1)),
                'catalyzed_by': list(getattr(m, 'catalyzed_by', [])),
                'catalyzes': list(getattr(m, 'catalyzes', []))
            }
            snapshot_data['modules'].append(module_data)
        except Exception as e:
            print(f"Error saving module {m.id}: {e}")
            continue
    
    if format == 'pickle':
        filename = f"ARCNET/experiments/lineage_snaps/{filename_base}_step_{step:04d}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(snapshot_data, f)
    elif format == 'json':
        filename = f"ARCNET/experiments/lineage_snaps/{filename_base}_step_{step:04d}.json"
        with open(filename, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
    
    return filename


def load_lineage_snapshot_from_file(filename):
    """Load a single snapshot from file - handles both Path objects and strings"""
    from pathlib import Path
    
    # Ensure we have a Path object for consistency
    if isinstance(filename, str):
        file_path = Path(filename)
    else:
        file_path = filename
    
    # Check file extension
    if file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def load_all_lineage_snapshots(filename_base, steps_range=None):
    """
    Load all snapshots for animation/analysis
    
    Args:
        filename_base: Base filename used when saving
        steps_range: Tuple (start, end) to load specific range, or None for all
    
    Returns:
        List of snapshot data dictionaries
    """
    snapshot_dir = Path("ARCNET/experiments/lineage_snaps")
    
    # Find all snapshot files
    pattern = f"{filename_base}_step_*.pkl"
    snapshot_files = sorted(snapshot_dir.glob(pattern))
    
    if not snapshot_files:
        # Try JSON format
        pattern = f"{filename_base}_step_*.json"
        snapshot_files = sorted(snapshot_dir.glob(pattern))
    
    snapshots = []
    for file_path in snapshot_files:
        # Extract step number from filename
        step_num = int(file_path.stem.split('_')[-1])
        
        if steps_range is None or (steps_range[0] <= step_num <= steps_range[1]):
            # Convert Path to string before passing to load function
            snapshot_data = load_lineage_snapshot_from_file(str(file_path))
            snapshots.append(snapshot_data)
    
    print(f"Loaded {len(snapshots)} snapshots")
    return snapshots


def animate_lineage_3d_from_files_gradient(filename_base, color_by='depth', output_name=None, 
                                          steps_range=None, frame_interval=1000):
    """
    Create animation with AGE-GRADIENT for both edges AND nodes: Red/Bright (newest) -> Gray/Faded (oldest)
    """
    
    # Load snapshots from files
    snapshot_data_list = load_all_lineage_snapshots(filename_base, steps_range)
    
    if not snapshot_data_list:
        print("No snapshots found!")
        return
    
    print(f"Found {len(snapshot_data_list)} snapshots")
    
    # Convert to plot-friendly format with GUARANTEED data types
    plot_snapshots = []
    for i, snapshot_data in enumerate(snapshot_data_list):
        print(f"Processing snapshot {i}: {len(snapshot_data['modules'])} modules")
        
        plot_modules = []
        for module_data in snapshot_data['modules']:
            try:
                class SafePlotModule:
                    def __init__(self, data):
                        self.id = int(data['id'])
                        self.parent_id = int(data['parent_id']) if data.get('parent_id') is not None else None
                        self.fitness = float(data['fitness'])
                        self.created_at = int(data.get('created_at', 0))  # Important for age calculation
                        
                        # ULTRA-SAFE position handling
                        pos_raw = data['position']
                        if isinstance(pos_raw, list):
                            pos_list = [float(x) for x in pos_raw]
                        elif isinstance(pos_raw, tuple):
                            pos_list = [float(x) for x in pos_raw]
                        elif hasattr(pos_raw, 'tolist'):
                            pos_list = [float(x) for x in pos_raw.tolist()]
                        elif hasattr(pos_raw, '__iter__'):
                            pos_list = [float(x) for x in pos_raw]
                        else:
                            print(f"Warning: Unexpected position type {type(pos_raw)} for module {self.id}")
                            pos_list = [0.5, 0.5, 0.5]
                        
                        if len(pos_list) != 3:
                            print(f"Warning: Invalid position length {len(pos_list)} for module {self.id}")
                            pos_list = [0.5, 0.5, 0.5]
                        
                        self.position = np.array(pos_list, dtype=np.float64)
                        self.assembly_index = int(data.get('assembly_index', 0))
                
                plot_modules.append(SafePlotModule(module_data))
                
            except Exception as e:
                print(f"Error processing module in snapshot {i}: {e}")
                continue
        
        if plot_modules:
            plot_snapshots.append(plot_modules)
        else:
            print(f"Warning: No valid modules in snapshot {i}")
    
    if not plot_snapshots:
        print("No valid plot snapshots created!")
        return
    
    print(f"Created {len(plot_snapshots)} plot snapshots")
    
    # Create animation
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Setup colormaps
    if color_by == 'depth':
        cmap_name = 'magma'
        dummy_sm = plt.cm.ScalarMappable(cmap=plt.cm.magma, norm=plt.Normalize(vmin=0, vmax=10))
        dummy_sm.set_array([])
        cbar = plt.colorbar(dummy_sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Generation Depth")
    elif color_by == 'fitness':
        cmap_name = 'viridis'
        dummy_sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
        dummy_sm.set_array([])
        cbar = plt.colorbar(dummy_sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Fitness")

    def build_lineage_edges_with_age(modules, current_step):
        """Build parent-child edges with age information"""
        edges_with_age = []
        module_dict = {m.id: m for m in modules}
        
        for module in modules:
            if module.parent_id is not None and module.parent_id in module_dict:
                parent = module_dict[module.parent_id]
                
                # Calculate edge age (when the child was created)
                edge_birth_step = module.created_at
                edge_age = current_step - edge_birth_step
                
                edges_with_age.append({
                    'parent_id': parent.id,
                    'child_id': module.id,
                    'age': edge_age,
                    'birth_step': edge_birth_step
                })
        
        return edges_with_age

    def get_edge_color_and_alpha(edge_age, max_age):
        """Calculate color and alpha based on edge age"""
        if max_age == 0:
            return 'red', 0.8, 2.5
        
        # Normalize age (0 = newest, 1 = oldest)
        age_ratio = edge_age / max_age
        
        # Color gradient: Red (new) -> Dark Gray (old)
        red_component = 1.0 - age_ratio
        gray_component = 0.3 + (0.4 * age_ratio)
        
        color = (red_component, gray_component, gray_component)
        
        # Alpha and width: Newer edges are more prominent
        alpha = 0.9 - (0.4 * age_ratio)  # 0.9 -> 0.5
        width = 3.0 - (1.0 * age_ratio)  # 3.0 -> 2.0
        
        return color, alpha, width

    def get_node_age_properties(module, current_step, max_node_age):
        """Calculate node properties based on age"""
        node_age = current_step - module.created_at
        
        if max_node_age == 0:
            return 1.0, 60  # Full opacity, standard size
        
        # Normalize age (0 = newest, 1 = oldest)
        age_ratio = node_age / max_node_age
        
        # Alpha: Newest nodes are fully opaque, oldest fade to 30%
        alpha = 1.0 - (0.7 * age_ratio)  # 1.0 -> 0.3
        
        # Size: Newer nodes are slightly larger
        size = 70 - (20 * age_ratio)  # 70 -> 50
        
        return alpha, size

    def compute_depth_map(modules, edges_with_age):
        """Compute generation depth for each module"""
        depth_map = {}
        module_dict = {m.id: m for m in modules}
        
        def compute_depth(module_id):
            if module_id not in depth_map:
                module = module_dict[module_id]
                if module.parent_id is None or module.parent_id not in module_dict:
                    depth_map[module_id] = 0
                else:
                    depth_map[module_id] = 1 + compute_depth(module.parent_id)
            return depth_map[module_id]
        
        for module in modules:
            compute_depth(module.id)
        
        return depth_map

    def safe_update(frame):
        """Update function with age-gradient for both edges AND nodes"""
        try:
            ax.clear()
            
            if frame < 0 or frame >= len(plot_snapshots):
                print(f"Frame {frame} out of bounds")
                return ax.scatter([], [], [], c='blue', s=40)
            
            modules = plot_snapshots[frame]
            step_num = snapshot_data_list[frame]['step']
            
            if not modules:
                print(f"No modules in frame {frame}")
                return ax.scatter([], [], [], c='blue', s=40)
            
            # Build lineage edges with age information
            edges_with_age = build_lineage_edges_with_age(modules, step_num)
            
            # Extract positions
            positions = {}
            xs, ys, zs = [], [], []
            
            for m in modules:
                try:
                    pos = m.position
                    if pos.shape == (3,):
                        positions[m.id] = pos
                        xs.append(float(pos[0]))
                        ys.append(float(pos[1]))
                        zs.append(float(pos[2]))
                except Exception as e:
                    continue
            
            if not xs:
                print(f"No valid positions in frame {frame}")
                return ax.scatter([], [], [], c='blue', s=40)
            
            # ZOOM IN: Calculate bounding box of actual node positions
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            z_min, z_max = min(zs), max(zs)
            
            # Add padding around the cluster
            x_range = max(x_max - x_min, 0.1)
            y_range = max(y_max - y_min, 0.1) 
            z_range = max(z_max - z_min, 0.1)
            
            padding = 0.15
            x_pad = x_range * padding
            y_pad = y_range * padding
            z_pad = z_range * padding
            
            # Set axis limits to zoom in on the actual cluster
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_zlim(z_min - z_pad, z_max + z_pad)
            
            # Calculate node ages for opacity/size scaling
            node_ages = [step_num - m.created_at for m in modules if m.id in positions]
            max_node_age = max(node_ages) if node_ages else 0
            
            # Compute colors for nodes
            try:
                if color_by == 'fitness':
                    values = [float(m.fitness) for m in modules if m.id in positions]
                    node_colors = values
                    if values:
                        dummy_sm.set_clim(vmin=min(values), vmax=max(values))
                else:  # depth
                    depth_map = compute_depth_map(modules, edges_with_age)
                    node_colors = [depth_map.get(m.id, 0) for m in modules if m.id in positions]
                    if node_colors:
                        dummy_sm.set_clim(vmin=min(node_colors), vmax=max(node_colors))
            except Exception as e:
                print(f"Error computing colors: {e}")
                node_colors = ['blue'] * len(xs)
            
            # Draw lineage edges with AGE GRADIENT
            edge_count = 0
            try:
                if edges_with_age:
                    max_edge_age = max(edge['age'] for edge in edges_with_age)
                    edges_with_age.sort(key=lambda x: -x['age'])  # Oldest first
                    
                    for edge in edges_with_age:
                        parent_id = edge['parent_id']
                        child_id = edge['child_id']
                        edge_age = edge['age']
                        
                        if parent_id in positions and child_id in positions:
                            parent_pos = positions[parent_id]
                            child_pos = positions[child_id]
                            
                            x_vals = [float(parent_pos[0]), float(child_pos[0])]
                            y_vals = [float(parent_pos[1]), float(child_pos[1])]
                            z_vals = [float(parent_pos[2]), float(child_pos[2])]
                            
                            # Get age-based color, alpha, and width
                            color, alpha, width = get_edge_color_and_alpha(edge_age, max_edge_age)
                            
                            # Draw edge with gradient styling
                            ax.plot(x_vals, y_vals, z_vals, c=color, alpha=alpha, linewidth=width)
                            edge_count += 1
                
            except Exception as e:
                print(f"Error drawing edges: {e}")
            
            # Draw nodes with AGE-BASED OPACITY AND SIZE
            try:
                # Group nodes by age for batch drawing (more efficient)
                age_groups = {}
                for i, m in enumerate([mod for mod in modules if mod.id in positions]):
                    alpha, size = get_node_age_properties(m, step_num, max_node_age)
                    age_key = (round(alpha, 2), int(size))  # Round for grouping
                    
                    if age_key not in age_groups:
                        age_groups[age_key] = {'xs': [], 'ys': [], 'zs': [], 'colors': [], 'indices': []}
                    
                    age_groups[age_key]['xs'].append(xs[i])
                    age_groups[age_key]['ys'].append(ys[i])
                    age_groups[age_key]['zs'].append(zs[i])
                    age_groups[age_key]['colors'].append(node_colors[i] if isinstance(node_colors[i], (int, float)) else 0)
                    age_groups[age_key]['indices'].append(i)
                
                # Draw each age group separately
                for (alpha, size), group_data in age_groups.items():
                    if group_data['xs']:  # Check if group has data
                        if all(isinstance(c, (int, float)) for c in group_data['colors']):
                            sc = ax.scatter(group_data['xs'], group_data['ys'], group_data['zs'], 
                                          c=group_data['colors'], cmap=cmap_name, 
                                          s=size, alpha=alpha, edgecolors='white', linewidth=0.8)
                        else:
                            sc = ax.scatter(group_data['xs'], group_data['ys'], group_data['zs'], 
                                          c='blue', s=size, alpha=alpha, edgecolors='white', linewidth=0.8)
                
            except Exception as e:
                print(f"Error drawing nodes: {e}")
                sc = ax.scatter(xs, ys, zs, c='blue', s=50, alpha=0.7)
            
            # Enhanced title with age information
            if edges_with_age and node_ages:
                newest_edge_age = min(edge['age'] for edge in edges_with_age)
                oldest_edge_age = max(edge['age'] for edge in edges_with_age)
                newest_node_age = min(node_ages)
                oldest_node_age = max(node_ages)
                title = f"Temporal Lineage Evolution Step {step_num}\n({len(modules)} modules, {edge_count} edges)\nNode Ages: {newest_node_age}-{oldest_node_age}, Edge Ages: {newest_edge_age}-{oldest_edge_age}"
            else:
                title = f"Temporal Lineage Evolution Step {step_num}\n({len(modules)} modules, {edge_count} edges)"
            
            ax.set_title(title, fontsize=10, pad=20)
            ax.set_xlabel("X Position", fontsize=9)
            ax.set_ylabel("Y Position", fontsize=9)
            ax.set_zlabel("Z Position", fontsize=9)
            
            # Enhanced legend for temporal visualization
            if frame == 0:
                from matplotlib.lines import Line2D
                from matplotlib.patches import Circle
                legend_elements = [
                    Line2D([0], [0], color='red', lw=3, alpha=0.9, label='Recent Edges'),
                    Line2D([0], [0], color='gray', lw=2, alpha=0.5, label='Ancient Edges'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                           markersize=8, alpha=1.0, label='Recent Nodes', linestyle='None'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                           markersize=6, alpha=0.4, label='Ancient Nodes', linestyle='None')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            # Rotating view
            ax.view_init(elev=20, azim=45 + frame * 2)
            
            return sc
            
        except Exception as e:
            print(f"Critical error in frame {frame}: {e}")
            import traceback
            traceback.print_exc()
            return ax.scatter([], [], [], c='red', s=40)

    try:
        # Create animation
        print(f"Creating temporal gradient animation with {len(plot_snapshots)} frames")
        anim = FuncAnimation(fig, safe_update, frames=len(plot_snapshots), 
                            interval=frame_interval, repeat=False, cache_frame_data=False)
        
        # Save animation
        if output_name is None:
            output_name = f"{filename_base}_{color_by}_temporal_gradient"
        
        output_path = f'data/{output_name}_{datetime.now().date()}.gif'
        print(f"Attempting to save animation to: {output_path}")
        
        Path("data").mkdir(exist_ok=True)
        
        anim.save(output_path, writer='pillow', fps=max(1, 1000//frame_interval), dpi=100)
        print(f"Temporal gradient animation saved successfully as {output_path}")
        
        # Show first frame as preview
        plt.figure(figsize=(10, 8))
        ax_preview = plt.axes(projection='3d')
        safe_update(0)
        plt.title("Preview: Temporal Gradient (Nodes + Edges)")
        plt.show()
        
    except Exception as e:
        print(f"Failed to save animation: {e}")
        import traceback
        traceback.print_exc()


# ========== Exporting Structures ==========
def export_best_models(population, top_k=5, save_path="ARCNET/experiments/trained_models", experiment_name="arcnet"):

    """
    Export the top-k best models for use as initial population in next run.
    Args:
        population: List of ConceptModule instances
        top_k: Number of best models to export
        save_path: Directory to save the exported models
        experiment_name: Base name for the exported files
    Returns:
        List of exported model filenames and metadata dictionary
    """
    
    # Create save directory
    Path(save_path).mkdir(exist_ok=True)
    
    # Sort by fitness and get top performers
    sorted_pop = sorted(population, key=lambda m: m.fitness, reverse=True)
    best_models = sorted_pop[:top_k]
    
    exported_models = []
    
    for i, model in enumerate(best_models):
        # Create a lightweight model export
        model_export = {
            'state_dict': model.state_dict(),
            'fitness': model.fitness,
            'assembly_index': model.assembly_index,
            'position': model.position.data.detach().cpu().numpy().tolist(),
            'hidden_dim': model.hidden_dim,
            'input_dim': model.input_dim,
            'output_dim': model.output_dim,
            'id': model.id,
            'reward_history': getattr(model, 'reward_history', []),
            
            # Q-learning state (CRITICAL for continuity)
            'q_function_state': None,
            'q_buffer': None,
            'alpha': model.alpha,
            'gamma': model.gamma,
            'epsilon': model.epsilon
        }
        
        # Export Q-learning knowledge
        if (model.q_learning_method == 'neural' and 
            model.q_function is not None):
            model_export['q_function_state'] = model.q_function.state_dict()
            model_export['q_buffer'] = model.q_function.replay_buffer[-50:]  # Keep recent experiences
            model_export['q_learning_method'] = 'neural'
        else:
            model_export['q_table'] = getattr(model, 'q_table', {})
            model_export['q_learning_method'] = 'table'
        
        # Save individual model
        filename = f"{save_path}/{experiment_name}_best_model_{i}_{model.id}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model_export, f)
        
        exported_models.append(filename)
        print(f"Exported model {i}: fitness={model.fitness:.4f}, assembly={model.assembly_index}")
    
    # Save metadata
    metadata = {
        'num_models': len(best_models),
        'best_fitness': best_models[0].fitness,
        'avg_fitness': sum(m.fitness for m in best_models) / len(best_models),
        'export_files': exported_models,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{save_path}/{experiment_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nExported {len(best_models)} best models to {save_path}/")
    return exported_models, metadata

def load_and_seed_population_fixed(load_path="ARCNET/experiments/trained_models", experiment_name="arcnet", 
                            total_population=60, input_dim=384, hidden_dim=128, output_dim=2):
    """Load exported models and use them to seed a new population - FIXED for Windows paths"""
    
    # Use os.path.join for cross-platform compatibility
    model_pattern = os.path.join(load_path, f"{experiment_name}_best_model_*.pkl")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print(f"No exported models found with pattern: {model_pattern}")
        print(f"Starting fresh population.")
        return [ConceptModule(input_dim, hidden_dim, output_dim, q_learning_method='neural') 
                for _ in range(total_population)]
    
    print(f"Found {len(model_files)} exported models. Loading...")
    
    seeded_population = []
    
    # Load each exported model
    for model_file in model_files:
        try:
            print(f"Loading: {model_file}")
            with open(model_file, 'rb') as f:
                model_export = pickle.load(f)
            
            # Create new ConceptModule
            new_module = ConceptModule(
                input_dim=model_export['input_dim'],
                hidden_dim=model_export['hidden_dim'], 
                output_dim=model_export['output_dim'],
                q_learning_method=model_export.get('q_learning_method', 'neural')
            )
            
            # Restore neural network weights
            new_module.load_state_dict(model_export['state_dict'])
            
            # Restore fitness and metadata
            new_module.fitness = model_export['fitness']
            new_module.assembly_index = model_export['assembly_index']
            new_module.position.data = torch.tensor(model_export['position'])
            new_module.reward_history = model_export.get('reward_history', [])
            
            # Restore Q-learning parameters
            new_module.alpha = model_export.get('alpha', 0.1)
            new_module.gamma = model_export.get('gamma', 0.9)
            new_module.epsilon = model_export.get('epsilon', 0.15)
            
            # CRITICAL: Restore Q-learning knowledge
            if model_export.get('q_learning_method') == 'neural':
                if model_export.get('q_function_state') is not None:
                    new_module.q_function.load_state_dict(model_export['q_function_state'])
                if model_export.get('q_buffer') is not None:
                    new_module.q_function.replay_buffer = model_export['q_buffer']
            else:
                new_module.q_table = model_export.get('q_table', {})
            
            seeded_population.append(new_module)
            print(f"✓ Loaded model: fitness={new_module.fitness:.4f}, assembly={new_module.assembly_index}")
            
        except Exception as e:
            print(f"✗ Error loading {model_file}: {e}")
            continue
    
    # Fill remaining slots with fresh modules if needed
    while len(seeded_population) < total_population:
        fresh_module = ConceptModule(input_dim, hidden_dim, output_dim, q_learning_method='neural')
        seeded_population.append(fresh_module)
    
    print(f"Created seeded population: {len(seeded_population)} total ({len(model_files)} loaded + {total_population - len(seeded_population)} fresh)")
    
    return seeded_population