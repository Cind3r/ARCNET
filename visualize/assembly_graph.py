import networkx as nx
from pyvis.network import Network
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def AssemblyLineageVisualizer(best_model, lineagesnap, assembly_registry=None, filename="comprehensive_assembly_lineage"):
    
    """
    Creates a comprehensive visualization showing the full assembly lineage
    with component reuse tracking and hierarchical organization.
    Uses all available information in assembly_registry if provided.

    (Pyvis)

    Args:
        - best_model (object | ConceptModule) : The best model instance containing assembly pathway and lineage information.
        - lineagesnap (object) : A snapshot of the lineage mapping (e.g., {parent_id: module_instance} ).
        - assembly_registry (object) : Optional registry containing component information (default: None).
        - filename (str) : Name of the output HTML file (default: "comprehensive_assembly_lineage") path is to docs folder by default.
    Returns:
        - G (networkx.DiGraph) : The directed graph representing the assembly lineage.
        - component_info (dict) : Comprehensive information about each component in the lineage.
        - longest_paths (list) : List of longest assembly paths found in the lineage.
    """
    
    # First, let's gather ALL modules in the lineage and organize by generation
    all_modules = {}  # generation -> list of modules
    lineage_modules = []
    
    # Traverse the lineage
    current = best_model
    visited = set()
    while current is not None and getattr(current, 'id', None) is not None and current.id not in visited:
        lineage_modules.append(current)
        visited.add(current.id)
        generation = getattr(current, 'created_at', 0)
        if generation not in all_modules:
                all_modules[generation] = []
        all_modules[generation].append(current)
        
        parent_id = getattr(current, 'parent_id', None)
        if parent_id is not None and parent_id in lineagesnap:
            current = lineagesnap[parent_id]
        else:
                break

    print(f"Found {len(lineage_modules)} modules across {len(all_modules)} generations")
    
    # Now build a comprehensive component graph that tracks actual reuse
    G = nx.DiGraph()
    component_info = {}  # component_id -> {module, generation, layer, complexity, etc.}
    component_hierarchy = {}  # track parent-child relationships
    
    # If assembly_registry is provided, use it to enrich component_info and graph
    if assembly_registry is not None:
        for comp_id, comp_data in assembly_registry.component_registry.items():
            # comp_data could be a dict or an object, try to extract info
            if isinstance(comp_data, dict):
                info = comp_data.copy()
                component = info.get('component', None)
            else:
                info = {}
                component = comp_data
            # Try to extract module, generation, complexity, etc.
            module = getattr(component, 'module', None) or info.get('module', None)
            generation = getattr(component, 'generation', None) or info.get('generation', None)
            if generation is None and module is not None:
                generation = getattr(module, 'created_at', 0)
            complexity = getattr(component, 'assembly_complexity', None) or info.get('assembly_complexity', 0)
            pathway_index = getattr(component, 'pathway_index', None) or info.get('pathway_index', None)
            layer_info = getattr(component, 'layer_info', None) or info.get('layer_info', None)
            # Compose info
            component_info[comp_id] = {
                'module': module,
                'generation': generation,
                'component': component,
                'pathway_index': pathway_index,
                'assembly_complexity': complexity,
                'layer_info': layer_info
            }
            G.add_node(comp_id)
            # Add parent relationships if available
            parents = getattr(component, 'parents', None) or info.get('parents', [])
            if parents:
                for parent in parents:
                    parent_id = getattr(parent, 'id', str(parent))
                    if parent_id != comp_id:
                        G.add_edge(parent_id, comp_id)
                        if comp_id not in component_hierarchy:
                            component_hierarchy[comp_id] = []
                        component_hierarchy[comp_id].append(parent_id)
    else:
        # Process each module's assembly pathway
        for mod in lineage_modules:
            generation = getattr(mod, 'created_at', 0)
            mod_id = getattr(mod, 'id', 'unknown')
            
            print(f"\nProcessing Module {mod_id} (Gen {generation}):")
            
            # Check assembly pathway
            if hasattr(mod, 'assembly_pathway') and mod.assembly_pathway:
                print(f"  Assembly pathway length: {len(mod.assembly_pathway)}")
                
                for idx, comp in enumerate(mod.assembly_pathway):
                    comp_id = getattr(comp, 'id', f'{mod_id}_comp_{idx}')
                    
                    # Store comprehensive component information
                    component_info[comp_id] = {
                        'module': mod,
                        'generation': generation,
                        'component': comp,
                        'pathway_index': idx,
                        'assembly_complexity': getattr(comp, 'assembly_complexity', 0),
                        'layer_info': f"Component {idx}"
                    }
                    
                    G.add_node(comp_id)
                    print(f"    Component {idx}: {comp_id} (complexity: {getattr(comp, 'assembly_complexity', 'N/A')})")
                    
                    # Track parent relationships
                    if hasattr(comp, 'parents') and comp.parents:
                        for parent in comp.parents:
                            parent_id = getattr(parent, 'id', str(parent))
                            if parent_id != comp_id:  # Avoid self-loops
                                G.add_edge(parent_id, comp_id)
                                print(f"      -> Parent: {parent_id}")
                                
                                # Store hierarchy info
                                if comp_id not in component_hierarchy:
                                    component_hierarchy[comp_id] = []
                                component_hierarchy[comp_id].append(parent_id)
            
            # Also check layer components for additional assembly info
            if hasattr(mod, 'layer_components'):
                print(f"  Layer components: {list(mod.layer_components.keys())}")
                for layer_name, layer_comp in mod.layer_components.items():
                    if hasattr(layer_comp, 'get_minimal_assembly_complexity'):
                        complexity = layer_comp.get_minimal_assembly_complexity()
                        print(f"    {layer_name}: complexity {complexity}")

    print(f"\nTotal components in graph: {G.number_of_nodes()}")
    print(f"Total edges (reuse relationships): {G.number_of_edges()}")
    
    # Identify different types of components
    final_components = set()
    if hasattr(best_model, 'assembly_pathway') and best_model.assembly_pathway:
        for comp in best_model.assembly_pathway:
            comp_id = getattr(comp, 'id', None)
            if comp_id:
                final_components.add(comp_id)
    
    # Find root components (no parents) and leaf components (no children)
    root_components = [n for n in G.nodes() if G.in_degree(n) == 0]
    leaf_components = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    print(f"Root components (no parents): {len(root_components)}")
    print(f"Leaf components (no children): {len(leaf_components)}")
    print(f"Final model components: {len(final_components)}")
    
    # Find longest paths to understand assembly chains
    longest_paths = []
    for root in root_components:
        for leaf in leaf_components:
            try:
                path = nx.shortest_path(G, root, leaf)
                longest_paths.append((len(path), path))
            except nx.NetworkXNoPath:
                continue
    
    longest_paths.sort(reverse=True)
    if longest_paths:
        print(f"Longest assembly chain: {longest_paths[0][0]} components")
        print(f"Path: {' -> '.join(longest_paths[0][1][:5])}{'...' if len(longest_paths[0][1]) > 5 else ''}")

    # Create enhanced PyVis visualization
    net = Network(height="900px", width="100%", directed=True, notebook=True)
    net.barnes_hut()
    
    # Enhanced options for better visualization
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 14,
          "color": "black",
          "face": "Arial"
        },
        "borderWidth": 2,
        "scaling": {
          "min": 20,
          "max": 80
        }
      },
      "edges": {
        "color": {
          "color": "#666666",
          "highlight": "#333333"
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1
          }
        },
        "smooth": {
          "enabled": true,
          "type": "dynamic",
          "roundness": 0.5
        },
        "width": 2
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "shakeTowards": "leaves"
        }
      },
      "physics": {
        "enabled": true,
        "hierarchicalRepulsion": {
          "centralGravity": 0.3,
          "springLength": 100,
          "springConstant": 0.01,
          "nodeDistance": 120,
          "damping": 0.09
        }
      },
      "interaction": {
        "hover": true,
        "selectConnectedEdges": true,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    # Add nodes with comprehensive information and size based on complexity
    for node in G.nodes():
        info = component_info.get(node, {})
        generation = info.get('generation', '?')
        complexity = info.get('assembly_complexity', 0)
        
        # Create detailed label and tooltip
        label = f"G{generation}\nC{complexity}"
        
        tooltip_parts = [
            f"<b>Component ID:</b> {node}",
            f"<b>Generation:</b> {generation}",
            f"<b>Assembly Complexity:</b> {complexity}",
            f"<b>Pathway Index:</b> {info.get('pathway_index', 'N/A')}"
        ]
        
        if 'module' in info and info['module'] is not None:
            mod = info['module']
            tooltip_parts.extend([
                f"<b>Module ID:</b> {getattr(mod, 'id', 'N/A')}",
                f"<b>Module Fitness:</b> {getattr(mod, 'fitness', 'N/A'):.4f}" if hasattr(mod, 'fitness') else "<b>Module Fitness:</b> N/A"
            ])
            
            # Add layer components info
            if hasattr(mod, 'layer_components'):
                tooltip_parts.append("<br><b>Layer Components:</b>")
                for layer_name, layer_comp in mod.layer_components.items():
                    if hasattr(layer_comp, 'get_minimal_assembly_complexity'):
                        layer_complexity = layer_comp.get_minimal_assembly_complexity()
                        tooltip_parts.append(f"• {layer_name}: {layer_complexity}")

        # Add parent/child information
        parents = component_hierarchy.get(node, [])
        children = list(G.successors(node))
        if parents:
            tooltip_parts.append(f"<b>Parents:</b> {', '.join(parents[:3])}" + ("..." if len(parents) > 3 else ""))
        if children:
            tooltip_parts.append(f"<b>Children:</b> {', '.join(children[:3])}" + ("..." if len(children) > 3 else ""))
        
        title = "<br>".join(tooltip_parts)
        
        # Determine node appearance based on role and complexity
        size = max(20, min(60, 20 + complexity * 2))  # Size based on complexity
        
        if node in final_components:
            # Final components - red
            color = {"background": "#ff4444", "border": "#cc0000", "highlight": {"background": "#ff6666", "border": "#ff0000"}}
        elif node in root_components:
            # Root components - green
            color = {"background": "#44ff44", "border": "#00cc00", "highlight": {"background": "#66ff66", "border": "#00ff00"}}
        elif complexity > 10:
            # High complexity intermediate - purple
            color = {"background": "#8844ff", "border": "#6600cc", "highlight": {"background": "#aa66ff", "border": "#8800ff"}}
        else:
            # Regular intermediate - orange
            color = {"background": "#ff9500", "border": "#cc7700", "highlight": {"background": "#ffaa33", "border": "#ff8800"}}
        
        net.add_node(node, label=label, title=title, color=color, size=size)
    
    # Add edges
    for source, target in G.edges():
        net.add_edge(source, target)

    # Generate and save
    net.show(f"docs/graphs/{filename}.html")
    add_graph_link_to_index(f"{filename}.html", display_name="Assembly Lineage Visualization", index_path="docs/index.html")
    print("🟢 Green = Root components | 🔴 Red = Final components | 🟣 Purple = High complexity | 🟠 Orange = Regular")
    return G, component_info, longest_paths

    # Usage:
    # If you have assembly_registry, pass it as a third argument:
    # assembly_graph, comp_info, chains = AssemblyLineageVisualizer(best_model, lineagesnap)


def TheoremGrapher(assembly_registry):

    """    Visualizes the assembly complexity history and verifies the theoretical bound
    of assembly complexity over time."""

    if hasattr(assembly_registry, "complexity_history"):
            complexity_history = assembly_registry.complexity_history
    elif hasattr(assembly_registry, "get_complexity_history"):
        complexity_history = assembly_registry.get_complexity_history()
    else:
        # Fallback: compute from component_registry if needed
        complexity_history = [
            comp["assembly_complexity"]
            for comp in assembly_registry.component_registry.values()
            if "assembly_complexity" in comp
        ]

    # Compute the theoretical bound (e.g., number of steps or components)
    num_steps = len(complexity_history)
    theoretical_bound = num_steps

    steps = np.arange(1, len(complexity_history)+1)
    A_sys = np.array(complexity_history)
    t = steps

    print("Maximum observed complexity:", max(complexity_history))
    print("Theoretical bound:", theoretical_bound)


    # The bound curve: A_sys(t) ≈ C * log(t) + A0 
    def log_bound(t, C, A0):
        return C * np.log(t) + A0

    # Fit the curve
    params, _ = curve_fit(log_bound, t, A_sys)
    C_fit, A0_fit = params

    # Compute the fitted bound
    A_bound = log_bound(t, C_fit, A0_fit)

    # Plot the results
    plt.figure(figsize=(8,5))
    plt.plot(t, A_sys, label="Observed $A_{sys}^{(t)}$")
    plt.plot(t, A_bound, '--', label=f"Fit: $C \\log(t) + A_0$\nC={C_fit:.2f}, A₀={A0_fit:.2f}")
    plt.xlabel("Step $t$")
    plt.ylabel("Assembly Complexity $A_{sys}^{(t)}$")
    plt.title("Empirical Verification of Assembly Theory Bound")
    plt.legend()
    plt.grid(True)
    plt.show()


# index.html updater
def add_graph_link_to_index(filename, display_name=None, index_path="index.html"):
    """
    Adds a new graph link to the index.html file.
    Args:
        filename (str): The HTML filename of the graph (e.g., "my_graph.html").
        display_name (str, optional): The text to display for the link. Defaults to the filename without extension.
        index_path (str): Path to the index.html file.
    """
    import os

    if display_name is None:
        display_name = os.path.splitext(os.path.basename(filename))[0].replace("_", " ").title()

    # Read the current index.html
    with open(index_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the <ul> and </ul> lines
    ul_start = None
    ul_end = None
    for i, line in enumerate(lines):
        if "<ul>" in line:
            ul_start = i
        if "</ul>" in line:
            ul_end = i
            break

    if ul_start is None or ul_end is None or ul_end <= ul_start:
        raise ValueError("Could not find <ul> section in index.html")

    # Prepare the new link line
    new_link = f'    <li><a href="graphs/{filename}" target="_blank">{display_name}</a></li>\n'

    # Check if the link already exists
    if new_link in lines:
        print("Link already exists in index.html")
        return

    # Insert the new link before </ul>
    lines.insert(ul_end, new_link)

    # Write back the updated file
    with open(index_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
