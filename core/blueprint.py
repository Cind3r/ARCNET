import networkx as nx
import matplotlib.pyplot as plt

class ArchitectureBlueprint:
    def __init__(self):
        self.modules = []
        self.connections = []  # list of (src_id, dst_id)

    def add_module(self, module):
        self.modules.append(module)

    def connect(self, src_id, dst_id):
        self.connections.append((src_id, dst_id))

    def get_module_by_id(self, module_id):
        for module in self.modules:
            if module.id == module_id:
                return module
        return None

    def visualize(self, experiment_name="arcnet"):
        filepath = f"ARCNET/experiments/architecture_graphs/{experiment_name}_architecture_graph.png"
        
        G = nx.DiGraph()
        for module in self.modules:
            G.add_node(module.id[:8])
        for src, dst in self.connections:
            G.add_edge(src[:8], dst[:8])
        plt.figure(figsize=(10, 6))
        nx.draw(G, with_labels=True, node_size=500, node_color="lightblue")
        plt.title("Architecture Blueprint Graph")
        plt.savefig(filepath)
        plt.close()
