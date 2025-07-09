import networkx as nx

class AssemblyGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    def register(self, module_id, parent_ids=[]):
        self.G.add_node(module_id)
        for p in parent_ids:
            self.G.add_edge(p, module_id)

    def divergence_score(self, model_a_ids, model_b_ids):
        a_sub = self.G.subgraph(model_a_ids)
        b_sub = self.G.subgraph(model_b_ids)
        return nx.graph_edit_distance(a_sub, b_sub)

    def visualize(self, path="graph.png"):
        import matplotlib.pyplot as plt
        nx.draw(self.G, with_labels=True, node_size=300, font_size=8)
        plt.savefig(path)
