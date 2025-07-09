import uuid

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
        parent_complexities = [parent.get_minimal_assembly_complexity(memo) for parent in self.parents]
        # Assembly theory: only count unique subcomponents
        unique_parent_ids = set([p.id for p in self.parents])
        memo[self.id] = 1 + sum([memo[pid] for pid in unique_parent_ids if pid in memo])
        return memo[self.id]