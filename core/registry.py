class ComponentRegistry:
    def __init__(self):
        self.data = {}

    def register(self, module):
        self.data[module.id] = {
            'type': type(module).__name__,
            'in_dim': module.linear.in_features,
            'out_dim': module.linear.out_features,
            'score': 0.0,
            'uses': 0
        }

    def update_score(self, module_id, delta):
        if module_id in self.data:
            self.data[module_id]['score'] += delta
            self.data[module_id]['uses'] += 1
