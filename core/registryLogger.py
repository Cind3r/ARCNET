# Add this simple file-based assembly tracker

import json
import os
from datetime import datetime
from registry import AssemblyTrackingRegistry
from registry import AssemblyTracker
from registry import AssemblyComponent

class FileBasedAssemblyTracker:
    """
    Simple file-based assembly tracker that writes events to disk and forgets them
    maybe more memory efficient than keeping everything in memory
    """
    
    def __init__(self, experiment_name="arcnet", output_dir="assembly_logs"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.step_counter = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f"{experiment_name}_assembly_{timestamp}.jsonl")
        
        # Write header info
        self._write_event({
            'type': 'experiment_start',
            'timestamp': timestamp,
            'experiment_name': experiment_name
        })
    
    def _write_event(self, event):
        """Write a single event to the log file"""
        with open(self.log_file, 'a') as f:
            json.dump(event, f)
            f.write('\n')
    
    def register_module_initialization(self, module, parent_modules=None):
        """Log module initialization"""
        event = {
            'type': 'module_init',
            'step': self.step_counter,
            'module_id': module.id,
            'parent_id': getattr(module, 'parent_id', None),
            'fitness': getattr(module, 'fitness', 0.0),
            'created_at': getattr(module, 'created_at', self.step_counter),
            'has_q_function': hasattr(module, 'q_function') and module.q_function is not None,
            'layer_count': len(list(module.named_parameters()))
        }
        
        # Add parent info if available
        if parent_modules:
            event['parent_count'] = len(parent_modules)
            event['parent_ids'] = [getattr(p, 'id', 'unknown') for p in parent_modules]
        
        self._write_event(event)
        
        # Set assembly index on module (simple version)
        module.assembly_index = len(parent_modules) if parent_modules else 1
        
        return module.assembly_index, []
    
    def track_mutation_with_lightweight_inheritance(self, parent_module, child_module, catalysts=None):
        """Log mutation event"""
        event = {
            'type': 'mutation',
            'step': self.step_counter,
            'parent_id': parent_module.id,
            'child_id': child_module.id,
            'parent_fitness': parent_module.fitness,
            'child_fitness': getattr(child_module, 'fitness', 0.0),
            'catalyst_count': len(catalysts) if catalysts else 0,
            'catalyst_ids': [c.id for c in catalysts] if catalysts else []
        }
        
        self._write_event(event)
        return event
    
    def step_forward(self):
        """Advance step counter and log step event"""
        self.step_counter += 1
        self._write_event({
            'type': 'step_advance',
            'step': self.step_counter
        })
    
    def update_generation_statistics(self, generation, population):
        """Log generation statistics"""
        fitnesses = [m.fitness for m in population]
        
        event = {
            'type': 'generation_stats',
            'step': self.step_counter,
            'generation': generation,
            'population_size': len(population),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'best_fitness': max(fitnesses),
            'worst_fitness': min(fitnesses),
            'q_modules': sum(1 for m in population if hasattr(m, 'q_function') and m.q_function is not None),
            'total_q_experiences': sum(len(m.q_function.replay_buffer) if hasattr(m, 'q_function') and m.q_function and hasattr(m.q_function, 'replay_buffer') else 0 for m in population)
        }
        
        self._write_event(event)
    
    def log_message_passing(self, population, step):
        """Log simple message passing event"""
        event = {
            'type': 'message_passing',
            'step': step,
            'population_size': len(population),
            'modules_with_messages': sum(1 for m in population if hasattr(m, 'messages') and m.messages)
        }
        self._write_event(event)
    
    def log_fitness_update(self, module, old_fitness, new_fitness):
        """Log significant fitness changes"""
        if abs(new_fitness - old_fitness) > 0.01:  # Only log significant changes
            event = {
                'type': 'fitness_update',
                'step': self.step_counter,
                'module_id': module.id,
                'old_fitness': old_fitness,
                'new_fitness': new_fitness,
                'fitness_delta': new_fitness - old_fitness
            }
            self._write_event(event)
    
    def get_assembly_statistics(self):
        """Return minimal stats for compatibility"""
        return {
            'total_components': 0,  # We don't track this in memory
            'reused_components': 0,
            'average_reuse': 0.0,
            'component_types': {},
            'log_file': self.log_file,
            'current_step': self.step_counter
        }
    
    def finalize_experiment(self, final_population):
        """Write final experiment summary"""
        fitnesses = [m.fitness for m in final_population]
        
        event = {
            'type': 'experiment_end',
            'final_step': self.step_counter,
            'final_population_size': len(final_population),
            'final_avg_fitness': sum(fitnesses) / len(fitnesses),
            'final_best_fitness': max(fitnesses),
            'best_module_id': max(final_population, key=lambda m: m.fitness).id
        }
        
        self._write_event(event)
        print(f"Assembly tracking logged to: {self.log_file}")