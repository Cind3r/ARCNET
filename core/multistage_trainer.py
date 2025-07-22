import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from core.trainer import Trainer
from visualize.assembly_graph import AssemblyLineageVisualizer
from data.analyzer import ARCNETParameterAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import defaultdict
import os
from datetime import datetime
from tqdm.notebook import tqdm
from IPython.display import display, HTML
display(HTML("""
<style>
.jp-ProgressBar .progress-bar {
    height: 6px !important;  /* Change 6px to your desired thickness */
}
</style>
"""))

def load_breast_cancer_data(test_size=0.2, random_state=42, normalize=False):
    data = load_breast_cancer()
    X, y = data.data, data.target
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def load_iris_data(test_size=0.2, random_state=42, normalize=False):
    data = load_iris()
    X, y = data.data, data.target
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def load_wine_data(test_size=0.2, random_state=42, normalize=False):
    data = load_wine()
    X, y = data.data, data.target
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def load_mnist_data(normalize=True, subset_size=None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    if subset_size:
        x_train = x_train[:subset_size]
        y_train = y_train[:subset_size]
        x_test = x_test[:subset_size//5]
        y_test = y_test[:subset_size//5]
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    if normalize:
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
    
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, x_test, y_train, y_test

def to_one_hot(labels, num_classes):
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot[range(len(labels)), labels] = 1
    return one_hot

def convert_to_python_type(value):
    """Convert numpy types to native Python types for JSON serialization"""
    try:
        if hasattr(value, 'dtype'):
            if 'int' in str(value.dtype):
                return int(value)
            elif 'float' in str(value.dtype):
                return float(value)
            elif 'bool' in str(value.dtype):
                return bool(value)
        # Handle other numpy scalar types
        if hasattr(value, 'item'):
            return value.item()
        return value
    except (ValueError, TypeError, AttributeError):
        # Fallback to string representation if conversion fails
        return str(value)

def MultiStageTrain(dataset_names, default_params=None, 
                           enable_advanced_analysis=True, save_results=True):
    """
    Multi-stage training function with advanced statistical analysis

    Args:
        - dataset_names (list): List of dataset names to be used for training
        - default_params (dict): Default parameters for training
        - enable_advanced_analysis (bool): Whether to run advanced statistical analysis
        - save_results (bool): Whether to save results to files

    Returns:
        - tuple: (all_results, summary_report, timestamp, advanced_analysis)
    """
    
    # Load datasets (same as before)
    datasets = {}
    print(f"Loading datasets: {dataset_names}")
    
    dataset_load_pbar = tqdm(dataset_names, desc="Loading Datasets")
    
    for dataset_name in dataset_load_pbar:
        dataset_load_pbar.set_postfix({'loading': dataset_name})
        
        try:
            if dataset_name == 'breast_cancer':
                datasets['breast_cancer'] = load_breast_cancer_data()
                print(f"Loaded breast_cancer: {datasets['breast_cancer'][0].shape[0]} samples")
            elif dataset_name == 'breast_cancer_norm':
                datasets['breast_cancer_norm'] = load_breast_cancer_data(normalize=True)
                print(f"Loaded breast_cancer_norm: {datasets['breast_cancer_norm'][0].shape[0]} samples")
            elif dataset_name == 'iris':
                datasets['iris'] = load_iris_data()
                print(f"Loaded iris: {datasets['iris'][0].shape[0]} samples")
            elif dataset_name == 'iris_norm':
                datasets['iris_norm'] = load_iris_data(normalize=True)
                print(f"Loaded iris_norm: {datasets['iris_norm'][0].shape[0]} samples")
            elif dataset_name == 'wine':
                datasets['wine'] = load_wine_data()
                print(f"Loaded wine: {datasets['wine'][0].shape[0]} samples")
            elif dataset_name == 'wine_norm':
                datasets['wine_norm'] = load_wine_data(normalize=True)
                print(f"Loaded wine_norm: {datasets['wine_norm'][0].shape[0]} samples")
            elif dataset_name == 'mnist_small':
                datasets['mnist_small'] = load_mnist_data(subset_size=10000)
                print(f"Loaded mnist_small: {datasets['mnist_small'][0].shape[0]} samples")
            elif dataset_name == 'mnist_full':
                datasets['mnist_full'] = load_mnist_data()
                print(f"Loaded mnist_full: {datasets['mnist_full'][0].shape[0]} samples")
            else:
                print(f"Warning: Unknown dataset '{dataset_name}' - skipping")
                
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue
    
    dataset_load_pbar.close()
    
    # Configuration and parameters (same as before)
    DATASET_CONFIGS = {
        'breast_cancer': {'output_size': 1, 'is_classification': True, 'binary': True},
        'iris': {'output_size': 3, 'is_classification': True, 'binary': False},
        'wine': {'output_size': 3, 'is_classification': True, 'binary': False},
        'mnist': {'output_size': 10, 'is_classification': True, 'binary': False},
        'mnist_small': {'output_size': 10, 'is_classification': True, 'binary': False},
        'mnist_full': {'output_size': 10, 'is_classification': True, 'binary': False}
    }
    
    PARAMETER_TESTS = {
        'hidden_dim': [16, 32, 64, 128],
        'initial_population': [10, 20, 50, 100],
        'steps': [10, 30, 50, 75],
        'epochs': [1, 5, 10, 20],
        'lineage_prune_rate': [1000, 1500, 2000],
        'lineage_kept': [500, 800, 1000],
        'num_survivors': [10, 20, 30, 50],
        'q_learning_method': ['neural', 'tabular'],
        'training_method': ['fitness', 'loss'],
        'enable_irxn': [True, False]
    }
    
    # Default parameters
    if default_params is None:
        default_params = {
            'hidden_dim': 30,
            'initial_population': 80,
            'steps': 75,
            'epochs': 5,
            'lineage_prune_rate': 1500,
            'lineage_kept': 800,
            'num_survivors': 33,
            'q_learning_method': 'neural',
            'training_method': 'loss',
            'enable_irxn': True
        }
    
    # Suppress warnings and set seeds
    warnings.filterwarnings("ignore")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize results storage
    all_results = defaultdict(list)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Enhanced parameter sweep function with better error handling
    def run_parameter_sweep(dataset_name, X_train, y_train, X_test, y_test, 
                           param_type='hidden_dim', max_tests=3):
        """Enhanced parameter sweep with better statistics collection"""
        results = []
        base_dataset_name = dataset_name.split('_norm')[0].split('_small')[0].split('_full')[0]
        config = DATASET_CONFIGS[base_dataset_name]
        
        test_values = PARAMETER_TESTS[param_type][:max_tests]
        param_pbar = tqdm(test_values, desc=f"Testing {param_type.replace('_', ' ').title()}", 
                         leave=False, position=1)
        
        for value in param_pbar:
            param_pbar.set_postfix({'current_value': value, 'dataset': dataset_name})
            
            test_params = default_params.copy()
            test_params[param_type] = value
            
            try:
                # Convert numpy arrays to torch tensors
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.FloatTensor(y_train)
                X_test_tensor = torch.FloatTensor(X_test)
                y_test_tensor = torch.FloatTensor(y_test)
                
                # Run ARCNET training
                allmods, lineagesnap, flname, bestmod, stats, assembly_registry = Trainer(
                    X_train=X_train_tensor,
                    y_train=y_train_tensor,
                    input_dim=X_train.shape[1],
                    hidden_dim=test_params['hidden_dim'],
                    output_dim=config['output_size'],
                    initial_population=test_params['initial_population'],
                    steps=test_params['steps'],
                    epochs=test_params['epochs'],
                    lineage_prune_rate=test_params['lineage_prune_rate'],
                    lineage_kept=test_params['lineage_kept'],
                    num_survivors=test_params['num_survivors'],
                    q_learning_method=test_params['q_learning_method'],
                    training_method=test_params['training_method'],
                    enable_irxn=test_params['enable_irxn'],
                    enable_model_save=False,
                    enable_bias_elimination=False,
                    enable_lineage_snap=False,
                    experiment_name=f"arcnet_{dataset_name}_{param_type}_{value}",
                    track_best_models=True,
                    top_k_per_generation=5,
                    debug=False
                )

                # Get the best model from the final population
                if allmods and len(allmods) > 0:
                    best_model = max(allmods, key=lambda m: m.fitness)
                else:
                    best_model = None
            
                # Initialize default values
                train_accuracy = 0.0
                test_accuracy = 0.0
                train_report = {'weighted avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}}
                test_report = {'weighted avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}}
                
                # Enhanced evaluation with more metrics
                if best_model is not None:
                    best_model.eval()
                    with torch.no_grad():
                        train_outputs = best_model(X_train_tensor)
                        test_outputs = best_model(X_test_tensor)

                        # Calculate accuracies and additional metrics
                        if config['binary']:
                            train_preds = (torch.sigmoid(train_outputs) > 0.5).float()
                            test_preds = (torch.sigmoid(test_outputs) > 0.5).float()
                            train_accuracy = (train_preds.squeeze() == y_train_tensor.squeeze()).float().mean().item()
                            test_accuracy = (test_preds.squeeze() == y_test_tensor.squeeze()).float().mean().item()
                        else:
                            train_preds = torch.argmax(train_outputs, dim=1)
                            test_preds = torch.argmax(test_outputs, dim=1)
                            if len(y_train_tensor.shape) > 1:
                                y_train_labels = torch.argmax(y_train_tensor, dim=1)
                                y_test_labels = torch.argmax(y_test_tensor, dim=1)
                            else:
                                y_train_labels = y_train_tensor.long()
                                y_test_labels = y_test_tensor.long()
                            train_accuracy = (train_preds == y_train_labels).float().mean().item()
                            test_accuracy = (test_preds == y_test_labels).float().mean().item()
                        
                        # Generate enhanced classification reports
                        if config['binary']:
                            train_report = classification_report(
                                y_train_tensor.squeeze().numpy(), 
                                train_preds.squeeze().numpy(),
                                output_dict=True, zero_division=0
                            )
                            test_report = classification_report(
                                y_test_tensor.squeeze().numpy(), 
                                test_preds.squeeze().numpy(),
                                output_dict=True, zero_division=0
                            )
                        else:
                            train_report = classification_report(
                                y_train_labels.numpy(), 
                                train_preds.numpy(),
                                output_dict=True, zero_division=0
                            )
                            test_report = classification_report(
                                y_test_labels.numpy(), 
                                test_preds.numpy(),
                                output_dict=True, zero_division=0
                            )
                
                # Enhanced result collection with additional metrics
                result = {
                    'dataset': dataset_name,
                    'parameter_type': param_type,
                    'parameter_value': convert_to_python_type(value),
                    'train_accuracy': convert_to_python_type(train_accuracy),
                    'test_accuracy': convert_to_python_type(test_accuracy),
                    'final_population_size': len(allmods),
                    'best_model_fitness': convert_to_python_type(best_model.fitness if best_model else 0),
                    'best_model_id': best_model.id if hasattr(best_model, 'id') else 'unknown',
                    'train_precision': convert_to_python_type(train_report.get('weighted avg', {}).get('precision', 0.0)),
                    'train_recall': convert_to_python_type(train_report.get('weighted avg', {}).get('recall', 0.0)),
                    'train_f1': convert_to_python_type(train_report.get('weighted avg', {}).get('f1-score', 0.0)),
                    'test_precision': convert_to_python_type(test_report.get('weighted avg', {}).get('precision', 0.0)),
                    'test_recall': convert_to_python_type(test_report.get('weighted avg', {}).get('recall', 0.0)),
                    'test_f1': convert_to_python_type(test_report.get('weighted avg', {}).get('f1-score', 0.0)),
                    'lineage_file': flname,
                    'generation_stats': stats if isinstance(stats, (list, dict)) else [],
                    # Additional metrics for enhanced analysis
                    'parameter_config': test_params.copy(),
                    'training_time': convert_to_python_type(getattr(best_model, 'training_time', 0)),
                    'convergence_generation': convert_to_python_type(getattr(best_model, 'generation', 0))
                }
                results.append(result)
                
                param_pbar.set_postfix({
                    'test_acc': f"{test_accuracy:.3f}",
                    'test_f1': f"{result['test_f1']:.3f}",
                    'fitness': f"{result['best_model_fitness']:.3f}",
                    'pop_size': len(allmods)
                })
                
            except Exception as e:
                param_pbar.set_postfix({'status': f"Error: {str(e)[:30]}..."})
                tqdm.write(f"Error with {param_type}={value}: {e}")
                continue
        
        param_pbar.close()
        return results
    
    # [Run all parameter sweeps - same logic as original but with enhanced error handling]
    dataset_pbar = tqdm(datasets.items(), desc="Processing Datasets", position=0)
    
    for dataset_name, (X_train, X_test, y_train, y_test) in dataset_pbar:
        dataset_pbar.set_postfix({'current_dataset': dataset_name})
        
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"TESTING DATASET: {dataset_name.upper()}")
        tqdm.write(f"{'='*60}")
        tqdm.write(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Run all parameter sweeps (same as original)
        for param_name, param_display in [
            ('hidden_dim', 'HIDDEN DIMENSION'),
            ('initial_population', 'POPULATION SIZE'), 
            ('steps', 'EVOLUTION STEPS'),
            ('epochs', 'EPOCHS'),
            ('lineage_prune_rate', 'LINEAGE PRUNE RATE'),
            ('lineage_kept', 'LINEAGE KEPT'),
            ('num_survivors', 'NUMBER OF SURVIVORS'),
            ('q_learning_method', 'Q-LEARNING METHOD'),
            ('training_method', 'TRAINING METHOD'),
            ('enable_irxn', 'INTERACTION ENABLE')
        ]:
            tqdm.write(f"\n{'-'*40}")
            tqdm.write(f"{param_display} ANALYSIS")
            tqdm.write(f"{'-'*40}")
            
            max_tests = 2 if param_name in ['q_learning_method', 'training_method', 'enable_irxn'] else 3
            param_results = run_parameter_sweep(dataset_name, X_train, y_train, X_test, y_test, 
                                              param_name, max_tests=max_tests)
            all_results[param_name].extend(param_results)
        
        # Update progress
        total_experiments = sum(len(results) for results in all_results.values())
        dataset_pbar.set_postfix({
            'completed_experiments': total_experiments,
            'current_dataset': dataset_name
        })
    
    dataset_pbar.close()
    
    # Enhanced Analysis Section
    tqdm.write("\n" + "="*80)
    tqdm.write("ENHANCED STATISTICAL ANALYSIS")
    tqdm.write("="*80)
    
    advanced_analysis = None
    
    if enable_advanced_analysis:
        try:
            # Create advanced analyzer
            analyzer = ARCNETParameterAnalyzer(results_dict=all_results)
            
            # Run comprehensive analysis
            tqdm.write("Running advanced statistical analysis...")
            advanced_analysis = analyzer.create_comprehensive_report()
            
            tqdm.write("Advanced analysis completed successfully!")
            
        except Exception as e:
            tqdm.write(f"Advanced analysis failed: {e}")
            tqdm.write("Continuing with basic analysis...")
    
    # Save results (enhanced with better error handling)
    if save_results:
        tqdm.write(f"\n{'='*60}")
        tqdm.write("SAVING ENHANCED RESULTS")
        tqdm.write(f"{'='*60}")
        
        # Create results directory
        os.makedirs('experiments/multistage_results', exist_ok=True)
        
        # Save individual parameter results
        save_pbar = tqdm(all_results.items(), desc="Saving Parameter Results", leave=False)
        
        for param_type, results in save_pbar:
            save_pbar.set_postfix({'saving': param_type})
            
            if results:
                df = pd.DataFrame(results)
                filename = f'experiments/multistage_results/arcnet_{param_type}_{timestamp}.csv'
                try:
                    df.to_csv(filename, index=False)
                    tqdm.write(f"Saved {param_type} results to {filename}")
                except Exception as e:
                    tqdm.write(f"Error saving {param_type}: {e}")
        
        save_pbar.close()
        
        # Create and save enhanced summary report
        summary_report = {
            'timestamp': timestamp,
            'datasets_tested': list(datasets.keys()),
            'parameters_tested': list(all_results.keys()),
            'total_experiments': sum(len(results) for results in all_results.values()),
            'best_configurations': {},
            'overall_performance': {},
            'advanced_analysis_enabled': enable_advanced_analysis
        }
        
        # Add best configurations with enhanced metrics
        for param_type, results in all_results.items():
            if results:
                df = pd.DataFrame(results)
                best_overall = df.loc[df['test_accuracy'].idxmax()]
                summary_report['best_configurations'][param_type] = {
                    'dataset': best_overall['dataset'],
                    'parameter_value': best_overall['parameter_value'],
                    'test_accuracy': float(best_overall['test_accuracy']),
                    'test_f1': float(best_overall['test_f1']),
                    'test_precision': float(best_overall['test_precision']),
                    'test_recall': float(best_overall['test_recall']),
                    'best_model_fitness': float(best_overall['best_model_fitness'])
                }
        
        # Enhanced overall performance statistics
        all_experiments = []
        for results in all_results.values():
            all_experiments.extend(results)
        
        if all_experiments:
            overall_df = pd.DataFrame(all_experiments)
            summary_report['overall_performance'] = {
                'mean_test_accuracy': float(overall_df['test_accuracy'].mean()),
                'std_test_accuracy': float(overall_df['test_accuracy'].std()),
                'max_test_accuracy': float(overall_df['test_accuracy'].max()),
                'min_test_accuracy': float(overall_df['test_accuracy'].min()),
                'mean_test_f1': float(overall_df['test_f1'].mean()),
                'std_test_f1': float(overall_df['test_f1'].std()),
                'max_test_f1': float(overall_df['test_f1'].max()),
                'mean_test_precision': float(overall_df['test_precision'].mean()),
                'mean_test_recall': float(overall_df['test_recall'].mean()),
                'mean_fitness': float(overall_df['best_model_fitness'].mean()),
                'max_fitness': float(overall_df['best_model_fitness'].max())
            }
        
        # Add advanced analysis results to summary
        if advanced_analysis:
            summary_report['advanced_analysis'] = advanced_analysis
        
        # Save enhanced summary report
        import json
        summary_filename = f'experiments/multistage_results/arcnet_enhanced_summary_{timestamp}.json'
        try:
            with open(summary_filename, 'w') as f:
                json.dump(summary_report, f, indent=2, default=convert_to_python_type)
            tqdm.write(f"Enhanced summary report saved to {summary_filename}")
        except Exception as e:
            tqdm.write(f"JSON serialization error: {e}")
            # Save as pickle backup
            import pickle
            backup_filename = f'experiments/multistage_results/arcnet_enhanced_summary_{timestamp}.pkl'
            with open(backup_filename, 'wb') as f:
                pickle.dump(summary_report, f)
            tqdm.write(f"Backup summary saved as pickle: {backup_filename}")
    
    # Final Summary
    tqdm.write(f"\n{'='*80}")
    tqdm.write("ENHANCED MULTISTAGE ANALYSIS COMPLETE")
    tqdm.write(f"{'='*80}")
    
    total_experiments = sum(len(results) for results in all_results.values())
    tqdm.write(f"Total experiments completed: {total_experiments}")
    tqdm.write(f"Datasets analyzed: {len(datasets)}")
    tqdm.write(f"Parameters tested: {len(all_results)}")
    
    if advanced_analysis:
        tqdm.write("Advanced statistical analysis: ✓ Completed")
        if 'summary_stats' in advanced_analysis:
            tqdm.write(f"Best overall performance: {advanced_analysis['summary_stats']['max_performance']:.4f}")
    
    return all_results, summary_report, timestamp, advanced_analysis

# Example usage function
# def run_comprehensive_arcnet_analysis(dataset_names=['breast_cancer', 'iris'], 
#                                      max_tests_per_param=3):
#     """
#     Run a comprehensive ARCNET analysis with advanced statistics
    
#     Args:
#         dataset_names: List of datasets to test
#         max_tests_per_param: Maximum parameter values to test per parameter
    
#     Returns:
#         Complete analysis results
#     """
#     print("Starting comprehensive ARCNET parameter analysis...")
    
#     results, summary, timestamp, advanced = EnhancedMultiStageTrain(
#         dataset_names=dataset_names,
#         enable_advanced_analysis=True,
#         save_results=True
#     )
    
#     print(f"\nAnalysis complete! Results saved with timestamp: {timestamp}")
#     print("Check the experiments/multistage_results/ directory for detailed outputs.")
    
#     return results, summary, timestamp, advanced