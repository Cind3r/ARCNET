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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import defaultdict
import os
from datetime import datetime
from tqdm.notebook import tqdm
import traceback

# Enhanced Data Loading Functions with additional preprocessing options
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

# =========================================
# ========== MAIN FUNCTION =============
# =========================================

def MultiStageTrain(dataset_names, norm=False, default_params=None):
    """Multi-stage training function with predefined parameters for testing purposes."""
    
    # Initialize timestamp first
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # IMMEDIATE OUTPUT TO CONFIRM FUNCTION IS RUNNING
    print("="*80)
    print("ARCNET MULTISTAGE TRAINING ANALYSIS STARTING")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Requested datasets: {dataset_names}")
    print(f"Default params provided: {default_params is not None}")
    
    # Create necessary directories immediately
    try:
        os.makedirs('experiments/multistage_results', exist_ok=True)
        os.makedirs('experiments/lineage_snaps', exist_ok=True)
        os.makedirs('docs', exist_ok=True)
        print("✓ Created necessary directories")
    except Exception as e:
        print(f"✗ Error creating directories: {e}")
    
    datasets = {}
    print(f"\nLoading datasets: {dataset_names}")
    
    # Load datasets with better error handling
    for dataset_name in dataset_names:
        print(f"Loading {dataset_name}...", end=" ")
        try:
            if dataset_name == 'breast_cancer':
                datasets['breast_cancer'] = load_breast_cancer_data()
                print(f"✓ {datasets['breast_cancer'][0].shape[0]} samples")
            elif dataset_name == 'breast_cancer_norm':
                datasets['breast_cancer_norm'] = load_breast_cancer_data(normalize=True)
                print(f"✓ {datasets['breast_cancer_norm'][0].shape[0]} samples")
            elif dataset_name == 'iris':
                datasets['iris'] = load_iris_data()
                print(f"✓ {datasets['iris'][0].shape[0]} samples")
            elif dataset_name == 'iris_norm':
                datasets['iris_norm'] = load_iris_data(normalize=True)
                print(f"✓ {datasets['iris_norm'][0].shape[0]} samples")
            elif dataset_name == 'wine':
                datasets['wine'] = load_wine_data()
                print(f"✓ {datasets['wine'][0].shape[0]} samples")
            elif dataset_name == 'wine_norm':
                datasets['wine_norm'] = load_wine_data(normalize=True)
                print(f"✓ {datasets['wine_norm'][0].shape[0]} samples")
            elif dataset_name == 'mnist_small':
                datasets['mnist_small'] = load_mnist_data(subset_size=10000)
                print(f"✓ {datasets['mnist_small'][0].shape[0]} samples")
            elif dataset_name == 'mnist_full':
                datasets['mnist_full'] = load_mnist_data()
                print(f"✓ {datasets['mnist_full'][0].shape[0]} samples")
            else:
                print(f"⚠ Unknown dataset '{dataset_name}' - skipping")
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    # Validate datasets
    if not datasets:
        print("ERROR: No datasets were successfully loaded!")
        # Still create empty files for debugging
        empty_report = {
            'timestamp': timestamp,
            'error': 'No datasets loaded',
            'datasets_requested': dataset_names
        }
        import json
        with open(f'experiments/multistage_results/arcnet_error_{timestamp}.json', 'w') as f:
            json.dump(empty_report, f, indent=2)
        return {}, empty_report, timestamp
    
    print(f"✓ Successfully loaded {len(datasets)} datasets")
    
    DATASET_CONFIGS = {
        'breast_cancer': {'output_size': 1, 'is_classification': True, 'binary': True},
        'iris': {'output_size': 3, 'is_classification': True, 'binary': False},
        'wine': {'output_size': 3, 'is_classification': True, 'binary': False},
        'mnist': {'output_size': 10, 'is_classification': True, 'binary': False},
        'mnist_small': {'output_size': 10, 'is_classification': True, 'binary': False},
        'mnist_full': {'output_size': 10, 'is_classification': True, 'binary': False}
    }   

    PARAMETER_TESTS = {
        'hidden_dim': [16, 32, 64],  # Reduced for faster testing
        'initial_population': [10, 20, 50],
        'steps': [10, 20, 30],  # Reduced for faster testing
        'epochs': [1, 2, 5],
        'lineage_prune_rate': [1000, 1500],
        'lineage_kept': [500, 800],
        'num_survivors': [10, 20, 30],
        'q_learning_method': ['neural', 'tabular'],
        'training_method': ['fitness', 'loss'],
        'enable_irxn': [True, False]
    }

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize results storage
    all_results = defaultdict(list)

    def run_parameter_sweep(dataset_name, X_train, y_train, X_test, y_test, 
                           param_type='hidden_dim', max_tests=2, default_params=default_params):
        """Run a parameter sweep for a specific parameter type"""
        
        print(f"\n--- Starting {param_type} sweep for {dataset_name} ---")
        results = []
        base_dataset_name = dataset_name.split('_norm')[0].split('_small')[0].split('_full')[0]
        config = DATASET_CONFIGS[base_dataset_name]
        
        # Default parameters - MUCH SMALLER FOR TESTING
        if default_params is None:
            default_params = {
                'hidden_dim': 16,
                'initial_population': 10,
                'steps': 5,  # Very small for testing
                'epochs': 1,
                'lineage_prune_rate': 1000,
                'lineage_kept': 500,
                'num_survivors': 5,
                'q_learning_method': 'neural',
                'training_method': 'loss',
                'enable_irxn': True
            }
        
        # Test different values for the specified parameter
        test_values = PARAMETER_TESTS[param_type][:max_tests]
        print(f"Testing values: {test_values}")
        
        for i, value in enumerate(test_values):
            print(f"\n  Test {i+1}/{len(test_values)}: {param_type}={value}")
            
            # Update parameter
            test_params = default_params.copy()
            test_params[param_type] = value
            print(f"  Parameters: {test_params}")
            
            try:
                # Convert numpy arrays to torch tensors
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.FloatTensor(y_train)
                X_test_tensor = torch.FloatTensor(X_test)
                y_test_tensor = torch.FloatTensor(y_test)
                
                print(f"  Starting ARCNET training...")
                
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
                    top_k_per_generation=3,
                    debug=False
                )
                
                print(f"  Training completed. Population size: {len(allmods)}")
                print(f"  Best model found: {bestmod is not None}")
                
                # Initialize default values
                train_accuracy = 0.0
                test_accuracy = 0.0
                train_report = {'weighted avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}}
                test_report = {'weighted avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}}
                
                # Evaluate the best model if it exists
                if bestmod is not None:
                    print("  Evaluating best model...")
                    bestmod.eval()
                    with torch.no_grad():
                        # Make predictions
                        train_outputs = bestmod(X_train_tensor)
                        test_outputs = bestmod(X_test_tensor)
                        
                        # Calculate accuracies based on task type
                        if config['binary']:
                            # Binary classification
                            train_preds = (torch.sigmoid(train_outputs) > 0.5).float()
                            test_preds = (torch.sigmoid(test_outputs) > 0.5).float()
                            train_accuracy = (train_preds.squeeze() == y_train_tensor.squeeze()).float().mean().item()
                            test_accuracy = (test_preds.squeeze() == y_test_tensor.squeeze()).float().mean().item()
                            
                            # Generate classification report
                            train_report = classification_report(
                                y_train_tensor.squeeze().numpy(), 
                                train_preds.squeeze().numpy(),
                                output_dict=True,
                                zero_division=0
                            )
                            test_report = classification_report(
                                y_test_tensor.squeeze().numpy(), 
                                test_preds.squeeze().numpy(),
                                output_dict=True,
                                zero_division=0
                            )
                        else:
                            # Multi-class classification
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
                            
                            # Generate classification report
                            train_report = classification_report(
                                y_train_labels.numpy(), 
                                train_preds.numpy(),
                                output_dict=True,
                                zero_division=0
                            )
                            test_report = classification_report(
                                y_test_labels.numpy(), 
                                test_preds.numpy(),
                                output_dict=True,
                                zero_division=0
                            )
                    
                    print(f"  Train accuracy: {train_accuracy:.4f}")
                    print(f"  Test accuracy: {test_accuracy:.4f}")
                else:
                    print("  Warning: No best model found, using default values")
                
                # Store results
                result = {
                    'dataset': dataset_name,
                    'parameter_type': param_type,
                    'parameter_value': value,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'final_population_size': len(allmods) if allmods else 0,
                    'best_model_id': bestmod.id if hasattr(bestmod, 'id') and bestmod else 'none',
                    'train_precision': train_report.get('weighted avg', {}).get('precision', 0.0),
                    'train_recall': train_report.get('weighted avg', {}).get('recall', 0.0),
                    'train_f1': train_report.get('weighted avg', {}).get('f1-score', 0.0),
                    'test_precision': test_report.get('weighted avg', {}).get('precision', 0.0),
                    'test_recall': test_report.get('weighted avg', {}).get('recall', 0.0),
                    'test_f1': test_report.get('weighted avg', {}).get('f1-score', 0.0),
                    'lineage_file': flname if flname else 'none',
                    'generation_stats': str(stats) if stats else 'none',
                    'experiment_completed': True,
                    'error_message': None
                }
                results.append(result)
                print(f"  ✓ Result stored: Test Acc={test_accuracy:.4f}, F1={result['test_f1']:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                traceback.print_exc()
                
                # Store error result
                error_result = {
                    'dataset': dataset_name,
                    'parameter_type': param_type,
                    'parameter_value': value,
                    'train_accuracy': 0.0,
                    'test_accuracy': 0.0,
                    'final_population_size': 0,
                    'best_model_id': 'error',
                    'train_precision': 0.0,
                    'train_recall': 0.0,
                    'train_f1': 0.0,
                    'test_precision': 0.0,
                    'test_recall': 0.0,
                    'test_f1': 0.0,
                    'lineage_file': 'error',
                    'generation_stats': 'error',
                    'experiment_completed': False,
                    'error_message': str(e)
                }
                results.append(error_result)
                continue
        
        print(f"--- Completed {param_type} sweep: {len(results)} results ---")
        return results

    # Process datasets
    print(f"\n" + "="*60)
    print("STARTING PARAMETER SWEEPS")
    print("="*60)
    
    experiment_count = 0
    
    for dataset_name, (X_train, X_test, y_train, y_test) in datasets.items():
        print(f"\n{'='*50}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*50}")
        print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Run a smaller set of tests to ensure completion
        print("\n1. HIDDEN DIMENSION ANALYSIS")
        print("-" * 30)
        hidden_results = run_parameter_sweep(dataset_name, X_train, y_train, X_test, y_test, 
                                           'hidden_dim', max_tests=2)
        all_results['hidden_dim'].extend(hidden_results)
        experiment_count += len(hidden_results)
        
        print("\n2. POPULATION SIZE ANALYSIS")
        print("-" * 30)
        pop_results = run_parameter_sweep(dataset_name, X_train, y_train, X_test, y_test, 
                                        'initial_population', max_tests=2)
        all_results['initial_population'].extend(pop_results)
        experiment_count += len(pop_results)
    
    # FORCE SAVE RESULTS EVEN IF INCOMPLETE
    print(f"\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    print(f"Total experiments completed: {experiment_count}")
    
    # Save individual parameter results
    for param_type, results in all_results.items():
        if results:
            df = pd.DataFrame(results)
            filename = f'experiments/multistage_results/arcnet_{param_type}_{timestamp}.csv'
            df.to_csv(filename, index=False)
            print(f"✓ Saved {len(results)} {param_type} results to {filename}")
        else:
            print(f"⚠ No results for {param_type}")
    
    # Create summary report
    summary_report = {
        'timestamp': timestamp,
        'datasets_tested': list(datasets.keys()),
        'parameters_tested': list(all_results.keys()),
        'total_experiments': experiment_count,
        'experiments_per_param': {k: len(v) for k, v in all_results.items()},
        'successful_experiments': sum(1 for results in all_results.values() for r in results if r.get('experiment_completed', False)),
        'failed_experiments': sum(1 for results in all_results.values() for r in results if not r.get('experiment_completed', True)),
        'best_configurations': {},
        'overall_performance': {}
    }
    
    # Add best configurations if we have results
    for param_type, results in all_results.items():
        if results:
            successful_results = [r for r in results if r.get('experiment_completed', False)]
            if successful_results:
                df = pd.DataFrame(successful_results)
                best_overall = df.loc[df['test_accuracy'].idxmax()]
                summary_report['best_configurations'][param_type] = {
                    'dataset': best_overall['dataset'],
                    'parameter_value': best_overall['parameter_value'],
                    'test_accuracy': float(best_overall['test_accuracy']),
                    'test_f1': float(best_overall['test_f1'])
                }

    # Calculate overall performance statistics
    all_successful_results = []
    for results in all_results.values():
        all_successful_results.extend([r for r in results if r.get('experiment_completed', False)])
    
    if all_successful_results:
        overall_df = pd.DataFrame(all_successful_results)
        summary_report['overall_performance'] = {
            'mean_test_accuracy': float(overall_df['test_accuracy'].mean()),
            'std_test_accuracy': float(overall_df['test_accuracy'].std()),
            'max_test_accuracy': float(overall_df['test_accuracy'].max()),
            'mean_test_f1': float(overall_df['test_f1'].mean()),
            'std_test_f1': float(overall_df['test_f1'].std()),
            'max_test_f1': float(overall_df['test_f1'].max())
        }
    
    # Save summary report
    import json
    summary_filename = f'experiments/multistage_results/arcnet_summary_{timestamp}.json'
    with open(summary_filename, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"✓ Summary report saved to {summary_filename}")
    
    # Print final summary
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total experiments: {experiment_count}")
    print(f"Successful: {summary_report.get('successful_experiments', 0)}")
    print(f"Failed: {summary_report.get('failed_experiments', 0)}")
    
    if summary_report.get('best_configurations'):
        print("\nBest configurations found:")
        for param_type, config in summary_report['best_configurations'].items():
            print(f"  {param_type}: {config['parameter_value']} (Acc: {config['test_accuracy']:.4f})")
    
    return all_results, summary_report, timestamp