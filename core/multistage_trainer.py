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

    """Multi-stage training function with predefined parameters for testing purposes.
    Args:
        - dataset_names (list): List of dataset names to be used for training.
            - This is primarily used for disabling/enabling MNIST as it is the largest and takes the longest. 
            - 'breast_cancer', 'breast_cancer_norm', 'iris', 'iris_norm', 'wine', 'wine_norm', 'mnist_small', 'mnist_full'
        - norm (bool): Whether to include normalize dataset training as well.
    """
    datasets = {}

    # Standard loading
    # Load only datasets specified in dataset_names
    if 'breast_cancer' in dataset_names:
        datasets['breast_cancer'] = load_breast_cancer_data()
    if 'breast_cancer_norm' in dataset_names:
        datasets['breast_cancer_norm'] = load_breast_cancer_data(normalize=True)
    if 'iris' in dataset_names:
        datasets['iris'] = load_iris_data()
    if 'iris_norm' in dataset_names:
        datasets['iris_norm'] = load_iris_data(normalize=True)
    if 'wine' in dataset_names:
        datasets['wine'] = load_wine_data()
    if 'wine_norm' in dataset_names:
        datasets['wine_norm'] = load_wine_data(normalize=True)
    if 'mnist_small' in dataset_names:
        datasets['mnist_small'] = load_mnist_data(subset_size=10000)
    if 'mnist_full' in dataset_names:
        datasets['mnist_full'] = load_mnist_data()
    
    DATASET_CONFIGS = {
    'breast_cancer': {
        'output_size': 1,
        'is_classification': True,
        'binary': True
    },
    'iris': {
        'output_size': 3,
        'is_classification': True,
        'binary': False
    },
    'wine': {
        'output_size': 3,
        'is_classification': True,
        'binary': False
    },
    'mnist': {
        'output_size': 10,
        'is_classification': True,
        'binary': False
    },
    'mnist_small': {
        'output_size': 10,
        'is_classification': True,
        'binary': False
    },
    'mnist_full': {
        'output_size': 10,
        'is_classification': True,
        'binary': False
    }
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

    default_params = default_params 

    #     allmods, lineagesnap, flname, bestmod, stats, assembly_registry = Trainer(
    #     X_train=X_train,
    #     y_train=y_train,
    #     input_dim=X_train.shape[1],
    #     hidden_dim=30,
    #     output_dim=2,
    #     initial_population=50,
    #     steps=15,
    #     epochs=1,
    #     lineage_prune_rate=1500,
    #     lineage_kept=800,
    #     num_survivors=25,
    #     q_learning_method='neural',
    #     training_method='fitness',
    #     enable_irxn=True,
    #     enable_model_save=False,
    #     enable_bias_elimination=False,
    #     enable_lineage_snap=False,
    #     experiment_name="arcnet_modular",
    #     track_best_models=True,
    #     top_k_per_generation=5,
    #     debug=False
    # )

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize results storage
    all_results = defaultdict(list)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*80)
    print("ARCNET MULTISTAGE TRAINING ANALYSIS")
    print("="*80)

    # Function to run parameter sweep
    # ...existing code...

    # Function to run parameter sweep
    def run_parameter_sweep(dataset_name, X_train, y_train, X_test, y_test, 
                           param_type='hidden_dim', max_tests=3, default_params=default_params):
        """
        Run a parameter sweep for a specific parameter type
        """
        results = []
        base_dataset_name = dataset_name.split('_norm')[0].split('_small')[0].split('_full')[0]
        config = DATASET_CONFIGS[base_dataset_name]
        
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
        
        # Test different values for the specified parameter
        test_values = PARAMETER_TESTS[param_type][:max_tests]

        # Create progress bar for parameter values
        param_pbar = tqdm(test_values, desc=f"Testing {param_type.replace('_', ' ').title()}", 
                         leave=False, position=1)
        
        for value in param_pbar:
            param_pbar.set_postfix({'current_value': value, 'dataset': dataset_name})
            
            # Update parameter
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
                    enable_lineage_snap=True,
                    experiment_name=f"arcnet_{dataset_name}_{param_type}_{value}",
                    track_best_models=True,
                    top_k_per_generation=5,
                    debug=False
                )
                
                # Evaluate the best model
                if bestmod is not None:
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
                        if config['binary']:
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
                
                # Store results
                result = {
                    'dataset': dataset_name,
                    'parameter_type': param_type,
                    'parameter_value': value,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'final_population_size': len(allmods),
                    'best_model_id': bestmod.id if hasattr(bestmod, 'id') else 'unknown',
                    'train_precision': train_report.get('weighted avg', {}).get('precision', 0.0),
                    'train_recall': train_report.get('weighted avg', {}).get('recall', 0.0),
                    'train_f1': train_report.get('weighted avg', {}).get('f1-score', 0.0),
                    'test_precision': test_report.get('weighted avg', {}).get('precision', 0.0),
                    'test_recall': test_report.get('weighted avg', {}).get('recall', 0.0),
                    'test_f1': test_report.get('weighted avg', {}).get('f1-score', 0.0),
                    'lineage_file': flname,
                    'generation_stats': stats if isinstance(stats, (list, dict)) else []
                }
                results.append(result)
                
                # Update progress bar with current results
                param_pbar.set_postfix({
                    'test_acc': f"{test_accuracy:.3f}",
                    'test_f1': f"{result['test_f1']:.3f}",
                    'pop_size': len(allmods)
                })
                
            except Exception as e:
                param_pbar.set_postfix({'status': f"Error: {str(e)[:30]}..."})
                tqdm.write(f"Error with {param_type}={value}: {e}")
                continue
        
        param_pbar.close()
        return results

    # Test each dataset with different parameter configurations
    dataset_pbar = tqdm(datasets.items(), desc="Processing Datasets", position=0)
    
    for dataset_name, (X_train, X_test, y_train, y_test) in dataset_pbar:
        dataset_pbar.set_postfix({'current_dataset': dataset_name})
        
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"TESTING DATASET: {dataset_name.upper()}")
        tqdm.write(f"{'='*60}")
        tqdm.write(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Test 1: Hidden Dimension Analysis
        tqdm.write(f"\n{'-'*40}")
        tqdm.write("1. HIDDEN DIMENSION ANALYSIS")
        tqdm.write(f"{'-'*40}")
        hidden_results = run_parameter_sweep(dataset_name, X_train, y_train, X_test, y_test, 
                                           'hidden_dim', max_tests=3)
        all_results['hidden_dim'].extend(hidden_results)
        
        # Test 2: Population Size Analysis
        tqdm.write(f"\n{'-'*40}")
        tqdm.write("2. POPULATION SIZE ANALYSIS")
        tqdm.write(f"{'-'*40}")
        pop_results = run_parameter_sweep(dataset_name, X_train, y_train, X_test, y_test, 
                                        'initial_population', max_tests=3)
        all_results['initial_population'].extend(pop_results)
        
        # Test 3: Evolution Steps Analysis
        tqdm.write(f"\n{'-'*40}")
        tqdm.write("3. EVOLUTION STEPS ANALYSIS")
        tqdm.write(f"{'-'*40}")
        steps_results = run_parameter_sweep(dataset_name, X_train, y_train, X_test, y_test, 
                                          'steps', max_tests=3)
        all_results['steps'].extend(steps_results)
        
        # Update main progress bar with completion stats
        total_experiments = sum(len(results) for results in all_results.values())
        dataset_pbar.set_postfix({
            'completed_experiments': total_experiments,
            'current_dataset': dataset_name
        })
    
    dataset_pbar.close()

    # Comprehensive Results Analysis
    tqdm.write("\n" + "="*80)
    tqdm.write("COMPREHENSIVE RESULTS ANALYSIS")
    tqdm.write("="*80)

    # Create summary DataFrames
    def create_summary_df(results, param_type):
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        summary = df.groupby(['dataset', 'parameter_value']).agg({
            'train_accuracy': ['mean', 'std', 'max'],
            'test_accuracy': ['mean', 'std', 'max'],
            'test_f1': ['mean', 'std', 'max'],
            'final_population_size': 'mean'
        }).round(4)
        
        return summary

    # Generate summaries for each parameter type with progress tracking
    summary_pbar = tqdm(all_results.items(), desc="Generating Summaries", leave=False)
    
    for param_type, results in summary_pbar:
        summary_pbar.set_postfix({'current_param': param_type})
        
        if results:
            tqdm.write(f"\n{'-'*60}")
            tqdm.write(f"{param_type.upper()} SUMMARY")
            tqdm.write(f"{'-'*60}")
            
            summary_df = create_summary_df(results, param_type)
            tqdm.write(summary_df.to_string())
    
    summary_pbar.close()

    # Best performing configurations
    tqdm.write(f"\n{'='*60}")
    tqdm.write("BEST PERFORMING CONFIGURATIONS")
    tqdm.write(f"{'='*60}")

    best_config_pbar = tqdm(all_results.items(), desc="Finding Best Configurations", leave=False)
    
    for param_type, results in best_config_pbar:
        best_config_pbar.set_postfix({'current_param': param_type})
        
        if results:
            df = pd.DataFrame(results)
            best_per_dataset = df.loc[df.groupby('dataset')['test_accuracy'].idxmax()]
            
            tqdm.write(f"\n{param_type.upper()} - Best per dataset:")
            for _, row in best_per_dataset.iterrows():
                tqdm.write(f"  {row['dataset']}: {row['parameter_value']} -> "
                          f"Test Accuracy: {row['test_accuracy']:.4f}, "
                          f"Test F1: {row['test_f1']:.4f}, "
                          f"Population: {row['final_population_size']}")
    
    best_config_pbar.close()

    # Generate comprehensive visualization
    tqdm.write(f"\n{'='*60}")
    tqdm.write("GENERATING VISUALIZATION PLOTS")
    tqdm.write(f"{'='*60}")

    # Create plots
    n_param_types = len(all_results)
    if n_param_types > 0:
        fig, axes = plt.subplots(2, min(3, n_param_types), figsize=(18, 12))
        if n_param_types == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        elif n_param_types == 2:
            axes = axes.reshape(2, 2)
        
        fig.suptitle('ARCNET Multistage Training Analysis Results', fontsize=16, fontweight='bold')

        plot_pbar = tqdm(all_results.items(), desc="Creating Plots", leave=False)
        plot_idx = 0
        
        for param_type, results in plot_pbar:
            plot_pbar.set_postfix({'current_plot': param_type})
            
            if results and plot_idx < 3:
                df = pd.DataFrame(results)
                
                # Test accuracy heatmap
                ax1 = axes[0, plot_idx]
                pivot_data = df.pivot_table(values='test_accuracy', 
                                          index='dataset', 
                                          columns='parameter_value', 
                                          aggfunc='mean')
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax1)
                ax1.set_title(f'{param_type.replace("_", " ").title()} vs Test Accuracy')
                ax1.set_xlabel('Parameter Value')
                ax1.set_ylabel('Dataset')
                
                # F1 score heatmap
                ax2 = axes[1, plot_idx]
                pivot_data_f1 = df.pivot_table(values='test_f1', 
                                             index='dataset', 
                                             columns='parameter_value', 
                                             aggfunc='mean')
                sns.heatmap(pivot_data_f1, annot=True, fmt='.3f', cmap='plasma', ax=ax2)
                ax2.set_title(f'{param_type.replace("_", " ").title()} vs Test F1')
                ax2.set_xlabel('Parameter Value')
                ax2.set_ylabel('Dataset')
                
                plot_idx += 1
        
        plot_pbar.close()

        plt.tight_layout()
        plt.savefig(f'docs/arcnet_multistage_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Save results to CSV
    tqdm.write(f"\n{'='*60}")
    tqdm.write("SAVING RESULTS")
    tqdm.write(f"{'='*60}")

    # Create results directory if it doesn't exist
    os.makedirs('experiments/multistage_results', exist_ok=True)

    # Save comprehensive results with progress tracking
    save_pbar = tqdm(all_results.items(), desc="Saving Results", leave=False)
    
    for param_type, results in save_pbar:
        save_pbar.set_postfix({'saving': param_type})
        
        if results:
            df = pd.DataFrame(results)
            filename = f'experiments/multistage_results/arcnet_{param_type}_{timestamp}.csv'
            df.to_csv(filename, index=False)
            tqdm.write(f"Saved {param_type} results to {filename}")
    
    save_pbar.close()

    # Create and save summary report
    summary_report = {
        'timestamp': timestamp,
        'datasets_tested': list(datasets.keys()),
        'parameters_tested': list(all_results.keys()),
        'total_experiments': sum(len(results) for results in all_results.values()),
        'best_configurations': {},
        'overall_performance': {}
    }

    # Add best configurations
    for param_type, results in all_results.items():
        if results:
            df = pd.DataFrame(results)
            best_overall = df.loc[df['test_accuracy'].idxmax()]
            summary_report['best_configurations'][param_type] = {
                'dataset': best_overall['dataset'],
                'parameter_value': best_overall['parameter_value'],
                'test_accuracy': float(best_overall['test_accuracy']),
                'test_f1': float(best_overall['test_f1'])
            }

    # Calculate overall performance statistics
    all_experiments = []
    for results in all_results.values():
        all_experiments.extend(results)
    
    if all_experiments:
        overall_df = pd.DataFrame(all_experiments)
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
    with open(f'experiments/multistage_results/arcnet_summary_{timestamp}.json', 'w') as f:
        json.dump(summary_report, f, indent=2)

    tqdm.write(f"\nSummary report saved to experiments/multistage_results/arcnet_summary_{timestamp}.json")

    # Final recommendations
    tqdm.write(f"\n{'='*80}")
    tqdm.write("FINAL RECOMMENDATIONS")
    tqdm.write(f"{'='*80}")

    rec_pbar = tqdm(datasets.keys(), desc="Generating Recommendations", leave=False)
    
    for dataset in rec_pbar:
        rec_pbar.set_postfix({'current_dataset': dataset})
        
        tqdm.write(f"\n{dataset.upper()} OPTIMAL CONFIGURATION:")
        dataset_recommendations = {}
        
        for param_type, results in all_results.items():
            if results:
                df = pd.DataFrame(results)
                dataset_results = df[df['dataset'] == dataset]
                if not dataset_results.empty:
                    best_config = dataset_results.loc[dataset_results['test_accuracy'].idxmax()]
                    dataset_recommendations[param_type] = {
                        'value': best_config['parameter_value'],
                        'test_accuracy': best_config['test_accuracy'],
                        'test_f1': best_config['test_f1']
                    }
                    tqdm.write(f"  {param_type.replace('_', ' ').title()}: {best_config['parameter_value']} "
                              f"(Acc: {best_config['test_accuracy']:.4f}, "
                              f"F1: {best_config['test_f1']:.4f})")
    
    rec_pbar.close()

    tqdm.write(f"\n{'='*80}")
    tqdm.write("MULTISTAGE ANALYSIS COMPLETE")
    tqdm.write(f"{'='*80}")
    
    return all_results, summary_report, timestamp


