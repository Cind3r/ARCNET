import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
from typing import Dict, List, Tuple, Any



# Functions purely made to report assembly stats/structure for readability
def AssemblyStats(modules):
    """
    Generate assembly statistics for all models in the provided list.
    This function prints detailed assembly information for the best model
    based on fitness, including assembly complexity, pathway, operations,
    and other relevant metrics.
    """
    if not modules:
        print("No models available for assembly statistics.")
        return

    # Sort models by fitness
    modules.sort(key=lambda m: m.fitness, reverse=True)

    # Print basic information about all models
    print("=== Assembly Statistics ===")
    print(f"Total models: {len(modules)}")
    print(f"Best model fitness: {modules[0].fitness:.4f}")
    print()

    # Print assembly information of best model
    best_model = max(modules, key=lambda m: m.fitness)

    print("=== Assembly Information for Best Model ===")
    print(f"Model ID: {best_model.id}")
    print(f"Fitness: {best_model.fitness:.4f}")
    print(f"Created at step: {best_model.created_at}")
    print(f"Parent ID: {best_model.parent_id}")
    print()

    # Assembly complexity information
    print("=== Assembly Complexity ===")
    assembly_complexity = best_model.get_assembly_complexity()
    print(f"Assembly Complexity: {assembly_complexity}")
    print(f"Assembly Index: {best_model.assembly_index}")
    print(f"Assembly Steps: {best_model.assembly_steps}")
    print(f"Copy Number: {best_model.copy_number}")
    print()

    # Assembly pathway information
    print("=== Assembly Pathway ===")
    if hasattr(best_model, 'assembly_pathway') and best_model.assembly_pathway:
        print(f"Assembly pathway length: {len(best_model.assembly_pathway)}")
        for i, component in enumerate(best_model.assembly_pathway[:5]):  # Show first 5 components
            if hasattr(component, 'id'):
                print(f"  Component {i}: {component.id}")
            else:
                print(f"  Component {i}: {type(component).__name__}")
        if len(best_model.assembly_pathway) > 5:
            print(f"  ... and {len(best_model.assembly_pathway) - 5} more components")
    else:
        print("No assembly pathway recorded")
    print()

    # Assembly operations
    print("=== Assembly Operations ===")
    if hasattr(best_model, 'assembly_operations') and best_model.assembly_operations:
        print(f"Number of assembly operations: {len(best_model.assembly_operations)}")
        for i, op in enumerate(best_model.assembly_operations):
            print(f"  Operation {i}: {op}")
    else:
        print("No assembly operations recorded")
    print()

    # Catalytic information
    print("=== Catalytic Information ===")
    print(f"Is autocatalytic: {best_model.is_autocatalytic}")
    print(f"Catalyzed by: {best_model.catalyzed_by}")
    print(f"Catalyzes: {best_model.catalyzes}")
    print()

    # Layer components assembly complexity
    print("=== Layer Components Assembly Complexity ===")
    if hasattr(best_model, 'layer_components'):
        for layer_name, component in best_model.layer_components.items():
            if hasattr(component, 'get_minimal_assembly_complexity'):
                complexity = component.get_minimal_assembly_complexity()
                print(f"  {layer_name}: {complexity}")
            else:
                print(f"  {layer_name}: No complexity method available")
    print()

    # System-level assembly complexity
    print("=== System Assembly Complexity ===")
    try:
        from models.arcnet import system_assembly_complexity
        sys_complexity = system_assembly_complexity(modules)
        print(f"Total system assembly complexity: {sys_complexity:.4f}")
    except Exception as e:
        print(f"Could not compute system complexity: {e}")
    print()

    # Position and manifold information
    print("=== Manifold Position Information ===")
    print(f"Manifold dimension: {best_model.manifold_dim}")
    print(f"Position: {best_model.position.data[:5].tolist()}...")  # Show first 5 dimensions / Consider updating to a better visualization for large dimensions
    print(f"Curvature: {best_model.curvature}")
    if best_model.position_info:
        print(f"Position info: {best_model.position_info}")
    print()

    # Q-learning information related to assembly
    print("=== Q-Learning Assembly Information ===")
    print(f"Q-learning method: {best_model.q_learning_method}")
    if hasattr(best_model, 'q_function') and best_model.q_function is not None:
        if hasattr(best_model.q_function, 'replay_buffer'):
            print(f"Q-function replay buffer size: {len(best_model.q_function.replay_buffer)}")
        print(f"Q-memory usage: {best_model.get_q_memory_usage():.2f} MB")
    else:
        print("No Q-function available")



# Model definitions for positively correlated assembly theory
def exponential_model(complexity, A_base, gamma, beta):
    """
    Exponential model for accuracy-complexity relationship:
    A(t) = A_base + γ * (1 - exp(-β * A_sys(t)))
    """
    return A_base + gamma * (1 - np.exp(-beta * complexity))

def linear_model(complexity, A_min, delta):
    """
    Linear model for accuracy-complexity relationship:
    A(t) ≈ A_min + δ * A_sys(t)
    """
    return A_min + delta * complexity

# Model definitions for inverse assembly theory
def inverse_exponential_model(complexity, A_max, decay_rate, A_min):
    """Inverse exponential: accuracy decreases exponentially with complexity"""
    return A_min + (A_max - A_min) * np.exp(-decay_rate * complexity)

def inverse_gaussian_model(complexity, A_max, mu_opt, sigma, A_min):
    """Inverse Gaussian: peak at low optimal complexity"""
    return A_min + A_max * np.exp(-((complexity - mu_opt)**2) / (2 * sigma**2))

def power_law_decay_model(complexity, A_max, alpha, A_min):
    """Power law decay: diminishing returns to complexity"""
    return A_min + A_max * np.power(complexity + 1, -alpha)

def sigmoid_decay_model(complexity, A_max, k, x0, A_min):
    """Sigmoid decay: S-curve relationship"""
    return A_min + (A_max - A_min) / (1 + np.exp(k * (complexity - x0)))

def validate_complexity_accuracy_relationship(tracker, accuracies, dataset_name):
    """
    Validate the relationship between complexity and accuracy
    """
    # Extract complexity values from tracker
    epochs = [data['epoch'] for data in tracker.epoch_data]
    complexities = [data['assembly_stats']['avg_assembly_index'] for data in tracker.epoch_data]
    
    # Extract corresponding accuracy values
    # Ensure we have matching data points
    acc_values = [accuracies[epoch] for epoch in epochs if epoch < len(accuracies)]
    complexities = complexities[:len(acc_values)]
    
    # Create a dataframe for analysis
    df = pd.DataFrame({
        'epoch': epochs[:len(acc_values)],
        'complexity': complexities,
        'accuracy': acc_values
    })
    
    # Fit the exponential model
    try:
        params_exp, _ = curve_fit(
            exponential_model, 
            df['complexity'], 
            df['accuracy'],
            bounds=([0, 0, 0], [1, 1, 100]),  # Reasonable bounds for parameters
            maxfev=10000
        )
        A_base, gamma, beta = params_exp
        
        # Calculate fitted values and metrics
        y_pred_exp = exponential_model(df['complexity'], A_base, gamma, beta)
        r2_exp = r2_score(df['accuracy'], y_pred_exp)
        rmse_exp = np.sqrt(mean_squared_error(df['accuracy'], y_pred_exp))
    except:
        print(f"Could not fit exponential model for {dataset_name}")
        A_base, gamma, beta = None, None, None
        r2_exp, rmse_exp = None, None
        y_pred_exp = np.zeros_like(df['accuracy'])
    
    # Fit the linear model
    try:
        params_lin, _ = curve_fit(
            linear_model, 
            df['complexity'], 
            df['accuracy'],
            bounds=([0, -1], [1, 1])  # Reasonable bounds for parameters
        )
        A_min, delta = params_lin
        
        # Calculate fitted values and metrics
        y_pred_lin = linear_model(df['complexity'], A_min, delta)
        r2_lin = r2_score(df['accuracy'], y_pred_lin)
        rmse_lin = np.sqrt(mean_squared_error(df['accuracy'], y_pred_lin))
    except:
        print(f"Could not fit linear model for {dataset_name}")
        A_min, delta = None, None
        r2_lin, rmse_lin = None, None
        y_pred_lin = np.zeros_like(df['accuracy'])
    
    # Visualization
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Accuracy vs Complexity Scatter
    plt.subplot(2, 2, 1)
    plt.scatter(df['complexity'], df['accuracy'], c=df['epoch'], cmap='viridis', s=50)
    plt.colorbar(label='Epoch')
    plt.xlabel('Assembly Complexity')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name}: Accuracy vs Complexity')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Fitted Models
    plt.subplot(2, 2, 2)
    plt.scatter(df['complexity'], df['accuracy'], alpha=0.6, label='Actual Data')
    
    # Sort for smooth curve plotting
    sorted_idx = np.argsort(df['complexity'])
    sorted_complexity = df['complexity'].values[sorted_idx]
    
    if A_base is not None:
        plt.plot(sorted_complexity, exponential_model(sorted_complexity, A_base, gamma, beta), 
                'r-', linewidth=2, label=f'Exponential Model (R²={r2_exp:.3f})')
    if A_min is not None:
        plt.plot(sorted_complexity, linear_model(sorted_complexity, A_min, delta), 
                'g--', linewidth=2, label=f'Linear Model (R²={r2_lin:.3f})')
    
    plt.xlabel('Assembly Complexity')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name}: Model Fitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy and Complexity over Epochs
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax1.plot(df['epoch'], df['accuracy'], 'b-', marker='o', linewidth=2, label='Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['complexity'], 'r-', marker='x', linewidth=2, label='Complexity')
    ax2.set_ylabel('Assembly Complexity', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title(f'{dataset_name}: Accuracy and Complexity Evolution')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Results summary
    print(f"\n=== {dataset_name} Complexity-Accuracy Relationship ===")
    
    if A_base is not None:
        print(f"Exponential Model: A(t) = {A_base:.4f} + {gamma:.4f} * (1 - exp(-{beta:.4f} * A_sys(t)))")
        print(f"  R² = {r2_exp:.4f}, RMSE = {rmse_exp:.4f}")
    
    if A_min is not None:
        print(f"Linear Model: A(t) = {A_min:.4f} + {delta:.4f} * A_sys(t)")
        print(f"  R² = {r2_lin:.4f}, RMSE = {rmse_lin:.4f}")
    
    # Return the best model and its metrics
    if r2_exp is not None and r2_lin is not None:
        best_model = "Exponential" if r2_exp > r2_lin else "Linear"
        print(f"Best fitting model: {best_model}")
        return df, best_model, max(r2_exp, r2_lin)
    elif r2_exp is not None:
        print(f"Best fitting model: Exponential")
        return df, "Exponential", r2_exp
    elif r2_lin is not None:
        print(f"Best fitting model: Linear")
        return df, "Linear", r2_lin
    else:
        print("No valid model fit")
        return df, None, None


def validate_inverse_complexity_models(training_results: Dict, dataset_name: str = "Unknown"):
    """
    Validate inverse complexity-accuracy relationships using multiple models
    
    Args:
        training_results: Dictionary containing training metrics
        dataset_name: Name of the dataset
        
    Returns:
        model_results: Dictionary containing model fitting results
    """
    
    print(f"\n{'='*60}")
    print(f"INVERSE COMPLEXITY MODEL VALIDATION: {dataset_name}")
    print(f"{'='*60}")
    
    # Extract data
    complexities = np.array(training_results['complexities'])
    accuracies = np.array(training_results['accuracies'])
    epochs = np.array(range(len(complexities)))
    
    # Remove any invalid data points
    valid_mask = ~(np.isnan(complexities) | np.isnan(accuracies))
    complexities = complexities[valid_mask]
    accuracies = accuracies[valid_mask]
    epochs = epochs[valid_mask]
    
    if len(complexities) < 5:
        print("Insufficient data points for model validation")
        return None
    
    # Test multiple inverse models
    models_tested = {}
    
    # 1. Inverse Exponential Model
    try:
        params, _ = curve_fit(
            inverse_exponential_model, complexities, accuracies,
            bounds=([0, 0, 0], [1, 5, 1]),
            maxfev=5000
        )
        pred = inverse_exponential_model(complexities, *params)
        r2 = r2_score(accuracies, pred)
        
        models_tested['inverse_exponential'] = {
            'params': params,
            'r2': r2,
            'rmse': np.sqrt(mean_squared_error(accuracies, pred)),
            'predictions': pred,
            'name': 'Inverse Exponential'
        }
    except:
        pass
    
    # 2. Inverse Gaussian Model
    try:
        # Force low optimal complexity
        params, _ = curve_fit(
            inverse_gaussian_model, complexities, accuracies,
            bounds=([0, np.min(complexities), 0.1, 0], 
                    [1, np.min(complexities) + 2, np.max(complexities), 1]),
            maxfev=5000
        )
        pred = inverse_gaussian_model(complexities, *params)
        r2 = r2_score(accuracies, pred)
        
        models_tested['inverse_gaussian'] = {
            'params': params,
            'r2': r2,
            'rmse': np.sqrt(mean_squared_error(accuracies, pred)),
            'predictions': pred,
            'name': 'Inverse Gaussian',
            'optimal_complexity': params[1]
        }
    except:
        pass
    
    # 3. Power Law Decay Model
    try:
        params, _ = curve_fit(
            power_law_decay_model, complexities, accuracies,
            bounds=([0, 0.1, 0], [1, 5, 1]),
            maxfev=5000
        )
        pred = power_law_decay_model(complexities, *params)
        r2 = r2_score(accuracies, pred)
        
        models_tested['power_law'] = {
            'params': params,
            'r2': r2,
            'rmse': np.sqrt(mean_squared_error(accuracies, pred)),
            'predictions': pred,
            'name': 'Power Law Decay'
        }
    except:
        pass
    
    # 4. Sigmoid Decay Model
    try:
        params, _ = curve_fit(
            sigmoid_decay_model, complexities, accuracies,
            bounds=([0, 0, np.min(complexities), 0], 
                    [1, 10, np.max(complexities), 1]),
            maxfev=5000
        )
        pred = sigmoid_decay_model(complexities, *params)
        r2 = r2_score(accuracies, pred)
        
        models_tested['sigmoid_decay'] = {
            'params': params,
            'r2': r2,
            'rmse': np.sqrt(mean_squared_error(accuracies, pred)),
            'predictions': pred,
            'name': 'Sigmoid Decay'
        }
    except:
        pass
    
    # 5. Simple Linear Inverse
    try:
        # Linear regression with inverse complexity
        inverse_complexity = 1.0 / (complexities + 0.1)
        linear_params = np.polyfit(inverse_complexity, accuracies, 1)
        linear_pred = np.poly1d(linear_params)(inverse_complexity)
        r2 = r2_score(accuracies, linear_pred)
        
        models_tested['linear_inverse'] = {
            'params': linear_params,
            'r2': r2,
            'rmse': np.sqrt(mean_squared_error(accuracies, linear_pred)),
            'predictions': linear_pred,
            'name': 'Linear Inverse (1/x)'
        }
    except:
        pass
    
    # Visualization
    if models_tested:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Sort models by R² score
        sorted_models = sorted(models_tested.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        for i, (model_key, model_data) in enumerate(sorted_models):
            if i < 5:  # Plot top 5 models
                ax = axes[i]
                
                # Scatter plot
                ax.scatter(complexities, accuracies, alpha=0.6, s=30, color='gray', label='Data')
                
                # Model fit
                if model_key == 'linear_inverse':
                    # Special handling for linear inverse
                    inverse_comp = 1.0 / (complexities + 0.1)
                    sort_idx = np.argsort(inverse_comp)
                    ax.plot(complexities[sort_idx], model_data['predictions'][sort_idx], 
                           'r-', linewidth=2, label=f"{model_data['name']} (R²={model_data['r2']:.3f})")
                else:
                    # Sort for smooth plotting
                    sort_idx = np.argsort(complexities)
                    ax.plot(complexities[sort_idx], model_data['predictions'][sort_idx], 
                           'r-', linewidth=2, label=f"{model_data['name']} (R²={model_data['r2']:.3f})")
                
                # Mark optimal complexity if available
                if 'optimal_complexity' in model_data:
                    opt_comp = model_data['optimal_complexity']
                    if np.min(complexities) <= opt_comp <= np.max(complexities):
                        opt_acc = inverse_gaussian_model(opt_comp, *model_data['params'])
                        ax.axvline(opt_comp, color='green', linestyle='--', alpha=0.7)
                        ax.plot(opt_comp, opt_acc, 'go', markersize=8, label=f'Optimal: {opt_comp:.2f}')
                
                ax.set_xlabel('Assembly Complexity')
                ax.set_ylabel('Accuracy')
                ax.set_title(f"{model_data['name']}")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
        
        # Summary plot
        if len(sorted_models) > 0:
            ax = axes[5]
            model_names = [model[1]['name'] for model in sorted_models]
            r2_scores = [model[1]['r2'] for model in sorted_models]
            
            bars = ax.bar(range(len(model_names)), r2_scores, alpha=0.7)
            ax.set_ylabel('R² Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=0)
            
            # Add value labels
            for bar, score in zip(bars, r2_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.suptitle(f'Inverse Complexity Model Validation: {dataset_name}', fontsize=16, y=0.98)
        plt.show()
        
        # Print results
        print(f"\nModel Performance Summary:")
        print("-" * 50)
        for model_key, model_data in sorted_models:
            print(f"{model_data['name']:<20}: R² = {model_data['r2']:.4f}, RMSE = {model_data['rmse']:.4f}")
            if 'optimal_complexity' in model_data:
                print(f"{'':>20}  Optimal complexity: {model_data['optimal_complexity']:.3f}")
        
        # Best model summary
        best_model = sorted_models[0]
        print(f"\n🏆 Best Model: {best_model[1]['name']} (R² = {best_model[1]['r2']:.4f})")
        
        return {
            'models': models_tested,
            'best_model': best_model[0],
            'best_r2': best_model[1]['r2'],
            'complexity_accuracy_correlation': np.corrcoef(complexities, accuracies)[0, 1]
        }
    
    else:
        print("No models could be successfully fitted to the data")
        return None


def analyze_complexity_accuracy_relationship(complexities, accuracies, dataset_name):
    """Analyze relationship between complexity and accuracy with better models"""
    
    # Create dataframe for analysis
    if len(complexities) < len(accuracies):
        track_every = len(accuracies) // len(complexities)
        sampled_epochs = list(range(0, len(accuracies), track_every))[:len(complexities)]
        sampled_accuracies = [accuracies[i] for i in sampled_epochs]
        df = pd.DataFrame({
            'complexity': complexities,
            'accuracy': sampled_accuracies,
            'epoch': sampled_epochs
        })
    else:
        # If lengths match or complexities are more frequent, just trim
        min_len = min(len(complexities), len(accuracies))
        df = pd.DataFrame({
            'complexity': complexities[:min_len],
            'accuracy': accuracies[:min_len],
            'epoch': list(range(min_len))
        })
    
    # Test multiple relationship models
    plt.figure(figsize=(12, 10))
    
    # 1. Scatter plot with epoch coloring
    plt.subplot(2, 2, 1)
    plt.scatter(df['complexity'], df['accuracy'], c=df['epoch'], cmap='viridis')
    plt.colorbar(label='Epoch')
    plt.xlabel('Complexity')
    plt.ylabel('Accuracy')
    plt.title('Complexity vs Accuracy (colored by epoch)')
    
    # 2. Linear model
    plt.subplot(2, 2, 2)
    
    # Try quadratic model to capture non-linear relationship
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    X = df['complexity'].values.reshape(-1, 1)
    y = df['accuracy'].values
    
    # Linear fit
    model_lin = LinearRegression()
    model_lin.fit(X, y)
    y_lin = model_lin.predict(X)
    r2_lin = r2_score(y, y_lin)
    
    # Quadratic fit
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model_quad = LinearRegression()
    model_quad.fit(X_poly, y)
    y_quad = model_quad.predict(X_poly)
    r2_quad = r2_score(y, y_quad)
    
    # Plot results
    plt.scatter(df['complexity'], df['accuracy'], alpha=0.7)
    
    # Sort for smooth curve plotting
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    y_lin_sorted = y_lin[sort_idx]
    y_quad_sorted = y_quad[sort_idx]
    
    plt.plot(X_sorted, y_lin_sorted, 'r-', label=f'Linear (R²={r2_lin:.3f})')
    plt.plot(X_sorted, y_quad_sorted, 'g-', label=f'Quadratic (R²={r2_quad:.3f})')
    plt.xlabel('Complexity')
    plt.ylabel('Accuracy')
    plt.title('Model Fitting')
    plt.legend()
    
    # 3. Inverted U-shape test (theory predicts optimal complexity)
    plt.subplot(2, 2, 3)
    
    # Fit an inverted U-shape model: y = a + b*x - c*x²
    def inverted_u(x, a, b, c):
        return a + b*x - c*x**2
    
    try:
        params, _ = curve_fit(inverted_u, df['complexity'], df['accuracy'])
        a, b, c = params
        
        x_range = np.linspace(df['complexity'].min(), df['complexity'].max(), 100)
        y_pred = inverted_u(x_range, a, b, c)
        
        # Calculate optimal complexity
        optimal_x = b / (2*c) if c > 0 else None
        
        plt.scatter(df['complexity'], df['accuracy'], alpha=0.7)
        plt.plot(x_range, y_pred, 'b-', label='Inverted U-model')
        
        if optimal_x is not None and optimal_x > df['complexity'].min() and optimal_x < df['complexity'].max():
            plt.axvline(x=optimal_x, color='k', linestyle='--', alpha=0.5)
            plt.text(optimal_x, df['accuracy'].min(), f'Optimal: {optimal_x:.2f}', 
                     ha='center', va='bottom')
        
        plt.xlabel('Complexity')
        plt.ylabel('Accuracy')
        plt.title('Inverted U-shape Test')
        plt.legend()
    except:
        plt.text(0.5, 0.5, 'Could not fit inverted U-model', 
                 ha='center', va='center', transform=plt.gca().transAxes)
    
    # 4. Time evolution
    plt.subplot(2, 2, 4)
    plt.plot(df['epoch'], df['complexity'], 'r-', label='Complexity')
    plt.plot(df['epoch'], df['accuracy'], 'b-', label='Accuracy')
    plt.xlabel('Epoch')
    plt.title('Evolution over Training')
    plt.legend()
    
    plt.tight_layout()
    # plt.savefig(f'{dataset_name}_complexity_analysis.png')
    plt.show()
    
    # Print analysis
    print(f"\n=== {dataset_name} Complexity-Accuracy Analysis ===")
    print(f"Linear model: accuracy = {model_lin.intercept_:.4f} + {model_lin.coef_[0]:.4f} * complexity")
    print(f"Linear R²: {r2_lin:.4f}")
    
    if r2_quad > r2_lin:
        print(f"Quadratic model better explains the relationship (R²={r2_quad:.4f})")
        b1, b2 = model_quad.coef_[1], model_quad.coef_[2]
        optimal = -b1 / (2*b2) if b2 != 0 else None
        if optimal is not None and optimal > 0:
            print(f"Estimated optimal complexity: {optimal:.2f}")
    
    return df

def analyze_enhanced_castle_training(training_results: Dict, tracker: Any, dataset_name: str = "Unknown"):
    """
    Comprehensive analysis of enhanced CASTLE training results
    
    Args:
        training_results: Dictionary containing enhanced training metrics
        tracker: MolecularAssemblyTracker instance
        dataset_name: Name of the dataset for labeling
    
    Returns:
        analysis_summary: Dictionary containing analysis results
    """
    
    print(f"{'='*80}")
    print(f"ENHANCED CASTLE TRAINING ANALYSIS: {dataset_name}")
    print(f"{'='*80}")
    
    # Extract basic metrics
    epochs = list(range(len(training_results['losses'])))
    losses = training_results['losses']
    accuracies = training_results['accuracies']
    complexities = training_results['complexities']
    complexity_rewards = training_results['complexity_rewards']
    parsimony_scores = training_results['parsimony_scores']
    learning_rates = training_results['learning_rates']
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Main training metrics
    ax1 = plt.subplot(3, 4, 1)
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(epochs, accuracies, 'b-', linewidth=2, label='Accuracy')
    line2 = ax1_twin.plot(epochs, losses, 'r-', linewidth=2, label='Loss')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='b')
    ax1_twin.set_ylabel('Loss', color='r')
    ax1.set_title('Training Progress')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Complexity evolution
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(epochs, complexities, 'g-', linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Assembly Complexity')
    ax2.set_title('Complexity Evolution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Complexity-Accuracy relationship
    ax3 = plt.subplot(3, 4, 3)
    scatter = ax3.scatter(complexities, accuracies, c=epochs, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ax=ax3, label='Epoch')
    ax3.set_xlabel('Assembly Complexity')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Complexity vs Accuracy')
    ax3.grid(True, alpha=0.3)
    
    # 4. Parsimony scores
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(epochs, parsimony_scores, 'purple', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Parsimony Score (Acc/Complexity)')
    ax4.set_title('Parsimony Evolution')
    ax4.grid(True, alpha=0.3)
    
    # 5. Complexity rewards
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(epochs, complexity_rewards, 'orange', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Complexity Reward')
    ax5.set_title('Complexity Rewards')
    ax5.grid(True, alpha=0.3)
    
    # 6. Learning rate adaptation
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(epochs, learning_rates, 'brown', linewidth=2)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Learning Rate')
    ax6.set_title('Adaptive Learning Rate')
    ax6.grid(True, alpha=0.3)
    
    # 7. Molecular statistics evolution
    ax7 = plt.subplot(3, 4, 7)
    if training_results['molecular_statistics']:
        # Extract reuse ratios over time
        reuse_ratios = []
        unique_molecules = []
        
        for epoch_stats in training_results['molecular_statistics']:
            if epoch_stats:
                avg_reuse = np.mean([stats['reuse_ratio'] for stats in epoch_stats.values()])
                avg_unique = np.mean([stats['unique_molecules'] for stats in epoch_stats.values()])
                reuse_ratios.append(avg_reuse)
                unique_molecules.append(avg_unique)
        
        if reuse_ratios:
            ax7_twin = ax7.twinx()
            line1 = ax7.plot(range(len(reuse_ratios)), reuse_ratios, 'g-', label='Reuse Ratio')
            line2 = ax7_twin.plot(range(len(unique_molecules)), unique_molecules, 'r-', label='Unique Molecules')
            
            ax7.set_ylabel('Reuse Ratio', color='g')
            ax7_twin.set_ylabel('Unique Molecules', color='r')
            ax7.set_title('Molecular Reuse Evolution')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax7.legend(lines, labels, loc='upper left')
    
    # 8. Layer-wise complexity analysis
    ax8 = plt.subplot(3, 4, 8)
    if training_results['layer_complexities']:
        layer_names = set()
        for epoch_data in training_results['layer_complexities'].values():
            layer_names.update(epoch_data.keys())
        
        layer_names = sorted(list(layer_names))
        colors = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))
        
        for i, layer_name in enumerate(layer_names):
            layer_complexities = []
            layer_epochs = []
            
            for epoch, epoch_data in training_results['layer_complexities'].items():
                if layer_name in epoch_data:
                    layer_complexities.append(epoch_data[layer_name])
                    layer_epochs.append(epoch)
            
            if layer_complexities:
                ax8.plot(layer_epochs, layer_complexities, color=colors[i], 
                        label=layer_name.split('.')[-1], linewidth=2)
        
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Layer Complexity')
        ax8.set_title('Layer-wise Complexity')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
    
    # 9. Entropy statistics (if available)
    ax9 = plt.subplot(3, 4, 9)
    if training_results['entropy_statistics']:
        mean_entropies = []
        for epoch_stats in training_results['entropy_statistics']:
            if epoch_stats:
                avg_entropy = np.mean([stats['mean_entropy'] for stats in epoch_stats.values() if stats])
                mean_entropies.append(avg_entropy)
        
        if mean_entropies:
            ax9.plot(range(len(mean_entropies)), mean_entropies, 'teal', linewidth=2)
            ax9.set_xlabel('Epoch')
            ax9.set_ylabel('Mean Molecular Entropy')
            ax9.set_title('Entropy Evolution')
            ax9.grid(True, alpha=0.3)
    
    # 10. Gradient statistics
    ax10 = plt.subplot(3, 4, 10)
    if training_results['gradient_statistics']:
        grad_norms = []
        for epoch_stats in training_results['gradient_statistics']:
            if epoch_stats:
                avg_grad_norm = np.mean([stats['grad_norm'] for stats in epoch_stats.values()])
                grad_norms.append(avg_grad_norm)
        
        if grad_norms:
            ax10.plot(range(len(grad_norms)), grad_norms, 'navy', linewidth=2)
            ax10.set_xlabel('Epoch')
            ax10.set_ylabel('Average Gradient Norm')
            ax10.set_title('Gradient Evolution')
            ax10.grid(True, alpha=0.3)
    
    # 11. Model performance distribution
    ax11 = plt.subplot(3, 4, 11)
    ax11.hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax11.axvline(np.mean(accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.3f}')
    ax11.axvline(np.max(accuracies), color='green', linestyle='--', 
                label=f'Max: {np.max(accuracies):.3f}')
    ax11.set_xlabel('Accuracy')
    ax11.set_ylabel('Frequency')
    ax11.set_title('Accuracy Distribution')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Summary statistics
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Calculate summary statistics
    final_accuracy = accuracies[-1]
    final_complexity = complexities[-1]
    final_parsimony = parsimony_scores[-1]
    max_accuracy = np.max(accuracies)
    min_complexity = np.min(complexities)
    
    summary_text = f"""
TRAINING SUMMARY
{'='*30}
Final Accuracy: {final_accuracy:.4f}
Max Accuracy: {max_accuracy:.4f}
Final Complexity: {final_complexity:.3f}
Min Complexity: {min_complexity:.3f}
Final Parsimony: {final_parsimony:.3f}

EFFICIENCY METRICS
{'='*30}
Acc/Complexity Ratio: {final_accuracy/final_complexity:.3f}
Complexity Reduction: {(complexities[0]-final_complexity)/complexities[0]*100:.1f}%
Accuracy Improvement: {(final_accuracy-accuracies[0])/accuracies[0]*100:.1f}%

MOLECULAR INSIGHTS
{'='*30}
Total Molecules: {len(tracker.atomic_library)}
Total Lattices: {len(tracker.lattice_library)}
Assembly Pathways: {len(tracker.assembly_pathways)}
    """
    
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle(f'Enhanced CASTLE Training Analysis: {dataset_name}', fontsize=16, y=0.98)
    plt.show()
    
    # Statistical analysis summary
    analysis_summary = {
        'dataset_name': dataset_name,
        'final_accuracy': final_accuracy,
        'max_accuracy': max_accuracy,
        'final_complexity': final_complexity,
        'min_complexity': min_complexity,
        'final_parsimony': final_parsimony,
        'complexity_reduction_percent': (complexities[0]-final_complexity)/complexities[0]*100,
        'accuracy_improvement_percent': (final_accuracy-accuracies[0])/accuracies[0]*100,
        'total_molecules': len(tracker.atomic_library),
        'total_lattices': len(tracker.lattice_library),
        'assembly_pathways': len(tracker.assembly_pathways)
    }
    
    return analysis_summary



def analyze_molecular_evolution(tracker: Any, training_results: Dict, dataset_name: str = "Unknown"):
    """
    Analyze the evolution of molecular structures throughout training
    
    Args:
        tracker: MolecularAssemblyTracker instance
        training_results: Dictionary containing training metrics
        dataset_name: Name of the dataset
        
    Returns:
        evolution_analysis: Dictionary containing evolution metrics
    """
    
    print(f"\n{'='*60}")
    print(f"MOLECULAR EVOLUTION ANALYSIS: {dataset_name}")
    print(f"{'='*60}")
    
    # Extract molecular evolution data
    epoch_data = tracker.epoch_data
    
    if not epoch_data:
        print("No molecular evolution data available")
        return None
    
    # Analyze molecular discovery and reuse patterns
    epochs = [data['epoch'] for data in epoch_data]
    total_molecules = [data['assembly_stats']['total_molecules'] for data in epoch_data]
    avg_complexity = [data['assembly_stats']['avg_assembly_index'] for data in epoch_data]
    max_complexity = [data['assembly_stats']['max_assembly_index'] for data in epoch_data]
    total_lattices = [data['assembly_stats']['total_lattices_library'] for data in epoch_data]
    
    # Create comprehensive molecular evolution visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. Molecular discovery rate
    ax1 = axes[0, 0]
    ax1.plot(epochs, total_molecules, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Unique Molecules')
    ax1.set_title('Molecular Discovery')
    ax1.grid(True, alpha=0.3)
    
    # 2. Complexity evolution
    ax2 = axes[0, 1]
    ax2.plot(epochs, avg_complexity, 'g-o', linewidth=2, markersize=4, label='Average')
    ax2.plot(epochs, max_complexity, 'r-s', linewidth=2, markersize=4, label='Maximum')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Assembly Complexity')
    ax2.set_title('Complexity Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Lattice library growth
    ax3 = axes[0, 2]
    ax3.plot(epochs, total_lattices, 'purple', linewidth=2, marker='o', markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Total Lattice Structures')
    ax3.set_title('Lattice Library Growth')
    ax3.grid(True, alpha=0.3)
    
    # 4. Molecular reuse analysis
    ax4 = axes[1, 0]
    reuse_counts = {}
    for molecule, lattices in tracker.molecule_reuse.items():
        reuse_count = len(lattices)
        reuse_counts[reuse_count] = reuse_counts.get(reuse_count, 0) + 1
    
    if reuse_counts:
        reuse_levels = sorted(reuse_counts.keys())
        reuse_frequencies = [reuse_counts[level] for level in reuse_levels]
        
        ax4.bar(reuse_levels, reuse_frequencies, alpha=0.7, color='orange')
        ax4.set_xlabel('Reuse Count')
        ax4.set_ylabel('Number of Molecules')
        ax4.set_title('Molecular Reuse Distribution')
        ax4.grid(True, alpha=0.3)
    
    # 5. Assembly pathway complexity
    ax5 = axes[1, 1]
    pathway_lengths = [len(pathway) for pathway in tracker.assembly_pathways.values()]
    
    if pathway_lengths:
        ax5.hist(pathway_lengths, bins=min(10, len(set(pathway_lengths))), 
                alpha=0.7, color='teal', edgecolor='black')
        ax5.set_xlabel('Pathway Length')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Assembly Pathway Complexity')
        ax5.grid(True, alpha=0.3)
    
    # 6. Molecular formula diversity
    ax6 = axes[1, 2]
    formula_diversity = []
    for data in epoch_data:
        formulas = set()
        for lattice_data in data['new_lattices']:
            formulas.add(lattice_data['molecular_formula'])
        formula_diversity.append(len(formulas))
    
    if formula_diversity:
        ax6.plot(epochs, formula_diversity, 'brown', linewidth=2, marker='o', markersize=4)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Unique Molecular Formulas')
        ax6.set_title('Formula Diversity Evolution')
        ax6.grid(True, alpha=0.3)
    
    # 7. Efficiency metrics
    ax7 = axes[2, 0]
    if len(training_results['accuracies']) == len(avg_complexity):
        efficiency = np.array(training_results['accuracies']) / (np.array(avg_complexity) + 0.1)
        ax7.plot(epochs, efficiency, 'red', linewidth=2, marker='o', markersize=4)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Accuracy/Complexity Ratio')
        ax7.set_title('Assembly Efficiency')
        ax7.grid(True, alpha=0.3)
    
    # 8. Molecular type distribution
    ax8 = axes[2, 1]
    molecule_types = {}
    for molecule in tracker.atomic_library:
        symbol = molecule.atomic_symbol[:2]  # Get base symbol
        molecule_types[symbol] = molecule_types.get(symbol, 0) + 1
    
    if molecule_types:
        types = list(molecule_types.keys())
        counts = list(molecule_types.values())
        
        ax8.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
        ax8.set_title('Molecular Type Distribution')
    
    # 9. Summary statistics
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    # Calculate summary statistics
    discovery_rate = (total_molecules[-1] - total_molecules[0]) / len(epochs) if len(epochs) > 1 else 0
    complexity_change = avg_complexity[-1] - avg_complexity[0] if len(avg_complexity) > 1 else 0
    reuse_efficiency = len([m for m, lattices in tracker.molecule_reuse.items() if len(lattices) > 1]) / len(tracker.atomic_library) if tracker.atomic_library else 0
    
    summary_text = f"""
MOLECULAR EVOLUTION SUMMARY
{'='*35}

Discovery Metrics:
• Total molecules discovered: {total_molecules[-1] if total_molecules else 0}
• Discovery rate: {discovery_rate:.2f} mol/epoch
• Total lattice structures: {total_lattices[-1] if total_lattices else 0}

Complexity Metrics:
• Initial complexity: {avg_complexity[0] if avg_complexity else 0:.2f}
• Final complexity: {avg_complexity[-1] if avg_complexity else 0:.2f}
• Complexity change: {complexity_change:.2f}

Reuse Metrics:
• Reuse efficiency: {reuse_efficiency:.1%}
• Assembly pathways: {len(tracker.assembly_pathways)}
• Highly reused molecules: {len([m for m, l in tracker.molecule_reuse.items() if len(l) > 2])}

Diversity Metrics:
• Molecular types: {len(molecule_types)}
• Unique formulas: {len(set(d['molecular_formula'] for d in epoch_data[-1]['new_lattices'])) if epoch_data else 0}
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle(f'Molecular Evolution Analysis: {dataset_name}', fontsize=16, y=0.98)
    plt.show()
    
    # Return analysis summary
    evolution_analysis = {
        'discovery_rate': discovery_rate,
        'complexity_change': complexity_change,
        'reuse_efficiency': reuse_efficiency,
        'total_molecules': total_molecules[-1] if total_molecules else 0,
        'total_lattices': total_lattices[-1] if total_lattices else 0,
        'assembly_pathways': len(tracker.assembly_pathways),
        'molecular_types': len(molecule_types),
        'highly_reused_molecules': len([m for m, l in tracker.molecule_reuse.items() if len(l) > 2])
    }
    
    return evolution_analysis

def generate_comprehensive_report(analysis_summary: Dict, model_results: Dict, 
                                evolution_analysis: Dict, dataset_name: str = "Unknown"):
    """
    Generate a comprehensive analysis report
    
    Args:
        analysis_summary: Results from analyze_enhanced_castle_training
        model_results: Results from validate_inverse_complexity_models
        evolution_analysis: Results from analyze_molecular_evolution
        dataset_name: Name of the dataset
    """
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE CASTLE ANALYSIS REPORT: {dataset_name}")
    print(f"{'='*80}")
    
    # Training Performance Section
    print(f"\n TRAINING PERFORMANCE")
    print(f"{'-'*40}")
    if analysis_summary:
        print(f"Final Accuracy: {analysis_summary['final_accuracy']:.4f}")
        print(f"Maximum Accuracy: {analysis_summary['max_accuracy']:.4f}")
        print(f"Accuracy Improvement: {analysis_summary['accuracy_improvement_percent']:.1f}%")
        print(f"Final Complexity: {analysis_summary['final_complexity']:.3f}")
        print(f"Complexity Reduction: {analysis_summary['complexity_reduction_percent']:.1f}%")
        print(f"Final Parsimony Score: {analysis_summary['final_parsimony']:.3f}")
    
    # Assembly Theory Validation
    print(f"\n🧬 ASSEMBLY THEORY VALIDATION")
    print(f"{'-'*40}")
    if model_results:
        print(f"Best Model: {model_results['best_model'].replace('_', ' ').title()}")
        print(f"Model R² Score: {model_results['best_r2']:.4f}")
        print(f"Complexity-Accuracy Correlation: {model_results['complexity_accuracy_correlation']:.4f}")
        
        # Determine validation result
        if model_results['complexity_accuracy_correlation'] < -0.3:
            validation_result = " STRONG support for inverse complexity theory"
        elif model_results['complexity_accuracy_correlation'] < -0.1:
            validation_result =" MODERATE support for inverse complexity theory"
        else:
            validation_result = " LIMITED support for inverse complexity theory"
        
        print(f"Theory Validation: {validation_result}")
    
    # Molecular Evolution Insights
    print(f"\n🔬 MOLECULAR EVOLUTION INSIGHTS")
    print(f"{'-'*40}")
    if evolution_analysis:
        print(f"Molecular Discovery Rate: {evolution_analysis['discovery_rate']:.2f} molecules/epoch")
        print(f"Total Molecules Discovered: {evolution_analysis['total_molecules']}")
        print(f"Reuse Efficiency: {evolution_analysis['reuse_efficiency']:.1%}")
        print(f"Complexity Evolution: {evolution_analysis['complexity_change']:+.2f}")
        print(f"Assembly Pathways Created: {evolution_analysis['assembly_pathways']}")
        print(f"Highly Reused Molecules: {evolution_analysis['highly_reused_molecules']}")
    
    # Theoretical Implications
    print(f"\n🎯 THEORETICAL IMPLICATIONS")
    print(f"{'-'*40}")
    
    implications = []
    
    if analysis_summary and analysis_summary['complexity_reduction_percent'] > 10:
        implications.append("• Training successfully reduces assembly complexity")
    
    if model_results and model_results['complexity_accuracy_correlation'] < -0.2:
        implications.append("• Inverse complexity-accuracy relationship validated")
        implications.append("• Supports parsimony principle in neural networks")
    
    if evolution_analysis and evolution_analysis['reuse_efficiency'] > 0.3:
        implications.append("• Molecular reuse patterns emerge naturally")
        implications.append("• Assembly theory principles guide optimization")
    
    if analysis_summary and analysis_summary['final_parsimony'] > 1.0:
        implications.append("• High parsimony score indicates efficient learning")
    
    for implication in implications:
        print(implication)
    
    if not implications:
        print("• Results require further investigation")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print(f"{'-'*40}")
    
    recommendations = []
    
    if analysis_summary and analysis_summary['final_complexity'] > 2.0:
        recommendations.append("• Consider stronger complexity penalties")
        recommendations.append("• Increase molecular simplification constraints")
    
    if model_results and model_results['best_r2'] < 0.3:
        recommendations.append("• Collect more training data for better model fitting")
        recommendations.append("• Experiment with different complexity reward schedules")
    
    if evolution_analysis and evolution_analysis['reuse_efficiency'] < 0.2:
        recommendations.append("• Strengthen molecular reuse incentives")
        recommendations.append("• Adjust gradient modification strategies")
    
    recommendations.extend([
        "• Monitor parsimony scores for early stopping",
        "• Experiment with different molecular block sizes",
        "• Validate findings across multiple datasets"
    ])
    
    for recommendation in recommendations:
        print(recommendation)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")