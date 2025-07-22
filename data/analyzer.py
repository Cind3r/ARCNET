import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ARCNETParameterAnalyzer:
    """
    Advanced statistical analysis and visualization for ARCNET parameter optimization results
    """
    
    def __init__(self, results_dict=None, results_dir=None, timestamp=None):
        """
        Initialize the analyzer with results data
        
        Args:
            results_dict: Dictionary of results from MultiStageTrain
            results_dir: Directory containing CSV files
            timestamp: Timestamp for file naming
        """
        self.results_dict = results_dict
        self.results_dir = results_dir
        self.timestamp = timestamp
        self.combined_df = None
        self.numeric_params = ['hidden_dim', 'initial_population', 'steps', 'epochs', 
                              'lineage_prune_rate', 'lineage_kept', 'num_survivors']
        self.categorical_params = ['q_learning_method', 'training_method', 'enable_irxn']
        
        # Color palettes for consistent visualization
        self.color_palettes = {
            'accuracy': 'viridis',
            'f1': 'plasma',
            'precision': 'inferno',
            'recall': 'cividis',
            'diverging': 'RdBu_r'
        }
        
        self._load_data()
    
    def _load_data(self):
        """Load and combine all parameter analysis results"""
        if self.results_dict:
            # Convert results dictionary to combined DataFrame
            all_results = []
            for param_type, results in self.results_dict.items():
                for result in results:
                    result['parameter_type'] = param_type
                    all_results.append(result)
            self.combined_df = pd.DataFrame(all_results)
        
        elif self.results_dir and self.timestamp:
            # Load from CSV files
            import glob
            csv_files = glob.glob(f"{self.results_dir}/arcnet_*_{self.timestamp}.csv")
            
            all_dfs = []
            for file in csv_files:
                df = pd.read_csv(file)
                all_dfs.append(df)
            
            if all_dfs:
                self.combined_df = pd.concat(all_dfs, ignore_index=True)
        
        if self.combined_df is not None:
            self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the combined dataset"""
        # Ensure parameter values are properly typed
        for param in self.numeric_params:
            mask = self.combined_df['parameter_type'] == param
            if mask.any():
                self.combined_df.loc[mask, 'parameter_value'] = pd.to_numeric(
                    self.combined_df.loc[mask, 'parameter_value'], errors='coerce'
                )
        
        # Create performance score (weighted combination of metrics)
        self.combined_df['performance_score'] = (
            0.4 * self.combined_df['test_accuracy'] +
            0.3 * self.combined_df['test_f1'] +
            0.2 * self.combined_df['test_precision'] +
            0.1 * self.combined_df['test_recall']
        )
        
        # Calculate relative performance within each dataset
        self.combined_df['relative_performance'] = self.combined_df.groupby('dataset')['performance_score'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
        )
    
    def correlation_analysis(self):
        """
        Comprehensive correlation analysis between parameters and performance metrics
        """
        if self.combined_df is None:
            print("No data loaded for analysis")
            return
        
        print("="*80)
        print("COMPREHENSIVE CORRELATION ANALYSIS")
        print("="*80)
        
        # Prepare data for correlation analysis
        numeric_data = []
        
        for param in self.numeric_params:
            param_data = self.combined_df[self.combined_df['parameter_type'] == param].copy()
            if not param_data.empty:
                param_data = param_data.rename(columns={'parameter_value': param})
                numeric_data.append(param_data[['dataset', param, 'test_accuracy', 'test_f1', 
                                               'test_precision', 'test_recall', 'performance_score']])
        
        if not numeric_data:
            print("No numeric parameter data found")
            return
        
        # Calculate correlations for each dataset separately
        datasets = self.combined_df['dataset'].unique()
        correlation_results = {}
        
        for dataset in datasets:
            print(f"\n{'-'*60}")
            print(f"CORRELATION ANALYSIS FOR {dataset.upper()}")
            print(f"{'-'*60}")
            
            dataset_correlations = {}
            
            for param in self.numeric_params:
                param_data = self.combined_df[
                    (self.combined_df['parameter_type'] == param) & 
                    (self.combined_df['dataset'] == dataset)
                ].copy()
                
                if len(param_data) < 3:  # Need at least 3 points for meaningful correlation
                    continue
                
                # Calculate correlations with different performance metrics
                correlations = {}
                for metric in ['test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'performance_score']:
                    pearson_r, pearson_p = pearsonr(param_data['parameter_value'], param_data[metric])
                    spearman_r, spearman_p = spearmanr(param_data['parameter_value'], param_data[metric])
                    
                    correlations[metric] = {
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'significant': pearson_p < 0.05
                    }
                
                dataset_correlations[param] = correlations
                
                # Print significant correlations
                print(f"\n{param.replace('_', ' ').title()}:")
                for metric, corr_data in correlations.items():
                    if corr_data['significant']:
                        print(f"  {metric}: r={corr_data['pearson_r']:.3f} (p={corr_data['pearson_p']:.3f}) *")
                    else:
                        print(f"  {metric}: r={corr_data['pearson_r']:.3f} (p={corr_data['pearson_p']:.3f})")
            
            correlation_results[dataset] = dataset_correlations
        
        return correlation_results
    
    def parameter_sensitivity_analysis(self):
        """
        Analyze parameter sensitivity and importance
        """
        if self.combined_df is None:
            print("No data loaded for analysis")
            return
        
        print("\n" + "="*80)
        print("PARAMETER SENSITIVITY ANALYSIS")
        print("="*80)
        
        sensitivity_results = {}
        
        for dataset in self.combined_df['dataset'].unique():
            print(f"\n{'-'*60}")
            print(f"SENSITIVITY ANALYSIS FOR {dataset.upper()}")
            print(f"{'-'*60}")
            
            dataset_data = self.combined_df[self.combined_df['dataset'] == dataset]
            param_sensitivity = {}
            
            for param in self.numeric_params:
                param_data = dataset_data[dataset_data['parameter_type'] == param]
                
                if len(param_data) < 2:
                    continue
                
                # Calculate coefficient of variation for performance across parameter values
                param_values = param_data['parameter_value'].unique()
                if len(param_values) < 2:
                    continue
                
                performance_by_param = []
                for val in param_values:
                    val_data = param_data[param_data['parameter_value'] == val]
                    if not val_data.empty:
                        performance_by_param.append(val_data['performance_score'].mean())
                
                if len(performance_by_param) > 1:
                    # Calculate range and coefficient of variation
                    param_range = max(performance_by_param) - min(performance_by_param)
                    param_cv = np.std(performance_by_param) / np.mean(performance_by_param) if np.mean(performance_by_param) > 0 else 0
                    
                    # Calculate optimal parameter value
                    best_idx = np.argmax(performance_by_param)
                    optimal_value = param_values[best_idx]
                    
                    param_sensitivity[param] = {
                        'range': param_range,
                        'coefficient_variation': param_cv,
                        'optimal_value': optimal_value,
                        'max_performance': max(performance_by_param),
                        'sensitivity_score': param_range * param_cv  # Combined metric
                    }
                    
                    print(f"{param.replace('_', ' ').title()}:")
                    print(f"  Range: {param_range:.4f}")
                    print(f"  CV: {param_cv:.4f}")
                    print(f"  Optimal Value: {optimal_value}")
                    print(f"  Sensitivity Score: {param_range * param_cv:.4f}")
            
            # Rank parameters by sensitivity
            if param_sensitivity:
                ranked_params = sorted(param_sensitivity.items(), 
                                     key=lambda x: x[1]['sensitivity_score'], 
                                     reverse=True)
                
                print(f"\nParameter Sensitivity Ranking:")
                for i, (param, data) in enumerate(ranked_params, 1):
                    print(f"  {i}. {param.replace('_', ' ').title()} (Score: {data['sensitivity_score']:.4f})")
            
            sensitivity_results[dataset] = param_sensitivity
        
        return sensitivity_results
    
    def create_advanced_heatmaps(self):
        """
        Create comprehensive heatmap visualizations
        """
        if self.combined_df is None:
            print("No data loaded for visualization")
            return
        
        # 1. Parameter-Performance Correlation Heatmap
        self._create_correlation_heatmap()
        
        # 2. Multi-metric Performance Heatmaps
        self._create_multi_metric_heatmaps()
        
        # 3. Dataset Comparison Heatmap
        self._create_dataset_comparison_heatmap()
        
        # 4. Interactive Parameter Space Exploration
        self._create_interactive_heatmaps()
    
    def _create_correlation_heatmap(self):
        """Create correlation heatmap between parameters and performance metrics"""
        
        # Prepare correlation matrix
        correlation_data = []
        datasets = self.combined_df['dataset'].unique()
        
        for dataset in datasets:
            dataset_data = self.combined_df[self.combined_df['dataset'] == dataset]
            
            correlations = {}
            for param in self.numeric_params:
                param_data = dataset_data[dataset_data['parameter_type'] == param]
                if len(param_data) >= 3:
                    corr_with_performance = param_data['parameter_value'].corr(param_data['performance_score'])
                    correlations[param] = corr_with_performance
                else:
                    correlations[param] = np.nan
            
            correlations['dataset'] = dataset
            correlation_data.append(correlations)
        
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data).set_index('dataset')
            
            plt.figure(figsize=(12, 8))
            mask = corr_df.isnull()
            sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, 
                       mask=mask, fmt='.3f', cbar_kws={'label': 'Correlation with Performance'})
            plt.title('Parameter-Performance Correlations Across Datasets', fontsize=16, fontweight='bold')
            plt.xlabel('Parameters', fontsize=12)
            plt.ylabel('Datasets', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    
    def _create_multi_metric_heatmaps(self):
        """Create comprehensive heatmaps for multiple performance metrics"""
        
        metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']
        
        for param in self.numeric_params:
            param_data = self.combined_df[self.combined_df['parameter_type'] == param]
            
            if param_data.empty:
                continue
            
            # Create pivot tables for each metric
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{param.replace("_", " ").title()} Performance Analysis', fontsize=16, fontweight='bold')
            
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                
                # Create pivot table
                pivot_data = param_data.pivot_table(
                    values=metric,
                    index='dataset',
                    columns='parameter_value',
                    aggfunc='mean'
                )
                
                if not pivot_data.empty:
                    sns.heatmap(pivot_data, annot=True, fmt='.3f', 
                              cmap=self.color_palettes.get(metric.split('_')[1], 'viridis'),
                              ax=ax, cbar_kws={'label': metric.replace('_', ' ').title()})
                    ax.set_title(f'{metric.replace("_", " ").title()}')
                    ax.set_xlabel('Parameter Value')
                    ax.set_ylabel('Dataset')
            
            plt.tight_layout()
            plt.show()
    
    def _create_dataset_comparison_heatmap(self):
        """Create heatmap comparing optimal parameter values across datasets"""
        
        optimal_params = {}
        
        for dataset in self.combined_df['dataset'].unique():
            dataset_data = self.combined_df[self.combined_df['dataset'] == dataset]
            optimal_params[dataset] = {}
            
            for param in self.numeric_params:
                param_data = dataset_data[dataset_data['parameter_type'] == param]
                if not param_data.empty:
                    best_row = param_data.loc[param_data['performance_score'].idxmax()]
                    optimal_params[dataset][param] = best_row['parameter_value']
        
        if optimal_params:
            optimal_df = pd.DataFrame(optimal_params).T
            
            # Normalize values for better visualization
            normalized_df = optimal_df.copy()
            for col in optimal_df.columns:
                col_data = optimal_df[col].dropna()
                if len(col_data) > 1:
                    normalized_df[col] = (optimal_df[col] - col_data.min()) / (col_data.max() - col_data.min())
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(normalized_df, annot=optimal_df, fmt='.0f', 
                       cmap='RdYlBu_r', cbar_kws={'label': 'Normalized Parameter Value'})
            plt.title('Optimal Parameter Values Across Datasets', fontsize=16, fontweight='bold')
            plt.xlabel('Parameters', fontsize=12)
            plt.ylabel('Datasets', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    
    def _create_interactive_heatmaps(self):
        """Create interactive heatmaps using Plotly"""
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create interactive correlation matrix
            datasets = self.combined_df['dataset'].unique()
            params = self.numeric_params
            
            correlation_matrix = np.zeros((len(datasets), len(params)))
            
            for i, dataset in enumerate(datasets):
                for j, param in enumerate(params):
                    param_data = self.combined_df[
                        (self.combined_df['dataset'] == dataset) & 
                        (self.combined_df['parameter_type'] == param)
                    ]
                    if len(param_data) >= 3:
                        corr = param_data['parameter_value'].corr(param_data['performance_score'])
                        correlation_matrix[i, j] = corr if not np.isnan(corr) else 0
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=[p.replace('_', ' ').title() for p in params],
                y=datasets,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlation"),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Interactive Parameter-Performance Correlation Matrix',
                xaxis_title='Parameters',
                yaxis_title='Datasets',
                height=600,
                width=1000
            )
            
            fig.show()
            
        except ImportError:
            print("Plotly not available for interactive visualizations")
    
    def parameter_interaction_analysis(self):
        """
        Analyze interactions between parameters
        """
        if self.combined_df is None:
            print("No data loaded for analysis")
            return
        
        print("\n" + "="*80)
        print("PARAMETER INTERACTION ANALYSIS")
        print("="*80)
        
        # For each dataset, look at parameter combinations
        for dataset in self.combined_df['dataset'].unique():
            print(f"\n{'-'*60}")
            print(f"INTERACTION ANALYSIS FOR {dataset.upper()}")
            print(f"{'-'*60}")
            
            dataset_data = self.combined_df[self.combined_df['dataset'] == dataset]
            
            # Create parameter combination analysis
            param_combinations = {}
            
            # Group by experiment (assuming same experiment has similar base parameters)
            for param_type in self.numeric_params:
                param_data = dataset_data[dataset_data['parameter_type'] == param_type]
                
                if len(param_data) > 1:
                    # Find best and worst performing parameter values
                    best_row = param_data.loc[param_data['performance_score'].idxmax()]
                    worst_row = param_data.loc[param_data['performance_score'].idxmin()]
                    
                    improvement = best_row['performance_score'] - worst_row['performance_score']
                    param_combinations[param_type] = {
                        'best_value': best_row['parameter_value'],
                        'worst_value': worst_row['parameter_value'],
                        'improvement': improvement,
                        'best_performance': best_row['performance_score']
                    }
            
            # Rank parameters by improvement potential
            if param_combinations:
                ranked_improvements = sorted(param_combinations.items(), 
                                           key=lambda x: x[1]['improvement'], 
                                           reverse=True)
                
                print("Parameter Impact Ranking:")
                for i, (param, data) in enumerate(ranked_improvements, 1):
                    print(f"  {i}. {param.replace('_', ' ').title()}: "
                          f"{data['improvement']:.4f} improvement "
                          f"(Best: {data['best_value']}, Performance: {data['best_performance']:.4f})")
    
    def generate_optimization_recommendations(self):
        """
        Generate optimization recommendations based on analysis
        """
        if self.combined_df is None:
            print("No data loaded for recommendations")
            return
        
        print("\n" + "="*80)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*80)
        
        recommendations = {}
        
        for dataset in self.combined_df['dataset'].unique():
            print(f"\n{'-'*60}")
            print(f"RECOMMENDATIONS FOR {dataset.upper()}")
            print(f"{'-'*60}")
            
            dataset_data = self.combined_df[self.combined_df['dataset'] == dataset]
            dataset_recommendations = {}
            
            # Find optimal values for each parameter
            for param in self.numeric_params + self.categorical_params:
                param_data = dataset_data[dataset_data['parameter_type'] == param]
                
                if not param_data.empty:
                    best_row = param_data.loc[param_data['performance_score'].idxmax()]
                    
                    # Calculate confidence based on number of trials and performance consistency
                    param_values = param_data['parameter_value'].unique()
                    if len(param_values) > 1:
                        performance_by_value = param_data.groupby('parameter_value')['performance_score'].agg(['mean', 'std', 'count'])
                        best_std = performance_by_value.loc[best_row['parameter_value'], 'std'] if not pd.isna(performance_by_value.loc[best_row['parameter_value'], 'std']) else 0
                        confidence = min(1.0, (1.0 - best_std) * (len(param_values) / 10))  # Confidence based on consistency and trials
                    else:
                        confidence = 0.5  # Medium confidence for single value
                    
                    dataset_recommendations[param] = {
                        'optimal_value': best_row['parameter_value'],
                        'performance': best_row['performance_score'],
                        'confidence': confidence,
                        'priority': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'
                    }
                    
                    print(f"{param.replace('_', ' ').title()}: {best_row['parameter_value']} "
                          f"(Performance: {best_row['performance_score']:.4f}, "
                          f"Confidence: {confidence:.2f}, Priority: {dataset_recommendations[param]['priority']})")
            
            recommendations[dataset] = dataset_recommendations
            
            # Generate configuration summary
            print(f"\nOptimal Configuration Summary:")
            high_priority = {k: v for k, v in dataset_recommendations.items() if v['priority'] == 'High'}
            if high_priority:
                print("High Priority Parameters:")
                for param, data in high_priority.items():
                    print(f"  {param}: {data['optimal_value']}")
        
        return recommendations
    
    def create_comprehensive_report(self, save_path=None):
        """
        Create a comprehensive analysis report
        """
        if self.combined_df is None:
            print("No data loaded for report generation")
            return
        
        # Run all analyses
        correlations = self.correlation_analysis()
        sensitivity = self.parameter_sensitivity_analysis()
        self.parameter_interaction_analysis()
        recommendations = self.generate_optimization_recommendations()
        
        # Create visualizations
        self.create_advanced_heatmaps()
        
        # Generate summary statistics
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY")
        print("="*80)
        
        total_experiments = len(self.combined_df)
        datasets_tested = self.combined_df['dataset'].nunique()
        parameters_tested = self.combined_df['parameter_type'].nunique()
        
        print(f"Total Experiments: {total_experiments}")
        print(f"Datasets Tested: {datasets_tested}")
        print(f"Parameters Analyzed: {parameters_tested}")
        print(f"Average Performance Score: {self.combined_df['performance_score'].mean():.4f}")
        print(f"Best Performance Score: {self.combined_df['performance_score'].max():.4f}")
        
        # Best performing configuration overall
        best_overall = self.combined_df.loc[self.combined_df['performance_score'].idxmax()]
        print(f"\nBest Overall Configuration:")
        print(f"  Dataset: {best_overall['dataset']}")
        print(f"  Parameter: {best_overall['parameter_type']} = {best_overall['parameter_value']}")
        print(f"  Performance Score: {best_overall['performance_score']:.4f}")
        print(f"  Test Accuracy: {best_overall['test_accuracy']:.4f}")
        print(f"  Test F1: {best_overall['test_f1']:.4f}")
        
        if save_path:
            # Save detailed results to file
            with open(save_path, 'w') as f:
                f.write("ARCNET Parameter Analysis Report\n")
                f.write("="*50 + "\n\n")
                f.write(f"Generated: {pd.Timestamp.now()}\n")
                f.write(f"Total Experiments: {total_experiments}\n")
                f.write(f"Datasets: {datasets_tested}\n")
                f.write(f"Parameters: {parameters_tested}\n\n")
                
                # Add recommendations
                f.write("OPTIMIZATION RECOMMENDATIONS\n")
                f.write("-"*30 + "\n")
                for dataset, recs in recommendations.items():
                    f.write(f"\n{dataset}:\n")
                    for param, data in recs.items():
                        f.write(f"  {param}: {data['optimal_value']} (Priority: {data['priority']})\n")
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        
        return {
            'correlations': correlations,
            'sensitivity': sensitivity,
            'recommendations': recommendations,
            'summary_stats': {
                'total_experiments': total_experiments,
                'datasets_tested': datasets_tested,
                'parameters_tested': parameters_tested,
                'mean_performance': self.combined_df['performance_score'].mean(),
                'max_performance': self.combined_df['performance_score'].max(),
                'best_config': best_overall.to_dict()
            }
        }

def analyze_arcnet_results(results_dict=None, results_dir=None, timestamp=None):
    """
    Convenience function to run complete ARCNET parameter analysis
    
    Args:
        results_dict: Dictionary from MultiStageTrain function
        results_dir: Path to directory containing CSV files
        timestamp: Timestamp for file identification
    
    Returns:
        Comprehensive analysis results
    """
    analyzer = ARCNETParameterAnalyzer(results_dict, results_dir, timestamp)
    return analyzer.create_comprehensive_report()

# Example usage:
if __name__ == "__main__":
    # Example of how to use with your existing results
    # analyzer = ARCNETParameterAnalyzer(results_dict=all_results)
    # comprehensive_analysis = analyzer.create_comprehensive_report()
    
    print("Advanced ARCNET Parameter Analyzer ready for use!")
    print("\nUsage:")
    print("1. With results dictionary: analyzer = ARCNETParameterAnalyzer(results_dict=your_results)")
    print("2. With CSV files: analyzer = ARCNETParameterAnalyzer(results_dir='path/to/results', timestamp='20250722_154910')")
    print("3. Generate report: analysis = analyzer.create_comprehensive_report()")