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
import networkx as nx
from matplotlib.patches import Circle
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import f_oneway, zscore
import streamlit as st
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
    
    def detect_outliers(self, z_thresh=3):
        """Remove outliers from performance_score using z-score."""
        if self.combined_df is None:
            return
        z_scores = zscore(self.combined_df['performance_score'])
        mask = np.abs(z_scores) < z_thresh
        self.combined_df = self.combined_df[mask]
        print(f"Outliers removed: {np.sum(~mask)}")

    def pairwise_heatmap(self, dataset):
        """Show pairwise heatmap for two parameters."""
        df = self.combined_df[self.combined_df['dataset'] == dataset]
        for p1 in self.numeric_params:
            for p2 in self.numeric_params:
                if p1 != p2:
                    pivot = df.pivot_table(index='parameter_value', columns='parameter_type', values='performance_score')
                    if p1 in pivot.index and p2 in pivot.columns:
                        plt.figure(figsize=(8,6))
                        sns.heatmap(pivot, annot=True, cmap='viridis')
                        plt.title(f"{p1} vs {p2} Heatmap - {dataset}")
                        plt.savefig(f"{dataset}_{p1}_vs_{p2}_heatmap.png")
                        plt.show()

    def multivariate_regression(self):
        """Fit RandomForestRegressor and show feature importances and SHAP values."""
        df = self.combined_df.copy()
        # Prepare X, y
        X = pd.get_dummies(df[self.numeric_params + self.categorical_params])
        y = df['performance_score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print(f"Random Forest R2: {r2_score(y_test, model.predict(X_test)):.3f}")
        # Feature importances
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("Feature importances:\n", importances)
        # SHAP
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig("shap_summary.png")
        plt.show()

    def anova_test(self):
        """Run ANOVA for each parameter."""
        for param in self.numeric_params:
            groups = [group['performance_score'].values for _, group in self.combined_df.groupby('parameter_value')]
            if len(groups) > 1:
                stat, p = f_oneway(*groups)
                print(f"ANOVA for {param}: F={stat:.3f}, p={p:.3g}")

    def permutation_test(self, param, n_permutations=1000):
        """Permutation test for parameter effect."""
        df = self.combined_df[self.combined_df['parameter_type'] == param]
        observed = df['performance_score'].corr(df['parameter_value'])
        permuted = []
        for _ in range(n_permutations):
            permuted.append(df['performance_score'].corr(np.random.permutation(df['parameter_value'])))
        p_value = np.mean(np.abs(permuted) >= np.abs(observed))
        print(f"Permutation test for {param}: observed={observed:.3f}, p={p_value:.3g}")

    def bayesian_optimization_stub(self):
        """Stub for Bayesian optimization (suggests next config)."""
        print("Bayesian optimization not implemented. Use optuna/skopt for full support.")
        # Example: suggest random config
        suggestion = {p: np.random.choice(self.combined_df[self.combined_df['parameter_type']==p]['parameter_value'].unique()) for p in self.numeric_params}
        print("Suggested next config:", suggestion)

    def export_all_plots(self, outdir="arcnet_analysis_plots"):
        """Export all plots to files."""
        import os
        os.makedirs(outdir, exist_ok=True)
        # Example: save all figures in matplotlib
        for i in plt.get_fignums():
            plt.figure(i)
            plt.savefig(f"{outdir}/figure_{i}.png")

    def summary_table(self):
        """Print summary table of best configs."""
        summary = self.combined_df.groupby(['dataset', 'parameter_type'])['performance_score'].max().unstack()
        print("Summary Table (max performance per parameter):")
        print(summary)
        summary.to_csv("arcnet_parameter_summary.csv")

    def dashboard(self):
        """Streamlit dashboard stub."""
        st.title("ARCNET Parameter Analysis Dashboard")
        st.write("Summary Table")
        st.dataframe(self.combined_df)
        st.write("Feature Importances")
        self.multivariate_regression()
        st.write("Pairwise Heatmaps")
        for dataset in self.combined_df['dataset'].unique():
            self.pairwise_heatmap(dataset)


    def create_advanced_visualizations(self):
        """
        Create comprehensive visualization suite (replacement for heatmaps)
        """
        if self.combined_df is None:
            print("No data loaded for visualization")
            return
        
        # 1. Correlation Network Graphs
        self._create_correlation_networks()
        
        # 2. Parallel Coordinates Plots
        self._create_parallel_coordinates()
        
        # 3. Interactive Scatter Plot Matrices
        self._create_scatter_matrices()
        
        # 4. Parameter Performance Radar Charts
        self._create_radar_charts()
        
        # 5. 3D Parameter Space Visualization
        self._create_3d_parameter_space()
        
        # 6. Distribution and Trend Analysis
        self._create_distribution_plots()
        
        # 7. Interactive Correlation Wheels
        self._create_correlation_wheels()
    
    def _create_correlation_networks(self):
        """Create network graphs showing parameter correlations"""
        print("Creating correlation network visualizations...")
        
        datasets = self.combined_df['dataset'].unique()
        
        for dataset in datasets:
            dataset_data = self.combined_df[self.combined_df['dataset'] == dataset]
            
            # Calculate correlation matrix between parameters
            param_correlations = {}
            performance_correlations = {}
            
            for param in self.numeric_params:
                param_data = dataset_data[dataset_data['parameter_type'] == param]
                if len(param_data) >= 3:
                    corr = param_data['parameter_value'].corr(param_data['performance_score'])
                    if not np.isnan(corr):
                        performance_correlations[param] = abs(corr)
            
            if not performance_correlations:
                continue
            
            # Create network graph
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create circular layout
            G = nx.Graph()
            params = list(performance_correlations.keys())
            G.add_nodes_from(params)
            
            # Add edges based on correlation strength
            threshold = 0.3  # Only show correlations above this threshold
            for i, param1 in enumerate(params):
                for j, param2 in enumerate(params[i+1:], i+1):
                    # Calculate correlation between parameter effects
                    corr_strength = abs(performance_correlations[param1] - performance_correlations[param2])
                    if corr_strength < threshold:
                        G.add_edge(param1, param2, weight=1-corr_strength)
            
            # Create circular layout
            pos = nx.circular_layout(G)
            
            # Draw network
            node_sizes = [performance_correlations[param] * 3000 for param in params]
            node_colors = [performance_correlations[param] for param in params]
            
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                 node_color=node_colors, cmap='viridis', 
                                 alpha=0.8, ax=ax)
            
            nx.draw_networkx_edges(G, pos, alpha=0.5, width=2, ax=ax)
            
            # Add labels
            labels = {param: param.replace('_', '\n').title() for param in params}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
            
            ax.set_title(f'Parameter Correlation Network - {dataset}', 
                        fontsize=16, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                     norm=plt.Normalize(vmin=min(performance_correlations.values()),
                                                       vmax=max(performance_correlations.values())))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Correlation Strength with Performance', rotation=270, labelpad=20)
            
            plt.tight_layout()
            plt.show()
    
    def _create_parallel_coordinates(self):
        """Create parallel coordinates plots for multi-dimensional analysis"""
        print("Creating parallel coordinates visualizations...")
        
        try:
            # Prepare data for parallel coordinates
            plot_data = []
            
            for dataset in self.combined_df['dataset'].unique():
                dataset_data = self.combined_df[self.combined_df['dataset'] == dataset]
                
                # Get parameter values for each experiment
                experiments = {}
                for _, row in dataset_data.iterrows():
                    exp_id = f"{dataset}_{row.name}"
                    if exp_id not in experiments:
                        experiments[exp_id] = {
                            'dataset': dataset,
                            'performance_score': row['performance_score'],
                            'test_accuracy': row['test_accuracy'],
                            'test_f1': row['test_f1']
                        }
                    experiments[exp_id][row['parameter_type']] = row['parameter_value']
                
                # Convert to list
                for exp_data in experiments.values():
                    if len([k for k in exp_data.keys() if k in self.numeric_params]) >= 3:
                        plot_data.append(exp_data)
            
            if not plot_data:
                print("Insufficient data for parallel coordinates")
                return
            
            df_plot = pd.DataFrame(plot_data)
            
            # Normalize parameters for better visualization
            params_to_plot = [p for p in self.numeric_params if p in df_plot.columns]
            if len(params_to_plot) < 3:
                print("Need at least 3 parameters for parallel coordinates")
                return
            
            df_normalized = df_plot.copy()
            for param in params_to_plot:
                if df_plot[param].max() != df_plot[param].min():
                    df_normalized[param] = (df_plot[param] - df_plot[param].min()) / (df_plot[param].max() - df_plot[param].min())
            
            # Create plotly parallel coordinates
            fig = go.Figure(data=go.Parcoords(
                line=dict(color=df_normalized['performance_score'],
                         colorscale='viridis',
                         showscale=True,
                         colorbar=dict(title="Performance Score")),
                dimensions=[
                    dict(range=[0, 1],
                         constraintrange=[0, 1],
                         label=param.replace('_', ' ').title(),
                         values=df_normalized[param]) for param in params_to_plot
                ] + [
                    dict(range=[df_plot['performance_score'].min(), df_plot['performance_score'].max()],
                         label="Performance Score",
                         values=df_plot['performance_score'])
                ]
            ))
            
            fig.update_layout(
                title='Parameter Space Exploration - Parallel Coordinates',
                font=dict(size=12),
                height=600,
                width=1200
            )
            
            fig.show()
            
        except Exception as e:
            print(f"Could not create parallel coordinates plot: {e}")
    
    def _create_scatter_matrices(self):
        """Create interactive scatter plot matrices"""
        print("Creating scatter plot matrices...")
        
        for dataset in self.combined_df['dataset'].unique():
            dataset_data = self.combined_df[self.combined_df['dataset'] == dataset]
            
            # Prepare data for scatter matrix
            scatter_data = {}
            scatter_data['Performance'] = []
            scatter_data['Dataset'] = []
            
            for param in self.numeric_params:
                scatter_data[param.replace('_', ' ').title()] = []
            
            # Collect data points
            for param in self.numeric_params:
                param_data = dataset_data[dataset_data['parameter_type'] == param]
                for _, row in param_data.iterrows():
                    scatter_data['Performance'].append(row['performance_score'])
                    scatter_data['Dataset'].append(dataset)
                    
                    # Fill parameter values
                    for p in self.numeric_params:
                        if p == param:
                            scatter_data[p.replace('_', ' ').title()].append(row['parameter_value'])
                        else:
                            scatter_data[p.replace('_', ' ').title()].append(None)
            
            # Convert to DataFrame and remove rows with too many nulls
            df_scatter = pd.DataFrame(scatter_data)
            df_scatter = df_scatter.dropna(thresh=3)  # Keep rows with at least 3 non-null values
            
            if len(df_scatter) < 5:
                continue
            
            # Create scatter matrix plot
            params_for_plot = [col for col in df_scatter.columns 
                             if col not in ['Performance', 'Dataset'] and df_scatter[col].notna().sum() > 2]
            
            if len(params_for_plot) >= 2:
                try:
                    fig = px.scatter_matrix(
                        df_scatter[params_for_plot + ['Performance']],
                        color='Performance',
                        title=f'Parameter Relationships - {dataset}',
                        color_continuous_scale='viridis',
                        height=800,
                        width=1000
                    )
                    fig.update_traces(diagonal_visible=False)
                    fig.show()
                except Exception as e:
                    print(f"Could not create scatter matrix for {dataset}: {e}")
    
    def _create_radar_charts(self):
        """Create radar charts showing parameter profiles"""
        print("Creating radar chart visualizations...")
        
        for dataset in self.combined_df['dataset'].unique():
            dataset_data = self.combined_df[self.combined_df['dataset'] == dataset]
            
            # Get best performing configuration
            best_configs = {}
            for param in self.numeric_params:
                param_data = dataset_data[dataset_data['parameter_type'] == param]
                if not param_data.empty:
                    best_row = param_data.loc[param_data['performance_score'].idxmax()]
                    best_configs[param] = best_row['parameter_value']
            
            if len(best_configs) < 3:
                continue
            
            # Normalize values for radar chart
            normalized_values = {}
            for param, value in best_configs.items():
                param_data = dataset_data[dataset_data['parameter_type'] == param]
                if len(param_data) > 1:
                    min_val = param_data['parameter_value'].min()
                    max_val = param_data['parameter_value'].max()
                    if max_val != min_val:
                        normalized_values[param] = (value - min_val) / (max_val - min_val)
                    else:
                        normalized_values[param] = 0.5
                else:
                    normalized_values[param] = 0.5
            
            # Create radar chart
            categories = list(normalized_values.keys())
            values = list(normalized_values.values())
            
            # Close the radar chart
            categories += [categories[0]]
            values += [values[0]]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=[cat.replace('_', ' ').title() for cat in categories],
                fill='toself',
                name=f'Optimal Config - {dataset}',
                line_color='rgb(255, 99, 71)',
                fillcolor='rgba(255, 99, 71, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title=f"Optimal Parameter Profile - {dataset}",
                height=600,
                width=600
            )
            
            fig.show()
    
    def _create_3d_parameter_space(self):
        """Create 3D visualizations of parameter space"""
        print("Creating 3D parameter space visualizations...")
        
        for dataset in self.combined_df['dataset'].unique():
            dataset_data = self.combined_df[self.combined_df['dataset'] == dataset]
            
            # Find top 3 most variable parameters
            param_variance = {}
            for param in self.numeric_params:
                param_data = dataset_data[dataset_data['parameter_type'] == param]
                if len(param_data) > 1:
                    param_variance[param] = param_data['parameter_value'].var()
            
            if len(param_variance) < 3:
                continue
            
            top_params = sorted(param_variance.items(), key=lambda x: x[1], reverse=True)[:3]
            param_names = [p[0] for p in top_params]
            
            # Prepare 3D data
            plot_data = {'x': [], 'y': [], 'z': [], 'performance': [], 'param_names': param_names}
            
            for i, param_x in enumerate([param_names[0]]):
                param_x_data = dataset_data[dataset_data['parameter_type'] == param_x]
                for _, row_x in param_x_data.iterrows():
                    for j, param_y in enumerate([param_names[1]]):
                        param_y_data = dataset_data[dataset_data['parameter_type'] == param_y]
                        for _, row_y in param_y_data.iterrows():
                            for k, param_z in enumerate([param_names[2]]):
                                param_z_data = dataset_data[dataset_data['parameter_type'] == param_z]
                                for _, row_z in param_z_data.iterrows():
                                    plot_data['x'].append(row_x['parameter_value'])
                                    plot_data['y'].append(row_y['parameter_value'])
                                    plot_data['z'].append(row_z['parameter_value'])
                                    # Average performance for this combination
                                    avg_perf = np.mean([row_x['performance_score'], 
                                                       row_y['performance_score'], 
                                                       row_z['performance_score']])
                                    plot_data['performance'].append(avg_perf)
            
            if len(plot_data['x']) > 0:
                fig = go.Figure(data=go.Scatter3d(
                    x=plot_data['x'],
                    y=plot_data['y'],
                    z=plot_data['z'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=plot_data['performance'],
                        colorscale='viridis',
                        showscale=True,
                        colorbar=dict(title="Performance Score")
                    ),
                    text=[f'Performance: {p:.3f}' for p in plot_data['performance']],
                    hovertemplate='<b>%{text}</b><br>' +
                                 f'{param_names[0]}: %{{x}}<br>' +
                                 f'{param_names[1]}: %{{y}}<br>' +
                                 f'{param_names[2]}: %{{z}}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f'3D Parameter Space - {dataset}',
                    scene=dict(
                        xaxis_title=param_names[0].replace('_', ' ').title(),
                        yaxis_title=param_names[1].replace('_', ' ').title(),
                        zaxis_title=param_names[2].replace('_', ' ').title()
                    ),
                    height=700,
                    width=900
                )
                
                fig.show()
    
    def _create_distribution_plots(self):
        """Create distribution and trend analysis plots"""
        print("Creating distribution and trend visualizations...")
        
        for dataset in self.combined_df['dataset'].unique():
            dataset_data = self.combined_df[self.combined_df['dataset'] == dataset]
            
            # Create subplots for each parameter
            n_params = len(self.numeric_params)
            if n_params == 0:
                continue
            
            cols = min(3, n_params)
            rows = (n_params + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            if rows == 1:
                axes = axes.reshape(1, -1) if n_params > 1 else [axes]
            
            param_idx = 0
            for i in range(rows):
                for j in range(cols):
                    if param_idx >= len(self.numeric_params):
                        axes[i, j].axis('off')
                        continue
                    
                    param = self.numeric_params[param_idx]
                    param_data = dataset_data[dataset_data['parameter_type'] == param]
                    
                    if not param_data.empty:
                        ax = axes[i, j] if rows > 1 else axes[j]
                        
                        # Create violin plot with scatter overlay
                        parts = ax.violinplot([param_data['performance_score']], 
                                            positions=[1], widths=0.8, showmeans=True)
                        
                        # Color the violin plot
                        for pc in parts['bodies']:
                            pc.set_facecolor('lightblue')
                            pc.set_alpha(0.7)
                        
                        # Scatter plot overlay
                        scatter = ax.scatter(np.ones(len(param_data)) + np.random.normal(0, 0.05, len(param_data)),
                                           param_data['performance_score'],
                                           c=param_data['parameter_value'],
                                           cmap='viridis',
                                           alpha=0.7,
                                           s=50)
                        
                        # Add trend line
                        if len(param_data) > 2:
                            z = np.polyfit(param_data['parameter_value'], param_data['performance_score'], 1)
                            p = np.poly1d(z)
                            param_range = np.linspace(param_data['parameter_value'].min(), 
                                                    param_data['parameter_value'].max(), 100)
                            
                            # Normalize trend line to violin plot scale
                            trend_normalized = (p(param_range) - param_data['performance_score'].min()) / \
                                             (param_data['performance_score'].max() - param_data['performance_score'].min())
                            trend_normalized = trend_normalized * 0.3 + 0.7  # Scale to violin plot width
                            
                            ax2 = ax.twinx()
                            ax2.plot(trend_normalized, param_range, 'r-', linewidth=2, alpha=0.8)
                            ax2.set_ylabel(f'{param.replace("_", " ").title()} Value', color='red')
                            ax2.tick_params(axis='y', labelcolor='red')
                        
                        ax.set_title(f'{param.replace("_", " ").title()}\nDistribution & Performance')
                        ax.set_xlabel('Distribution')
                        ax.set_ylabel('Performance Score')
                        ax.set_xlim(0.5, 1.5)
                        
                        # Add colorbar for scatter
                        plt.colorbar(scatter, ax=ax, label='Parameter Value', shrink=0.6)
                    
                    param_idx += 1
            
            plt.suptitle(f'Parameter Distributions and Trends - {dataset}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
    
    def _create_correlation_wheels(self):
        """Create circular correlation wheel visualizations"""
        print("Creating correlation wheel visualizations...")
        
        for dataset in self.combined_df['dataset'].unique():
            dataset_data = self.combined_df[self.combined_df['dataset'] == dataset]
            
            # Calculate correlations with performance
            correlations = {}
            for param in self.numeric_params:
                param_data = dataset_data[dataset_data['parameter_type'] == param]
                if len(param_data) >= 3:
                    corr = param_data['parameter_value'].corr(param_data['performance_score'])
                    if not np.isnan(corr):
                        correlations[param] = corr
            
            if len(correlations) < 3:
                continue
            
            # Create circular plot
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Calculate angles for parameters
            params = list(correlations.keys())
            angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False)
            values = [correlations[param] for param in params]
            
            # Create the wheel
            bars = ax.bar(angles, np.abs(values), width=2*np.pi/len(params)*0.8, 
                         bottom=0.0, alpha=0.7)
            
            # Color bars based on correlation sign and strength
            for bar, value in zip(bars, values):
                if value > 0:
                    bar.set_color(plt.cm.RdYlBu_r(0.8))  # Red for positive
                else:
                    bar.set_color(plt.cm.RdYlBu_r(0.2))  # Blue for negative
                bar.set_alpha(min(1.0, abs(value) * 2))  # Alpha based on strength
            
            # Add parameter labels
            ax.set_xticks(angles)
            ax.set_xticklabels([param.replace('_', '\n').title() for param in params])
            
            # Add correlation values as text
            for angle, value, param in zip(angles, values, params):
                ax.text(angle, abs(value) + 0.1, f'{value:.2f}', 
                       ha='center', va='center', fontweight='bold')
            
            # Customize the plot
            ax.set_ylim(0, 1.2)
            ax.set_title(f'Parameter-Performance Correlation Wheel\n{dataset}', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=plt.cm.RdYlBu_r(0.8), label='Positive Correlation'),
                             Patch(facecolor=plt.cm.RdYlBu_r(0.2), label='Negative Correlation')]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            plt.tight_layout()
            plt.show()
    
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
        #self.detect_outliers()
        try:
            self.anova_test()
            self.multivariate_regression()
            self.summary_table()
            self.export_all_plots()
            self.bayesian_optimization_stub()
        except Exception as e:
            print(f"Error during analysis: {e}")
            


        correlations = self.correlation_analysis()
        sensitivity = self.parameter_sensitivity_analysis()
        self.parameter_interaction_analysis()
        recommendations = self.generate_optimization_recommendations()
        
        # Create visualizations (new advanced visualizations instead of heatmaps)
        # self.create_advanced_visualizations()
        
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

# # Example usage:
# if __name__ == "__main__":
#     # Example of how to use with your existing results
#     # analyzer = ARCNETParameterAnalyzer(results_dict=all_results)
#     # comprehensive_analysis = analyzer.create_comprehensive_report()
    
#     print("Advanced ARCNET Parameter Analyzer ready for use!")
#     print("\nUsage:")
#     print("1. With results dictionary: analyzer = ARCNETParameterAnalyzer(results_dict=your_results)")
#     print("2. With CSV files: analyzer = ARCNETParameterAnalyzer(results_dir='path/to/results', timestamp='20250722_154910')")
#     print("3. Generate report: analysis = analyzer.create_comprehensive_report()")
#     print("\nNew visualization methods include:")
#     print("- Correlation Networks: Show parameter relationships as connected graphs")
#     print("- Parallel Coordinates: Multi-dimensional parameter space exploration")
#     print("- 3D Parameter Space: Interactive 3D visualization of parameter interactions")
#     print("- Radar Charts: Parameter profiles and optimal configurations")
#     print("- Distribution Plots: Parameter distributions with performance trends")
#     print("- Correlation Wheels: Circular correlation displays")
#     print("- Interactive Scatter Matrices: Pairwise parameter relationships")