#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CTX Length Ablation Study for PLIP
Run from: /home/abhishek/desktop/VLM_Cal/CalibPrompt/experiments/abalations
"""

import yaml
import os
import subprocess
import pandas as pd
import re
import argparse
import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns

class CtxLengthAblationStudy:
    def __init__(self):
        # Fixed to PLIP only for ablation study
        self.model_type = 'plip'
        
        # Define paths
        self.base_dir = os.path.abspath("../..")  # Go up two levels from abalations
        self.base_config_path = os.path.join(self.base_dir, f"configs/trainers/CoOp/vit_b32_{self.model_type}_c16_ep50_batch16.yaml")
        self.results_dir = os.path.join(self.base_dir, f"output/ctx_length_ablation_{self.model_type}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Seeds (using just one seed for efficiency as in the original code)
        self.seeds = [1]
        
        # N_CTX values for ablation study
        self.ctx_list = [2, 4, 6, 8, 10, 12, 14, 16, 32, 64]
        
        # Fixed parameters
        self.fixed_params = {
            'lr': 0.002,
            'epochs': 50,
            'shots': 8  # Fix shots to 8 for this ablation
        }
        
        # Datasets
        self.datasets = ["kather", "pannuke", "digestpath"]
        
        # Load and prepare loss configurations based on datasets
        self.dataset_configs = self.get_dataset_configs()
        self.loss_configs_per_dataset = {
            dataset: self.get_loss_configs(dataset) for dataset in self.datasets
        }
        
        print(f"Ablation study configuration:")
        print(f"- Model: {self.model_type}")
        print(f"- Datasets: {', '.join(self.datasets)}")
        print(f"- CTX Lengths: {', '.join(map(str, self.ctx_list))}")
        print(f"- Fixed shots: {self.fixed_params['shots']}")
        print(f"- Results directory: {self.results_dir}")

    def get_dataset_configs(self):
        """Get dataset-specific configurations for PLIP with only the required loss functions."""
        return {
            'kather': {
                'CE_SMAC': {'SMAC': {'ALPHA': 0.07}},
                'CE_SMAC_CL': {'SMAC': {'ALPHA': 0.07}, 'AS': {'WEIGHT': 0.01}},
                'FL_SMAC': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.03}},
                'FL_SMAC_CL': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.03}, 'AS': {'WEIGHT': 0.01}},
                'LS_SMAC': {'LS': {'ALPHA': 0.05}, 'SMAC': {'ALPHA': 0.05}},
                'LS_SMAC_CL': {'LS': {'ALPHA': 0.05}, 'SMAC': {'ALPHA': 0.05}, 'AS': {'WEIGHT': 0.01}}
            },
            'pannuke': {
                'CE_SMAC': {'SMAC': {'ALPHA': 0.2}},
                'CE_SMAC_CL': {'SMAC': {'ALPHA': 0.2}, 'AS': {'WEIGHT': 0.1}},
                'FL_SMAC': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.2}},
                'FL_SMAC_CL': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.2}, 'AS': {'WEIGHT': 0.1}},
                'LS_SMAC': {'LS': {'ALPHA': 0.2}, 'SMAC': {'ALPHA': 0.2}},
                'LS_SMAC_CL': {'LS': {'ALPHA': 0.2}, 'SMAC': {'ALPHA': 0.2}, 'AS': {'WEIGHT': 0.1}}
            },
            'digestpath': {
                'CE_SMAC': {'SMAC': {'ALPHA': 0.03}},
                'CE_SMAC_CL': {'SMAC': {'ALPHA': 0.03}, 'AS': {'WEIGHT': 0.001}},
                'FL_SMAC': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.03}},
                'FL_SMAC_CL': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.03}, 'AS': {'WEIGHT': 0.001}},
                'LS_SMAC': {'LS': {'ALPHA': 0.03}, 'SMAC': {'ALPHA': 0.03}},
                'LS_SMAC_CL': {'LS': {'ALPHA': 0.03}, 'SMAC': {'ALPHA': 0.03}, 'AS': {'WEIGHT': 0.001}}
            }
        }

    def get_loss_configs(self, dataset):
        """Convert dataset-specific loss configs into the format needed for experiments."""
        dataset_params = self.dataset_configs[dataset]
        
        loss_configs = []
        for loss_name, params in dataset_params.items():
            # Determine enabled losses based on loss name and parameters
            enabled_losses = []
            
            # Add base loss type (CE, FL, LS)
            if loss_name.startswith('CE_'):
                enabled_losses.append("CE")
            elif loss_name.startswith('FL_'):
                enabled_losses.append("FL")
            elif loss_name.startswith('LS_'):
                enabled_losses.append("LS")
            
            # Add additional losses from parameters
            for loss_type in params.keys():
                if loss_type not in enabled_losses:
                    enabled_losses.append(loss_type)
            
            loss_configs.append({
                'name': loss_name,
                'enabled_losses': enabled_losses,
                'params': params
            })
            
        return loss_configs

    def modify_config(self, dataset, loss_config, ctx_length):
        """Create a modified config file for the experiment with specified CTX length."""
        try:
            # Read base config
            with open(self.base_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set fixed parameters
            config['OPTIM']['LR'] = self.fixed_params['lr']
            config['OPTIM']['MAX_EPOCH'] = self.fixed_params['epochs']
            
            # Set the CTX length parameter
            config['TRAINER']['COOP']['N_CTX'] = ctx_length
            
            # Removed FINETUNE_MODE which was causing the error
            
            # Set loss configuration
            loss_config_yaml = config['TRAINER']['COOP']['LOSS']
            loss_config_yaml['ENABLED_LOSSES'] = loss_config['enabled_losses']
            
            # Configure each individual loss
            for loss_name, loss_params in loss_config['params'].items():
                if loss_name not in loss_config_yaml:
                    loss_config_yaml[loss_name] = {}
                
                # Set all parameters for this loss
                for param_name, param_value in loss_params.items():
                    loss_config_yaml[loss_name][param_name] = param_value
            
            # Save config file
            config_name = f"config_{self.model_type}_{dataset}_{loss_config['name']}_ctx{ctx_length}.yaml"
            config_path = os.path.join(self.results_dir, config_name)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            return config_path
            
        except Exception as e:
            print(f"Error modifying config: {e}")
            return None

    def extract_metrics(self, log_path):
        """Extract relevant metrics from experiment log file."""
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                metrics = {}
                
                # Metrics to extract
                patterns = {
                    'accuracy': r'\* accuracy: ([\d.]+)%',
                    'error_rate': r'\* error: ([\d.]+)%',
                    'confidence': r'\* confidence: ([\d.]+)%',
                    'ece': r'\* ece: ([\d.]+)%',
                    'mce': r'\* mce: ([\d.]+)%',
                    'ace': r'\* ace: ([\d.]+)%',
                    'macro_f1': r'\* macro_f1: ([\d.]+)%',
                    'ece_kde': r'\* ece_kde: ([\d.]+)%'
                }
                
                for metric_name, pattern in patterns.items():
                    match = re.search(pattern, content)
                    if not match:
                        print(f"Warning: Missing {metric_name} in log")
                        metrics[metric_name] = float('nan')
                    else:
                        metrics[metric_name] = float(match.group(1))
                
                return metrics
        except Exception as e:
            print(f"Error reading log {log_path}: {e}")
            return None

    def run_experiment(self, dataset, loss_config, seed, ctx_length):
        """Run a single experiment with the specified parameters."""
        try:
            # Get modified config
            config_path = self.modify_config(dataset, loss_config, ctx_length)
            if not config_path:
                print(f"Error: Failed to create config for {dataset}, {loss_config['name']}, ctx={ctx_length}")
                return None
            
            # Setup experiment directory - note the fixed shots parameter
            exp_name = f"{dataset}/shots_{self.fixed_params['shots']}/ctx_{ctx_length}/CoOp/prompt/{loss_config['name']}"
            output_dir = os.path.join(self.results_dir, exp_name, f"seed{seed}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Check for completed experiment
            log_file = os.path.join(output_dir, "log.txt")
            if os.path.exists(log_file):
                print(f"Experiment exists: {dataset}, {loss_config['name']}, ctx={ctx_length}, seed={seed}")
                metrics = self.extract_metrics(log_file)
                if metrics:
                    self.save_experiment_result(dataset, ctx_length, loss_config['name'], seed, metrics)
                return metrics
            
            # Command to run the experiment
            cmd = [
                "python", os.path.join(self.base_dir, "train.py"),
                "--root", "/home/abhishek/desktop/VLM_Cal/CalibPrompt/DATA",  # Updated path
                "--seed", str(seed),
                "--trainer", "CoOp",
                "--dataset-config-file", os.path.join(self.base_dir, f"configs/datasets/{dataset}.yaml"),
                "--config-file", config_path,
                "--output-dir", output_dir,
                "DATASET.NUM_SHOTS", str(self.fixed_params['shots']),  # Fixed shots
                "DATASET.SUBSAMPLE_CLASSES", "all",
                "MODEL.NAME", self.model_type,
                "MODEL_ROOT", os.path.join(self.base_dir, "models")
            ]
            
            # Run the experiment
            print(f"\nRunning: {dataset}, {loss_config['name']}, ctx={ctx_length}, seed={seed}")
            subprocess.run(cmd, check=True)
            
            # Extract metrics and save results
            metrics = self.extract_metrics(log_file)
            if metrics:
                self.save_experiment_result(dataset, ctx_length, loss_config['name'], seed, metrics)
            return metrics
            
        except Exception as e:
            print(f"Error in experiment: {e}")
            return None

    def save_experiment_result(self, dataset, ctx_length, loss_name, seed, metrics):
        """Save a single experiment result to CSV."""
        try:
            # Create result row
            result_row = {
                'dataset': dataset,
                'ctx_length': ctx_length,
                'losses': loss_name,
                'seed': seed,
                'accuracy': metrics['accuracy'],
                'ece': metrics['ece'],
                'mce': metrics['mce'],
                'ace': metrics['ace'],
                'macro_f1': metrics['macro_f1'],
                'confidence': metrics.get('confidence', float('nan')),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Define CSV path for individual results
            individual_csv_path = os.path.join(
                self.results_dir, 
                f'{dataset}_ctx{ctx_length}_{loss_name}_seed{seed}_result.csv'
            )
            
            # Save individual result
            pd.DataFrame([result_row]).to_csv(individual_csv_path, index=False)
            
            # Also append to dataset results file
            dataset_csv_path = os.path.join(
                self.results_dir, 
                f'{dataset}_{self.model_type}_all_ctx_results.csv'
            )
            
            if os.path.exists(dataset_csv_path):
                # Read existing CSV
                existing_df = pd.read_csv(dataset_csv_path)
                
                # Check if this exact experiment already exists
                mask = (
                    (existing_df['dataset'] == dataset) & 
                    (existing_df['ctx_length'] == ctx_length) & 
                    (existing_df['losses'] == loss_name) & 
                    (existing_df['seed'] == seed)
                )
                
                if mask.any():
                    # Update existing row
                    existing_df.loc[mask, 'accuracy'] = metrics['accuracy']
                    existing_df.loc[mask, 'ece'] = metrics['ece']
                    existing_df.loc[mask, 'mce'] = metrics['mce']
                    existing_df.loc[mask, 'ace'] = metrics['ace']
                    existing_df.loc[mask, 'macro_f1'] = metrics['macro_f1']
                    existing_df.loc[mask, 'confidence'] = metrics.get('confidence', float('nan'))
                    existing_df.loc[mask, 'timestamp'] = result_row['timestamp']
                else:
                    # Append new row
                    existing_df = pd.concat([existing_df, pd.DataFrame([result_row])], ignore_index=True)
                
                existing_df.to_csv(dataset_csv_path, index=False)
            else:
                # Create new CSV with this row
                pd.DataFrame([result_row]).to_csv(dataset_csv_path, index=False)
                
            print(f"Saved result for {dataset}, {loss_name}, ctx={ctx_length}, seed={seed}")
            return True
        except Exception as e:
            print(f"Error saving experiment result: {e}")
            return False

    def run_all_experiments(self):
        """Run all experiments for the CTX length ablation study."""
        all_results = []
        
        # Track total experiments and completed
        total_experiments = len(self.datasets) * len(self.ctx_list) * len(self.loss_configs_per_dataset[self.datasets[0]]) * len(self.seeds)
        completed_experiments = 0
        
        # Iterate through all combinations
        for dataset in self.datasets:
            print(f"\n=== Dataset: {dataset} ===")
            dataset_results = []
            
            for ctx_length in self.ctx_list:
                print(f"\n--- CTX Length: {ctx_length} ---")
                
                for loss_config in self.loss_configs_per_dataset[dataset]:
                    print(f"\nTesting: {loss_config['name']}")
                    
                    for seed in self.seeds:
                        # Run experiment and get metrics
                        metrics = self.run_experiment(dataset, loss_config, seed, ctx_length)
                        completed_experiments += 1
                        
                        if metrics:
                            result_row = {
                                'dataset': dataset,
                                'ctx_length': ctx_length,
                                'losses': loss_config['name'],
                                'seed': seed,
                                'accuracy': metrics['accuracy'],
                                'ece': metrics['ece'],
                                'mce': metrics['mce'],
                                'ace': metrics['ace'],
                                'macro_f1': metrics['macro_f1'],
                                'confidence': metrics.get('confidence', float('nan'))
                            }
                            dataset_results.append(result_row)
                            all_results.append(result_row)
                
                print(f"Progress: {completed_experiments}/{total_experiments} experiments completed")
                
                # Generate summary statistics for this dataset
                self.generate_dataset_summaries(dataset)
            
            # Save combined dataset results
            print(f"All experiments for {dataset} completed")
        
        # Create comprehensive summaries across all datasets
        self.generate_combined_summaries()
        
        return True
    
    def generate_dataset_summaries(self, dataset):
        """Generate summary statistics for a dataset."""
        try:
            # Path to dataset results
            dataset_csv_path = os.path.join(
                self.results_dir, 
                f'{dataset}_{self.model_type}_all_ctx_results.csv'
            )
            
            if not os.path.exists(dataset_csv_path):
                print(f"Warning: No results file found for {dataset}")
                return False
            
            # Load results
            results_df = pd.read_csv(dataset_csv_path)
            
            # Remove any existing summary rows
            results_df = results_df[results_df['seed'] != 'mean_std']
            
            # Create summary rows
            summary_rows = []
            
            for ctx_length in self.ctx_list:
                for loss_name in set(results_df['losses']):
                    # Filter for this ctx/loss combination
                    combo_df = results_df[
                        (results_df['ctx_length'] == ctx_length) & 
                        (results_df['losses'] == loss_name)
                    ]
                    
                    if len(combo_df) == 0:
                        continue
                    
                    # Calculate mean and std for all metrics
                    mean_acc = combo_df['accuracy'].mean()
                    std_acc = combo_df['accuracy'].std() if len(combo_df) > 1 else 0
                    mean_ece = combo_df['ece'].mean()
                    std_ece = combo_df['ece'].std() if len(combo_df) > 1 else 0
                    mean_mce = combo_df['mce'].mean()
                    std_mce = combo_df['mce'].std() if len(combo_df) > 1 else 0
                    mean_ace = combo_df['ace'].mean()
                    std_ace = combo_df['ace'].std() if len(combo_df) > 1 else 0
                    mean_f1 = combo_df['macro_f1'].mean()
                    std_f1 = combo_df['macro_f1'].std() if len(combo_df) > 1 else 0
                    
                    # Create summary row
                    summary_row = {
                        'dataset': dataset,
                        'ctx_length': ctx_length,
                        'losses': loss_name,
                        'seed': 'mean_std',
                        'accuracy': f"{mean_acc:.2f}±{std_acc:.2f}",
                        'ece': f"{mean_ece:.2f}±{std_ece:.2f}",
                        'mce': f"{mean_mce:.2f}±{std_mce:.2f}",
                        'ace': f"{mean_ace:.2f}±{std_ace:.2f}",
                        'macro_f1': f"{mean_f1:.2f}±{std_f1:.2f}",
                        'mean_accuracy': mean_acc,
                        'std_accuracy': std_acc,
                        'mean_ece': mean_ece,
                        'std_ece': std_ece,
                        'mean_mce': mean_mce,
                        'std_mce': std_mce,
                        'mean_ace': mean_ace,
                        'std_ace': std_ace,
                        'mean_macro_f1': mean_f1,
                        'std_macro_f1': std_f1
                    }
                    summary_rows.append(summary_row)
            
            # Add summary rows to results and save
            if summary_rows:
                combined_df = pd.concat([results_df, pd.DataFrame(summary_rows)], ignore_index=True)
                combined_df.to_csv(dataset_csv_path, index=False)
                
                # Also save just the summary rows to a separate file
                summary_df = pd.DataFrame(summary_rows)
                summary_path = os.path.join(
                    self.results_dir, 
                    f'{dataset}_{self.model_type}_summary.csv'
                )
                summary_df.to_csv(summary_path, index=False)
                
                print(f"Generated summary statistics for {dataset}")
                
                # Create pivot tables for this dataset
                self.create_dataset_pivot_tables(dataset, summary_df)
                
                # Generate plots for this dataset
                self.generate_dataset_plots(dataset, summary_df)
                
                return True
            else:
                print(f"Warning: No data to generate summaries for {dataset}")
                return False
                
        except Exception as e:
            print(f"Error generating dataset summaries for {dataset}: {e}")
            return False
    
    def create_dataset_pivot_tables(self, dataset, summary_df):
        """Create pivot tables for a specific dataset."""
        try:
            # Create pivot tables for accuracy and ECE
            for metric in ['accuracy', 'ece']:
                pivot_data = []
                
                for loss_name in sorted(summary_df['losses'].unique()):
                    row_data = {'Loss': loss_name}
                    
                    for ctx_length in self.ctx_list:
                        row = summary_df[
                            (summary_df['losses'] == loss_name) & 
                            (summary_df['ctx_length'] == ctx_length)
                        ]
                        
                        if not row.empty:
                            row_data[f'ctx={ctx_length}'] = row[metric].values[0]
                        else:
                            row_data[f'ctx={ctx_length}'] = 'N/A'
                    
                    pivot_data.append(row_data)
                
                if pivot_data:
                    pivot_df = pd.DataFrame(pivot_data)
                    pivot_path = os.path.join(
                        self.results_dir,
                        f'{dataset}_{metric}_ctx_pivot.csv'
                    )
                    pivot_df.to_csv(pivot_path, index=False)
                    print(f"Created {metric} pivot table for {dataset}")
        
        except Exception as e:
            print(f"Error creating pivot tables for {dataset}: {e}")
    
    def generate_dataset_plots(self, dataset, summary_df):
        """Generate plots for a specific dataset."""
        try:
            # Plot Accuracy vs CTX Length
            plt.figure(figsize=(12, 6))
            
            # Prepare data for plotting
            for loss in sorted(summary_df['losses'].unique()):
                loss_data = summary_df[summary_df['losses'] == loss].copy()
                
                # Ensure ctx_length is numeric for sorting
                loss_data['ctx_length'] = pd.to_numeric(loss_data['ctx_length'])
                
                # Sort by ctx_length and extract mean accuracy
                loss_data = loss_data.sort_values('ctx_length')
                
                if 'mean_accuracy' in loss_data.columns:
                    y_values = loss_data['mean_accuracy']
                else:
                    # Extract from string format if needed
                    y_values = loss_data['accuracy'].apply(
                        lambda x: float(x.split('±')[0]) if isinstance(x, str) else x
                    )
                
                plt.plot(loss_data['ctx_length'], y_values, marker='o', linestyle='-', label=loss)
            
            plt.title(f'Accuracy vs CTX Length for {dataset} dataset (PLIP)')
            plt.xlabel('CTX Length')
            plt.ylabel('Accuracy (%)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='Loss Function')
            plt.tight_layout()
            
            acc_plot_path = os.path.join(self.plots_dir, f'{dataset}_accuracy_vs_ctx.png')
            plt.savefig(acc_plot_path, dpi=300)
            plt.close()
            
            # Plot ECE vs CTX Length
            plt.figure(figsize=(12, 6))
            
            for loss in sorted(summary_df['losses'].unique()):
                loss_data = summary_df[summary_df['losses'] == loss].copy()
                
                # Ensure ctx_length is numeric for sorting
                loss_data['ctx_length'] = pd.to_numeric(loss_data['ctx_length'])
                
                # Sort by ctx_length and extract mean ECE
                loss_data = loss_data.sort_values('ctx_length')
                
                if 'mean_ece' in loss_data.columns:
                    y_values = loss_data['mean_ece']
                else:
                    # Extract from string format if needed
                    y_values = loss_data['ece'].apply(
                        lambda x: float(x.split('±')[0]) if isinstance(x, str) else x
                    )
                
                plt.plot(loss_data['ctx_length'], y_values, marker='o', linestyle='-', label=loss)
            
            plt.title(f'ECE vs CTX Length for {dataset} dataset (PLIP)')
            plt.xlabel('CTX Length')
            plt.ylabel('ECE (%)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='Loss Function')
            plt.tight_layout()
            
            ece_plot_path = os.path.join(self.plots_dir, f'{dataset}_ece_vs_ctx.png')
            plt.savefig(ece_plot_path, dpi=300)
            plt.close()
            
            print(f"Generated plots for {dataset}")
            
        except Exception as e:
            print(f"Error generating plots for {dataset}: {e}")
    
    def generate_combined_summaries(self):
        """Generate combined summaries across all datasets."""
        try:
            # Collect all summary dataframes
            all_summaries = []
            
            for dataset in self.datasets:
                summary_path = os.path.join(
                    self.results_dir, 
                    f'{dataset}_{self.model_type}_summary.csv'
                )
                
                if os.path.exists(summary_path):
                    summary_df = pd.read_csv(summary_path)
                    all_summaries.append(summary_df)
            
            if not all_summaries:
                print("No summary data available")
                return
            
            # Combine all summaries
            combined_summary = pd.concat(all_summaries, ignore_index=True)
            combined_path = os.path.join(
                self.results_dir,
                f'combined_{self.model_type}_summary.csv'
            )
            combined_summary.to_csv(combined_path, index=False)
            
            print("Generated combined summaries")
            
        except Exception as e:
            print(f"Error generating combined summaries: {e}")

    def cleanup(self):
        """Clean up temporary configuration files."""
        try:
            for f in os.listdir(self.results_dir):
                if f.startswith('config_') and f.endswith('.yaml'):
                    os.remove(os.path.join(self.results_dir, f))
            print("Temporary config files cleaned up")
        except Exception as e:
            print(f"Error in cleanup: {e}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run CTX length ablation study for PLIP')
    
    # Add option to run for specific dataset and/or ctx length
    parser.add_argument('--dataset', type=str, choices=['kather', 'pannuke', 'digestpath'],
                        help='Run on a specific dataset only')
    parser.add_argument('--ctx', type=int, help='Run for a specific CTX length only')
    parser.add_argument('--loss', type=str, help='Run for a specific loss function only')
    parser.add_argument('--skip-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    try:
        print("Starting CTX length ablation study for PLIP...")
        ablation = CtxLengthAblationStudy()
        
        # Modify parameters if specified
        if args.dataset:
            ablation.datasets = [args.dataset]
            print(f"Running on dataset: {args.dataset} only")
        
        if args.ctx:
            ablation.ctx_list = [args.ctx]
            print(f"Running with CTX length: {args.ctx} only")
        
        if args.loss:
            for dataset in ablation.datasets:
                ablation.loss_configs_per_dataset[dataset] = [
                    conf for conf in ablation.loss_configs_per_dataset[dataset] 
                    if conf['name'] == args.loss
                ]
            print(f"Running with loss function: {args.loss} only")
        
        # Run the experiments
        ablation.run_all_experiments()
        
        # Clean up temp files
        ablation.cleanup()
        
        print("\nAblation study completed successfully!")
        
    except Exception as e:
        print(f"Error in ablation study: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """
# Shots Ablation Study for PLIP
# Run from: /home/abhishek/desktop/VLM_Cal/CLIP_Calibration/experiments/abalations
# """

# import yaml
# import os
# import subprocess
# import pandas as pd
# import re
# import argparse
# import numpy as np
# from datetime import datetime
# import sys
# import matplotlib.pyplot as plt
# import seaborn as sns

# class ShotsAblationStudy:
#     def __init__(self):
#         # Fixed to PLIP only for ablation study
#         self.model_type = 'plip'
        
#         # Define paths
#         self.base_dir = os.path.abspath("../..")  # Go up two levels from abalations
#         self.base_config_path = os.path.join(self.base_dir, f"configs/trainers/CoOp/vit_b32_{self.model_type}_c16_ep50_batch16.yaml")
#         self.results_dir = os.path.join(self.base_dir, f"output/shots_ablation_{self.model_type}")
#         os.makedirs(self.results_dir, exist_ok=True)
        
#         # Create subdirectories
#         self.plots_dir = os.path.join(self.results_dir, 'plots')
#         self.latex_dir = os.path.join(self.results_dir, 'latex')
#         os.makedirs(self.plots_dir, exist_ok=True)
#         os.makedirs(self.latex_dir, exist_ok=True)
        
#         # Seeds (using just one seed for efficiency as in the original code)
#         self.seeds = [1]
        
#         # Shots for ablation study
#         self.shots_list = [2, 4, 8, 12, 16, 32]
        
#         # Fixed parameters
#         self.fixed_params = {
#             'lr': 0.002,
#             'epochs': 50
#         }
        
#         # Datasets
#         self.datasets = ["kather", "pannuke", "digestpath"]
        
#         # Load and prepare loss configurations based on datasets
#         self.dataset_configs = self.get_dataset_configs()
#         self.loss_configs_per_dataset = {
#             dataset: self.get_loss_configs(dataset) for dataset in self.datasets
#         }
        
#         print(f"Ablation study configuration:")
#         print(f"- Model: {self.model_type}")
#         print(f"- Datasets: {', '.join(self.datasets)}")
#         print(f"- Shots: {', '.join(map(str, self.shots_list))}")
#         print(f"- Results directory: {self.results_dir}")

#     def get_dataset_configs(self):
#         """Get dataset-specific configurations for PLIP with only the required loss functions."""
#         return {
#             'kather': {
#                 'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.07}},
#                 'CE_SLMDCA_CL': {'SLMDCA': {'ALPHA': 0.07}, 'COSINE': {'WEIGHT': 0.01}},
#                 'FL_SLMDCA': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.03}},
#                 'FL_SLMDCA_CL': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.03}, 'COSINE': {'WEIGHT': 0.01}},
#                 'LS_SLMDCA': {'LS': {'ALPHA': 0.05}, 'SLMDCA': {'ALPHA': 0.05}},
#                 'LS_SLMDCA_CL': {'LS': {'ALPHA': 0.05}, 'SLMDCA': {'ALPHA': 0.05}, 'COSINE': {'WEIGHT': 0.01}}
#             },
#             'pannuke': {
#                 'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.2}},
#                 'CE_SLMDCA_CL': {'SLMDCA': {'ALPHA': 0.2}, 'COSINE': {'WEIGHT': 0.1}},
#                 'FL_SLMDCA': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.2}},
#                 'FL_SLMDCA_CL': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.2}, 'COSINE': {'WEIGHT': 0.1}},
#                 'LS_SLMDCA': {'LS': {'ALPHA': 0.2}, 'SLMDCA': {'ALPHA': 0.2}},
#                 'LS_SLMDCA_CL': {'LS': {'ALPHA': 0.2}, 'SLMDCA': {'ALPHA': 0.2}, 'COSINE': {'WEIGHT': 0.1}}
#             },
#             'digestpath': {
#                 'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.03}},
#                 'CE_SLMDCA_CL': {'SLMDCA': {'ALPHA': 0.03}, 'COSINE': {'WEIGHT': 0.001}},
#                 'FL_SLMDCA': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.03}},
#                 'FL_SLMDCA_CL': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.03}, 'COSINE': {'WEIGHT': 0.001}},
#                 'LS_SLMDCA': {'LS': {'ALPHA': 0.03}, 'SLMDCA': {'ALPHA': 0.03}},
#                 'LS_SLMDCA_CL': {'LS': {'ALPHA': 0.03}, 'SLMDCA': {'ALPHA': 0.03}, 'COSINE': {'WEIGHT': 0.001}}
#             }
#         }

#     def get_loss_configs(self, dataset):
#         """Convert dataset-specific loss configs into the format needed for experiments."""
#         dataset_params = self.dataset_configs[dataset]
        
#         loss_configs = []
#         for loss_name, params in dataset_params.items():
#             # Determine enabled losses based on loss name
#             enabled_losses = []
#             for part in loss_name.split('_'):
#                 if part in ['CE', 'FL', 'LS', 'DCA', 'MMCE', 'MDCA', 'SLMDCA', 'CL', 'COSINE']:
#                     if part == 'CL':  # CL is actually COSINE in the config
#                         enabled_losses.append('COSINE')
#                     else:
#                         enabled_losses.append(part)
            
#             loss_configs.append({
#                 'name': loss_name,
#                 'enabled_losses': enabled_losses,
#                 'params': params
#             })
            
#         return loss_configs

#     def modify_config(self, dataset, loss_config):
#         """Create a modified config file for the experiment."""
#         try:
#             # Read base config
#             with open(self.base_config_path, 'r') as f:
#                 config = yaml.safe_load(f)
            
#             # Set fixed parameters
#             config['OPTIM']['LR'] = self.fixed_params['lr']
#             config['OPTIM']['MAX_EPOCH'] = self.fixed_params['epochs']
            
#             # Add the FINETUNE_MODE for prompt learning
#             config['TRAINER']['COOP']['FINETUNE_MODE'] = 'prompt'
            
#             # Remove the PROMPT_TEMPLATE setting if it exists
#             if 'PROMPT_TEMPLATE' in config['TRAINER']['COOP']:
#                 del config['TRAINER']['COOP']['PROMPT_TEMPLATE']
            
#             # Set loss configuration
#             loss_config_yaml = config['TRAINER']['COOP']['LOSS']
#             loss_config_yaml['ENABLED_LOSSES'] = loss_config['enabled_losses']
            
#             # Configure each individual loss
#             for loss_name, loss_params in loss_config['params'].items():
#                 if loss_name not in loss_config_yaml:
#                     loss_config_yaml[loss_name] = {}
                
#                 # Set all parameters for this loss
#                 for param_name, param_value in loss_params.items():
#                     loss_config_yaml[loss_name][param_name] = param_value
            
#             # Save config file
#             config_name = f"config_{self.model_type}_{dataset}_{loss_config['name']}.yaml"
#             config_path = os.path.join(self.results_dir, config_name)
            
#             with open(config_path, 'w') as f:
#                 yaml.dump(config, f)
            
#             return config_path
            
#         except Exception as e:
#             print(f"Error modifying config: {e}")
#             return None

#     def extract_metrics(self, log_path):
#         """Extract relevant metrics from experiment log file."""
#         try:
#             with open(log_path, 'r') as f:
#                 content = f.read()
#                 metrics = {}
                
#                 # Metrics to extract
#                 patterns = {
#                     'accuracy': r'\* accuracy: ([\d.]+)%',
#                     'error_rate': r'\* error: ([\d.]+)%',
#                     'confidence': r'\* confidence: ([\d.]+)%',
#                     'ece': r'\* ece: ([\d.]+)%',
#                     'mce': r'\* mce: ([\d.]+)%',
#                     'ace': r'\* ace: ([\d.]+)%',
#                     'macro_f1': r'\* macro_f1: ([\d.]+)%'
#                 }
                
#                 for metric_name, pattern in patterns.items():
#                     match = re.search(pattern, content)
#                     if not match:
#                         print(f"Warning: Missing {metric_name} in log")
#                         metrics[metric_name] = float('nan')
#                     else:
#                         metrics[metric_name] = float(match.group(1))
                
#                 return metrics
#         except Exception as e:
#             print(f"Error reading log {log_path}: {e}")
#             return None

#     def run_experiment(self, dataset, loss_config, seed, shots):
#         """Run a single experiment with the specified parameters."""
#         try:
#             # Get modified config
#             config_path = self.modify_config(dataset, loss_config)
#             if not config_path:
#                 print(f"Error: Failed to create config for {dataset}, {loss_config['name']}, shots={shots}")
#                 return None
            
#             # Setup experiment directory
#             exp_name = f"{dataset}/shots_{shots}/CoOp/prompt/{loss_config['name']}"
#             output_dir = os.path.join(self.results_dir, exp_name, f"seed{seed}")
#             os.makedirs(output_dir, exist_ok=True)
            
#             # Check for completed experiment
#             log_file = os.path.join(output_dir, "log.txt")
#             if os.path.exists(log_file):
#                 print(f"Experiment exists: {dataset}, {loss_config['name']}, shots={shots}, seed={seed}")
#                 metrics = self.extract_metrics(log_file)
#                 if metrics:
#                     self.save_experiment_result(dataset, shots, loss_config['name'], seed, metrics)
#                 return metrics
            
#             # Command to run the experiment
#             cmd = [
#                 "python", os.path.join(self.base_dir, "train.py"),
#                 "--root", "/home/abhishek/desktop/VLM_Cal/CLIP_Calibration/$DATA",
#                 "--seed", str(seed),
#                 "--trainer", "CoOp",
#                 "--dataset-config-file", os.path.join(self.base_dir, f"configs/datasets/{dataset}.yaml"),
#                 "--config-file", config_path,
#                 "--output-dir", output_dir,
#                 "DATASET.NUM_SHOTS", str(shots),
#                 "DATASET.SUBSAMPLE_CLASSES", "all",
#                 "MODEL.NAME", self.model_type,
#                 "MODEL_ROOT", os.path.join(self.base_dir, "models")
#             ]
            
#             # Run the experiment
#             print(f"\nRunning: {dataset}, {loss_config['name']}, shots={shots}, seed={seed}")
#             subprocess.run(cmd, check=True)
            
#             # Extract metrics and save results
#             metrics = self.extract_metrics(log_file)
#             if metrics:
#                 self.save_experiment_result(dataset, shots, loss_config['name'], seed, metrics)
#             return metrics
            
#         except Exception as e:
#             print(f"Error in experiment: {e}")
#             return None

#     def save_experiment_result(self, dataset, shots, loss_name, seed, metrics):
#         """Save a single experiment result to CSV."""
#         try:
#             # Create result row
#             result_row = {
#                 'dataset': dataset,
#                 'shots': shots,
#                 'losses': loss_name,
#                 'seed': seed,
#                 'accuracy': metrics['accuracy'],
#                 'ece': metrics['ece'],
#                 'mce': metrics['mce'],
#                 'ace': metrics['ace'],
#                 'macro_f1': metrics['macro_f1'],
#                 'confidence': metrics.get('confidence', float('nan')),
#                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             }
            
#             # Define CSV path for individual results
#             individual_csv_path = os.path.join(
#                 self.results_dir, 
#                 f'{dataset}_shots{shots}_{loss_name}_seed{seed}_result.csv'
#             )
            
#             # Save individual result
#             pd.DataFrame([result_row]).to_csv(individual_csv_path, index=False)
            
#             # Also append to dataset results file
#             dataset_csv_path = os.path.join(
#                 self.results_dir, 
#                 f'{dataset}_{self.model_type}_all_shots_results.csv'
#             )
            
#             if os.path.exists(dataset_csv_path):
#                 # Read existing CSV
#                 existing_df = pd.read_csv(dataset_csv_path)
                
#                 # Check if this exact experiment already exists
#                 mask = (
#                     (existing_df['dataset'] == dataset) & 
#                     (existing_df['shots'] == shots) & 
#                     (existing_df['losses'] == loss_name) & 
#                     (existing_df['seed'] == seed)
#                 )
                
#                 if mask.any():
#                     # Update existing row
#                     existing_df.loc[mask, 'accuracy'] = metrics['accuracy']
#                     existing_df.loc[mask, 'ece'] = metrics['ece']
#                     existing_df.loc[mask, 'mce'] = metrics['mce']
#                     existing_df.loc[mask, 'ace'] = metrics['ace']
#                     existing_df.loc[mask, 'macro_f1'] = metrics['macro_f1']
#                     existing_df.loc[mask, 'confidence'] = metrics.get('confidence', float('nan'))
#                     existing_df.loc[mask, 'timestamp'] = result_row['timestamp']
#                 else:
#                     # Append new row
#                     existing_df = pd.concat([existing_df, pd.DataFrame([result_row])], ignore_index=True)
                
#                 existing_df.to_csv(dataset_csv_path, index=False)
#             else:
#                 # Create new CSV with this row
#                 pd.DataFrame([result_row]).to_csv(dataset_csv_path, index=False)
                
#             print(f"Saved result for {dataset}, {loss_name}, shots={shots}, seed={seed}")
#             return True
#         except Exception as e:
#             print(f"Error saving experiment result: {e}")
#             return False

#     def run_all_experiments(self):
#         """Run all experiments for the shots ablation study."""
#         all_results = []
        
#         # Track total experiments and completed
#         total_experiments = len(self.datasets) * len(self.shots_list) * 6 * len(self.seeds)  # 6 loss functions
#         completed_experiments = 0
        
#         # Iterate through all combinations
#         for dataset in self.datasets:
#             print(f"\n=== Dataset: {dataset} ===")
#             dataset_results = []
            
#             for shots in self.shots_list:
#                 print(f"\n--- Shots: {shots} ---")
                
#                 for loss_config in self.loss_configs_per_dataset[dataset]:
#                     print(f"\nTesting: {loss_config['name']}")
                    
#                     for seed in self.seeds:
#                         # Run experiment and get metrics
#                         metrics = self.run_experiment(dataset, loss_config, seed, shots)
#                         completed_experiments += 1
                        
#                         if metrics:
#                             result_row = {
#                                 'dataset': dataset,
#                                 'shots': shots,
#                                 'losses': loss_config['name'],
#                                 'seed': seed,
#                                 'accuracy': metrics['accuracy'],
#                                 'ece': metrics['ece'],
#                                 'mce': metrics['mce'],
#                                 'ace': metrics['ace'],
#                                 'macro_f1': metrics['macro_f1'],
#                                 'confidence': metrics.get('confidence', float('nan'))
#                             }
#                             dataset_results.append(result_row)
#                             all_results.append(result_row)
                
#                 print(f"Progress: {completed_experiments}/{total_experiments} experiments completed")
                
#                 # Generate summary statistics for this dataset
#                 self.generate_dataset_summaries(dataset)
            
#             # Save combined dataset results
#             print(f"All experiments for {dataset} completed")
        
#         # Create comprehensive summaries across all datasets
#         self.generate_combined_summaries()
        
#         return True
    
#     def generate_dataset_summaries(self, dataset):
#         """Generate summary statistics for a dataset."""
#         try:
#             # Path to dataset results
#             dataset_csv_path = os.path.join(
#                 self.results_dir, 
#                 f'{dataset}_{self.model_type}_all_shots_results.csv'
#             )
            
#             if not os.path.exists(dataset_csv_path):
#                 print(f"Warning: No results file found for {dataset}")
#                 return False
            
#             # Load results
#             results_df = pd.read_csv(dataset_csv_path)
            
#             # Remove any existing summary rows
#             results_df = results_df[results_df['seed'] != 'mean_std']
            
#             # Create summary rows
#             summary_rows = []
            
#             for shots in self.shots_list:
#                 for loss_name in set(results_df['losses']):
#                     # Filter for this shots/loss combination
#                     combo_df = results_df[
#                         (results_df['shots'] == shots) & 
#                         (results_df['losses'] == loss_name)
#                     ]
                    
#                     if len(combo_df) == 0:
#                         continue
                    
#                     # Calculate mean and std for all metrics
#                     mean_acc = combo_df['accuracy'].mean()
#                     std_acc = combo_df['accuracy'].std() if len(combo_df) > 1 else 0
#                     mean_ece = combo_df['ece'].mean()
#                     std_ece = combo_df['ece'].std() if len(combo_df) > 1 else 0
#                     mean_mce = combo_df['mce'].mean()
#                     std_mce = combo_df['mce'].std() if len(combo_df) > 1 else 0
#                     mean_ace = combo_df['ace'].mean()
#                     std_ace = combo_df['ace'].std() if len(combo_df) > 1 else 0
#                     mean_f1 = combo_df['macro_f1'].mean()
#                     std_f1 = combo_df['macro_f1'].std() if len(combo_df) > 1 else 0
                    
#                     # Create summary row
#                     summary_row = {
#                         'dataset': dataset,
#                         'shots': shots,
#                         'losses': loss_name,
#                         'seed': 'mean_std',
#                         'accuracy': f"{mean_acc:.2f}±{std_acc:.2f}",
#                         'ece': f"{mean_ece:.2f}±{std_ece:.2f}",
#                         'mce': f"{mean_mce:.2f}±{std_mce:.2f}",
#                         'ace': f"{mean_ace:.2f}±{std_ace:.2f}",
#                         'macro_f1': f"{mean_f1:.2f}±{std_f1:.2f}",
#                         'mean_accuracy': mean_acc,
#                         'std_accuracy': std_acc,
#                         'mean_ece': mean_ece,
#                         'std_ece': std_ece,
#                         'mean_mce': mean_mce,
#                         'std_mce': std_mce,
#                         'mean_ace': mean_ace,
#                         'std_ace': std_ace,
#                         'mean_macro_f1': mean_f1,
#                         'std_macro_f1': std_f1
#                     }
#                     summary_rows.append(summary_row)
            
#             # Add summary rows to results and save
#             if summary_rows:
#                 combined_df = pd.concat([results_df, pd.DataFrame(summary_rows)], ignore_index=True)
#                 combined_df.to_csv(dataset_csv_path, index=False)
                
#                 # Also save just the summary rows to a separate file
#                 summary_df = pd.DataFrame(summary_rows)
#                 summary_path = os.path.join(
#                     self.results_dir, 
#                     f'{dataset}_{self.model_type}_summary.csv'
#                 )
#                 summary_df.to_csv(summary_path, index=False)
                
#                 print(f"Generated summary statistics for {dataset}")
                
#                 # Create pivot tables for this dataset
#                 self.create_dataset_pivot_tables(dataset, summary_df)
                
#                 # Generate plots for this dataset
#                 self.generate_dataset_plots(dataset, summary_df)
                
#                 return True
#             else:
#                 print(f"Warning: No data to generate summaries for {dataset}")
#                 return False
                
#         except Exception as e:
#             print(f"Error generating dataset summaries for {dataset}: {e}")
#             return False
    
#     def create_dataset_pivot_tables(self, dataset, summary_df):
#         """Create pivot tables for a specific dataset."""
#         try:
#             # Create pivot tables for accuracy and ECE
#             for metric in ['accuracy', 'ece']:
#                 pivot_data = []
                
#                 for loss_name in sorted(summary_df['losses'].unique()):
#                     row_data = {'Loss': loss_name}
                    
#                     for shots in self.shots_list:
#                         row = summary_df[
#                             (summary_df['losses'] == loss_name) & 
#                             (summary_df['shots'] == shots)
#                         ]
                        
#                         if not row.empty:
#                             row_data[f'{shots} shots'] = row[metric].values[0]
#                         else:
#                             row_data[f'{shots} shots'] = 'N/A'
                    
#                     pivot_data.append(row_data)
                
#                 if pivot_data:
#                     pivot_df = pd.DataFrame(pivot_data)
#                     pivot_path = os.path.join(
#                         self.results_dir,
#                         f'{dataset}_{metric}_shots_pivot.csv'
#                     )
#                     pivot_df.to_csv(pivot_path, index=False)
#                     print(f"Created {metric} pivot table for {dataset}")
        
#         except Exception as e:
#             print(f"Error creating pivot tables for {dataset}: {e}")
    
#     def generate_dataset_plots(self, dataset, summary_df):
#         """Generate plots for a specific dataset."""
#         try:
#             # Plot Accuracy vs Shots
#             plt.figure(figsize=(12, 6))
            
#             # Prepare data for plotting
#             for loss in sorted(summary_df['losses'].unique()):
#                 loss_data = summary_df[summary_df['losses'] == loss].copy()
                
#                 # Ensure shots is numeric for sorting
#                 loss_data['shots'] = pd.to_numeric(loss_data['shots'])
                
#                 # Sort by shots and extract mean accuracy
#                 loss_data = loss_data.sort_values('shots')
                
#                 if 'mean_accuracy' in loss_data.columns:
#                     y_values = loss_data['mean_accuracy']
#                 else:
#                     # Extract from string format if needed
#                     y_values = loss_data['accuracy'].apply(
#                         lambda x: float(x.split('±')[0]) if isinstance(x, str) else x
#                     )
                
#                 plt.plot(loss_data['shots'], y_values, marker='o', linestyle='-', label=loss)
            
#             plt.title(f'Accuracy vs Shots for {dataset} dataset (PLIP)')
#             plt.xlabel('Number of Shots')
#             plt.ylabel('Accuracy (%)')
#             plt.grid(True, linestyle='--', alpha=0.7)
#             plt.legend(title='Loss Function')
#             plt.tight_layout()
            
#             acc_plot_path = os.path.join(self.plots_dir, f'{dataset}_accuracy_vs_shots.png')
#             plt.savefig(acc_plot_path, dpi=300)
#             plt.close()
            
#             # Plot ECE vs Shots
#             plt.figure(figsize=(12, 6))
            
#             for loss in sorted(summary_df['losses'].unique()):
#                 loss_data = summary_df[summary_df['losses'] == loss].copy()
                
#                 # Ensure shots is numeric for sorting
#                 loss_data['shots'] = pd.to_numeric(loss_data['shots'])
                
#                 # Sort by shots and extract mean ECE
#                 loss_data = loss_data.sort_values('shots')
                
#                 if 'mean_ece' in loss_data.columns:
#                     y_values = loss_data['mean_ece']
#                 else:
#                     # Extract from string format if needed
#                     y_values = loss_data['ece'].apply(
#                         lambda x: float(x.split('±')[0]) if isinstance(x, str) else x
#                     )
                
#                 plt.plot(loss_data['shots'], y_values, marker='o', linestyle='-', label=loss)
            
#             plt.title(f'ECE vs Shots for {dataset} dataset (PLIP)')
#             plt.xlabel('Number of Shots')
#             plt.ylabel('ECE (%)')
#             plt.grid(True, linestyle='--', alpha=0.7)
#             plt.legend(title='Loss Function')
#             plt.tight_layout()
            
#             ece_plot_path = os.path.join(self.plots_dir, f'{dataset}_ece_vs_shots.png')
#             plt.savefig(ece_plot_path, dpi=300)
#             plt.close()
            
#             print(f"Generated plots for {dataset}")
            
#         except Exception as e:
#             print(f"Error generating plots for {dataset}: {e}")
    
#     def generate_combined_summaries(self):
#         """Generate combined summaries across all datasets."""
#         try:
#             # Collect all summary dataframes
#             all_summaries = []
            
#             for dataset in self.datasets:
#                 summary_path = os.path.join(
#                     self.results_dir, 
#                     f'{dataset}_{self.model_type}_summary.csv'
#                 )
                
#                 if os.path.exists(summary_path):
#                     summary_df = pd.read_csv(summary_path)
#                     all_summaries.append(summary_df)
            
#             if not all_summaries:
#                 print("No summary data available")
#                 return
            
#             # Combine all summaries
#             combined_summary = pd.concat(all_summaries, ignore_index=True)
#             combined_path = os.path.join(
#                 self.results_dir,
#                 f'combined_{self.model_type}_summary.csv'
#             )
#             combined_summary.to_csv(combined_path, index=False)
            
#             # Create LaTeX tables
#             self.create_latex_tables(combined_summary)
            
#             print("Generated combined summaries")
            
#         except Exception as e:
#             print(f"Error generating combined summaries: {e}")
    
#     def create_latex_tables(self, summary_df):
#         """Create LaTeX tables for the paper."""
#         try:
#             # Create individual tables per dataset
#             for dataset in self.datasets:
#                 dataset_summary = summary_df[summary_df['dataset'] == dataset]
                
#                 if dataset_summary.empty:
#                     continue
                
#                 for metric_name, metric in [('Accuracy', 'accuracy'), ('ECE', 'ece')]:
#                     latex_path = os.path.join(
#                         self.latex_dir,
#                         f'{dataset}_{metric}_shots_table.tex'
#                     )
                    
#                     with open(latex_path, 'w') as f:
#                         f.write("\\begin{table}[ht]\n")
#                         f.write("\\centering\n")
#                         f.write(f"\\caption{{Effect of varying shots on {metric_name} (\\%) for {dataset} dataset with PLIP model}}\n")
#                         f.write("\\begin{tabular}{l" + "c" * len(self.shots_list) + "}\n")
#                         f.write("\\toprule\n")
                        
#                         # Header row
#                         f.write("Loss Function")
#                         for shots in self.shots_list:
#                             f.write(f" & {shots} shots")
#                         f.write(" \\\\\n")
#                         f.write("\\midrule\n")
                        
#                         # Data rows
#                         for loss_name in sorted(dataset_summary['losses'].unique()):
#                             f.write(f"{loss_name}")
                            
#                             for shots in self.shots_list:
#                                 row = dataset_summary[
#                                     (dataset_summary['losses'] == loss_name) & 
#                                     (dataset_summary['shots'] == shots)
#                                 ]
                                
#                                 if not row.empty and metric in row.columns:
#                                     f.write(f" & {row[metric].values[0]}")
#                                 else:
#                                     f.write(" & --")
                            
#                             f.write(" \\\\\n")
                        
#                         f.write("\\bottomrule\n")
#                         f.write("\\end{tabular}\n")
#                         f.write(f"\\label{{tab:{dataset}_{metric}_shots_ablation}}\n")
#                         f.write("\\end{table}\n")
                    
#                     print(f"Created LaTeX table for {dataset} {metric}")
            
#             # Create comprehensive table for all datasets
#             for metric_name, metric in [('Accuracy', 'accuracy'), ('ECE', 'ece')]:
#                 latex_path = os.path.join(
#                     self.latex_dir,
#                     f'all_datasets_{metric}_shots_table.tex'
#                 )
                
#                 with open(latex_path, 'w') as f:
#                     f.write("\\begin{table}[ht]\n")
#                     f.write("\\centering\n")
#                     f.write(f"\\caption{{Effect of shots on {metric_name} (\\%) across datasets with PLIP model}}\n")
#                     f.write("\\begin{tabular}{lc" + "c" * len(self.shots_list) + "}\n")
#                     f.write("\\toprule\n")
                    
#                     # Header row
#                     f.write("Dataset & Loss Function")
#                     for shots in self.shots_list:
#                         f.write(f" & {shots}")
#                     f.write(" \\\\\n")
#                     f.write("\\midrule\n")
                    
#                     # Data rows grouped by dataset
#                     for dataset in self.datasets:
#                         dataset_summary = summary_df[summary_df['dataset'] == dataset]
#                         loss_functions = sorted(dataset_summary['losses'].unique())
                        
#                         if not loss_functions:
#                             continue
                        
#                         # Use multirow for dataset name
#                         f.write(f"\\multirow{{{len(loss_functions)}}}{{*}}{{{dataset}}}")
                        
#                         for i, loss_name in enumerate(loss_functions):
#                             if i > 0:
#                                 f.write(" ")
                            
#                             f.write(f" & {loss_name}")
                            
#                             for shots in self.shots_list:
#                                 row = dataset_summary[
#                                     (dataset_summary['losses'] == loss_name) & 
#                                     (dataset_summary['shots'] == shots)
#                                 ]
                                
#                                 if not row.empty and metric in row.columns:
#                                     f.write(f" & {row[metric].values[0]}")
#                                 else:
#                                     f.write(" & --")
                            
#                             f.write(" \\\\\n")
                        
#                         if dataset != self.datasets[-1]:
#                             f.write("\\midrule\n")
                    
#                     f.write("\\bottomrule\n")
#                     f.write("\\end{tabular}\n")
#                     f.write(f"\\label{{tab:all_datasets_{metric}_shots_ablation}}\n")
#                     f.write("\\end{table}\n")
                
#                 print(f"Created comprehensive LaTeX table for {metric}")
            
#         except Exception as e:
#             print(f"Error creating LaTeX tables: {e}")

#     def cleanup(self):
#         """Clean up temporary configuration files."""
#         try:
#             for f in os.listdir(self.results_dir):
#                 if f.startswith('config_') and f.endswith('.yaml'):
#                     os.remove(os.path.join(self.results_dir, f))
#             print("Temporary config files cleaned up")
#         except Exception as e:
#             print(f"Error in cleanup: {e}")

# def main():
#     """Main execution function."""
#     parser = argparse.ArgumentParser(description='Run shots ablation study for PLIP')
    
#     # Add option to run for specific dataset and/or shots
#     parser.add_argument('--dataset', type=str, choices=['kather', 'pannuke', 'digestpath'],
#                         help='Run on a specific dataset only')
#     parser.add_argument('--shots', type=int, help='Run for a specific shots value only')
#     parser.add_argument('--loss', type=str, help='Run for a specific loss function only')
#     parser.add_argument('--skip-plots', action='store_true', help='Skip generating plots')
#     parser.add_argument('--skip-latex', action='store_true', help='Skip generating LaTeX tables')
    
#     args = parser.parse_args()
    
#     try:
#         print("Starting shots ablation study for PLIP...")
#         ablation = ShotsAblationStudy()
        
#         # Modify parameters if specified
#         if args.dataset:
#             ablation.datasets = [args.dataset]
#             print(f"Running on dataset: {args.dataset} only")
        
#         if args.shots:
#             ablation.shots_list = [args.shots]
#             print(f"Running with shots: {args.shots} only")
        
#         if args.loss:
#             for dataset in ablation.datasets:
#                 ablation.loss_configs_per_dataset[dataset] = [
#                     conf for conf in ablation.loss_configs_per_dataset[dataset] 
#                     if conf['name'] == args.loss
#                 ]
#             print(f"Running with loss function: {args.loss} only")
        
#         # Run the experiments
#         ablation.run_all_experiments()
        
#         # Clean up temp files
#         ablation.cleanup()
        
#         print("\nAblation study completed successfully!")
        
#     except Exception as e:
#         print(f"Error in ablation study: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()