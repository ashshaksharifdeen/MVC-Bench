import yaml
import os
import subprocess
import pandas as pd
import re
import argparse
import numpy as np
from datetime import datetime

class BaseLinePromptLearningExperiments:
    def __init__(self, model_type, dataset=None):
        self.model_type = model_type
        
        # Set the base config path based on model type with correct ViT-B/32 architecture
        if model_type == 'medclip':
            self.base_config_path = f"../../configs/trainers/CoOp_MedCLIP/vit_b32_medclip_c16_ep50_batch16.yaml"
            self.trainer_name = "CoOp_MedCLIP"
        elif model_type == 'biomedclip':
            self.base_config_path = f"../../configs/trainers/CoOp_BioMedCLIP/vit_b32_biomedclip_c16_ep50_batch16.yaml"
            self.trainer_name = "CoOp_BioMedCLIP"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.results_dir = f"../../output/medclip_biomedclip_baseline/prompt_{model_type}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Using seeds 1, 2, 3
        self.seeds = [1]
        
        # Fixed parameters
        self.fixed_params = {
            'lr': 0.002,
            'epochs': 50,
            'shots': 8
        }
        
        # Datasets - use specific dataset if provided, otherwise use both
        if dataset:
            self.datasets = [dataset]
        else:
            self.datasets = ["covid", "rsna18"]
        
        # Create dataset-specific loss configurations
        self.loss_configs = self._create_dataset_specific_configs()
            
        # Keep all results for progress tracking
        self.all_results = []

    def _create_dataset_specific_configs(self):
        """Create dataset-specific loss configurations."""
        # Base configurations (shared across datasets)
        base_configs = [
            {'name': 'LS', 'enabled_losses': ['LS'], 'params': {'LS': {'ALPHA': 0.05}}},
            {'name': 'LS_MDCA', 'enabled_losses': ['LS', 'MDCA'], 'params': {'LS': {'ALPHA': 0.05}, 'MDCA': {'WEIGHT': 1.0}}},
            {'name': 'ECE_KDE', 'enabled_losses': ['ECE_KDE'], 'params': {'ECE_KDE': {'WEIGHT': 1.0}}}
        ]
        
        # COVID dataset configs for MedCLIP
        covid_medclip_configs = [
            {'name': 'LS_SLMDCA', 'enabled_losses': ['LS', 'SLMDCA'], 'params': {'LS': {'ALPHA': 0.05}, 'SLMDCA': {'ALPHA': 0.05}}},
            {'name': 'LS_SLMDCA_COSINE', 'enabled_losses': ['LS', 'SLMDCA', 'COSINE'], 'params': {
                'LS': {'ALPHA': 0.05}, 
                'SLMDCA': {'ALPHA': 0.05}, 
                'COSINE': {'WEIGHT': 0.001}
            }}
        ]
        
        # RSNA dataset configs for MedCLIP
        rsna_medclip_configs = [
            {'name': 'LS_SLMDCA', 'enabled_losses': ['LS', 'SLMDCA'], 'params': {'LS': {'ALPHA': 0.17}, 'SLMDCA': {'ALPHA': 0.17}}},
            {'name': 'LS_SLMDCA_COSINE', 'enabled_losses': ['LS', 'SLMDCA', 'COSINE'], 'params': {
                'LS': {'ALPHA': 0.17}, 
                'SLMDCA': {'ALPHA': 0.17}, 
                'COSINE': {'WEIGHT': 0.01}
            }}
        ]
        
        # COVID dataset configs for BioMedCLIP
        covid_biomedclip_configs = [
            {'name': 'LS_SLMDCA', 'enabled_losses': ['LS', 'SLMDCA'], 'params': {'LS': {'ALPHA': 0.05}, 'SLMDCA': {'ALPHA': 0.05}}},
            {'name': 'LS_SLMDCA_COSINE', 'enabled_losses': ['LS', 'SLMDCA', 'COSINE'], 'params': {
                'LS': {'ALPHA': 0.05}, 
                'SLMDCA': {'ALPHA': 0.05}, 
                'COSINE': {'WEIGHT': 0.001}
            }}
        ]
        
        # RSNA dataset configs for BioMedCLIP
        rsna_biomedclip_configs = [
            {'name': 'LS_SLMDCA', 'enabled_losses': ['LS', 'SLMDCA'], 'params': {'LS': {'ALPHA': 0.1}, 'SLMDCA': {'ALPHA': 0.1}}},
            {'name': 'LS_SLMDCA_COSINE', 'enabled_losses': ['LS', 'SLMDCA', 'COSINE'], 'params': {
                'LS': {'ALPHA': 0.1}, 
                'SLMDCA': {'ALPHA': 0.1}, 
                'COSINE': {'WEIGHT': 0.01}
            }}
        ]
        
        # We'll use this to store our dataset-specific configurations
        self.dataset_specific_configs = {
            'covid': {
                'medclip': covid_medclip_configs,
                'biomedclip': covid_biomedclip_configs
            },
            'rsna18': {
                'medclip': rsna_medclip_configs,
                'biomedclip': rsna_biomedclip_configs
            }
        }
        
        # For backward compatibility, return the base configs
        # We'll select from dataset_specific_configs during run_experiment
        return base_configs

    def get_loss_configs_for_dataset(self, dataset):
        """Get loss configurations specific to the dataset and model type."""
        # Start with base configs
        configs = self.loss_configs.copy()
        
        # Add dataset-specific configs if available
        if dataset in self.dataset_specific_configs and self.model_type in self.dataset_specific_configs[dataset]:
            dataset_configs = self.dataset_specific_configs[dataset][self.model_type]
            # Filter out any duplicates by name
            existing_names = {config['name'] for config in configs}
            for config in dataset_configs:
                if config['name'] not in existing_names:
                    configs.append(config)
                else:
                    # Replace with dataset-specific version if name exists
                    for i, existing_config in enumerate(configs):
                        if existing_config['name'] == config['name']:
                            configs[i] = config
                            break
        
        return configs

    def modify_config(self, dataset, loss_config):
        """Modify configuration file for the experiment."""
        try:
            # Read base config
            with open(self.base_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set fixed parameters
            config['OPTIM']['LR'] = self.fixed_params['lr']
            config['OPTIM']['MAX_EPOCH'] = self.fixed_params['epochs']
            
            # Ensure correct backbone specification
            if 'MODEL' not in config:
                config['MODEL'] = {}
            if 'BACKBONE' not in config['MODEL']:
                config['MODEL']['BACKBONE'] = {}
            config['MODEL']['BACKBONE']['NAME'] = "ViT-B/32"
            
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
            
            # Print configuration for verification
            print(f"\nConfiguration for {dataset} - {loss_config['name']}:")
            print(f"Enabled losses: {loss_config['enabled_losses']}")
            for loss_name, loss_params in loss_config['params'].items():
                print(f"{loss_name}: {loss_params}")
            
            # Save config file
            config_name = f"config_{self.model_type}_{dataset}_{loss_config['name']}.yaml"
            config_path = os.path.join(self.results_dir, config_name)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            return config_path
            
        except Exception as e:
            print(f"Error modifying config: {e}")
            raise

    def run_experiment(self, dataset, loss_config, seed):
        """Run a single experiment with specified parameters."""
        try:
            # Get modified config
            config_path = self.modify_config(dataset, loss_config)
            
            # Setup experiment directory
            exp_name = f"{dataset}/shots_{self.fixed_params['shots']}/{self.trainer_name}/{loss_config['name']}"
            output_dir = os.path.join(self.results_dir, exp_name, f"seed{seed}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Check for completed experiment
            log_file = os.path.join(output_dir, "log.txt")
            if os.path.exists(log_file):
                print(f"Experiment exists: {dataset}, {loss_config['name']}, seed={seed}")
                return self.extract_metrics(log_file)
            
            # Prepare command
            cmd = [
                "python", "../../train.py",
                "--root", "/home/abhishek/desktop/VLM_Cal/CLIP_Calibration/$DATA",
                "--seed", str(seed),
                "--trainer", self.trainer_name,
                "--dataset-config-file", f"../../configs/datasets/{dataset}.yaml",
                "--config-file", config_path,
                "--output-dir", output_dir,
                "DATASET.NUM_SHOTS", str(self.fixed_params['shots']),
                "DATASET.SUBSAMPLE_CLASSES", "all",
                "MODEL.NAME", self.model_type,
                "MODEL_ROOT", "../../models"
            ]
            
            # Run experiment
            print(f"\nRunning: {dataset}, {loss_config['name']}, seed={seed}")
            subprocess.run(cmd, check=True)
            return self.extract_metrics(log_file)
            
        except Exception as e:
            print(f"Error in experiment: {e}")
            return None

    def extract_metrics(self, log_path):
        """Extract metrics from experiment log file."""
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
                    'ece_kde': r'\* ece_kde: ([\d.]+)%',  # Added ECE-KDE metric
                    'mce': r'\* mce: ([\d.]+)%',
                    'ace': r'\* ace: ([\d.]+)%',
                    'macro_f1': r'\* macro_f1: ([\d.]+)%'
                }
                
                # Extract each metric
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

    def run_experiments(self):
        """Run all experiments with specified configurations."""
        all_results = []
        
        # Iterate through all combinations
        for dataset in self.datasets:
            print(f"\n=== Dataset: {dataset} ===")
            dataset_results = []
            
            # Get dataset-specific loss configurations
            dataset_loss_configs = self.get_loss_configs_for_dataset(dataset)
            
            for loss_config in dataset_loss_configs:
                print(f"\nTesting: {loss_config['name']}")
                
                seed_results = []
                for seed in self.seeds:
                    result = self.run_experiment(dataset, loss_config, seed)
                    
                    if result:
                        # Record individual seed results
                        result_row = {
                            'dataset': dataset,
                            'lr': self.fixed_params['lr'],
                            'epochs': self.fixed_params['epochs'],
                            'shots': self.fixed_params['shots'],
                            'losses': loss_config['name'],
                            'seed': seed,
                            'ece': result['ece'],
                            'ece_kde': result.get('ece_kde', float('nan')),  # Added ECE-KDE metric
                            'accuracy': result['accuracy'],
                            'mce': result['mce'],
                            'ace': result['ace'],
                            'macro_f1': result['macro_f1'],
                            'confidence': result.get('confidence', float('nan'))
                        }
                        seed_results.append(result_row)
                        dataset_results.append(result_row)
                        all_results.append(result_row)
                
                # Calculate aggregate statistics across seeds
                if seed_results:
                    seed_df = pd.DataFrame(seed_results)
                    
                    # Calculate mean and std for all metrics
                    mean_acc = seed_df['accuracy'].mean()
                    std_acc = seed_df['accuracy'].std()
                    mean_ece = seed_df['ece'].mean()
                    std_ece = seed_df['ece'].std()
                    mean_ece_kde = seed_df['ece_kde'].mean()  # Added ECE-KDE metric
                    std_ece_kde = seed_df['ece_kde'].std()    # Added ECE-KDE metric
                    mean_mce = seed_df['mce'].mean()
                    std_mce = seed_df['mce'].std()
                    mean_ace = seed_df['ace'].mean()
                    std_ace = seed_df['ace'].std()
                    mean_f1 = seed_df['macro_f1'].mean()
                    std_f1 = seed_df['macro_f1'].std()
                    
                    # Add summary row
                    summary_row = {
                        'dataset': dataset,
                        'lr': self.fixed_params['lr'],
                        'epochs': self.fixed_params['epochs'],
                        'shots': self.fixed_params['shots'],
                        'losses': loss_config['name'],
                        'seed': 'mean_std',
                        'accuracy': f"{mean_acc:.2f}±{std_acc:.2f}",
                        'ece': f"{mean_ece:.2f}±{std_ece:.2f}",
                        'ece_kde': f"{mean_ece_kde:.2f}±{std_ece_kde:.2f}",  # Added ECE-KDE metric
                        'mce': f"{mean_mce:.2f}±{std_mce:.2f}",
                        'ace': f"{mean_ace:.2f}±{std_ace:.2f}",
                        'macro_f1': f"{mean_f1:.2f}±{std_f1:.2f}",
                        'mean_accuracy': mean_acc,
                        'std_accuracy': std_acc,
                        'mean_ece': mean_ece,
                        'std_ece': std_ece,
                        'mean_ece_kde': mean_ece_kde,  # Added ECE-KDE metric
                        'std_ece_kde': std_ece_kde     # Added ECE-KDE metric
                    }
                    dataset_results.append(summary_row)
                    all_results.append(summary_row)
                    
                    # Print summary statistics
                    print(f"\nSummary for {dataset} - {loss_config['name']}:")
                    print(f"Accuracy: {mean_acc:.2f}±{std_acc:.2f}%")
                    print(f"ECE: {mean_ece:.2f}±{std_ece:.2f}%")
                    print(f"ECE-KDE: {mean_ece_kde:.2f}±{std_ece_kde:.2f}%")  # Added ECE-KDE metric
                    print(f"MCE: {mean_mce:.2f}±{std_mce:.2f}%")
                    print(f"ACE: {mean_ace:.2f}±{std_ace:.2f}%")
                    print(f"Macro F1: {mean_f1:.2f}±{std_f1:.2f}%")
                
                # Immediately save ALL results gathered so far for this dataset 
                # This ensures the CSV always contains all results up to the current point
                results_df = pd.DataFrame(dataset_results)
                results_path = os.path.join(
                    self.results_dir, 
                    f'{dataset}_{self.model_type}_prompt_results.csv'
                )
                results_df.to_csv(results_path, index=False)
                print(f"Updated results saved to: {results_path}")
                
                # Also save the combined results file after each loss function
                combined_results_df = pd.DataFrame(all_results)
                combined_path = os.path.join(
                    self.results_dir, 
                    f'all_prompt_results_{self.model_type}.csv'
                )
                combined_results_df.to_csv(combined_path, index=False)
                print(f"Combined results saved to: {combined_path}")
        
        # Store all results for final summary
        self.all_results = all_results
        
        # Create summary dataframe 
        self.create_summary(all_results)
        
        return pd.DataFrame(all_results)
    
    def create_summary(self, results):
        """Create a summary table."""
        try:
            summary_data = []
            
            # Filter only the summary rows
            results_df = pd.DataFrame(results)
            summary_rows = results_df[results_df['seed'] == 'mean_std']
            
            if summary_rows.empty:
                print("No summary data available yet")
                return
                
            # Create a table for each dataset
            for dataset in self.datasets:
                dataset_summary = summary_rows[summary_rows['dataset'] == dataset]
                
                if dataset_summary.empty:
                    continue
                    
                for _, row in dataset_summary.iterrows():
                    summary_data.append({
                        'Dataset': dataset,
                        'Loss Function': row['losses'],
                        'Accuracy (%)': row['accuracy'],
                        'ECE (%)': row['ece'],
                        'ECE-KDE (%)': row['ece_kde'],  # Added ECE-KDE metric
                        'MCE (%)': row['mce'],
                        'ACE (%)': row['ace'],
                        'F1 Score (%)': row['macro_f1']
                    })
            
            # Create and save the summary table
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(
                self.results_dir,
                f'prompt_summary_{self.model_type}.csv'
            )
            summary_df.to_csv(summary_path, index=False)
            
            # Also create a LaTeX formatted table
            latex_path = os.path.join(
                self.results_dir,
                f'prompt_summary_{self.model_type}.tex'
            )
            
            with open(latex_path, 'w') as f:
                f.write("\\begin{table}[ht]\n")
                f.write("\\centering\n")
                f.write("\\caption{Performance metrics for prompt learning with " + self.model_type + "}\n")
                f.write("\\begin{tabular}{lcccc}\n")  # Added an extra column for ECE-KDE
                f.write("\\toprule\n")
                f.write("Dataset & Loss Function & Accuracy (\\%) & ECE (\\%) & ECE-KDE (\\%) \\\\\n")  # Added ECE-KDE
                f.write("\\midrule\n")
                
                # Group by dataset for better organization
                for dataset in self.datasets:
                    dataset_rows = summary_df[summary_df['Dataset'] == dataset]
                    if dataset_rows.empty:
                        continue
                        
                    f.write("\\multirow{" + str(len(dataset_rows)) + "}{*}{" + dataset + "}")
                    
                    first_row = True
                    for _, row in dataset_rows.iterrows():
                        if first_row:
                            f.write(f" & {row['Loss Function']} & {row['Accuracy (%)']} & {row['ECE (%)']} & {row['ECE-KDE (%)']} \\\\\n")
                            first_row = False
                        else:
                            f.write(f" & {row['Loss Function']} & {row['Accuracy (%)']} & {row['ECE (%)']} & {row['ECE-KDE (%)']} \\\\\n")
                    
                    f.write("\\cmidrule{1-5}\n")  # Updated to include the new column
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\label{tab:prompt_performance_" + self.model_type + "}\n")
                f.write("\\end{table}\n")
                
            print(f"Summary saved to: {summary_path}")
            print(f"LaTeX table saved to: {latex_path}")
            
        except Exception as e:
            print(f"Error creating summary: {e}")

    def print_all_summaries(self):
        """Print summaries for all loss functions."""
        if not hasattr(self, 'all_results') or not self.all_results:
            print("No results available yet")
            return
            
        results_df = pd.DataFrame(self.all_results)
        summary_rows = results_df[results_df['seed'] == 'mean_std']
        
        print("\n===== SUMMARY OF RESULTS =====")
        for dataset in self.datasets:
            print(f"\nDataset: {dataset}")
            dataset_rows = summary_rows[summary_rows['dataset'] == dataset]
            
            # Create a table format
            print(f"{'Loss Function':<20} {'Accuracy (%)':<20} {'ECE (%)':<15} {'ECE-KDE (%)':<15}")  # Added ECE-KDE
            print("-" * 70)  # Increased width
            
            for _, row in dataset_rows.iterrows():
                print(f"{row['losses']:<20} {row['accuracy']:<20} {row['ece']:<15} {row['ece_kde']:<15}")  # Added ECE-KDE

    def cleanup(self):
        """Clean up temporary configuration files."""
        try:
            for f in os.listdir(self.results_dir):
                if f.startswith('config_') and f.endswith('.yaml'):
                    os.remove(os.path.join(self.results_dir, f))
        except Exception as e:
            print(f"Error in cleanup: {e}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run prompt learning experiments for MedCLIP and BioMedCLIP')
    parser.add_argument('--model', type=str, choices=['medclip', 'biomedclip'], required=True,
                      help='Model type to run experiments for')
    parser.add_argument('--dataset', type=str, choices=['covid', 'rsna18'],
                      help='Specific dataset to run experiments on')
    parser.add_argument('--seeds', type=str, default="1,2,3",
                      help='Comma-separated list of seeds to use (default: 1,2,3)')
    args = parser.parse_args()
    
    try:
        # Parse seeds from arguments
        seeds = [int(s) for s in args.seeds.split(',')]
        
        experiments = BaseLinePromptLearningExperiments(args.model, args.dataset)
        # Override seeds with those provided as arguments
        experiments.seeds = seeds
        
        print(f"Starting prompt learning experiments for {args.model}" + 
              (f" on {args.dataset}" if args.dataset else ""))
        print(f"Using seeds: {experiments.seeds}")
        
        # Get and display dataset-specific loss configs
        if args.dataset:
            loss_configs = experiments.get_loss_configs_for_dataset(args.dataset)
        else:
            # Show configurations for both datasets
            print("Loss configurations by dataset:")
            for dataset in experiments.datasets:
                configs = experiments.get_loss_configs_for_dataset(dataset)
                print(f"\n{dataset} - {args.model} loss configurations:")
                for config in configs:
                    print(f"  - {config['name']}")
        
        experiments.run_experiments()
        
        # Print summary for ALL loss functions
        experiments.print_all_summaries()
        
        experiments.cleanup()
        print("\nExperiments completed!")
        
    except Exception as e:
        print(f"Error in execution: {e}")

if __name__ == "__main__":
    main()