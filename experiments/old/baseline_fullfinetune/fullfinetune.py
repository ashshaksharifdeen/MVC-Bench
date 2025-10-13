import yaml
import os
import subprocess
import pandas as pd
import re
import argparse
import numpy as np
from datetime import datetime

class FullFinetuneExperiments:
    def __init__(self, model_type, dataset=None):
        self.model_type = model_type
        self.base_config_path = f"../../configs/trainers/CoOp/vit_b32_{model_type}_c16_ep50_batch16.yaml"
        self.results_dir = f"../../output/baseline_fullfinetune_{model_type}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Modified: Using 3 seeds as requested for MICCAI
        self.seeds = [44, 9999, 1234]
        
        # Dataset-specific prompt templates
        self.prompt_templates = {
            'kather': "An H&E image patch of {}.",
            'digestpath': "An H&E image patch of {} tissue.",
            'pannuke': "An H&E image patch of {} skin tissue."
        }
        
        # Dataset-specific configurations
        self.dataset_configs = self.get_dataset_configs()
        
        # Fixed parameters
        self.fixed_params = {
            'lr': 0.0000002,
            'epochs': 50,
            'shots': 8
        }
        
        # Datasets - use specific dataset if provided
        if dataset:
            self.datasets = [dataset]
        else:
            self.datasets = ["kather", "pannuke", "digestpath"]
            
        # Store loss configs per dataset
        self.loss_configs_per_dataset = {
            dataset: self.get_loss_configs(dataset) for dataset in self.datasets
        }

    def get_dataset_configs(self):
        """Get dataset-specific configurations."""
        if self.model_type == 'plip':
            return {
                'kather': {
                    'CE': {},
                    'CE_DCA': {'DCA': {'WEIGHT': 9.0}},
                    'CE_MMCE': {'MMCE': {'WEIGHT': 1.0}},
                    'CE_MDCA': {'MDCA': {'WEIGHT': 1.0}},
                    'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.07}},
                    'CE_SLMDCA_CL': {'SLMDCA': {'ALPHA': 0.07}, 'COSINE': {'WEIGHT': 0.01}},
                    'FL': {'FL': {'GAMMA': 3.0}},
                    'FL_MDCA': {'FL': {'GAMMA': 3.0}, 'MDCA': {'WEIGHT': 1.0}},
                    'FL_SLMDCA': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.03}},
                    'FL_SLMDCA_CL': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.03}, 'COSINE': {'WEIGHT': 0.01}},
                    'LS': {'LS': {'ALPHA': 0.05}},
                    'LS_MDCA': {'LS': {'ALPHA': 0.05}, 'MDCA': {'WEIGHT': 1.0}},
                    'LS_SLMDCA': {'LS': {'ALPHA': 0.05}, 'SLMDCA': {'ALPHA': 0.05}},
                    'LS_SLMDCA_CL': {'LS': {'ALPHA': 0.05}, 'SLMDCA': {'ALPHA': 0.05}, 'COSINE': {'WEIGHT': 0.01}}
                },
                'pannuke': {
                    'CE': {},
                    'CE_DCA': {'DCA': {'WEIGHT': 9.0}},
                    'CE_MMCE': {'MMCE': {'WEIGHT': 1.0}},
                    'CE_MDCA': {'MDCA': {'WEIGHT': 1.0}},
                    'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.2}},
                    'CE_SLMDCA_CL': {'SLMDCA': {'ALPHA': 0.2}, 'COSINE': {'WEIGHT': 0.1}},
                    'FL': {'FL': {'GAMMA': 3.0}},
                    'FL_MDCA': {'FL': {'GAMMA': 3.0}, 'MDCA': {'WEIGHT': 1.0}},
                    'FL_SLMDCA': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.2}},
                    'FL_SLMDCA_CL': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.2}, 'COSINE': {'WEIGHT': 0.1}},
                    'LS': {'LS': {'ALPHA': 0.05}},
                    'LS_MDCA': {'LS': {'ALPHA': 0.05}, 'MDCA': {'WEIGHT': 1.0}},
                    'LS_SLMDCA': {'LS': {'ALPHA': 0.2}, 'SLMDCA': {'ALPHA': 0.2}},
                    'LS_SLMDCA_CL': {'LS': {'ALPHA': 0.2}, 'SLMDCA': {'ALPHA': 0.2}, 'COSINE': {'WEIGHT': 0.1}}
                },
                'digestpath': {
                    'CE': {},
                    'CE_DCA': {'DCA': {'WEIGHT': 9.0}},
                    'CE_MMCE': {'MMCE': {'WEIGHT': 1.0}},
                    'CE_MDCA': {'MDCA': {'WEIGHT': 1.0}},
                    'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.03}},
                    'CE_SLMDCA_CL': {'SLMDCA': {'ALPHA': 0.03}, 'COSINE': {'WEIGHT': 0.001}},
                    'FL': {'FL': {'GAMMA': 3.0}},
                    'FL_MDCA': {'FL': {'GAMMA': 3.0}, 'MDCA': {'WEIGHT': 1.0}},
                    'FL_SLMDCA': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.03}},
                    'FL_SLMDCA_CL': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.03}, 'COSINE': {'WEIGHT': 0.001}},
                    'LS': {'LS': {'ALPHA': 0.05}},
                    'LS_MDCA': {'LS': {'ALPHA': 0.05}, 'MDCA': {'WEIGHT': 1.0}},
                    'LS_SLMDCA': {'LS': {'ALPHA': 0.03}, 'SLMDCA': {'ALPHA': 0.03}},
                    'LS_SLMDCA_CL': {'LS': {'ALPHA': 0.03}, 'SLMDCA': {'ALPHA': 0.03}, 'COSINE': {'WEIGHT': 0.001}}
                }
            }
        else:  # quiltnet
            return {
                'kather': {
                    'CE': {},
                    'CE_DCA': {'DCA': {'WEIGHT': 9.0}},
                    'CE_MMCE': {'MMCE': {'WEIGHT': 1.0}},
                    'CE_MDCA': {'MDCA': {'WEIGHT': 1.0}},
                    'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.07}},
                    'CE_SLMDCA_CL': {'SLMDCA': {'ALPHA': 0.07}, 'COSINE': {'WEIGHT': 1.0}},
                    'FL': {'FL': {'GAMMA': 3.0}},
                    'FL_MDCA': {'FL': {'GAMMA': 3.0}, 'MDCA': {'WEIGHT': 1.0}},
                    'FL_SLMDCA': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.03}},
                    'FL_SLMDCA_CL': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.03}, 'COSINE': {'WEIGHT': 1.0}},
                    'LS': {'LS': {'ALPHA': 0.05}},
                    'LS_MDCA': {'LS': {'ALPHA': 0.05}, 'MDCA': {'WEIGHT': 1.0}},
                    'LS_SLMDCA': {'LS': {'ALPHA': 0.01}, 'SLMDCA': {'ALPHA': 0.01}},
                    'LS_SLMDCA_CL': {'LS': {'ALPHA': 0.01}, 'SLMDCA': {'ALPHA': 0.01}, 'COSINE': {'WEIGHT': 1.0}}
                },
                'pannuke': {
                    'CE': {},
                    'CE_DCA': {'DCA': {'WEIGHT': 100.0}},
                    'CE_MMCE': {'MMCE': {'WEIGHT': 1.0}},
                    'CE_MDCA': {'MDCA': {'WEIGHT': 0.00001}},
                    'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.1}},
                    'CE_SLMDCA_CL': {'SLMDCA': {'ALPHA': 0.1}, 'COSINE': {'WEIGHT': 0.001}},
                    'FL': {'FL': {'GAMMA': 3.0}},
                    'FL_MDCA': {'FL': {'GAMMA': 3.0}, 'MDCA': {'WEIGHT': 0.00001}},
                    'FL_SLMDCA': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.1}},
                    'FL_SLMDCA_CL': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.1}, 'COSINE': {'WEIGHT': 0.001}},
                    'LS': {'LS': {'ALPHA': 0.05}},
                    'LS_MDCA': {'LS': {'ALPHA': 0.05}, 'MDCA': {'WEIGHT': 0.00001}},
                    'LS_SLMDCA': {'LS': {'ALPHA': 0.1}, 'SLMDCA': {'ALPHA': 0.1}},
                    'LS_SLMDCA_CL': {'LS': {'ALPHA': 0.1}, 'SLMDCA': {'ALPHA': 0.1}, 'COSINE': {'WEIGHT': 0.001}}
                },
                'digestpath': {
                    'CE': {},
                    'CE_DCA': {'DCA': {'WEIGHT': 9.0}},
                    'CE_MMCE': {'MMCE': {'WEIGHT': 1.0}},
                    'CE_MDCA': {'MDCA': {'WEIGHT': 1.0}},
                    'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.05}},
                    'CE_SLMDCA_CL': {'SLMDCA': {'ALPHA': 0.05}, 'COSINE': {'WEIGHT': 0.001}},
                    'FL': {'FL': {'GAMMA': 3.0}},
                    'FL_MDCA': {'FL': {'GAMMA': 3.0}, 'MDCA': {'WEIGHT': 1.0}},
                    'FL_SLMDCA': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.05}},
                    'FL_SLMDCA_CL': {'FL': {'GAMMA': 3.0}, 'SLMDCA': {'ALPHA': 0.05}, 'COSINE': {'WEIGHT': 0.001}},
                    'LS': {'LS': {'ALPHA': 0.05}},
                    'LS_MDCA': {'LS': {'ALPHA': 0.05}, 'MDCA': {'WEIGHT': 1.0}},
                    'LS_SLMDCA': {'LS': {'ALPHA': 0.05}, 'SLMDCA': {'ALPHA': 0.05}},
                    'LS_SLMDCA_CL': {'LS': {'ALPHA': 0.05}, 'SLMDCA': {'ALPHA': 0.05}, 'COSINE': {'WEIGHT': 0.001}}
                }
            }

    def get_loss_configs(self, dataset):
        """Generate loss configurations based on dataset."""
        dataset_params = self.dataset_configs[dataset]
        
        loss_configs = []
        for loss_name, params in dataset_params.items():
            # Determine enabled losses based on loss name
            enabled_losses = []
            for part in loss_name.split('_'):
                if part in ['CE', 'FL', 'LS', 'DCA', 'MMCE', 'MDCA', 'SLMDCA', 'CL', 'COSINE']:
                    if part == 'CL':  # CL is actually COSINE in our config
                        enabled_losses.append('COSINE')
                    else:
                        enabled_losses.append(part)
            
            loss_configs.append({
                'name': loss_name,
                'enabled_losses': enabled_losses,
                'params': params
            })
            
        return loss_configs

    def modify_config(self, dataset, loss_config):
        """Modify configuration file for the experiment."""
        try:
            # Read base config
            with open(self.base_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set fixed parameters
            config['OPTIM']['LR'] = self.fixed_params['lr']
            config['OPTIM']['MAX_EPOCH'] = self.fixed_params['epochs']
            
            # Add the FINETUNE_MODE for full model fine-tuning with hard prompts
            config['TRAINER']['COOP']['FINETUNE_MODE'] = 'full_hard'
            
            # Set the prompt template for this dataset
            config['TRAINER']['COOP']['PROMPT_TEMPLATE'] = self.prompt_templates[dataset]
            
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
            print(f"Prompt template: {self.prompt_templates[dataset]}")
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
            exp_name = f"{dataset}/shots_{self.fixed_params['shots']}/CoOp/full_hard/{loss_config['name']}"
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
                "--trainer", "CoOp",
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
        results = []
        
        # Iterate through all combinations
        for dataset in self.datasets:
            print(f"\n=== Dataset: {dataset} ===")
            dataset_results = []
            
            for loss_config in self.loss_configs_per_dataset[dataset]:
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
                            'accuracy': result['accuracy'],
                            'mce': result['mce'],
                            'ace': result['ace'],
                            'macro_f1': result['macro_f1']
                        }
                        seed_results.append(result_row)
                        dataset_results.append(result_row)
                
                # Calculate aggregate statistics across seeds
                if seed_results:
                    seed_df = pd.DataFrame(seed_results)
                    
                    # Calculate mean and std for all metrics
                    mean_acc = seed_df['accuracy'].mean()
                    std_acc = seed_df['accuracy'].std()
                    mean_ece = seed_df['ece'].mean()
                    std_ece = seed_df['ece'].std()
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
                        'mce': f"{mean_mce:.2f}±{std_mce:.2f}",
                        'ace': f"{mean_ace:.2f}±{std_ace:.2f}",
                        'macro_f1': f"{mean_f1:.2f}±{std_f1:.2f}",
                        'mean_accuracy': mean_acc,
                        'std_accuracy': std_acc,
                        'mean_ece': mean_ece,
                        'std_ece': std_ece
                    }
                    dataset_results.append(summary_row)
                    
                    # Print summary statistics
                    print(f"\nSummary for {dataset} - {loss_config['name']}:")
                    print(f"Accuracy: {mean_acc:.2f}±{std_acc:.2f}%")
                    print(f"ECE: {mean_ece:.2f}±{std_ece:.2f}%")
                    print(f"MCE: {mean_mce:.2f}±{std_mce:.2f}%")
                    print(f"ACE: {mean_ace:.2f}±{std_ace:.2f}%")
                    print(f"Macro F1: {mean_f1:.2f}±{std_f1:.2f}%")
                
                # Immediately save results for this dataset
                results_df = pd.DataFrame(dataset_results)
                results_path = os.path.join(
                    self.results_dir, 
                    f'{dataset}_{self.model_type}_full_hard_results.csv'
                )
                results_df.to_csv(results_path, index=False)
                print(f"Updated results saved to: {results_path}")
            
            # Append to overall results
            results.extend(dataset_results)
        
        # Create summary dataframe specifically formatted for MICCAI
        self.create_miccai_summary(results)
        
        return pd.DataFrame(results)
    
    def create_miccai_summary(self, results):
        """Create a summary table formatted specifically for MICCAI publication."""
        try:
            summary_data = []
            
            # Filter only the summary rows
            results_df = pd.DataFrame(results)
            summary_rows = results_df[results_df['seed'] == 'mean_std']
            
            if summary_rows.empty:
                print("No summary data available yet")
                return
                
            # Create a pivot table for each dataset
            for dataset in self.datasets:
                dataset_summary = summary_rows[summary_rows['dataset'] == dataset]
                
                if dataset_summary.empty:
                    continue
                    
                for _, row in dataset_summary.iterrows():
                    summary_data.append({
                        'Dataset': dataset,
                        'Loss Function': row['losses'],
                        'Accuracy (%)': row['accuracy'],
                        'ECE (%)': row['ece']
                    })
            
            # Create and save the summary table
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(
                self.results_dir,
                f'miccai_summary_{self.model_type}.csv'
            )
            summary_df.to_csv(summary_path, index=False)
            
            # Also create a LaTeX formatted table for direct inclusion in the paper
            latex_path = os.path.join(
                self.results_dir,
                f'miccai_summary_{self.model_type}.tex'
            )
            
            with open(latex_path, 'w') as f:
                f.write("\\begin{table}[ht]\n")
                f.write("\\centering\n")
                f.write("\\caption{Performance metrics for " + self.model_type + " across different datasets and loss functions}\n")
                f.write("\\begin{tabular}{lccc}\n")
                f.write("\\toprule\n")
                f.write("Dataset & Loss Function & Accuracy (\\%) & ECE (\\%) \\\\\n")
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
                            f.write(f" & {row['Loss Function']} & {row['Accuracy (%)']} & {row['ECE (%)']} \\\\\n")
                            first_row = False
                        else:
                            f.write(f" & {row['Loss Function']} & {row['Accuracy (%)']} & {row['ECE (%)']} \\\\\n")
                    
                    f.write("\\cmidrule{1-4}\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\label{tab:performance_" + self.model_type + "}\n")
                f.write("\\end{table}\n")
                
            print(f"MICCAI summary saved to: {summary_path}")
            print(f"LaTeX table saved to: {latex_path}")
            
        except Exception as e:
            print(f"Error creating MICCAI summary: {e}")

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
    parser = argparse.ArgumentParser(description='Run full model fine-tuning experiments')
    parser.add_argument('--model', type=str, choices=['plip', 'quiltnet'], required=True,
                      help='Model type to run experiments for')
    parser.add_argument('--dataset', type=str, choices=['kather', 'pannuke', 'digestpath'],
                      help='Specific dataset to run experiments on')
    args = parser.parse_args()
    
    try:
        experiments = FullFinetuneExperiments(args.model, args.dataset)
        print(f"Starting full-finetune experiments for {args.model}" + 
              (f" on {args.dataset}" if args.dataset else ""))
        
        # Print information about the seeds being used
        print(f"Running experiments with seeds: {experiments.seeds}")
        
        results = experiments.run_experiments()
        print("\nExperiments completed!")
        
        experiments.cleanup()
    except Exception as e:
        print(f"Error in execution: {e}")

if __name__ == "__main__":
    main()