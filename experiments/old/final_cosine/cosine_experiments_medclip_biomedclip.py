import yaml
import os
import subprocess
import pandas as pd
import re
import argparse
from datetime import datetime

class CosineExperiments:
    def __init__(self, model_type, dataset=None):
        self.model_type = model_type  # 'medclip' or 'biomedclip'
        
        # Set appropriate configuration path and trainer name based on model
        if model_type == 'medclip':
            self.base_config_path = f"../../configs/trainers/CoOp_MedCLIP/vit_b32_medclip_c16_ep50_batch16.yaml"
            self.trainer_name = "CoOp_MedCLIP"
        elif model_type == 'biomedclip':
            self.base_config_path = f"../../configs/trainers/CoOp_BioMedCLIP/vit_b32_biomedclip_c16_ep50_batch16.yaml"
            self.trainer_name = "CoOp_BioMedCLIP"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.results_dir = f"../../output/final_cosine_experiments_{model_type}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Single seed
        self.seeds = [1]
        
        # Updated Cosine weights
        self.cosine_weights = [
            0.0, 0.001, 0.01, 0.5, 0.1, 1.0, 3.0, 5.0, 10.0, 15.0
        ]
        
        # Dataset-specific configurations for MedCLIP and BioMedCLIP
        # Starting with reasonable values - these can be adjusted based on your findings
        self.dataset_configs = {
            'medclip': {
                'covid': {
                    'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.05}},
                    'FL_SLMDCA': {'SLMDCA': {'ALPHA': 0.05}, 'FL': {'GAMMA': 3.0}},
                    'LS_SLMDCA': {'SLMDCA': {'ALPHA': 0.05}, 'LS': {'ALPHA': 0.05}}
                },
                'rsna18': {
                    'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.25}},
                    'FL_SLMDCA': {'SLMDCA': {'ALPHA': 0.25}, 'FL': {'GAMMA': 3.0}},
                    'LS_SLMDCA': {'SLMDCA': {'ALPHA': 0.17}, 'LS': {'ALPHA': 0.17}}
                }
            },
            'biomedclip': {
                'covid': {
                    'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.05}},
                    'FL_SLMDCA': {'SLMDCA': {'ALPHA': 0.05}, 'FL': {'GAMMA': 3.0}},
                    'LS_SLMDCA': {'SLMDCA': {'ALPHA': 0.05}, 'LS': {'ALPHA': 0.05}}
                },
                'rsna18': {
                    'CE_SLMDCA': {'SLMDCA': {'ALPHA': 0.25}},
                    'FL_SLMDCA': {'SLMDCA': {'ALPHA': 0.25}, 'FL': {'GAMMA': 3.0}},
                    'LS_SLMDCA': {'SLMDCA': {'ALPHA': 0.1}, 'LS': {'ALPHA': 0.1}}
                }
            }
        }
        
        # Generate loss configurations based on dataset
        def get_loss_configs(dataset):
            dataset_params = self.dataset_configs[self.model_type][dataset]
            return [
                {'name': 'CE_SLMDCA_COSINE',
                 'losses': ['CE', 'SLMDCA', 'COSINE'],
                 'weights': [1.0, 1.0],
                 'extra_params': dataset_params['CE_SLMDCA']},
                
                {'name': 'FL_SLMDCA_COSINE',
                 'losses': ['FL', 'SLMDCA', 'COSINE'],
                 'weights': [1.0, 1.0],
                 'extra_params': dataset_params['FL_SLMDCA']},
                
                {'name': 'LS_SLMDCA_COSINE',
                 'losses': ['LS', 'SLMDCA', 'COSINE'],
                 'weights': [1.0, 1.0],
                 'extra_params': dataset_params['LS_SLMDCA']}
            ]
        
        # Fixed parameters
        self.fixed_params = {
            'lr': 0.002,
            'epochs': 50,
            'shots': 8
        }
        
        # Datasets - use specific dataset if provided
        if dataset:
            self.datasets = [dataset]
        else:
            self.datasets = ["covid", "rsna18"]
            
        # Store loss configs per dataset
        self.loss_configs_per_dataset = {
            dataset: get_loss_configs(dataset) for dataset in self.datasets
        }

    def modify_config(self, dataset, loss_config, cosine_weight):
        """Modify configuration file for the experiment."""
        try:
            # Read base config
            with open(self.base_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set fixed parameters
            config['OPTIM']['LR'] = self.fixed_params['lr']
            config['OPTIM']['MAX_EPOCH'] = self.fixed_params['epochs']
            
            # Ensure backbone is set correctly
            if 'MODEL' not in config:
                config['MODEL'] = {}
            if 'BACKBONE' not in config['MODEL']:
                config['MODEL']['BACKBONE'] = {}
            config['MODEL']['BACKBONE']['NAME'] = "ViT-B/32"
            
            # Set loss configuration
            loss_config_yaml = config['TRAINER']['COOP']['LOSS']
            loss_config_yaml['ENABLED_LOSSES'] = loss_config['losses']
            
            # Initialize loss configurations
            for idx, loss_name in enumerate(loss_config['losses'][:-1]):  # Exclude COSINE
                if loss_name not in loss_config_yaml:
                    loss_config_yaml[loss_name] = {}
                    
                loss_config_yaml[loss_name]['WEIGHT'] = loss_config['weights'][idx]
                
                # Add extra parameters if any
                if loss_name in loss_config['extra_params']:
                    loss_config_yaml[loss_name].update(loss_config['extra_params'][loss_name])
            
            # Configure COSINE loss separately
            if 'COSINE' not in loss_config_yaml:
                loss_config_yaml['COSINE'] = {}
            loss_config_yaml['COSINE']['WEIGHT'] = cosine_weight
            
            # Print configuration for verification
            print(f"\nConfiguration for {dataset}:")
            for loss in loss_config['losses']:
                weight = loss_config_yaml[loss]['WEIGHT']
                print(f"{loss}: weight={weight}")
                if loss in loss_config['extra_params']:
                    print(f"  extra params: {loss_config['extra_params'][loss]}")
            
            # Save config file
            config_name = f"config_{self.model_type}_{dataset}_{loss_config['name']}_cosine_{cosine_weight}.yaml"
            config_path = os.path.join(self.results_dir, config_name)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            return config_path
            
        except Exception as e:
            print(f"Error modifying config: {e}")
            raise

    def run_experiment(self, dataset, loss_config, cosine_weight, seed):
        """Run a single experiment with specified parameters."""
        try:
            # Get modified config
            config_path = self.modify_config(dataset, loss_config, cosine_weight)
            
            # Setup experiment directory
            exp_name = f"{dataset}/shots_{self.fixed_params['shots']}/{self.trainer_name}/{loss_config['name']}_cosine_{cosine_weight}"
            output_dir = os.path.join(self.results_dir, exp_name, f"seed{seed}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Check for completed experiment
            log_file = os.path.join(output_dir, "log.txt")
            if os.path.exists(log_file):
                print(f"Experiment exists: {dataset}, {loss_config['name']}, cosine={cosine_weight}, seed={seed}")
                return self.extract_metrics(log_file)
            
            # Prepare command
            cmd = [
                "python", "../../train.py",
                "--root", r"/home/abhishek/desktop/VLM_Cal/CLIP_Calibration/$DATA",
                "--seed", str(seed),
                "--trainer", self.trainer_name,  # Use appropriate trainer name
                "--dataset-config-file", f"../../configs/datasets/{dataset}.yaml",
                "--config-file", config_path,
                "--output-dir", output_dir,
                "DATASET.NUM_SHOTS", str(self.fixed_params['shots']),
                "DATASET.SUBSAMPLE_CLASSES", "all",
                "MODEL.NAME", self.model_type,
                "MODEL_ROOT", "../../models"
            ]
            
            # Run experiment
            print(f"\nRunning: {dataset}, {loss_config['name']}, cosine={cosine_weight}, seed={seed}")
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
                
                # Get the base loss and its parameters
                base_loss = loss_config['name'].split('_')[0]  # CE, FL, or LS
                slmdca_alpha = loss_config['extra_params']['SLMDCA']['ALPHA']
                
                # Get additional parameters based on loss type
                extra_param_str = ""
                if base_loss == 'FL':
                    gamma = loss_config['extra_params']['FL']['GAMMA']
                    extra_param_str = f"_GAMMA_{gamma}"
                elif base_loss == 'LS':
                    ls_alpha = loss_config['extra_params']['LS']['ALPHA']
                    extra_param_str = f"_ALPHA_{ls_alpha}"
                
                for cosine_weight in self.cosine_weights:
                    for seed in self.seeds:
                        result = self.run_experiment(dataset, loss_config, cosine_weight, seed)
                        
                        if result:
                            # Create new loss name format
                            loss_name = f"{base_loss}{extra_param_str}_SLMDCA_ALPHA_{slmdca_alpha}_COSINE_{cosine_weight}"
                            
                            # Record results
                            result_row = {
                                'dataset': dataset,
                                'lr': self.fixed_params['lr'],
                                'epochs': self.fixed_params['epochs'],
                                'shots': self.fixed_params['shots'],
                                'losses': loss_name,
                                'seed': seed,
                                'ece': result['ece'],
                                'accuracy': result['accuracy'],
                                'mce': result.get('mce', float('nan')),  # Add MCE
                                'ace': result.get('ace', float('nan')),  # Add ACE
                                'macro_f1': result.get('macro_f1', float('nan'))  # Add F1
                            }
                            dataset_results.append(result_row)
                            
                            # Immediately save results for this dataset
                            results_df = pd.DataFrame(dataset_results)
                            results_path = os.path.join(
                                self.results_dir, 
                                f'{dataset}_{self.model_type}_results.csv'
                            )
                            results_df.to_csv(results_path, index=False)
                            print(f"Updated results saved to: {results_path}")
            
            # Append to overall results
            results.extend(dataset_results)
            
            # Create visualizations for this dataset
            self.create_visualizations(dataset, dataset_results)
        
        return pd.DataFrame(results)

    def create_visualizations(self, dataset, results):
        """Create simple CSV files for visualization of results."""
        if not results:
            return
            
        results_df = pd.DataFrame(results)
        
        # Create directory for visualizations
        viz_dir = os.path.join(self.results_dir, "visualizations", dataset)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Process results for each loss type
        for loss_type in ['CE', 'FL', 'LS']:
            # Filter for this loss type
            loss_results = results_df[results_df['losses'].str.startswith(loss_type)]
            
            if loss_results.empty:
                continue
                
            # Group by cosine weight value
            grouped = loss_results.groupby('losses')
            
            # Save summary data
            summary_rows = []
            for loss_name, group in grouped:
                # Extract cosine weight from loss name
                cosine_weight = float(loss_name.split('COSINE_')[1])
                
                # Get metrics
                mean_acc = group['accuracy'].mean()
                mean_ece = group['ece'].mean()
                
                summary_rows.append({
                    'cosine_weight': cosine_weight,
                    'accuracy': mean_acc,
                    'ece': mean_ece
                })
            
            # Save to CSV
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows).sort_values('cosine_weight')
                summary_path = os.path.join(viz_dir, f"{loss_type}_cosine_summary.csv")
                summary_df.to_csv(summary_path, index=False)
                
                # Find best settings
                best_ece_row = summary_df.loc[summary_df['ece'].idxmin()]
                best_acc_row = summary_df.loc[summary_df['accuracy'].idxmax()]
                
                with open(os.path.join(viz_dir, f"{loss_type}_best_cosine.txt"), 'w') as f:
                    f.write(f"Best Cosine Weight for {dataset} with {loss_type} (by ECE): {best_ece_row['cosine_weight']}\n")
                    f.write(f"  - ECE: {best_ece_row['ece']}, Accuracy: {best_ece_row['accuracy']}\n\n")
                    f.write(f"Best Cosine Weight for {dataset} with {loss_type} (by Accuracy): {best_acc_row['cosine_weight']}\n")
                    f.write(f"  - Accuracy: {best_acc_row['accuracy']}, ECE: {best_acc_row['ece']}\n")

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
    parser = argparse.ArgumentParser(description='Run cosine weight experiments for MedCLIP and BioMedCLIP')
    parser.add_argument('--model', type=str, choices=['medclip', 'biomedclip'], required=True,
                      help='Model type to run experiments for')
    parser.add_argument('--dataset', type=str, choices=['covid', 'rsna18'],
                      help='Specific dataset to run experiments on')
    args = parser.parse_args()
    
    try:
        experiments = CosineExperiments(args.model, args.dataset)
        print(f"Starting experiments for {args.model}" + 
              (f" on {args.dataset}" if args.dataset else ""))
        
        results = experiments.run_experiments()
        print("\nExperiments completed!")
        
        # Create overall summary
        print("\nBest Cosine Weights:")
        for dataset in experiments.datasets:
            print(f"\nDataset: {dataset}")
            viz_dir = os.path.join(experiments.results_dir, "visualizations", dataset)
            
            for loss_type in ['CE', 'FL', 'LS']:
                summary_file = os.path.join(viz_dir, f"{loss_type}_best_cosine.txt")
                if os.path.exists(summary_file):
                    with open(summary_file, 'r') as f:
                        print(f"{loss_type}:")
                        for line in f:
                            print(f"  {line.strip()}")
        
        experiments.cleanup()
    except Exception as e:
        print(f"Error in execution: {e}")

if __name__ == "__main__":
    main()