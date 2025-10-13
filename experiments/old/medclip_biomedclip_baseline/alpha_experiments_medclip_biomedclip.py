import yaml
import os
import subprocess
import pandas as pd
import re
from datetime import datetime
import argparse

class AlphaExperiments:
    def __init__(self, model_type):
        self.model_type = model_type  # 'medclip' or 'biomedclip'
        
        # Set appropriate paths and trainer name based on model type
        if model_type == 'medclip':
            self.base_config_path = f"../../configs/trainers/CoOp_MedCLIP/vit_b32_medclip_c16_ep50_batch16.yaml"
            self.trainer_name = "CoOp_MedCLIP"
        elif model_type == 'biomedclip':
            self.base_config_path = f"../../configs/trainers/CoOp_BioMedCLIP/vit_b32_biomedclip_c16_ep50_batch16.yaml"
            self.trainer_name = "CoOp_BioMedCLIP"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.results_dir = f"../../output/alpha_experiments_{model_type}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Alpha values to test - explicitly as floats, including 0
        self.alpha_values = [0.0, 0.01, 0.05, 0.1, 0.15, 0.17, 0.20, 0.25]
        
        # Loss configurations
        self.loss_configs = [
            {'name': 'CE_SLMDCA', 'losses': ['CE', 'SLMDCA'], 'weights': [1.0, 1.0]},
            {'name': 'FL_SLMDCA', 'losses': ['FL', 'SLMDCA'], 'weights': [1.0, 1.0]},
            {'name': 'LS_SLMDCA', 'losses': ['LS', 'SLMDCA'], 'weights': [1.0, 1.0]}
        ]
        
        # Fixed parameters
        self.fixed_params = {
            'lr': 0.002,
            'epochs': 50,
            'shots': 8
        }
        
        # Datasets for medical imaging
        self.datasets = ["covid", "rsna18"]
        
        # Seeds (using 1 by default as in the original)
        self.seed = 1
        
        # Keep track of all results
        self.all_results = []

    def modify_config(self, losses, weights, alpha):
        try:
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
            loss_config = config['TRAINER']['COOP']['LOSS']
            loss_config['ENABLED_LOSSES'] = losses
            
            # Initialize all possible loss configurations
            for loss_name, loss_params in {
                'CE': {'WEIGHT': 1.0},
                'FL': {'WEIGHT': 1.0, 'GAMMA': 3.0},
                'LS': {'WEIGHT': 1.0, 'ALPHA': 0.05},  # Fixed default LS alpha to 0.05
                'SLMDCA': {'WEIGHT': 1.0, 'ALPHA': 0.05},  # Fixed default SLMDCA alpha to 0.05
                'DCA': {'WEIGHT': 9.0},
                'MDCA': {'WEIGHT': 1.0},
                'MMCE': {'WEIGHT': 2.0},
            }.items():
                loss_config[loss_name] = loss_params.copy()
            
            # Special handling for LS + SLMDCA - both alphas should be equal
            if 'LS' in losses and 'SLMDCA' in losses:
                # Set both alphas to the same value
                loss_config['LS']['ALPHA'] = float(alpha)
                loss_config['SLMDCA']['ALPHA'] = float(alpha)
                print(f"Setting both LS and SLMDCA alpha to {alpha}")
            else:
                # Update weights and parameters for enabled losses
                for loss, weight in zip(losses, weights):
                    loss_config[loss]['WEIGHT'] = weight
                    if loss in ['LS', 'SLMDCA']:
                        loss_config[loss]['ALPHA'] = float(alpha)
                        print(f"Setting {loss} alpha to {alpha}")
            
            # Create unique config filename
            config_name = f"config_{self.model_type}_{'_'.join(losses)}_alpha_{alpha}.yaml"
            config_path = os.path.join(self.results_dir, config_name)
            
            print(f"\nFinal loss config:")
            print(yaml.dump(loss_config))
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                
            return config_path
            
        except Exception as e:
            print(f"Error modifying config: {e}")
            raise

    def run_experiment(self, dataset, loss_config, alpha):
        try:
            # Modify config and get new config path
            config_path = self.modify_config(loss_config['losses'], loss_config['weights'], alpha)
            
            # Build experiment directory structure
            exp_name = f"{dataset}/shots_{self.fixed_params['shots']}/{self.trainer_name}/{loss_config['name']}_alpha_{alpha}"
            output_dir = os.path.join(self.results_dir, exp_name, f"seed{self.seed}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if experiment is already completed
            log_file = os.path.join(output_dir, "log.txt")
            if os.path.exists(log_file):
                print(f"Experiment already completed: {dataset}, {loss_config['name']}, alpha={alpha}")
                return self.extract_metrics(log_file)
            
            # Prepare command
            cmd = [
                "python", "../../train.py",
                "--root", r"/home/abhishek/desktop/VLM_Cal/CLIP_Calibration/$DATA",
                "--seed", str(self.seed),
                "--trainer", self.trainer_name,  # Using appropriate trainer
                "--dataset-config-file", f"../../configs/datasets/{dataset}.yaml",
                "--config-file", config_path,
                "--output-dir", output_dir,
                "DATASET.NUM_SHOTS", str(self.fixed_params['shots']),
                "DATASET.SUBSAMPLE_CLASSES", "all",
                "MODEL.NAME", self.model_type,
                "MODEL_ROOT", "../../models"
            ]
            
            # Run the experiment
            print(f"\nRunning experiment: {dataset}, {loss_config['name']}, alpha={alpha}")
            subprocess.run(cmd, check=True)
            return self.extract_metrics(log_file)
            
        except Exception as e:
            print(f"Error running experiment: {e}")
            return None

    def extract_metrics(self, log_path):
        """Extract metrics from log file"""
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                metrics = {}
                
                patterns = {
                    'accuracy': r'\* accuracy: ([\d.]+)%',
                    'error_rate': r'\* error: ([\d.]+)%',
                    'confidence': r'\* confidence: ([\d.]+)%',
                    'ece': r'\* ece: ([\d.]+)%',
                    'mce': r'\* mce: ([\d.]+)%',
                    'ace': r'\* ace: ([\d.]+)%',
                    'macro_f1': r'\* macro_f1: ([\d.]+)%'
                }
                
                for metric_name, pattern in patterns.items():
                    match = re.search(pattern, content)
                    if not match:
                        print(f"Warning: Could not find {metric_name} in log file")
                        metrics[metric_name] = float('nan')
                    else:
                        metrics[metric_name] = float(match.group(1))
                
                return metrics
        except Exception as e:
            print(f"Error reading log file {log_path}: {e}")
            return None

    def run_experiments(self):
        results = []
        
        # For each dataset
        for dataset in self.datasets:
            print(f"\n=== Running experiments for {dataset} ===")
            
            # For each loss configuration
            for loss_config in self.loss_configs:
                print(f"\nTesting loss configuration: {loss_config['name']}")
                
                # For each alpha value
                for alpha in self.alpha_values:
                    result = self.run_experiment(dataset, loss_config, alpha)
                    
                    if result:
                        # Save results
                        result_row = {
                            'dataset': dataset,
                            'lr': self.fixed_params['lr'],
                            'epochs': self.fixed_params['epochs'],
                            'shots': self.fixed_params['shots'],
                            'losses': loss_config['name'],
                            'alpha': alpha,
                            'ece': result['ece'],
                            'accuracy': result['accuracy'],
                            'mce': result['mce'],
                            'ace': result['ace'],
                            'macro_f1': result.get('macro_f1', float('nan'))
                        }
                        results.append(result_row)
                        self.all_results.append(result_row)
                        
                        # Save current results to CSV
                        df = pd.DataFrame(results)
                        df.to_csv(os.path.join(self.results_dir, f'{self.model_type}_results.csv'), index=False)
                        print(f"Updated results saved to {self.model_type}_results.csv")
                        
                    # After each alpha value, create visualizations
                    self.create_visualizations()
        
        return pd.DataFrame(results)
    
    def create_visualizations(self):
        """Create visualization of ECE vs alpha and Accuracy vs alpha"""
        if not self.all_results:
            return  # No results to visualize yet
            
        results_df = pd.DataFrame(self.all_results)
        
        # Create directory for visualizations
        viz_dir = os.path.join(self.results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate CSV data for plotting
        for dataset in self.datasets:
            dataset_results = results_df[results_df['dataset'] == dataset]
            
            if dataset_results.empty:
                continue
                
            for loss_name in [config['name'] for config in self.loss_configs]:
                loss_results = dataset_results[dataset_results['losses'] == loss_name]
                
                if loss_results.empty or len(loss_results) < 2:  # Need at least 2 points for visualization
                    continue
                    
                # Sort by alpha
                loss_results = loss_results.sort_values('alpha')
                
                # Save data for plotting
                plot_data_path = os.path.join(viz_dir, f"{dataset}_{loss_name}_alpha_plot_data.csv")
                loss_results.to_csv(plot_data_path, index=False)
                print(f"Visualization data saved to {plot_data_path}")
                
                # Create simple text file with best alpha
                best_ece_row = loss_results.loc[loss_results['ece'].idxmin()]
                best_acc_row = loss_results.loc[loss_results['accuracy'].idxmax()]
                
                summary_path = os.path.join(viz_dir, f"{dataset}_{loss_name}_best_alpha.txt")
                with open(summary_path, 'w') as f:
                    f.write(f"Best alpha for {dataset} with {loss_name} (by ECE): {best_ece_row['alpha']}\n")
                    f.write(f"  - ECE: {best_ece_row['ece']}, Accuracy: {best_ece_row['accuracy']}\n\n")
                    f.write(f"Best alpha for {dataset} with {loss_name} (by Accuracy): {best_acc_row['alpha']}\n")
                    f.write(f"  - Accuracy: {best_acc_row['accuracy']}, ECE: {best_acc_row['ece']}\n")

    def cleanup(self):
        """Clean up temporary config files"""
        try:
            for f in os.listdir(self.results_dir):
                if f.startswith('config_') and f.endswith('.yaml'):
                    os.remove(os.path.join(self.results_dir, f))
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run alpha parameter experiments for MedCLIP and BioMedCLIP')
    parser.add_argument('--model', type=str, choices=['medclip', 'biomedclip'], required=True,
                      help='Model type to run experiments for')
    parser.add_argument('--dataset', type=str, choices=['covid', 'rsna18'],
                      help='Specific dataset to run (default: run both)')
    args = parser.parse_args()
    
    try:
        experiments = AlphaExperiments(args.model)
        print(f"Starting alpha experiments for {args.model}...")
        
        # If dataset is specified, only use that one
        if args.dataset:
            experiments.datasets = [args.dataset]
            
        results = experiments.run_experiments()
        print("\nAll experiments completed!")
        
        # Print a summary of the best alpha values for each configuration
        print("\nBest Alpha Values:")
        for dataset in experiments.datasets:
            dataset_results = results[results['dataset'] == dataset]
            for loss_name in [config['name'] for config in experiments.loss_configs]:
                loss_results = dataset_results[dataset_results['losses'] == loss_name]
                if not loss_results.empty:
                    best_ece = loss_results.loc[loss_results['ece'].idxmin()]
                    best_acc = loss_results.loc[loss_results['accuracy'].idxmax()]
                    print(f"{dataset} - {loss_name}:")
                    print(f"  Best alpha for ECE: {best_ece['alpha']} (ECE={best_ece['ece']:.2f}, Acc={best_ece['accuracy']:.2f})")
                    print(f"  Best alpha for Accuracy: {best_acc['alpha']} (Acc={best_acc['accuracy']:.2f}, ECE={best_acc['ece']:.2f})")
        
        experiments.cleanup()
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()