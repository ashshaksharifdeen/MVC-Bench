import yaml
import os
import subprocess
import pandas as pd
import re
from datetime import datetime

class AlphaExperiments:
    def __init__(self, model_type):
        self.model_type = model_type  # 'plip' or 'quiltnet'
        self.base_config_path = f"../../configs/trainers/CoOp/vit_b32_{model_type}_c16_ep50_batch16.yaml"
        self.results_dir = f"../../output/alpha_experiments_{model_type}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Alpha values to test - explicitly as floats, including 0
        self.alpha_values = [0.0, 0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35]
        
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
        
        # Datasets
        self.datasets = ["kather", "pannuke", "digestpath"]

    def modify_config(self, losses, weights, alpha):
        try:
            with open(self.base_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set fixed parameters
            config['OPTIM']['LR'] = self.fixed_params['lr']
            config['OPTIM']['MAX_EPOCH'] = self.fixed_params['epochs']
            
            # Set loss configuration
            loss_config = config['TRAINER']['COOP']['LOSS']
            loss_config['ENABLED_LOSSES'] = losses
            
            # Initialize all possible loss configurations
            for loss_name, loss_params in {
                'CE': {'WEIGHT': 1.0},
                'FL': {'WEIGHT': 1.0},
                'LS': {'WEIGHT': 1.0, 'ALPHA': 0.1},
                'SLMDCA': {'WEIGHT': 1.0, 'ALPHA': 0.1},
                'DCA': {'WEIGHT': 1.0},
                'MDCA': {'WEIGHT': 1.0},
                'MMCE': {'WEIGHT': 2.0},
            }.items():
                loss_config[loss_name] = loss_params.copy()
            
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
            exp_name = f"{dataset}/shots_{self.fixed_params['shots']}/{loss_config['name']}_alpha_{alpha}"
            output_dir = os.path.join(self.results_dir, exp_name, "seed1")
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
                "--seed", "1",
                "--trainer", "CoOp",
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
                            'losses': f"{loss_config['name']}_alpha_{alpha}",
                            'ece': result['ece'],
                            'accuracy': result['accuracy']
                        }
                        results.append(result_row)
                        
                        # Save current results to CSV
                        df = pd.DataFrame(results)
                        df.to_csv(os.path.join(self.results_dir, f'{self.model_type}_results.csv'), index=False)
                        print(f"Updated results saved to {self.model_type}_results.csv")
        
        return pd.DataFrame(results)

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
    import argparse
    parser = argparse.ArgumentParser(description='Run alpha parameter experiments')
    parser.add_argument('--model', type=str, choices=['plip', 'quiltnet'], required=True,
                      help='Model type to run experiments for')
    args = parser.parse_args()
    
    try:
        experiments = AlphaExperiments(args.model)
        print(f"Starting alpha experiments for {args.model}...")
        results = experiments.run_experiments()
        print("\nAll experiments completed!")
        print("\nResults Summary:")
        print(results.to_string())
        experiments.cleanup()
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()