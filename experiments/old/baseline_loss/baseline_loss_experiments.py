import yaml
import os
import subprocess
import pandas as pd
import re
from datetime import datetime

class BaselineLossExperiments:
    def __init__(self):
        self.base_config_path = "../../configs/trainers/CoOp/vit_b32_plip_c16_ep50_batch16.yaml"
        self.results_dir = "../../output/baseline_loss"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define loss combinations
        self.loss_configs = [
            {'losses': ['CE'], 'weights': [1.0]},
            {'losses': ['FL'], 'weights': [1.0]},
            {'losses': ['LS'], 'weights': [1.0]},
            {'losses': ['CE', 'MDCA'], 'weights': [1.0, 1.0]},
            {'losses': ['FL', 'MDCA'], 'weights': [1.0, 1.0]},
            {'losses': ['LS', 'MDCA'], 'weights': [1.0, 1.0]},
            {'losses': ['CE', 'RCR'], 'weights': [1.0, 1.0]},
            {'losses': ['FL', 'RCR'], 'weights': [1.0, 1.0]},
            {'losses': ['BS'], 'weights': [1.0]},
            {'losses': ['CE', 'DCA'], 'weights': [1.0, 15.0]},
            {'losses': ['CE', 'MMCE'], 'weights': [1.0, 2.0]},
            {'losses': ['FLSD'], 'weights': [1.0]}
        ]
        
        # Fixed parameters from Phase 1
        self.fixed_params = {
            'lr': 0.002,
            'epochs': 50,
            'shots': 8
        }
        
        # Datasets
        self.datasets = ["kather", "pannuke", "digestpath"]

    def modify_config(self, losses, weights):
        try:
            with open(self.base_config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Set fixed parameters
            config['OPTIM']['LR'] = self.fixed_params['lr']
            config['OPTIM']['MAX_EPOCH'] = self.fixed_params['epochs']
            
            # Set loss configuration
            config['TRAINER']['COOP']['LOSS']['ENABLED_LOSSES'] = losses
            
            # Configure loss weights
            for loss, weight in zip(losses, weights):
                if loss not in config['TRAINER']['COOP']['LOSS']:
                    config['TRAINER']['COOP']['LOSS'][loss] = {}
                config['TRAINER']['COOP']['LOSS'][loss]['WEIGHT'] = weight
                
            # Create unique config filename
            config_name = f"config_{'_'.join(losses)}.yaml"
            config_path = os.path.join(self.results_dir, config_name)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                
            return config_path
            
        except Exception as e:
            print(f"Error modifying config: {e}")
            raise

    def run_experiment(self, dataset, losses, weights):
        try:
            # Modify config and get new config path
            config_path = self.modify_config(losses, weights)
            
            # Build experiment directory structure
            loss_str = '_'.join(losses)
            exp_name = f"{dataset}/shots_{self.fixed_params['shots']}/{loss_str}"
            output_dir = os.path.join(self.results_dir, exp_name, "seed1")
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if experiment is already completed
            log_file = os.path.join(output_dir, "log.txt")
            if os.path.exists(log_file):
                print(f"Experiment already completed: {dataset}, losses={loss_str}")
                return self.extract_metrics(log_file)
            
            # Prepare command
            cmd = [
                "python", "../../train.py",
                "--root", "/home/abhishek/desktop/VLM_Cal/CLIP_Calibration/$DATA",
                "--seed", "1",
                "--trainer", "CoOp",
                "--dataset-config-file", f"../../configs/datasets/{dataset}.yaml",
                "--config-file", config_path,
                "--output-dir", output_dir,
                "DATASET.NUM_SHOTS", str(self.fixed_params['shots']),
                "DATASET.SUBSAMPLE_CLASSES", "all",
                "MODEL.NAME", "plip",
                "MODEL_ROOT", "/home/abhishek/desktop/VLM_Cal/CLIP_Calibration/models"
            ]
            
            # Run the experiment
            print(f"\nRunning experiment: {dataset}, losses={loss_str}")
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
        all_results = []
        
        # For each dataset
        for dataset in self.datasets:
            print(f"\n=== Running experiments for {dataset} ===")
            
            # For each loss configuration
            for loss_config in self.loss_configs:
                print(f"\nTesting loss configuration: {loss_config['losses']}")
                result = self.run_experiment(dataset, loss_config['losses'], loss_config['weights'])
                
                if result:
                    result_row = {
                        'dataset': dataset,
                        'lr': self.fixed_params['lr'],
                        'epochs': self.fixed_params['epochs'],
                        'shots': self.fixed_params['shots'],
                        'losses': '_'.join(loss_config['losses'])
                    }
                    result_row.update(result)
                    all_results.append(result_row)
                    
                    # Save intermediate results
                    df = pd.DataFrame(all_results)
                    results_file = os.path.join(self.results_dir, 'baseline_loss_results.csv')
                    df.to_csv(results_file, index=False)
                    print(f"Updated results saved to {results_file}")
        
        return pd.DataFrame(all_results)

    def cleanup(self):
        """Clean up temporary config files"""
        try:
            for f in os.listdir(self.results_dir):
                if f.startswith('config_') and f.endswith('.yaml'):
                    os.remove(os.path.join(self.results_dir, f))
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    try:
        experiments = BaselineLossExperiments()
        print("Starting baseline loss experiments...")
        results = experiments.run_experiments()
        print("\nAll experiments completed!")
        print("\nFinal Results Summary:")
        print(results.to_string())
        experiments.cleanup()
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()