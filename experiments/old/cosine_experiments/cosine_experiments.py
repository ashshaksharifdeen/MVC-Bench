import yaml
import os
import subprocess
import pandas as pd
import re
import argparse
from datetime import datetime

class CosineExperiments:
    def __init__(self, model_type, dataset=None):
        self.model_type = model_type
        self.base_config_path = f"../../configs/trainers/CoOp/vit_b32_{model_type}_c16_ep50_batch16.yaml"
        self.results_dir = f"../../output/cosine_experiments_{model_type}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Single seed
        self.seeds = [1]
        
        # Cosine weights to test (ascending order)
        self.cosine_weights = [
            0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 
            18.0, 21.0, 25.0, 28.0, 35.0, 38.0, 42.0, 45.0, 50.0, 75.0, 100.0
        ]
        
        # DCA weight based on model type
        dca_weight = 13.0 if model_type == 'plip' else 9.0
        print(f"Using DCA weight: {dca_weight} for {model_type}")
        
        # Loss configurations
        self.loss_configs = [
            {'name': 'CE_COSINE', 
             'losses': ['CE', 'COSINE'], 
             'weights': [1.0], 
             'extra_params': {}},
            
            {'name': 'CE_DCA_COSINE', 
             'losses': ['CE', 'DCA', 'COSINE'], 
             'weights': [1.0, dca_weight], 
             'extra_params': {}},
            
            {'name': 'CE_MDCA_COSINE', 
             'losses': ['CE', 'MDCA', 'COSINE'], 
             'weights': [1.0, 1.0], 
             'extra_params': {}},
            
            {'name': 'CE_MMCE_COSINE', 
             'losses': ['CE', 'MMCE', 'COSINE'], 
             'weights': [1.0, 2.0], 
             'extra_params': {}},
            
            {'name': 'FL_COSINE', 
             'losses': ['FL', 'COSINE'], 
             'weights': [1.0], 
             'extra_params': {}},
            
            {'name': 'FL_MDCA_COSINE', 
             'losses': ['FL', 'MDCA', 'COSINE'], 
             'weights': [1.0, 1.0], 
             'extra_params': {}},
            
            {'name': 'LS_COSINE', 
             'losses': ['LS', 'COSINE'], 
             'weights': [1.0], 
             'extra_params': {'LS': {'ALPHA': 0.05}}},
            
            {'name': 'LS_MDCA_COSINE', 
             'losses': ['LS', 'MDCA', 'COSINE'], 
             'weights': [1.0, 1.0], 
             'extra_params': {'LS': {'ALPHA': 0.05}}}
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
            self.datasets = ["kather", "pannuke", "digestpath"]

    def modify_config(self, loss_config, cosine_weight):
        """Modify configuration file for the experiment."""
        try:
            # Read base config
            with open(self.base_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set fixed parameters
            config['OPTIM']['LR'] = self.fixed_params['lr']
            config['OPTIM']['MAX_EPOCH'] = self.fixed_params['epochs']
            
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
            print("\nLoss Configuration:")
            for loss in loss_config['losses']:
                weight = loss_config_yaml[loss]['WEIGHT']
                print(f"{loss}: weight={weight}")
                if loss in loss_config['extra_params']:
                    print(f"  extra params: {loss_config['extra_params'][loss]}")
            
            # Save config file
            config_name = f"config_{self.model_type}_{loss_config['name']}_cosine_{cosine_weight}.yaml"
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
            config_path = self.modify_config(loss_config, cosine_weight)
            
            # Setup experiment directory
            exp_name = f"{dataset}/shots_{self.fixed_params['shots']}/{loss_config['name']}_cosine_{cosine_weight}"
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
            
            for loss_config in self.loss_configs:
                print(f"\nTesting: {loss_config['name']}")
                
                for cosine_weight in self.cosine_weights:
                    for seed in self.seeds:
                        result = self.run_experiment(dataset, loss_config, cosine_weight, seed)
                        
                        if result:
                            # Record results
                            result_row = {
                                'dataset': dataset,
                                'lr': self.fixed_params['lr'],
                                'epochs': self.fixed_params['epochs'],
                                'shots': self.fixed_params['shots'],
                                'losses': f"{loss_config['name']}_{cosine_weight}",
                                'seed': seed,
                                'ece': result['ece'],
                                'accuracy': result['accuracy']
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
        
        return pd.DataFrame(results)

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
    parser = argparse.ArgumentParser(description='Run cosine weight experiments')
    parser.add_argument('--model', type=str, choices=['plip', 'quiltnet'], required=True,
                      help='Model type to run experiments for')
    parser.add_argument('--dataset', type=str, choices=['kather', 'pannuke', 'digestpath'],
                      help='Specific dataset to run experiments on')
    args = parser.parse_args()
    
    try:
        experiments = CosineExperiments(args.model, args.dataset)
        print(f"Starting experiments for {args.model}" + 
              (f" on {args.dataset}" if args.dataset else ""))
        
        results = experiments.run_experiments()
        print("\nExperiments completed!")
        
        experiments.cleanup()
    except Exception as e:
        print(f"Error in execution: {e}")

if __name__ == "__main__":
    main()