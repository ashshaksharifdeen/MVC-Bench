import yaml
import os
import subprocess
import pandas as pd
import re
import argparse
from datetime import datetime

class CosineMarginExperiments:
    def __init__(self, model_type, dataset=None):
        self.model_type = model_type
        self.base_config_path = f"../../configs/trainers/CoOp/vit_b32_{model_type}_c16_ep50_batch16.yaml"
        self.results_dir = f"../../output/cosine_margin_experiments_{model_type}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Single seed
        self.seeds = [1]
        
        # Margin values to test (in radians)
        self.margins = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        # DCA weight based on model type
        dca_weight = 13.0 if model_type == 'plip' else 9.0
        print(f"Using DCA weight: {dca_weight} for {model_type}")
        
        # Loss configurations
        self.loss_configs = [
            {'name': 'CE_COSINE_MARGIN', 
             'losses': ['CE', 'COSINE_MARGIN'], 
             'weights': [1.0], 
             'extra_params': {}},
            
            {'name': 'CE_DCA_COSINE_MARGIN', 
             'losses': ['CE', 'DCA', 'COSINE_MARGIN'], 
             'weights': [1.0, dca_weight], 
             'extra_params': {}},
            
            {'name': 'CE_MDCA_COSINE_MARGIN', 
             'losses': ['CE', 'MDCA', 'COSINE_MARGIN'], 
             'weights': [1.0, 1.0], 
             'extra_params': {}},
            
            {'name': 'CE_MMCE_COSINE_MARGIN', 
             'losses': ['CE', 'MMCE', 'COSINE_MARGIN'], 
             'weights': [1.0, 2.0], 
             'extra_params': {}},
            
            {'name': 'FL_COSINE_MARGIN', 
             'losses': ['FL', 'COSINE_MARGIN'], 
             'weights': [1.0], 
             'extra_params': {}},
            
            {'name': 'FL_MDCA_COSINE_MARGIN', 
             'losses': ['FL', 'MDCA', 'COSINE_MARGIN'], 
             'weights': [1.0, 1.0], 
             'extra_params': {}},
            
            {'name': 'LS_COSINE_MARGIN', 
             'losses': ['LS', 'COSINE_MARGIN'], 
             'weights': [1.0], 
             'extra_params': {'LS': {'ALPHA': 0.05}}},
            
            {'name': 'LS_MDCA_COSINE_MARGIN', 
             'losses': ['LS', 'MDCA', 'COSINE_MARGIN'], 
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

    def modify_config(self, loss_config, margin):
        """Modify configuration file for the experiment."""
        try:
            with open(self.base_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set fixed parameters
            config['OPTIM']['LR'] = self.fixed_params['lr']
            config['OPTIM']['MAX_EPOCH'] = self.fixed_params['epochs']
            
            # Set loss configuration
            loss_config_yaml = config['TRAINER']['COOP']['LOSS']
            loss_config_yaml['ENABLED_LOSSES'] = loss_config['losses']
            
            # Initialize non-COSINE_MARGIN losses
            for idx, loss_name in enumerate(loss_config['losses'][:-1]):
                if loss_name not in loss_config_yaml:
                    loss_config_yaml[loss_name] = {}
                loss_config_yaml[loss_name]['WEIGHT'] = loss_config['weights'][idx]
                
                # Add extra parameters if any
                if loss_name in loss_config['extra_params']:
                    loss_config_yaml[loss_name].update(loss_config['extra_params'][loss_name])
            
            # Configure COSINE_MARGIN
            if 'COSINE_MARGIN' not in loss_config_yaml:
                loss_config_yaml['COSINE_MARGIN'] = {}
            loss_config_yaml['COSINE_MARGIN'].update({
                'WEIGHT': 1.0,
                'MARGIN': float(margin)  # Ensure margin is float
            })
            
            # Print configuration for verification
            print("\nLoss Configuration:")
            print(f"Margin (radians): {margin:.3f}, (degrees): {margin * 180/3.14159:.2f}°")
            for loss in loss_config['losses']:
                if loss == 'COSINE_MARGIN':
                    print(f"{loss}: weight=1.0, margin={margin:.3f}")
                else:
                    weight = loss_config_yaml[loss]['WEIGHT']
                    print(f"{loss}: weight={weight}")
                    if loss in loss_config['extra_params']:
                        print(f"  extra params: {loss_config['extra_params'][loss]}")
            
            # Save config
            config_name = f"config_{self.model_type}_{loss_config['name']}_margin_{margin:.3f}.yaml"
            config_path = os.path.join(self.results_dir, config_name)
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            return config_path
            
        except Exception as e:
            print(f"Error modifying config: {e}")
            raise

    def run_experiment(self, dataset, loss_config, margin, seed):
        try:
            # Get modified config
            config_path = self.modify_config(loss_config, margin)
            
            # Setup experiment directory
            exp_name = f"{dataset}/shots_{self.fixed_params['shots']}/{loss_config['name']}_margin_{margin:.3f}"
            output_dir = os.path.join(self.results_dir, exp_name, f"seed{seed}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Check for completed experiment
            log_file = os.path.join(output_dir, "log.txt")
            if os.path.exists(log_file):
                print(f"Experiment exists: {dataset}, {loss_config['name']}, margin={margin}, seed={seed}")
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
            print(f"\nRunning: {dataset}, {loss_config['name']}, margin={margin:.3f}, seed={seed}")
            subprocess.run(cmd, check=True)
            return self.extract_metrics(log_file)
            
        except Exception as e:
            print(f"Error in experiment: {e}")
            return None

    def extract_metrics(self, log_path):
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
                        print(f"Warning: Missing {metric_name} in log")
                        metrics[metric_name] = float('nan')
                    else:
                        metrics[metric_name] = float(match.group(1))
                
                return metrics
        except Exception as e:
            print(f"Error reading log {log_path}: {e}")
            return None

    def run_experiments(self):
        results = []
        
        for dataset in self.datasets:
            print(f"\n=== Dataset: {dataset} ===")
            
            for loss_config in self.loss_configs:
                print(f"\nTesting: {loss_config['name']}")
                
                for margin in self.margins:
                    for seed in self.seeds:
                        result = self.run_experiment(dataset, loss_config, margin, seed)
                        
                        if result:
                            # Record results
                            result_row = {
                                'dataset': dataset,
                                'lr': self.fixed_params['lr'],
                                'epochs': self.fixed_params['epochs'],
                                'shots': self.fixed_params['shots'],
                                'losses': f"{loss_config['name']}_margin_{margin:.3f}",
                                'margin_rad': margin,
                                'margin_deg': margin * 180/3.14159,
                                'seed': seed,
                                'ece': result['ece'],
                                'accuracy': result['accuracy']
                            }
                            results.append(result_row)
                            
                            # Save progress
                            df = pd.DataFrame(results)
                            csv_path = os.path.join(self.results_dir, f'{self.model_type}_{dataset}_results.csv')
                            df.to_csv(csv_path, index=False)
                            print(f"Updated results: {csv_path}")
        
        return pd.DataFrame(results)

    def cleanup(self):
        try:
            for f in os.listdir(self.results_dir):
                if f.startswith('config_') and f.endswith('.yaml'):
                    os.remove(os.path.join(self.results_dir, f))
        except Exception as e:
            print(f"Error in cleanup: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run cosine margin experiments')
    parser.add_argument('--model', type=str, choices=['plip', 'quiltnet'], required=True,
                      help='Model type to run experiments for')
    parser.add_argument('--dataset', type=str, choices=['kather', 'pannuke', 'digestpath'],
                      help='Specific dataset to run experiments on')
    args = parser.parse_args()
    
    try:
        experiments = CosineMarginExperiments(args.model, args.dataset)
        print(f"Starting experiments for {args.model}" + 
              (f" on {args.dataset}" if args.dataset else ""))
        
        results = experiments.run_experiments()
        print("\nExperiments completed!")
        
        experiments.cleanup()
    except Exception as e:
        print(f"Error in execution: {e}")

if __name__ == "__main__":
    main()