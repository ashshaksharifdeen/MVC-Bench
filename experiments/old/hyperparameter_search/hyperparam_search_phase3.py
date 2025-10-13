import yaml
import os
import subprocess
import pandas as pd
import re
from datetime import datetime

class HyperparamSearch:
    def __init__(self):
        self.base_config_path = "../configs/trainers/CoOp/vit_b32_plip_c16_ep50_batch16.yaml"
        self.results_dir = "../output/hyperparam_search_phase3"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define datasets
        self.datasets = ["kather", "pannuke", "digestpath"]
        
        # Phase 3 search space
        self.search_space = {
            'lr': 0.002,  # Fixed from Phase 1
            'epochs': 50,
            'shots': [4, 8, 16, 32, 64],
            'losses': ['CE', 'CE_MDCA']  # Test both configurations
        }
        
    def extract_metrics_from_log(self, log_path):
        """Extract metrics directly from log file"""
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                metrics = {}
                
                # Extract metrics after "=> result"
                result_section = content.split("=> result")[-1]
                
                # Extract all available metrics
                metric_patterns = {
                    'accuracy': r'\* accuracy: ([\d.]+)%',
                    'ece': r'\* ece: ([\d.]+)%',
                    'mce': r'\* mce: ([\d.]+)%',
                    'ace': r'\* ace: ([\d.]+)%',
                    'macro_f1': r'\* macro_f1: ([\d.]+)%'
                }
                
                for metric_name, pattern in metric_patterns.items():
                    match = re.search(pattern, result_section)
                    metrics[metric_name] = float(match.group(1)) if match else None
                
                return metrics
        except Exception as e:
            print(f"Error reading log file {log_path}: {e}")
            return None

    def modify_config(self, losses, shots):
        with open(self.base_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Set fixed hyperparameters
        config['OPTIM']['LR'] = self.search_space['lr']
        config['OPTIM']['MAX_EPOCH'] = self.search_space['epochs']
        
        # Set loss configuration
        if losses == 'CE':
            config['TRAINER']['COOP']['LOSS']['ENABLED_LOSSES'] = ['CE']
            config['TRAINER']['COOP']['LOSS']['CE']['WEIGHT'] = 1.0
        else:  # CE_MDCA
            config['TRAINER']['COOP']['LOSS']['ENABLED_LOSSES'] = ['CE', 'MDCA']
            config['TRAINER']['COOP']['LOSS']['CE']['WEIGHT'] = 1.0
            if 'MDCA' not in config['TRAINER']['COOP']['LOSS']:
                config['TRAINER']['COOP']['LOSS']['MDCA'] = {}
            config['TRAINER']['COOP']['LOSS']['MDCA']['WEIGHT'] = 1.0
            
        # Create unique config filename
        config_name = f"config_lr{self.search_space['lr']}_ep{self.search_space['epochs']}_shots{shots}_{losses}.yaml"
        config_path = os.path.join(self.results_dir, config_name)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        return config_path

    def run_experiment(self, dataset, shots, losses):
        # Modify config and get new config path
        config_path = self.modify_config(losses, shots)
        
        # Build experiment directory structure
        exp_name = f"{dataset}/shots_{shots}/{losses}"
        output_dir = os.path.join(self.results_dir, exp_name, "seed1")
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if experiment is already completed
        log_file = os.path.join(output_dir, "log.txt")
        if os.path.exists(log_file):
            print(f"Experiment already completed: {dataset}, shots={shots}, losses={losses}")
            results = self.extract_metrics_from_log(log_file)
            return results
        
        # Prepare command
        cmd = [
            "python", "../train.py",
            "--root", "/home/abhishek/desktop/VLM_Cal/CLIP_Calibration/$DATA",
            "--seed", "1",
            "--trainer", "CoOp",
            "--dataset-config-file", f"../configs/datasets/{dataset}.yaml",
            "--config-file", config_path,
            "--output-dir", output_dir,
            "DATASET.NUM_SHOTS", str(shots),
            "DATASET.SUBSAMPLE_CLASSES", "all",
            "MODEL.NAME", "plip",
            "MODEL_ROOT", "/home/abhishek/desktop/VLM_Cal/CLIP_Calibration/models"
        ]
        
        # Run the experiment
        try:
            print(f"\nRunning experiment: {dataset}, shots={shots}, losses={losses}")
            subprocess.run(cmd, check=True)
            
            # Extract results from log file
            results = self.extract_metrics_from_log(log_file)
            if results:
                print(f"Results:")
                for metric, value in results.items():
                    print(f"{metric}: {value}%")
                
            return results
            
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment: {e}")
            return None

    def run_experiments(self):
        all_results = []
        
        # For each dataset
        for dataset in self.datasets:
            print(f"\n=== Running experiments for {dataset} ===")
            
            # For each shot count
            for shots in self.search_space['shots']:
                # For each loss configuration
                for losses in self.search_space['losses']:
                    print(f"\nRunning: Dataset={dataset}, Shots={shots}, Losses={losses}")
                    result = self.run_experiment(dataset, shots, losses)
                    
                    if result:
                        result_row = {
                            'dataset': dataset,
                            'lr': self.search_space['lr'],
                            'epochs': self.search_space['epochs'],
                            'shots': shots,
                            'losses': losses
                        }
                        result_row.update(result)  # Add all metrics
                        all_results.append(result_row)
                        
                        # Save intermediate results
                        df = pd.DataFrame(all_results)
                        df.to_csv(os.path.join(self.results_dir, 'phase3_results.csv'), index=False)
                        print(f"Updated results saved to {self.results_dir}/phase3_results.csv")
        
        return pd.DataFrame(all_results)

    def cleanup(self):
        """Clean up temporary config files"""
        for f in os.listdir(self.results_dir):
            if f.startswith('config_') and f.endswith('.yaml'):
                os.remove(os.path.join(self.results_dir, f))

def main():
    search = HyperparamSearch()
    print("Starting Phase 3: Testing shots progression with CE and CE+MDCA...")
    results = search.run_experiments()
    print("\nAll experiments completed!")
    print("\nFinal Results Summary:")
    print(results)
    search.cleanup()

if __name__ == "__main__":
    main()