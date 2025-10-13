import yaml
import os
import subprocess
import pandas as pd
import re
from datetime import datetime

class BaselineDCAWeightExperiments:
    def __init__(self):
        self.base_config_path = "../../configs/trainers/CoOp/vit_b32_quiltnet_c16_ep50_batch16.yaml"
        self.results_dir = "../../output/baseline_dca_weight_quiltnet"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define DCA weight combinations to test
        self.dca_weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
        self.loss_configs = [
            {'losses': ['CE', 'DCA'], 'weights': [1.0, w]} for w in self.dca_weights
        ]
        
        # Fixed parameters
        self.fixed_params = {
            'lr': 0.002,
            'epochs': 50,
            'shots': 8
        }
        
        self.datasets = ["kather", "pannuke", "digestpath"]

    def modify_config(self, losses, weights):
        try:
            with open(self.base_config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            config['OPTIM']['LR'] = self.fixed_params['lr']
            config['OPTIM']['MAX_EPOCH'] = self.fixed_params['epochs']
            config['TRAINER']['COOP']['LOSS']['ENABLED_LOSSES'] = losses
            
            for loss, weight in zip(losses, weights):
                if loss not in config['TRAINER']['COOP']['LOSS']:
                    config['TRAINER']['COOP']['LOSS'][loss] = {}
                config['TRAINER']['COOP']['LOSS'][loss]['WEIGHT'] = weight
                
            config_name = f"config_CE_DCA_{weights[1]}.yaml"
            config_path = os.path.join(self.results_dir, config_name)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                
            return config_path
            
        except Exception as e:
            print(f"Error modifying config: {e}")
            raise

    def run_experiment(self, dataset, losses, weights):
        try:
            config_path = self.modify_config(losses, weights)
            
            exp_name = f"{dataset}/shots_{self.fixed_params['shots']}/CE_DCA_{weights[1]}"
            output_dir = os.path.join(self.results_dir, exp_name, "seed1")
            os.makedirs(output_dir, exist_ok=True)
            
            log_file = os.path.join(output_dir, "log.txt")
            if os.path.exists(log_file):
                print(f"Experiment already completed: {dataset}, DCA weight={weights[1]}")
                return self.extract_metrics(log_file)
            
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
                "MODEL.NAME", "quiltnet",
                "MODEL_ROOT", "/home/abhishek/desktop/VLM_Cal/CLIP_Calibration/models"
            ]
            
            print(f"\nRunning experiment: {dataset}, DCA weight={weights[1]}")
            subprocess.run(cmd, check=True)
            return self.extract_metrics(log_file)
            
        except Exception as e:
            print(f"Error running experiment: {e}")
            return None

    def extract_metrics(self, log_path):
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                metrics = {}
                
                patterns = {
                    'accuracy': r'\* accuracy: ([\d.]+)%',
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
        
        for dataset in self.datasets:
            print(f"\n=== Running experiments for {dataset} ===")
            
            for loss_config in self.loss_configs:
                print(f"\nTesting DCA weight: {loss_config['weights'][1]}")
                result = self.run_experiment(dataset, loss_config['losses'], loss_config['weights'])
                
                if result:
                    result_row = {
                        'dataset': dataset,
                        'dca_weight': loss_config['weights'][1],
                        'lr': self.fixed_params['lr'],
                        'epochs': self.fixed_params['epochs'],
                        'shots': self.fixed_params['shots']
                    }
                    result_row.update(result)
                    all_results.append(result_row)
                    
                    df = pd.DataFrame(all_results)
                    results_file = os.path.join(self.results_dir, 'dca_weight_results.csv')
                    df.to_csv(results_file, index=False)
                    print(f"Updated results saved to {results_file}")
        
        return pd.DataFrame(all_results)

    def cleanup(self):
        try:
            for f in os.listdir(self.results_dir):
                if f.startswith('config_') and f.endswith('.yaml'):
                    os.remove(os.path.join(self.results_dir, f))
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    try:
        experiments = BaselineDCAWeightExperiments()
        print("Starting DCA weight experiments...")
        results = experiments.run_experiments()
        print("\nAll experiments completed!")
        print("\nFinal Results Summary:")
        print(results.to_string())
        experiments.cleanup()
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()