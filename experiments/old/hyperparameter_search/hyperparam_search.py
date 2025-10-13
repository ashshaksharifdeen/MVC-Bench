import yaml
import os
import subprocess
import pandas as pd
import re
from datetime import datetime

class HyperparamSearch:
    def __init__(self):
        self.base_config_path = "../configs/trainers/CoOp/vit_b32_plip_c16_ep50_batch16.yaml"
        self.results_dir = "../output/hyperparam_search"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define datasets
        self.datasets = ["kather", "pannuke", "digestpath"]
        
        # Define search space
        self.search_space = {
            'phase1': {
                'lr': [0.002, 0.02, 0.2],
                'epochs': [50],  # Fixed for phase 1
                'shots': [4],    # Fixed for phase 1
                'losses': [['CE']]  # Only CE for phase 1
            }
        }

    def extract_metrics_from_log(self, log_path):
        """Extract metrics directly from log file"""
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                metrics = {}
                
                # Extract metrics after "=> result"
                result_section = content.split("=> result")[-1]
                
                # Extract accuracy
                acc_match = re.search(r'\* accuracy: ([\d.]+)%', result_section)
                metrics['accuracy'] = float(acc_match.group(1)) if acc_match else None
                
                # Extract ECE
                ece_match = re.search(r'\* ece: ([\d.]+)%', result_section)
                metrics['ece'] = float(ece_match.group(1)) if ece_match else None
                
                return metrics
        except Exception as e:
            print(f"Error reading log file {log_path}: {e}")
            return None
        
    def modify_config(self, lr, epochs, shots, losses):
        with open(self.base_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Modify hyperparameters
        config['OPTIM']['LR'] = lr
        config['OPTIM']['MAX_EPOCH'] = epochs
        config['TRAINER']['COOP']['LOSS']['ENABLED_LOSSES'] = losses
        
        # Set loss weights
        if 'DCA' in losses:
            if 'DCA' not in config['TRAINER']['COOP']['LOSS']:
                config['TRAINER']['COOP']['LOSS']['DCA'] = {}
            config['TRAINER']['COOP']['LOSS']['DCA']['WEIGHT'] = 20.0
            
        if 'MDCA' in losses:
            if 'MDCA' not in config['TRAINER']['COOP']['LOSS']:
                config['TRAINER']['COOP']['LOSS']['MDCA'] = {}
            config['TRAINER']['COOP']['LOSS']['MDCA']['WEIGHT'] = 1.0
            
        # Create unique config filename based on the loss directory format
        config_name = f"config_lr{lr}_ep{epochs}_shots{shots}_{'_'.join(losses)}.yaml"
        config_path = os.path.join(self.results_dir, config_name)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        return config_path

    def run_experiment(self, dataset, lr, epochs, shots, losses):
        # Modify config and get new config path
        config_path = self.modify_config(lr, epochs, shots, losses)
        
        # Build experiment directory structure
        exp_name = f"{dataset}/lr{lr}_ep{epochs}_shots{shots}_{'_'.join(losses)}"
        output_dir = os.path.join(self.results_dir, exp_name, "seed1")
        os.makedirs(output_dir, exist_ok=True)
        
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
            subprocess.run(cmd, check=True)
            
            # Extract results from log file
            log_file = os.path.join(output_dir, "log.txt")
            results = self.extract_metrics_from_log(log_file)
            
            if results:
                print(f"\nResults for {dataset} with LR={lr}:")
                print(f"Accuracy: {results['accuracy']}%")
                print(f"ECE: {results['ece']}%")
                
            return results
            
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment with dataset={dataset}, LR={lr}: {e}")
            return None

    def run_phase1(self):
        all_results = []
        
        # For each dataset
        for dataset in self.datasets:
            print(f"\n=== Running experiments for {dataset} ===")
            
            # Generate all combinations for phase 1
            for lr in self.search_space['phase1']['lr']:
                for epochs in self.search_space['phase1']['epochs']:
                    for shots in self.search_space['phase1']['shots']:
                        for losses in self.search_space['phase1']['losses']:
                            print(f"\nRunning experiment: Dataset={dataset}, LR={lr}, Epochs={epochs}, Shots={shots}, Losses={losses}")
                            result = self.run_experiment(dataset, lr, epochs, shots, losses)
                            
                            if result:
                                all_results.append({
                                    'dataset': dataset,
                                    'lr': lr,
                                    'epochs': epochs,
                                    'shots': shots,
                                    'losses': '_'.join(losses),
                                    'ece': result['ece'],
                                    'accuracy': result['accuracy']
                                })
                
            # Print summary for this dataset
            df_dataset = pd.DataFrame([r for r in all_results if r['dataset'] == dataset])
            print(f"\nSummary for {dataset}:")
            print(df_dataset)
            
            # Print best results for this dataset
            best_ece = df_dataset.loc[df_dataset['ece'].idxmin()]
            best_acc = df_dataset.loc[df_dataset['accuracy'].idxmax()]
            
            print(f"\nBest configuration for {dataset} by ECE:")
            print(f"Learning Rate: {best_ece['lr']}")
            print(f"ECE: {best_ece['ece']}%")
            print(f"Accuracy: {best_ece['accuracy']}%")
            
            print(f"\nBest configuration for {dataset} by Accuracy:")
            print(f"Learning Rate: {best_acc['lr']}")
            print(f"ECE: {best_acc['ece']}%")
            print(f"Accuracy: {best_acc['accuracy']}%")
                
        # Save all results
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(self.results_dir, 'phase1_results.csv'), index=False)
        return df

    def cleanup(self):
        """Clean up temporary config files"""
        for f in os.listdir(self.results_dir):
            if f.startswith('config_') and f.endswith('.yaml'):
                os.remove(os.path.join(self.results_dir, f))

def main():
    search = HyperparamSearch()
    print("Starting Phase 1: Finding optimal learning rate for each dataset...")
    phase1_results = search.run_phase1()
    print("\nAll Results:")
    print(phase1_results)
    
    # Print summary for each dataset
    for dataset in search.datasets:
        df_dataset = phase1_results[phase1_results['dataset'] == dataset]
        
        print(f"\n=== Summary for {dataset} ===")
        best_ece = df_dataset.loc[df_dataset['ece'].idxmin()]
        best_acc = df_dataset.loc[df_dataset['accuracy'].idxmax()]
        
        print(f"\nBest configuration by ECE:")
        print(f"Learning Rate: {best_ece['lr']}")
        print(f"ECE: {best_ece['ece']}%")
        print(f"Accuracy: {best_ece['accuracy']}%")
        
        print(f"\nBest configuration by Accuracy:")
        print(f"Learning Rate: {best_acc['lr']}")
        print(f"ECE: {best_acc['ece']}%")
        print(f"Accuracy: {best_acc['accuracy']}%")
    
    search.cleanup()

if __name__ == "__main__":
    main()