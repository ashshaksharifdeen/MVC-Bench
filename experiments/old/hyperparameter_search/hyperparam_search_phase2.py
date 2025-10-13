import yaml
import os
import subprocess
import pandas as pd
import re
from datetime import datetime

class HyperparamSearch:
    def __init__(self):
        self.base_config_path = "../configs/trainers/CoOp/vit_b32_plip_c16_ep50_batch16.yaml"
        self.results_dir = "../output/hyperparam_search_phase2"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define datasets
        self.datasets = ["kather", "pannuke", "digestpath"]
        
        # Phase 2 search space
        self.search_space = {
            'lr': 0.002,  # Fixed based on Phase 1
            'epochs': 50,
            'shots': 4,
            'losses': [
                ['CE'],  # Baseline
                ['CE', 'DCA'],
                ['CE', 'MDCA']
            ]
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
        
    def modify_config(self, losses):
        with open(self.base_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Set fixed hyperparameters
        config['OPTIM']['LR'] = self.search_space['lr']
        config['OPTIM']['MAX_EPOCH'] = self.search_space['epochs']
        config['TRAINER']['COOP']['LOSS']['ENABLED_LOSSES'] = losses
        
        # Set loss weights
        if 'DCA' in losses:
            if 'DCA' not in config['TRAINER']['COOP']['LOSS']:
                config['TRAINER']['COOP']['LOSS']['DCA'] = {}
            config['TRAINER']['COOP']['LOSS']['DCA']['WEIGHT'] = 15.0
            
        if 'MDCA' in losses:
            if 'MDCA' not in config['TRAINER']['COOP']['LOSS']:
                config['TRAINER']['COOP']['LOSS']['MDCA'] = {}
            config['TRAINER']['COOP']['LOSS']['MDCA']['WEIGHT'] = 1.0
            
        # Create unique config filename
        config_name = f"config_lr{self.search_space['lr']}_ep{self.search_space['epochs']}_shots{self.search_space['shots']}_{'_'.join(losses)}.yaml"
        config_path = os.path.join(self.results_dir, config_name)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        return config_path

    def run_experiment(self, dataset, losses):
        # Modify config and get new config path
        config_path = self.modify_config(losses)
        
        # Build experiment directory structure
        exp_name = f"{dataset}/{'_'.join(losses)}"
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
            "DATASET.NUM_SHOTS", str(self.search_space['shots']),
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
                print(f"\nResults for {dataset} with losses={'_'.join(losses)}:")
                print(f"Accuracy: {results['accuracy']}%")
                print(f"ECE: {results['ece']}%")
                
            return results
            
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment with dataset={dataset}, losses={losses}: {e}")
            return None

    def run_experiments(self):
        all_results = []
        
        # For each dataset
        for dataset in self.datasets:
            print(f"\n=== Running experiments for {dataset} ===")
            
            # Test each loss combination
            for losses in self.search_space['losses']:
                print(f"\nRunning experiment: Dataset={dataset}, Losses={losses}")
                result = self.run_experiment(dataset, losses)
                
                if result:
                    all_results.append({
                        'dataset': dataset,
                        'lr': self.search_space['lr'],
                        'epochs': self.search_space['epochs'],
                        'shots': self.search_space['shots'],
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
            print(f"Losses: {best_ece['losses']}")
            print(f"ECE: {best_ece['ece']}%")
            print(f"Accuracy: {best_ece['accuracy']}%")
            
            print(f"\nBest configuration for {dataset} by Accuracy:")
            print(f"Losses: {best_acc['losses']}")
            print(f"ECE: {best_acc['ece']}%")
            print(f"Accuracy: {best_acc['accuracy']}%")
                
        # Save all results
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(self.results_dir, 'phase2_results.csv'), index=False)
        return df

    def cleanup(self):
        """Clean up temporary config files"""
        for f in os.listdir(self.results_dir):
            if f.startswith('config_') and f.endswith('.yaml'):
                os.remove(os.path.join(self.results_dir, f))

def main():
    search = HyperparamSearch()
    print("Starting Phase 2: Testing different loss combinations...")
    results = search.run_experiments()
    print("\nAll Results:")
    print(results)
    
    # Print summary for each dataset
    for dataset in search.datasets:
        df_dataset = results[results['dataset'] == dataset]
        
        print(f"\n=== Summary for {dataset} ===")
        print("\nAll configurations:")
        print(df_dataset[['losses', 'ece', 'accuracy']])
        
        best_ece = df_dataset.loc[df_dataset['ece'].idxmin()]
        best_acc = df_dataset.loc[df_dataset['accuracy'].idxmax()]
        
        print(f"\nBest by ECE: {best_ece['losses']}")
        print(f"ECE: {best_ece['ece']}%")
        print(f"Accuracy: {best_ece['accuracy']}%")
        
        print(f"\nBest by Accuracy: {best_acc['losses']}")
        print(f"ECE: {best_acc['ece']}%")
        print(f"Accuracy: {best_acc['accuracy']}%")
    
    search.cleanup()

if __name__ == "__main__":
    main()