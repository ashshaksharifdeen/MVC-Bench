import yaml
import os
import subprocess
import pandas as pd
import re
import argparse
import numpy as np

# ======= LOSS CONFIGURATIONS =======
def get_loss_configs(model_type):
    """Centralized loss configurations"""
    if model_type == 'medclip':
        return {
            'rsna18': {
                'CE': {},
                'CE_DCA': {'DCA': {'WEIGHT': 9.0}},
                'CE_MMCE': {'MMCE': {'WEIGHT': 2.0}},
                'CE_MDCA': {'MDCA': {'WEIGHT': 1.0}},
                'CE_SMAC_0_1': {'SMAC': {'ALPHA': 0.1}},
                'CE_SMAC_0_1_AS_3_0': {'SMAC': {'ALPHA': 0.1}, 'AS': {'WEIGHT': 3.0}},
                'FL': {'FL': {'GAMMA': 3.0}},
                'FL_MDCA': {'FL': {'GAMMA': 3.0}, 'MDCA': {'WEIGHT': 1.0}},
                'FL_SMAC_0_1': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.1}},
                'FL_SMAC_0_1_AS_3_0': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.1}, 'AS': {'WEIGHT': 3.0}},
                'LS_0_2': {'LS': {'ALPHA': 0.2}},
                'LS_MDCA_0_2': {'LS': {'ALPHA': 0.2}, 'MDCA': {'WEIGHT': 1.0}},
                'LS_SMAC_0_2': {'LS': {'ALPHA': 0.2}, 'SMAC': {'ALPHA': 0.2}},
                'LS_SMAC_AS_1_0': {'LS': {'ALPHA': 0.2}, 'SMAC': {'ALPHA': 0.2}, 'AS': {'WEIGHT': 1.0}}
            },
            'covid': {
                'CE': {},
                'CE_DCA': {'DCA': {'WEIGHT': 9.0}},
                'CE_MMCE': {'MMCE': {'WEIGHT': 2.0}},
                'CE_MDCA': {'MDCA': {'WEIGHT': 1.0}},
                'CE_SMAC_0_1': {'SMAC': {'ALPHA': 0.1}},
                'CE_SMAC_0_1_AS_3_0': {'SMAC': {'ALPHA': 0.1}, 'AS': {'WEIGHT': 3.0}},
                'FL': {'FL': {'GAMMA': 3.0}},
                'FL_MDCA': {'FL': {'GAMMA': 3.0}, 'MDCA': {'WEIGHT': 1.0}},
                'FL_SMAC_0_1': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.1}},
                'FL_SMAC_0_1_AS_3_0': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.1}, 'AS': {'WEIGHT': 3.0}},
                'LS_0_2': {'LS': {'ALPHA': 0.2}},
                'LS_MDCA_0_2': {'LS': {'ALPHA': 0.2}, 'MDCA': {'WEIGHT': 1.0}},
                'LS_SMAC_0_2': {'LS': {'ALPHA': 0.2}, 'SMAC': {'ALPHA': 0.2}},
                'LS_SMAC_AS_1_0': {'LS': {'ALPHA': 0.2}, 'SMAC': {'ALPHA': 0.2}, 'AS': {'WEIGHT': 1.0}}
            }
        }
    elif model_type == 'biomedclip':
        return {
            'rsna18': {
                'CE': {},
                'CE_DCA': {'DCA': {'WEIGHT': 9.0}},
                'CE_MMCE': {'MMCE': {'WEIGHT': 2.0}},
                'CE_MDCA': {'MDCA': {'WEIGHT': 1.0}},
                'CE_SMAC_0_1': {'SMAC': {'ALPHA': 0.1}},
                'CE_SMAC_0_1_AS_3_0': {'SMAC': {'ALPHA': 0.1}, 'AS': {'WEIGHT': 3.0}},
                'FL': {'FL': {'GAMMA': 3.0}},
                'FL_MDCA': {'FL': {'GAMMA': 3.0}, 'MDCA': {'WEIGHT': 1.0}},
                'FL_SMAC_0_1': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.1}},
                'FL_SMAC_0_1_AS_3_0': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.1}, 'AS': {'WEIGHT': 3.0}},
                'LS_0_2': {'LS': {'ALPHA': 0.2}},
                'LS_MDCA_0_2': {'LS': {'ALPHA': 0.2}, 'MDCA': {'WEIGHT': 1.0}},
                'LS_SMAC_0_1': {'LS': {'ALPHA': 0.1}, 'SMAC': {'ALPHA': 0.1}},
                'LS_SMAC_AS_3_0': {'LS': {'ALPHA': 0.1}, 'SMAC': {'ALPHA': 0.1}, 'AS': {'WEIGHT': 3.0}}
            },
            'covid': {
                'CE': {},
                'CE_DCA': {'DCA': {'WEIGHT': 9.0}},
                'CE_MMCE': {'MMCE': {'WEIGHT': 2.0}},
                'CE_MDCA': {'MDCA': {'WEIGHT': 1.0}},
                'CE_SMAC_0_1': {'SMAC': {'ALPHA': 0.1}},
                'CE_SMAC_0_1_AS_3_0': {'SMAC': {'ALPHA': 0.1}, 'AS': {'WEIGHT': 3.0}},
                'FL': {'FL': {'GAMMA': 3.0}},
                'FL_MDCA': {'FL': {'GAMMA': 3.0}, 'MDCA': {'WEIGHT': 1.0}},
                'FL_SMAC_0_1': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.1}},
                'FL_SMAC_0_1_AS_3_0': {'FL': {'GAMMA': 3.0}, 'SMAC': {'ALPHA': 0.1}, 'AS': {'WEIGHT': 3.0}},
                'LS_0_2': {'LS': {'ALPHA': 0.2}},
                'LS_MDCA_0_2': {'LS': {'ALPHA': 0.2}, 'MDCA': {'WEIGHT': 1.0}},
                'LS_SMAC_0_1': {'LS': {'ALPHA': 0.1}, 'SMAC': {'ALPHA': 0.1}},
                'LS_SMAC_AS_3_0': {'LS': {'ALPHA': 0.1}, 'SMAC': {'ALPHA': 0.1}, 'AS': {'WEIGHT': 3.0}}
            }
        }

class MedicalPromptLearningExperiments:
    def __init__(self, model_type, dataset=None):
        self.model_type = model_type
        # Use correct trainer names
        self.trainer_name = "CoOp_MedCLIP" if model_type == "medclip" else "CoOp_BioMedCLIP"
        # Updated with exact path format
        self.base_config_path = f"/home/abhishek/desktop/VLM_Cal/CalibPrompt/configs/trainers/{self.trainer_name}/vit_b32_{model_type}_c16_ep50_batch16.yaml"
        self.results_dir = f"/home/abhishek/desktop/VLM_Cal/CalibPrompt/output/baseline_prompt_{model_type}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Using specific seeds
        self.seeds = [1]
        
        # Dataset-specific configurations
        self.dataset_configs = get_loss_configs(model_type)
        
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
            self.datasets = ["rsna18", "covid"]
            
        # Store loss configs per dataset
        self.loss_configs_per_dataset = {
            dataset: self.get_loss_configs(dataset) for dataset in self.datasets
        }
        
        # Debug print all configurations
        print("======== DEBUGGING ALL CONFIGURATIONS ========")
        for dataset in self.datasets:
            print(f"\nDataset: {dataset}")
            for config in self.loss_configs_per_dataset[dataset]:
                print(f"  Config: {config['name']}")
                print(f"    Enabled losses: {config['enabled_losses']}")
                print(f"    Params: {config['params']}")

    def get_loss_configs(self, dataset):
        """Generate loss configurations based on dataset."""
        dataset_params = self.dataset_configs[dataset]
        
        loss_configs = []
        for loss_name, params in dataset_params.items():
            # Determine enabled losses based on loss name by splitting it
            enabled_losses = []
            for part in loss_name.split('_'):
                if part in ['CE', 'FL', 'LS', 'DCA', 'MMCE', 'MDCA', 'SMAC', 'AS']:
                    enabled_losses.append(part)
            
            # Double check that all losses in params are enabled
            for loss in params.keys():
                if loss not in enabled_losses:
                    enabled_losses.append(loss)
            
            loss_configs.append({
                'name': loss_name,
                'enabled_losses': enabled_losses,
                'params': params
            })
            
        return loss_configs

    def modify_config(self, dataset, loss_config):
        """Modify configuration file for the experiment."""
        try:
            # Read base config
            with open(self.base_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set fixed parameters
            config['OPTIM']['LR'] = self.fixed_params['lr']
            config['OPTIM']['MAX_EPOCH'] = self.fixed_params['epochs']
            
            # Set loss configuration
            if 'LOSS' not in config['TRAINER']['COOP']:
                config['TRAINER']['COOP']['LOSS'] = {}
                
            loss_config_yaml = config['TRAINER']['COOP']['LOSS']
            loss_config_yaml['ENABLED_LOSSES'] = loss_config['enabled_losses']
            
            # Configure each individual loss
            for loss_name, loss_params in loss_config['params'].items():
                if loss_name not in loss_config_yaml:
                    loss_config_yaml[loss_name] = {}
                
                # Set all parameters for this loss
                for param_name, param_value in loss_params.items():
                    loss_config_yaml[loss_name][param_name] = param_value
            
            # Debug checks and verification
            print(f"\n=== DETAILED CONFIGURATION FOR {dataset} - {loss_config['name']} ===")
            print(f"Enabled losses: {loss_config_yaml['ENABLED_LOSSES']}")
            
            # Verify that AS is correctly configured if it should be used
            if 'AS' in loss_config['params']:
                as_weight = loss_config['params']['AS'].get('WEIGHT', 'Not set')
                print(f"AS weight in params: {as_weight}")
                
                # Check if AS is in the YAML config
                if 'AS' in loss_config_yaml:
                    as_weight_yaml = loss_config_yaml['AS'].get('WEIGHT', 'Not set in YAML')
                    print(f"AS weight in YAML: {as_weight_yaml}")
                    
                    # Confirm AS is in enabled losses
                    if 'AS' not in loss_config_yaml['ENABLED_LOSSES']:
                        print("WARNING: AS is configured but not in ENABLED_LOSSES!")
                        print("Adding AS to enabled losses")
                        loss_config_yaml['ENABLED_LOSSES'].append('AS')
                else:
                    print("WARNING: AS is not in YAML config!")
            
            for loss_name, loss_params in loss_config['params'].items():
                print(f"{loss_name} in YAML: {loss_config_yaml.get(loss_name, 'Not in YAML')}")
            
            # Debug print entire YAML config
            print("\nFull LOSS configuration:")
            print(yaml.dump(loss_config_yaml))
            
            # Save config file
            config_name = f"config_{self.model_type}_{dataset}_{loss_config['name']}.yaml"
            config_path = os.path.join(self.results_dir, config_name)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Debug verify saved config
            print(f"Config saved to: {config_path}")
            
            return config_path
            
        except Exception as e:
            print(f"Error modifying config: {e}")
            raise

    def run_experiment(self, dataset, loss_config, seed):
        """Run a single experiment with specified parameters."""
        try:
            # Get modified config
            config_path = self.modify_config(dataset, loss_config)
            
            # Setup experiment directory
            exp_name = f"{dataset}/shots_{self.fixed_params['shots']}/{self.trainer_name}/{loss_config['name']}"
            output_dir = os.path.join(self.results_dir, exp_name, f"seed{seed}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Check for completed experiment
            log_file = os.path.join(output_dir, "log.txt")
            if os.path.exists(log_file):
                print(f"Experiment exists: {dataset}, {loss_config['name']}, seed={seed}")
                return self.extract_metrics(log_file)
            
            # Prepare command
            cmd = [
                "python", "/home/abhishek/desktop/VLM_Cal/CalibPrompt/train.py",
                "--root", "/home/abhishek/desktop/VLM_Cal/CalibPrompt/DATA",
                "--seed", str(seed),
                "--trainer", self.trainer_name,
                "--dataset-config-file", f"/home/abhishek/desktop/VLM_Cal/CalibPrompt/configs/datasets/{dataset}.yaml",
                "--config-file", config_path,
                "--output-dir", output_dir,
                "DATASET.NUM_SHOTS", str(self.fixed_params['shots']),
                "DATASET.SUBSAMPLE_CLASSES", "all",
                "MODEL.NAME", self.model_type,
                "MODEL_ROOT", "/home/abhishek/desktop/VLM_Cal/CalibPrompt/models"
            ]
            
            # Print the full command for debugging
            print(f"\nRunning command: {' '.join(cmd)}")
            
            # Run experiment
            print(f"\nRunning: {dataset}, {loss_config['name']}, seed={seed}")
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
                    'macro_f1': r'\* macro_f1: ([\d.]+)%',
                    'ece_kde': r'\* ece_kde: ([\d.]+)%'
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
                
                seed_results = []
                for seed in self.seeds:
                    result = self.run_experiment(dataset, loss_config, seed)
                    
                    if result:
                        # Record individual seed results
                        result_row = {
                            'dataset': dataset,
                            'lr': self.fixed_params['lr'],
                            'epochs': self.fixed_params['epochs'],
                            'shots': self.fixed_params['shots'],
                            'losses': loss_config['name'],
                            'seed': seed,
                            'ece': result['ece'],
                            'ece_kde': result['ece_kde'],
                            'accuracy': result['accuracy'],
                            'mce': result['mce'],
                            'ace': result['ace'],
                            'macro_f1': result['macro_f1'],
                            'confidence': result.get('confidence', float('nan'))
                        }
                        seed_results.append(result_row)
                        dataset_results.append(result_row)
                
                # For single seed, just print the result directly without calculating mean/std
                if seed_results:
                    result = seed_results[0]  # Get the single result
                    
                    # Print result statistics
                    print(f"\nResult for {dataset} - {loss_config['name']}:")
                    print(f"Accuracy: {result['accuracy']:.2f}%")
                    print(f"ECE: {result['ece']:.2f}%")
                    print(f"MCE: {result['mce']:.2f}%")
                    print(f"ACE: {result['ace']:.2f}%")
                    print(f"Macro F1: {result['macro_f1']:.2f}%")
                
                # Immediately save results for this dataset
                results_df = pd.DataFrame(dataset_results)
                results_path = os.path.join(
                    self.results_dir, 
                    f'{dataset}_{self.model_type}_prompt_results.csv'
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
    parser = argparse.ArgumentParser(description='Run prompt learning experiments for medical images')
    parser.add_argument('--model', type=str, choices=['medclip', 'biomedclip'], required=True,
                      help='Model type to run experiments for')
    parser.add_argument('--dataset', type=str, choices=['rsna18', 'covid'],
                      help='Specific dataset to run experiments on')
    parser.add_argument('--gpu', type=int, default=1,
                      help='GPU ID to use for experiments (default: 1)')
    args = parser.parse_args()
    
    try:
        # Set the GPU to use
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        
        experiments = MedicalPromptLearningExperiments(args.model, args.dataset)
        print(f"Starting prompt learning experiments for {args.model}" + 
              (f" on {args.dataset}" if args.dataset else "") + 
              f" using GPU {args.gpu}")
        
        # Print information about the seeds being used
        print(f"Running experiments with seeds: {experiments.seeds}")
        
        results = experiments.run_experiments()
        print("\nExperiments completed!")
        
        # Print summary of all configurations by lowest ECE
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            print("\nAll configurations by lowest ECE:")
            for dataset in experiments.datasets:
                dataset_results = results_df[results_df['dataset'] == dataset].sort_values(by='ece')
                print(f"\n{dataset} Results (sorted by ECE):")
                for _, row in dataset_results.iterrows():
                    print(f"  {row['losses']}: Acc = {row['accuracy']:.2f}%, ECE = {row['ece']:.2f}%")
        
        experiments.cleanup()
    except Exception as e:
        print(f"Error in execution: {e}")

if __name__ == "__main__":
    main()