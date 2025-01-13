import subprocess

# Define parameter ranges
seeds = [1, 2, 3]
batch_sizes = [1]
worker_counts = [16, 32, 64]
local_iteration_nums = [4, 8, 16, 32, 64]
optimizers = ['SLowcalSGD', 'LocalSGD', 'MinibatchSGD']
learning_rates = {'SLowcalSGD': 0.01, 'LocalSGD': 0.01, 'MinibatchSGD': 0.1}

# Define default parameters
config_folder_path = './config'
dataset = 'mnist'
model = 'logistic_regression'
num_epochs = 1
eval_interval = 1
use_alpha_t = True
use_wandb = False
weight_decay = 0.0
experiment_name = 'SLowcalSGD Experiment'

# Iterate over parameter combinations and execute commands
for seed in seeds:
    for batch_size in batch_sizes:
        for worker_count in worker_counts:
            for local_iteration_num in local_iteration_nums:
                for optimizer in optimizers:
                    # Construct the command
                    command = [
                        'python', 'main.py',
                        '--workers_num', str(worker_count),
                        '--config_folder_path', config_folder_path,
                        '--dataset', dataset,
                        '--model', model,
                        '--epoch_num', str(num_epochs),
                        '--eval_interval', str(eval_interval),
                        '--local_iterations_num', str(local_iteration_num),
                        '--optimizer', optimizer,
                        '--learning_rate', str(learning_rates[optimizer]),
                        '--batch_size', str(batch_size),
                        '--seed', str(seed),
                        '--weight_decay', str(weight_decay),
                        '--dirichlet_alpha', '0.1',
                        '--experiment_name', experiment_name
                    ]

                    # Add optional flags
                    if use_alpha_t:
                        command.append('--use_alpha_t')
                    if use_wandb:
                        command.append('--use_wandb')

                    # Execute the command
                    subprocess.run(command, check=True)
