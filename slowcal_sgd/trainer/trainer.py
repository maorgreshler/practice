import os
import json
import csv
import torch
import torch.nn as nn
import torchmetrics
import wandb


class Trainer:
    """
    A class to handle the training and evaluation of a model with multiple workers in a distributed setup.
    Supports logging, checkpointing, and metrics tracking.
    """

    def __init__(self, model, test_dataloader, params, workers, device, experiment_name=None, num_classes=10):
        """
        Initializes the Trainer instance.

        Args:
            model (torch.nn.Module): The model to be trained.
            test_dataloader (torch.utils.data.DataLoader): Dataloader for the test dataset.
            params (argparse.Namespace): Parsed command-line arguments containing training parameters.
            workers (list[Worker]): List of worker instances for distributed training.
            device (torch.device): The device to run the training process.
            experiment_name (str, optional): Name of the experiment for logging and checkpointing.
            num_classes (int, optional): Number of classes for the dataset.
        """
        self.workers_num = len(workers)
        self.model = model
        self.device = device
        self.model = model.to(device)
        self.experiment_name = experiment_name
        self.test_dataloader = test_dataloader
        self.workers = workers
        self.wandb = wandb

        # Store training parameters and initialize paths
        self.params = params.__dict__
        self.param_shapes = [p.shape for p in model.parameters()]
        self.split_sizes = [torch.prod(torch.tensor(shape)).item() for shape in self.param_shapes]
        self.checkpoint_path = None
        self.log_file_path = None
        self.run_directory = "./"

        # Initialize metrics and logging
        self.use_wandb = self.params.get("use_wandb", False)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.best_accuracy = 0.0
        self.metrics_data = []
        self.iter_results = {
            "iteration": 0,
            "train_loss": 0,
            "test_loss": 0,
            "test_acc": 0,
            **self.params
        }

    def train(self):
        """
        Prepares the environment for training by creating result directories and saving parameters.
        """
        # Set up results directory
        results_dir = 'results' if self.experiment_name is None else f'results-{self.experiment_name}'
        os.makedirs(results_dir, exist_ok=True)

        # Create a new numbered run directory
        existing_folders = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        next_folder_num = len(existing_folders) + 1
        run_directory = os.path.join(results_dir, f'run{next_folder_num}')
        os.makedirs(run_directory)
        self.checkpoint_path = os.path.join(run_directory, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.log_file_path = os.path.join(run_directory, 'metrics_log.txt')
        self.run_directory = run_directory

        # Save parameters to a JSON file
        params_file_path = os.path.join(run_directory, 'params.json')
        with open(params_file_path, 'w') as params_file:
            json.dump(self.params, params_file, indent=4)

    def evaluate(self):
        """
        Evaluates the model on the test dataset and computes accuracy and loss.

        Returns:
            tuple: (final_accuracy, test_loss)
                final_accuracy (float): Accuracy of the model on the test dataset.
                test_loss (float): Average loss on the test dataset.
        """
        self.model.eval()
        self.accuracy_metric.reset()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                self.accuracy_metric.update(predictions, labels)

        final_accuracy = self.accuracy_metric.compute()
        test_loss = total_loss / len(self.test_dataloader)
        self.model.train()
        return final_accuracy, test_loss

    def make_evaluation_step(self, iteration, total_iterations, eval_interval, running_loss, metrics_table, iter_title):
        """
        Performs an evaluation step, logs metrics, and saves the model if the accuracy improves.

        Args:
            iteration (int): Current iteration number.
            total_iterations (int): Total number of iterations.
            eval_interval (int): Interval at which evaluations are performed.
            running_loss (float): Accumulated loss over the interval.
            metrics_table (prettytable.PrettyTable): Table for displaying metrics.
            iter_title (str): Title for the current iteration in the log.
        """
        train_accuracy = self.accuracy_metric.compute()
        average_loss = running_loss / eval_interval

        # Evaluate on test dataset
        test_accuracy, test_loss = self.evaluate()

        # Log metrics to the table
        metrics_table.add_row([
            f"{iteration + 1}/{total_iterations}", f"{average_loss:.4f}", f"{train_accuracy:.2%}",
            f"{test_loss:.4f}", f"{test_accuracy:.2%}"
        ])
        print(metrics_table)

        # Append metrics to the log file
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"{metrics_table.get_string()}\n\n")
        metrics_table.clear_rows()

        # Collect metrics for further analysis
        self.metrics_data.append({
            iter_title: iteration,
            "Train Loss": average_loss,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy.item()
        })

        # Log metrics to Weights & Biases if enabled
        if self.use_wandb:
            self.wandb.log({
                "Train Loss": average_loss,
                "Test Loss": test_loss,
                "Test Accuracy": test_accuracy
            })

        # Save model checkpoint if current accuracy is the best
        if test_accuracy > self.best_accuracy:
            self.best_accuracy = test_accuracy
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, "best.pth"))

        # Reset metrics for the next interval
        self.accuracy_metric.reset()

    def save_metrics_and_params(self):
        """
        Saves the collected metrics and parameters to a CSV file.
        """
        csv_file_path = os.path.join(self.run_directory, 'results.csv')
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        if not self.metrics_data:
            print("No metrics data to save.")
            return

        # Combine parameters with metrics and write to CSV
        combined_rows = [{**metrics, **self.params} for metrics in self.metrics_data]
        fieldnames = list(combined_rows[0].keys())

        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)

    def make_optimization_step(self, inputs, targets, first_step=False):
        """
        Performs a single optimization step during training.

        Args:
            inputs (torch.Tensor): Input data for the current step.
            targets (torch.Tensor): Ground truth labels for the current step.
            first_step (bool, optional): Flag indicating if this is the first step.
        """
        pass  # Implementation to be added
