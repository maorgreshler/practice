import os
import math
import torch
import yaml
from tqdm import tqdm
from prettytable import PrettyTable
from .trainer import Trainer


class SyncTrainer(Trainer):
    """
    A class for synchronous distributed training, inheriting from the base Trainer class.
    Supports logging, checkpointing, and parameter synchronization across multiple workers.
    """

    def __init__(self, model, test_dataloader, params, workers, device, trainset_length,
                 experiment_name=None):
        """
        Initializes the SyncTrainer instance.

        Args:
            model (torch.nn.Module): The model to be trained.
            test_dataloader (torch.utils.data.DataLoader): Dataloader for the test dataset.
            params (argparse.Namespace): Training parameters.
            workers (list[Worker]): List of worker instances.
            device (torch.device): Device to run training on (CPU or GPU).
            experiment_name (str, optional): Name of the experiment for logging.
        """
        super().__init__(model, test_dataloader, params, workers, device, experiment_name)
        self.R = math.ceil(trainset_length / self.params['workers_num'] / self.params['local_iterations_num'])
        self.local_iterations = self.params['local_iterations_num'] if self.params['optimizer'] != "MinibatchSGD" else 1

        # Initialize Weights & Biases (wandb) if enabled
        if self.params["use_wandb"]:
            with open(os.path.join(params.config_folder_path, "wandb.yaml"), 'r') as file:
                self.wandb_conf = yaml.safe_load(file)
            project = self.wandb_conf["project"] if experiment_name is None else experiment_name
            self.wandb.init(
                project=project,
                entity=self.wandb_conf["entity"],
                name=(
                    f"{self.params['optimizer']}--{self.params['dataset']}--{self.params['model']}"
                    f"--LR: {self.params['learning_rate']}--Seed: {self.params['seed']}"
                    f"--Batch: {self.params['batch_size']}--Workers: {self.params['workers_num']}"
                )
            )
            self.wandb.config.update(self.params)

    def train(self, epoch_num: int = 100, eval_interval: int = 10):
        """
        Trains the model for a specified number of epochs with periodic evaluation.

        Args:
            epoch_num (int): Total number of epochs to train.
            eval_interval (int): Interval (in iterations) at which the model is evaluated.
        """
        super().train()
        self.model.train()
        self.accuracy_metric.reset()

        metrics_table = PrettyTable()
        metrics_table.field_names = ["Local Iteration", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"]

        for epoch in range(epoch_num):
            for r in tqdm(range(self.R), desc=f"Epoch {epoch + 1}/{epoch_num}"):
                running_loss = 0.0
                for _ in range(self.local_iterations):
                    for worker in self.workers:
                        loss = worker.step()
                        running_loss += loss

                self.update_new_parameters()

                if (r * epoch + r) % eval_interval == 0:
                    self.make_evaluation_step(
                        iteration=r * epoch + r,
                        total_iterations=self.R * epoch_num,
                        eval_interval=self.local_iterations * len(self.workers),
                        running_loss=running_loss,
                        metrics_table=metrics_table,
                        iter_title="Local Iteration"
                    )

        self.save_metrics_and_params()
        print('Finished Training')

    def update_new_parameters(self):
        """
        Abstract method to update model parameters.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_workers_model_parameters(self):
        """
        Gathers and averages model parameters from all workers.

        Returns:
            dict: A dictionary containing the averaged model parameters.
        """
        param_sum = None

        for worker in self.workers:
            worker_params = worker.model.state_dict()
            if param_sum is None:
                param_sum = {key: torch.zeros_like(param) for key, param in worker_params.items()}
            for key in param_sum:
                param_sum[key] += worker_params[key]

        num_workers = len(self.workers)
        avg_params = {key: param / num_workers for key, param in param_sum.items()}
        return avg_params

    def update_global_model(self):
        """
        Updates the global model with the averaged parameters from all workers.
        """
        model_parameters = self.get_workers_model_parameters()
        self.model.load_state_dict(model_parameters)

    def update_workers_model(self):
        """
        Updates all workers' models with the global model parameters.
        """
        global_model_params = self.model.state_dict()
        for worker in self.workers:
            worker.model.load_state_dict(global_model_params)


class LocalSGD(SyncTrainer):
    """
    Implements Local SGD for synchronous distributed training.
    """

    def __init__(self, model, test_dataloader, params, workers, device, trainset_length, experiment_name=None):
        super().__init__(model, test_dataloader, params, workers, device, trainset_length, experiment_name)

    def update_new_parameters(self):
        """
        Updates the global model by averaging parameters across workers
        and synchronizes all workers with the global model.
        """
        self.update_global_model()
        self.update_workers_model()


class SLowcalSGD(SyncTrainer):
    """
    Implements SLowcal SGD for synchronous distributed training,
    """

    def __init__(self, model, test_dataloader, params, workers, device, trainset_length, experiment_name=None):
        super().__init__(model, test_dataloader, params, workers, device, trainset_length, experiment_name)

    def get_w_parameters(self):
        """
        Gathers and averages optimizer-specific parameters from all workers.

        Returns:
            list[dict]: A list of averaged optimizer parameters.
        """
        w_param_sum = None

        for worker in self.workers:
            worker_w_params = worker.optimizer.w
            if w_param_sum is None:
                w_param_sum = [
                    {key: [torch.zeros_like(p) for p in param] if key == "params" else 0.0 for key, param in
                     group.items()}
                    for group in worker_w_params
                ]

            for i, group in enumerate(w_param_sum):
                for key in group:
                    if key == "params":
                        for j in range(len(group[key])):
                            group[key][j].add_(worker_w_params[i][key][j])
                    else:
                        group[key] += worker_w_params[i][key]

        num_workers = len(self.workers)
        avg_w_params = [
            {key: [p.div_(num_workers) for p in param] if key == "params" else param / num_workers for key, param in
             group.items()}
            for group in w_param_sum
        ]
        return avg_w_params

    def update_new_parameters(self):
        """
        Updates the global model and optimizer parameters by averaging them across all workers,
        and synchronizes all workers with the updated values.
        """
        self.update_global_model()
        self.update_workers_model()

        # Update workers' optimizers
        w_parameters = self.get_w_parameters()
        for worker in self.workers:
            for t, param_group in enumerate(worker.optimizer.w):
                for param_name, param in param_group.items():
                    if param_name == "params":
                        for j in range(len(param)):
                            worker.optimizer.w[t][param_name][j].data.copy_(w_parameters[t][param_name][j])
