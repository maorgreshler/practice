import argparse
from slowcal_sgd.dataset import DATASET_REGISTRY
from slowcal_sgd.model import MODEL_REGISTRY
from slowcal_sgd.optimizer import OPTIMIZER_REGISTRY
from torch.utils.data import DataLoader
from slowcal_sgd.utils import set_seed, get_device, split_dataset
from slowcal_sgd.worker import Worker
from slowcal_sgd.trainer import TRAINER_REGISTRY
import json

# first first comment 
def parse_arguments():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Training script for synchronous Byzantine machine learning.")

    # Argument definitions
    parser.add_argument('--workers_num', type=int, default=16,
                        help='Number of workers used for training.')
    parser.add_argument('--config_folder_path', type=str, default='./config',
                        help='Path to the configuration folder.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'],
                        help='Dataset to be used.')
    parser.add_argument('--model', type=str, default='logistic_regression', choices=MODEL_REGISTRY.keys(),
                        help='Model architecture to be used.')
    parser.add_argument('--epoch_num', type=int, default=1, help='Number of epochs for training.')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Interval (in epochs) for evaluation.')
    parser.add_argument('--local_iterations_num', type=int, default=64,
                        help='Number of local iterations per worker.')
    parser.add_argument('--optimizer', type=str, default='SLowcalSGD',
                        choices=['LocalSGD', 'SLowcalSGD', 'MinibatchSGD'],
                        help='Optimizer to be used for training.')
    parser.add_argument('--learning_rate', type=float,
                        default=0.1, help='Learning rate for the optimizer.')
    parser.add_argument('--use_alpha_t', action='store_true',
                        help='Enable use of alpha_t=t in the optimizer.')
    parser.add_argument('--query_point_momentum', type=float, default=0.1,
                        help='Fixed momentum for the query point if alpha_t is not used.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--seed', type=int, default=3, help='Random seed for reproducibility.')
    parser.add_argument('--use_wandb', action='store_true', help='Enable logging with Weights & Biases.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer.')
    parser.add_argument('--experiment_name', type=str,
                        help='Name of the experiment for logging and identification.')
    parser.add_argument('--dirichlet_alpha', type=float, default=None,
                        help='Alpha parameter for Dirichlet distribution to control data heterogeneity among workers. '
                             'If set, data will be sampled non-uniformly across workers based on this parameter.')

    return parser.parse_args()


def main():
    """Main function for initializing and running the training process."""
    args = parse_arguments()
    set_seed(args.seed)

    device = get_device()
    model = MODEL_REGISTRY[args.model]().to(device)
    # Load dataset and prepare dataloaders
    dataset = DATASET_REGISTRY[args.dataset]()
    test_dataloader = DataLoader(dataset.testset, batch_size=args.batch_size * args.workers_num, shuffle=False)
    if args.optimizer in ["LocalSGD", "SLowcalSGD"]:
        train_dataloaders = split_dataset(dataset=dataset.trainset,
                                          num_splits=args.workers_num,
                                          batch_size=args.batch_size,
                                          seed=args.seed,
                                          dirichlet_alpha=args.dirichlet_alpha,
                                          plot_distribution=True,
                                          shuffle=True)
    else:  # MinibatchSGD
        train_dataloaders = split_dataset(dataset=dataset.trainset,
                                          num_splits=args.workers_num,
                                          batch_size=args.batch_size * args.local_iterations_num,
                                          seed=args.seed,
                                          dirichlet_alpha=args.dirichlet_alpha,
                                          plot_distribution=True,
                                          shuffle=True)

    # Configure optimizer and parameters
    if args.optimizer in ["LocalSGD", "MinibatchSGD"]:
        optimizer = OPTIMIZER_REGISTRY["sgd"]
        optimizer_params = {
            "lr": args.learning_rate,
            "momentum": 0.0,
            "weight_decay": args.weight_decay
        }
    else:
        optimizer = OPTIMIZER_REGISTRY["anytime_sgd"]
        optimizer_params = {
            "lr": args.learning_rate,
            "gamma": args.query_point_momentum,
            "use_alpha_t": args.use_alpha_t,
            "weight_decay": args.weight_decay
        }

    # Initialize workers
    workers = []
    for i in range(args.workers_num):
        worker_model = MODEL_REGISTRY[args.model]().to(device)
        worker_optimizer = optimizer(worker_model.parameters(), **optimizer_params)
        workers.append(Worker(worker_optimizer, train_dataloaders[i], worker_model, device))

    # Initialize trainer
    trainer = TRAINER_REGISTRY[args.optimizer](model, test_dataloader, args, workers,
                                               device, trainset_length=len(dataset.trainset),
                                               experiment_name=args.experiment_name)

    # Start training
    trainer.train(epoch_num=args.epoch_num, eval_interval=args.eval_interval)


if __name__ == "__main__":
    main()
