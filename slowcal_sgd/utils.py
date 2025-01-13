import torch
import numpy as np
import random
import inspect
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
import seaborn as sns


def split_dataset(dataset, num_splits, batch_size, seed, shuffle=True, dirichlet_alpha=None, plot_distribution=False):
    """
    Splits a dataset into `num_splits` subsets, ensuring class distribution is preserved using stratified k-fold splitting.
    Optionally, controls data heterogeneity via a Dirichlet distribution and provides an option to visualize class distribution.

    Args:
        dataset (torch.utils.data.Dataset): The input dataset to be split.
        num_splits (int): The number of subsets to generate.
        batch_size (int): Batch size to be used for the resulting DataLoader instances.
        seed (int): Seed value for reproducibility of random operations.
        shuffle (bool): Whether to shuffle data within each subset. Defaults to True.
        dirichlet_alpha (float, optional): Parameter controlling heterogeneity when using a Dirichlet distribution. Defaults to None.
        plot_distribution (bool, optional): Flag to indicate whether to plot the class distribution across subsets. Defaults to False.

    Returns:
        list[torch.utils.data.DataLoader]: A list containing DataLoader instances for each generated subset.
    """
    set_seed(seed)

    # Convert dataset targets to a NumPy array
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    total_samples = len(dataset)

    if dirichlet_alpha is not None:
        # Create class-wise indices for Dirichlet-based splitting
        indices_per_class = {c: np.where(targets == c)[0] for c in range(num_classes)}
        subset_indices = [[] for _ in range(num_splits)]

        # Generate Dirichlet proportions for each class
        proportions = np.random.dirichlet([dirichlet_alpha] * num_splits, num_classes)

        # Initialize ordered proportions and calculate the threshold
        ordered_proportions = np.zeros_like(proportions)
        proportions_threshold = num_classes / num_splits
        samples_per_split = total_samples // num_splits

        # Assign proportions ensuring subset balance
        for c in range(num_classes):
            for i, p in enumerate(proportions[c]):
                assigned = False
                for j, op in enumerate(ordered_proportions[c]):
                    if ((ordered_proportions[:, j].sum() + p < proportions_threshold
                         or ordered_proportions[:, j].sum() == 0) and op == 0):
                        ordered_proportions[c, j] = p
                        assigned = True
                        break
                if not assigned:
                    indices = np.where(ordered_proportions[c] == 0)
                    min_idx = np.argmin(ordered_proportions[:, indices[0]].sum(axis=0))
                    ordered_proportions[c, [indices[0][min_idx]]] = p

        # Normalize proportions to ensure consistent subset sizes
        classes_amounts = ordered_proportions * total_samples // num_classes

        for c, indices in indices_per_class.items():
            random_indices = np.random.permutation(indices)
            current_idx = 0
            split_sizes = classes_amounts[c].astype(int)

            for i, size in enumerate(split_sizes):
                subset_indices[i].extend(random_indices[current_idx:current_idx + size])
                current_idx += size

        subsets = [Subset(dataset, indices) for indices in subset_indices]
    else:
        # Use StratifiedKFold for class-balanced splits
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
        subsets = [Subset(dataset, indices) for _, indices in skf.split(np.zeros(len(targets)), targets)]

    if plot_distribution:
        plot_class_distribution(subsets, num_classes)

    # Create DataLoader instances for each subset
    dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=shuffle) for subset in subsets]

    return dataloaders


def plot_class_distribution(subsets, num_classes, palette=None):
    """
    Plots the class distribution across subsets as a scatter plot.
    The size of each scatter point represents the number of samples, and the y-axis represents the class ID.

    Args:
        subsets (list[torch.utils.data.Subset]): List of dataset subsets.
        num_classes (int): Number of unique classes in the dataset.
        palette (list or None, optional): List of colors to use for plotting. If None, a color palette is generated dynamically.
    """
    class_counts = np.zeros((len(subsets), num_classes))

    for i, subset in enumerate(subsets):
        targets = np.array([subset.dataset.targets[idx] for idx in subset.indices])
        counts = np.bincount(targets, minlength=num_classes)
        class_counts[i] = counts

    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate a dynamic palette if not provided
    if palette is None:
        palette = sns.color_palette('tab10', num_classes)  # Use tab10 for up to 10 distinct colors

    # Plot each class with its corresponding color
    for c in range(num_classes):
        ax.scatter(
            np.arange(len(subsets)),
            [c] * len(subsets),
            s=class_counts[:, c],  # Scale size by number of samples
            color=palette[c % len(palette)],  # Cycle through palette if needed
            alpha=0.7
        )

    ax.set_xlabel('Worker ID')
    ax.set_ylabel('Class ID')
    plt.tight_layout()
    plt.savefig(f'class_distribution_{len(subsets)}.png')
    plt.show()



def set_seed(seed):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to be set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_valid_args(object_class, **kwargs):
    """
    Filters keyword arguments to only include those valid for the specified object's initializer.

    Args:
        object_class (type): The class whose initializer signature is used for filtering.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        dict: A dictionary containing only valid arguments for the specified object's initializer.
    """
    init_signature = inspect.signature(object_class.__init__)
    valid_params = set(init_signature.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return filtered_kwargs


def get_device():
    """
    Returns the appropriate computing device (CPU, CUDA, or MPS) based on availability.

    Returns:
        torch.device: The available computing device.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
