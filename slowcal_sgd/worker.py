import torch.nn as nn


class Worker:
    """
    Represents a training worker responsible for performing local updates
    using a given model, optimizer, and dataloader.
    """

    def __init__(self, optimizer, dataloader, model, device):
        """
        Initializes the Worker instance.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer used for model updates.
            dataloader (torch.utils.data.DataLoader): Dataloader providing training data.
            model (torch.nn.Module): Neural network model to be trained.
            device (torch.device): Device on which the model and data will be loaded.
        """
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.data_iterator = iter(self.dataloader)  # Initialize the data iterator

    def step(self):
        """
        Performs a single optimization step for the worker.

        Returns:
            float: The loss value for the current step.
        """
        try:
            # Fetch the next batch of data
            images, labels = next(self.data_iterator)
        except StopIteration:
            # Reinitialize the data iterator if the dataset is exhausted
            self.data_iterator = iter(self.dataloader)
            images, labels = next(self.data_iterator)

        # Move data to the specified device
        images, labels = images.to(self.device), labels.to(self.device)

        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
