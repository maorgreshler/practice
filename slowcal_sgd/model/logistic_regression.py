import torch.nn as nn


class LogisticRegressionModel(nn.Module):
    """
    A simple logistic regression model for image classification.

    This model consists of a single linear layer that maps input features
    to output classes. It is designed for datasets where the input images
    are flattened into vectors of size 784 (e.g., 28x28 grayscale images like MNIST).

    Attributes:
        linear (nn.Linear): The linear layer for mapping input features to output classes.
    """

    def __init__(self):
        """
        Initializes the LogisticRegressionModel with a single linear layer.
        """
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=784, out_features=10)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 10), representing
                          class scores for each input.
        """
        # Flatten the input tensor to (batch_size, 784)
        x = x.view(x.size(0), -1)

        # Apply the linear layer
        out = self.linear(x)
        return out
