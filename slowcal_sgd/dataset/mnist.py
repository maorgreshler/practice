import torchvision
import torchvision.transforms as transforms


class MNIST:
    """
    A class for loading and preprocessing the MNIST dataset with standard transformations
    for training and testing.

    Attributes:
        trainset (torchvision.datasets.MNIST): The training dataset with applied transformations.
        testset (torchvision.datasets.MNIST): The test dataset with applied transformations.
    """

    def __init__(self):
        """
        Initializes the MNIST class by setting up data transformations and loading
        the MNIST training and test datasets.
        """
        # Define transformations for both training and test datasets
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        # Load the training dataset
        self.trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )

        # Load the test dataset
        self.testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
