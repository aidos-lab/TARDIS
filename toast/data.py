"""Provides samples of more complicated data sets."""

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader


def sample_vision_data_set(name, n_samples):
    """Sample vision data set.

    Parameters
    ----------
    name : str
        Name of the data set. Currently, only "MNIST" and "FashionMNIST"
        are supported here.

    n_samples : int
        Number of samples to retrieve.

    Returns
    -------
    np.array
        Sampled data points
    """
    assert name in ["MNIST", "FashionMNIST"]

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

    transform = transforms.ToTensor()

    if name == "MNIST":
        cls = datasets.MNIST
    elif name == "FashionMNIST":
        cls = datasets.FashionMNIST

    data = cls(root="../data", train=True, download=True, transform=transform)

    data_loader = DataLoader(dataset=data, batch_size=n_samples, shuffle=True)

    X, _ = next(iter(data_loader))
    X = X.reshape(n_samples, -1)
    X = X.numpy()

    return X
