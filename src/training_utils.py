from typing import Tuple, Type
from jaxtyping import Float
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset as TorchDataset
import math
import matplotlib.pyplot as plt


MNIST_FEATURE_SIZE = 784


def convert_torch_dataset_to_tensor(
    dataset: Type[TorchDataset],
    flatten: bool = True,
) -> Tuple[Float[Tensor, "n f"], Float[Tensor, "n 1"]]:
    """Convert a torch dataset to a x and y tensors.

    Args:
        dataset: The torch dataset to convert.
        flatten: If the x tensor should be flattened. True if using MLP
    """
    if flatten:
        xs = torch.stack([x.flatten() for x, y in dataset])
    else:
        xs = torch.stack([x for x, y in dataset])

    ys = torch.Tensor([y for x, y in dataset])

    return (xs, ys)


def calculate_grid_size(num_images: int) -> Tuple[int, int]:
    """Helper for finding image sizes when debugging"""

    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    return rows, cols


def plot_cifar_image(images: Float[Tensor, "n 3 h w"]) -> None:
    """Helper for plotting images when debugging"""

    # Create a figure with subplots
    n = images.shape[0]
    n_cols, n_rows = calculate_grid_size(n)
    plt.figure(figsize=(n_cols * 2, n_rows * 2))

    for i in range(images.shape[0]):
        img = images[i].permute(1, 2, 0).detach().cpu().numpy()  # Transpose the image
        # Normalize the image if it's not in the range [0, 1]
        plt.subplot(n_rows, n_cols, i + 1)  # Create a subplot for each image
        plt.imshow(img)
        plt.axis("off")  # Turn off axis numbers and labels

    plt.tight_layout()
    plt.show()
