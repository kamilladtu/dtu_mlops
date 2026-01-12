import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1:] != (1, 28, 28):
            raise ValueError("Expected each sample to have shape [1, 28, 28]")

        x = torch.flatten(x, 1)   # (B, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)           # (B, 10)
        return x
