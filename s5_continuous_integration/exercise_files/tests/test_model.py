import torch
import pytest
from s1_development_environment.exercise_files.final_exercise.model import MyAwesomeModel

@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_model_output_shape_parametrized(batch_size):
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)

    assert y.shape == (batch_size, 10), (
        f"Expected output shape {(batch_size, 10)}, got {y.shape}"
    )

def test_model():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)


# opgave e

import pytest

def test_error_on_wrong_shape():
    model = MyAwesomeModel()

    with pytest.raises(ValueError, match="4D"):
        model(torch.randn(1, 28, 28))

    with pytest.raises(ValueError, match=r"\[1, 28, 28\]"):
        model(torch.randn(1, 1, 28, 29))
