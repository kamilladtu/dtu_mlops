import os
import pytest
import torch

from s1_development_environment.exercise_files.final_exercise.data import corrupt_mnist
from s5_continuous_integration.exercise_files.tests import _PATH_DATA


skip_if_no_data = pytest.mark.skipif(
    not os.path.exists(_PATH_DATA),
    reason="Data files not found",
)


@skip_if_no_data
def test_data_sanity():
    assert os.path.exists(_PATH_DATA), "Data directory should exist"


@skip_if_no_data
def test_data():
    train, test = corrupt_mnist()

    assert len(train) > 0, "Train set was empty"
    assert len(test) > 0, "Test set was empty"
    assert len(train) > len(test), "Train set should be larger than test set"

    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "Each x should be [1, 28, 28]"
            assert y in range(10), "Labels should be in 0..9"

    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all(), "Train should contain all labels 0..9"

    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0, 10)).all(), "Test should contain all labels 0..9"
