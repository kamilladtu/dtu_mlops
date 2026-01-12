import torch
from s1_development_environment.exercise_files.final_exercise.model import MyAwesomeModel

def test_training_forward_pass():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)

    assert y.shape[0] == 1, "Model output batch size should match input batch size"
