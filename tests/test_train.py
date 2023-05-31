import pytest
from gender_classification.scripts.train import train


def test_train_file_not_found():
    with pytest.raises(FileNotFoundError):
        train(config="not/existing/path")
