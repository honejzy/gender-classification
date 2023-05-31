from abc import ABCMeta, abstractmethod
from torch import nn
from torch import Tensor
from .feature_extractor import FeatureExtractorCNN


class BaseModel(nn.Module, metaclass=ABCMeta):
    """
    Abstract NN model
    """
    def __init__(self, config: dict, task: str = 'inference'):
        super().__init__()
        # init feature extractor
        self.feature_extractor = FeatureExtractorCNN(
                                    config,
                                    task=task
        )

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (B, ?)
        Returns:
            (?)
        """
