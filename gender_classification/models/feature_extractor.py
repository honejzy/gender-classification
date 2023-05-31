import torch
from torch import nn
from torchaudio.transforms import (
    Spectrogram,
    FrequencyMasking,
    TimeMasking,
    TimeStretch
)

class FeatureExtractorCNN(nn.Module):
    def __init__(self, config: dict,
                 task: str = 'inference'):
        super().__init__()
        self.task = task
        self.config = config
        feature_extractor = self.config['model']['cnn']['feature_extractor']
        self.transform = self.get_transform(feature_extractor)
        augmentation = self.config['augmentation']
        self.augmentations = nn.Sequential() if task == 'inference' \
                                else self.get_augmentations(augmentation) 

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.transform(waveform) \
                   if self.task == 'train' \
                   else \
                   self.transform(waveform).unsqueeze(0)
        
        if self.config['augmentation']:
            for aug in self.augmentations:
                features = aug(features)

        return features

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'(transform): {self.transform},\n'
                f'(augmentations): {self.augmentations}\n'
                ')')

    def get_transform(self, config: dict) -> nn.Sequential:
        name = config["name"]
        try:
            params = config[name]
        except KeyError:
            raise ValueError(f"Invalid feature extractor '{name}'")

        if name == 'spectrogram':
            transform = Spectrogram(
                n_fft=params['n_fft'],
                hop_length=params['hop_length'],
                normalized=params['normalized'],
                power=params['power']
            )

        transform = nn.Sequential(
            transform
        )

        return transform

    def get_augmentations(self, config: dict) -> nn.Sequential:
        if not config['name']:
            return nn.Sequential()

        augmentations = nn.Sequential()
        
        for aug in config['name']:
            if aug == 'freq_mask':
                augmentations.append(FrequencyMasking(
                    freq_mask_param=config[aug]['freq_mask_param']
                ))

            if aug == 'time_mask':
                augmentations.append(TimeMasking(
                    time_mask_param=config[aug]['time_mask_param']
                ))

            if aug == 'time_stretch':
                augmentations.append(TimeStretch(
                    fixed_rate=config[aug]['time_stretch_rate']
                ))

        return augmentations
