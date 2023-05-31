from typing import Union
import torch.nn.functional as F
import torch
from torch import nn

class PadToMaxDuration(nn.Module):
    def __init__(self, max_duration: Union[int, float]):
        super().__init__()
        self.max_duration = max_duration

    def forward(self, x):
        raise NotImplementedError()

    def __call__(self, waveforms):
        padded_waveforms = []
        for waveform in waveforms:
            length = waveform.size(-1)
            if length >= self.max_duration:
                padded_waveforms.append(waveform[:, :self.max_duration])
            else:
                padding = self.max_duration - length
                padded_waveforms.append(F.pad(waveform, (0, padding)))
        return torch.stack(padded_waveforms)
