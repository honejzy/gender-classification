import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseModel

class GenderClassificationModelCNN(BaseModel):
    def __init__(self, cfg, task: str = 'inference'):
        super().__init__(cfg, task)
        self.cfg = cfg
        self.num_conv_layers = cfg['model']['cnn']["num_conv_layers"]
        self.conv_layers = nn.ModuleList()
        input_size = self.compute_input_size()

        for i in range(self.num_conv_layers):
            in_channels = cfg['model']['cnn'][f"conv{i+1}"]["in_channels"]
            out_channels = cfg['model']['cnn'][f"conv{i+1}"]["out_channels"]
            kernel_size = cfg['model']['cnn'][f"conv{i+1}"]["kernel_size"]
            padding = cfg['model']['cnn'][f"conv{i+1}"]["padding"]

            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding
            )
            
            self.conv_layers.append(conv_layer)
        
        self.pool = nn.MaxPool2d(
            kernel_size=cfg['model']['cnn']["pool"]["kernel_size"],
            stride=cfg['model']['cnn']["pool"]["stride"]
        )
        size1 = input_size[0] // (2 ** self.num_conv_layers)
        size2 = input_size[1] // (2 ** self.num_conv_layers)
        self.fc1_inp = self.conv_layers[-1].out_channels * size1 * size2
        self.fc1 = nn.Linear(
            in_features=self.fc1_inp,
            out_features=cfg['model']['cnn']["fc1"]["out_features"]
        )
        self.fc2 = nn.Linear(
            in_features=cfg['model']['cnn']["fc2"]["in_features"],
            out_features=cfg['model']['cnn']["fc2"]["out_features"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        for i in range(self.num_conv_layers):
            x = self.conv_layers[i](x)
            x = F.relu(x)
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def compute_input_size(self):
        sample_rate = self.cfg['model']['sample_rate']
        max_len_sample = self.cfg['dataset']['max_len_sample']
        input_sample = sample_rate * max_len_sample
        feature_extractor = self.cfg['model']['cnn']['feature_extractor']

        if feature_extractor['name'] == 'spectrogram':
            n_fft = feature_extractor['spectrogram']['n_fft']
            hop_length = feature_extractor['spectrogram']['hop_length']
            input_size = (int(n_fft/2 + 1), (input_sample // hop_length) + 1)
        return input_size

    def load_best(self, weights_path: str):
        model_state_dict = torch.load(weights_path, map_location='cpu')
        self.load_state_dict(model_state_dict, strict=False)
