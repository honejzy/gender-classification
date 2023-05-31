import logging
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class GenderDatasetDefault(Dataset):
    def __init__(self, csv_file: str, sample_rate: int):
        self.data = pd.read_csv(csv_file)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.loc[idx, 'audio_path']
        waveform, sample_rate = torchaudio.load(audio_path)
        transform = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
        waveform = transform(waveform)

        label = self.data.loc[idx, 'gender']
        if label == 'male':
            label = 0
        else:
            label = 1
        return waveform, label
