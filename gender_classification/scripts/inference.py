from os import path as osp
import csv
import logging
import argparse
from typing import Union, Optional
from glob import glob
import torch
import torchaudio
from torchaudio.transforms import Resample
from gender_classification.models.cnn import \
    GenderClassificationModelCNN
from ..utils import utils
from ..utils.padding import PadToMaxDuration

DEFAULT_CONFIG = 'gender_classification/config/config.yaml'

logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class GenderClassificationInference:
    def __init__(self, config: Optional[Union[dict, str]] = None):
        """ Initialization

        Args:
            config (dict, str, None): 
                - config dict
                - path/to/config.yaml
                - If None, load DEFAULT_CONFIG
        """
        if not config:
            config = DEFAULT_CONFIG
        if isinstance(config, str):
            # load from yaml
            config = utils.config_from_yaml(DEFAULT_CONFIG)
        
        self.pad = PadToMaxDuration(config['dataset']['max_len_sample'] * \
                                    config['model']['sample_rate'])
        self.device = torch.device('cuda' if torch.cuda.is_available() \
                                   else 'cpu')
        self.sample_rate = config['model']['sample_rate']
        self.classes = config["classes"]
        self.model = GenderClassificationModelCNN(config, 
                                                  task='inference')
        weights = config['model']['cnn']['best_model']
        if weights:
            # it's able to skip loading weights (for utittests)
            self.model.load_best(weights)
        self.model.eval()

    def preprocess(self, wav_path: str) -> torch.Tensor:
        if not osp.exists(wav_path):
            raise FileNotFoundError(f"File '{wav_path}' does not exist.")
        
        if not wav_path.endswith(('.wav', '.mp3')):
            raise ValueError(f"Unsupported file format for '{wav_path}'. \
                             Only .wav and .mp3 are supported.")
        
        duration = torchaudio.info(wav_path).num_frames / \
                    torchaudio.info(wav_path).sample_rate
        if duration < 1.0 or duration > 15.0:
            raise ValueError(f"File duration is less than 1 second or \
                             longer than 15 second for '{wav_path}'.")

        waveform, sample_rate = torchaudio.load(wav_path)
        resample = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
        waveform_resample = resample(waveform)
        waveform = self.pad([waveform_resample]).squeeze(0)
        return waveform

    def inference(self, input: str) -> Union[str, dict, list]:
        """
        Args:
            input (str): 
            - path/to/audio.wav - single audio
            - path/to/*.wav - multiple audios
            - path/to/*.csv - folder with csv
        Returns:
            str: "male" or "female" for single audio
            dict: {"<wavpath>: <gender>} for multiple audios
            list: list of overwritten csv files
        """
        files = glob(input)
        if not files:
            raise FileNotFoundError(f"Not files for '{input}'")
        if input.endswith('.wav'):
            output = {}
            for wavpath in files:
                prediction = self.classification(wavpath)
                output[wavpath] = prediction
                print(f'Prediction for {wavpath}: {prediction}')
            return output if '*' in input else prediction
        
        for csv_file in files:
            self.inference_csv(csv_file)
        return files

    def postprocess(self, output: torch.Tensor) -> str:
        label = torch.argmax(output, dim=1).item()
        return self.classes[label]

    def classification(self, path: str) -> str:
        spectrogram = self.preprocess(path)
        output = self.model(spectrogram)
        prediction = self.postprocess(output)
        return prediction
    
    def inference_csv(self, csv_path: str):
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            for i in range(1, len(rows)):
                row = rows[i]
                _, _, _, path, _ = row
                audio_path = path.replace('stt', 'gender')
                prediction = self.classification(audio_path)
                row[-1] = prediction
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)

def main(args):
    if isinstance(args.config, str):
        config = utils.config_from_yaml(args.config)

    if args.wav_path:
        input = args.wav_path
    elif args.wav_folder:
        input = osp.join(args.csv_folder, '*.wav')
    else:
        input = osp.join(args.csv_folder, '*.csv')

    gender = GenderClassificationInference(config)
    gender.inference(input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Inference', 
        description='Inference for gender classification task'
    )
    parser.add_argument('--config', type=str,
                        default='gender_classification/config/config.yaml',
                        help='config with differents parameters')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--wav_path', type=str, 
                       help='Path to the input WAV file')
    group.add_argument('--wav_folder', type=str, 
                       help='Path to the input WAVs folder')
    group.add_argument('--csv_folder', type=str, required=False,
                       help='path to folder with csv files')
    args = parser.parse_args()
    main(args)
