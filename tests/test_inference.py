import unittest
import yaml
from gender_classification.scripts.inference \
    import GenderClassificationInference

CONFIG_YAML = 'gender_classification/config/config.yaml'


class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # load config
        with open(CONFIG_YAML) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        # skip loading weights
        config['model']['cnn']['best_model'] = None
        cls.instance = GenderClassificationInference(config)

    def test_correct_wavpath(self):
        """
        Test correct wavpath
        """
        wavpath = 'data/wavs/example.wav'
        gender = self.__class__.instance.inference(wavpath)
        self.assertTrue(gender in ["male", "female"])

    def test_correct_wavs(self):
        """
        Test correct wavpath
        """
        wavs = 'data/wavs/example*.wav'
        output = self.__class__.instance.inference(wavs)
        self.assertTrue(isinstance(output, dict))
        self.assertEqual(len(output), 1)
        wavpath = list(output.keys())[0]
        self.assertEqual(wavpath, 'data/wavs/example.wav')
        gender = list(output.values())[0]
        self.assertTrue(gender in ["male", "female"])


if __name__ == "__main__":
    unittest.main()
