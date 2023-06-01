<div align="center">    
 
# Gender Classification  
</div>
 
## Table of Contents

- [Description](#decription)
- [Architecture](#architecture)
- [Install](#install)
- [Train](#train)
- [Inference](#inference)
- [Data](#data)
- [Best models](models/README.md)
- [TODO](#todo)

## Description   
Gender classification uses russian speech signal in order to classify into 2 classes: **male** and **female**

## Architecture
Models:
CNN

## Install   
Firstly, clone project  
```bash
git clone https://github.com/honejzy/gender-classification
 ```   
Create docker image from Dockerfile
```bash
cd gender-classification/docker

docker build -t gender_class .
```  

## Train
Run training process:
```python
python3 -m gender_classification.scripts.train --config gender_classification/config/config.yaml
```

You can specify experiment folder name with argument: `--experiment`.
In the name of this folder will be saved best weights.
```python
python3 -m gender_classification.scripts.cnn_train --config gender_classification/config/config.yaml --experiment baseline
# experiments/results_cnn/baseline/train/001/best/best.pt
```

Also you can specify additional comment for your experiment using argument: `--comment`.
```python
python3 -m gender_classification.scripts.train --config gender_classification/config/config.yaml --experiment baseline --comment first
# experiments/results_cnn/baseline/train/001_first/best/best.pt
```

You can call one of managers for experiments using argument: `--clearml` or `--mlflow`.
```python
python3 -m gender_classification.scripts.train --config gender_classification/config/config.yaml --clearml
```

## Inference
```python
# gender classification for one audio file
python3 -m gender_classification.scripts.inference --config gender_classification/config/config.yaml --wav_path PATH/TO/AUDIO

# gender classification for folder with audio files
python3 -m gender_classification.scripts.inference --config gender_classification/config/config.yaml --wav_folder PATH/TO/AUDIO_FOLDER

# gender classification for csv file with audio pathes
# csv file for each audio file looks like:
# ch  start  end          path              gender
#  0  1.375  2.21875      /mnt/data/1.wav      
#  0  2.9375  10.03125    /mnt/data/1.wav      

python3 -m gender_classification.scripts.inference --config gender_classification/config/config.yaml --csv_folder PATH/TO/CSV_FOLDER
```

## Data
[Data format](docs/data.md) <br/>
For running data trasformation tasks use
```bash
python3 -m python3 -m gender_classification.data.make_dataset --task <TASK> --input <IN_MANIFEST_CSV> --output <OUT_MANIFEST_CSV>
```
See all available tasks using
```bash
python3 -m python3 -m gender_classification.data.make_dataset -h
```
To run data pipeline
```bash
dvc repro
```
See `dvc.yaml` for more detail <br/>
To push data into remote storage set credentials
```bash
dvc remote modify --local storage access_key_id <LOGIN>
dvc remote modify --local storage secret_access_key <PASSWORD>
```
>NOTE: to overcome error "Failed SSL validation" use `use_ssl = false` in `.dvc/config.local`
Then call 
```bash
dvc push
```

## TODO
*
*