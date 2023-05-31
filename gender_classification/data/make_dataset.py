"""
Functions for process dataset
Each step of pipeline represents transformation of manifest.csv
Steps are described in dvc.yaml
See manifest.csv format in docs/data.md
"""
import argparse
from typing import Optional
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

VALID_CLASSES = ["male", "female"]
NECESSARY_COLUMNS = ["path", "gender"]
CSV_SEP = ','


def concatenate_data(manifest_in: str, manifest_out: str, separator=CSV_SEP):
    """ Concatenate necessary manifests in one with only useful columns

    Args:
        manifest_in (str): list of path to manifests divided by whitespace
        manifest_out (str): path to output manifest
        separator (str): separator in csv
    """
    dataframes = []
    for manifest in manifest_in.split():
        dataframe = pd.read_csv(manifest, sep='\t')
        print(manifest)
        dataframes.append(dataframe)
    result_dataframe = pd.concat(dataframes)
    result_dataframe.to_csv(manifest_out, sep=separator,
                            index=False, header=True, columns=NECESSARY_COLUMNS)


def drop_duplicates(manifest_in: str, 
                    manifest_out: Optional[str] = None):
    """
    Remove duplicates of audios
    Args:
        manifest_in (str): path/to/input/manifest.csv
        manifest_out (str, None): path/to/output/manifest.csv
            If None, 'path/to/input/manifest.dropped-duplicates.csv'
    """
    dataframe = pd.read_csv(manifest_in, sep=CSV_SEP)

    dataframe.drop_duplicates(subset=['path'], inplace=True)

    if not manifest_out:
        manifest_out = manifest_in.replace('.csv',
                                           '.dropped-duplicates.csv')
    dataframe.to_csv(manifest_out, sep=CSV_SEP, 
                     index=False, header=True)


def drop_invalid_classes(manifest_in: str, 
                         manifest_out: Optional[str] = None):
    """
    Remove duplicates of audios
    Args:
        manifest_in (str): path/to/input/manifest.csv
        manifest_out (str, None): path/to/output/manifest.csv
            If None, 'path/to/input/manifest.valid-classes.csv'
    """
    dataframe = pd.read_csv(manifest_in, sep=CSV_SEP)

    dataframe = dataframe[dataframe['gender'].isin(VALID_CLASSES)]

    if not manifest_out:
        manifest_out = manifest_in.replace('.csv',
                                           '.valid-classes.csv')
    dataframe.to_csv(manifest_out, sep=CSV_SEP, 
                     index=False, header=True)

def delete_short_audio(manifest_in: str, 
                       manifest_out: Optional[str] = None):
    """Delete rows from manifest_in with audio less than 1 sec.

    Args:
        manifest_in (str): path/to/interim/manifest.csv
        manifest_out (str, None): path/to/interim/manifest.csv
            If None, 'path/to/input/manifest.valid-classes-dur.csv'
    """
    dataframe = pd.read_csv(manifest_in, sep=CSV_SEP)
    durations = []
    for audio_file in dataframe['audio_path']:
        sample, sample_rate = librosa.load(audio_file, sr=None)
        duration = librosa.get_duration(y=sample, sr=sample_rate)
        durations.append(duration)
    dataframe = dataframe.loc[durations >= 1.0]  # type: ignore

    if not manifest_out:
        manifest_out = manifest_in.replace('.csv',
                                           '.valid-classes-dur.csv')
    dataframe.to_csv(manifest_out, sep=CSV_SEP, 
                     index=False, header=True)

def split_to_train_test(manifest_in: str,
                        manifest_out: str):
    """Split manifest_in to train.csv and test.csv taking into 
    account the balance of classes and in the ratio of 80% to 20%.
    Save to processed folder
    Args:
        manifest_in (str): path/to/input/manifest.csv
        manifest_out (str):  path to output manifest (processed folder)
    """
    data = pd.read_csv(manifest_in)
    
    male_count = len(data[data['gender'] == 'male'])
    female_count = len(data[data['gender'] == 'female'])
    
    # Calculate minimum count for balanced classes
    min_count = min(male_count, female_count)
    target_count = 2 * min_count
    
    male_rows = data[data['gender'] == 'male'].sample(n=target_count//2, 
                                                        random_state=42)
    female_rows = data[data['gender'] == 'female'].sample(n=target_count//2, 
                                                            random_state=42)
    
    # Concatenate and shuffle rows
    selected_rows = pd.concat([male_rows, female_rows])
    selected_rows = selected_rows.sample(frac=1, random_state=42)
    
    train, test = train_test_split(selected_rows, test_size=0.2, 
                                   stratify=selected_rows['gender'], 
                                   random_state=42)

    train_out, test_out = manifest_out.split()

    train.to_csv(train_out, 
                 index=False, 
                 header=['audio_path', 'gender'])
    test.to_csv(test_out, 
                index=False, 
                header=['audio_path', 'gender'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation')

    tasks = {
        "concatenate_data": concatenate_data,
        "drop_duplicates": drop_duplicates,
        "drop_invalid_classes": drop_invalid_classes,
        "delete_short_audio": delete_short_audio,
        "split_to_train_test": split_to_train_test
    }

    parser.add_argument('--task', '-t', type=str,
                        choices=tasks.keys(),
                        help='what task of pipeline to run',
                        required=True)
    parser.add_argument('--input', '-i', type=str,
                        required=True,
                        help='path/to/input/manifest.csv')
    parser.add_argument('--output', '-o', type=str,
                        default=None,
                        help='path/to/output/manifest.csv')
    args = parser.parse_args()

    task_name = args.task
    manifest_in = args.input
    manifest_out = args.output

    tasks[task_name](manifest_in, manifest_out)  # type: ignore
