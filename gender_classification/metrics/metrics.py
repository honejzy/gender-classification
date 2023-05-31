import os
import glob
import argparse
from typing import Tuple, Dict, Union
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, \
                            f1_score, \
                            recall_score, \
                            precision_score


def compute_metrics(preds_and_labels: Tuple[
                    Union[np.ndarray, torch.Tensor], 
                    Union[np.ndarray, torch.Tensor]
                    ]) -> Dict[str, float]:

    predictions, labels = preds_and_labels
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1_value = f1_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    precision = precision_score(labels, predictions, zero_division=0)

    return {"acc": acc, "f1": f1_value, "recall": recall, "prec": precision}

def compute_metrics_folders(gt_folder: str, 
                            pred_folder: str, 
                            print_miss: bool = False):

    gt_files = sorted(glob.glob(os.path.join(gt_folder, '*.csv')))
    pred_files = sorted(glob.glob(os.path.join(pred_folder, '*.csv')))
    y_true = []
    y_pred = []
    misclassified_samples = []
    
    for gt_file, pred_file in zip(gt_files, pred_files):
        df_gt = pd.read_csv(gt_file)
        df_pred = pd.read_csv(pred_file)

        gt_gender = df_gt['gender']
        pred_gender = df_pred['gender']

        # Remove samples with gender = 'other'
        mask = gt_gender != 'other'
        gt_gender = gt_gender[mask]
        pred_gender = pred_gender[mask]

        y_true.extend(gt_gender.tolist())
        y_pred.extend(pred_gender.tolist())

        # Keep track of misclassified samples
        path_ls = df_gt.loc[mask][gt_gender != pred_gender]['path'].tolist()
        misclassified_samples.extend(path_ls)
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', 
                                zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1_value = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 score: {f1_value:.4f}')

    # Print misclassified samples
    if len(misclassified_samples) > 0 and print_miss:
        print('\nMisclassified samples:')
        for sample in misclassified_samples:
            print(sample)
    print('Count of errors = ', len(misclassified_samples))


def main(args):
    gt_folder = args.gt_folder
    pred_folder = args.pred_folder
    compute_metrics_folders(gt_folder, pred_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='Metrics', 
            description='Metrics for gender classification task')
    parser.add_argument('--gt_folder', type=str,
                        default='./gt',
                        help='folder with ground truth csv files',
                        required=True)
    parser.add_argument('--pred_folder', type=str,
                        default='./pred',
                        help='folder for pred csv files',
                        required=True)
    args = parser.parse_args()
    main(args)
