import os
from os.path import join, exists
from typing import Union, Tuple, Dict, Optional
import logging
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from ..metrics import compute_metrics
from ..data.dataset import GenderDatasetDefault
from ..models.cnn import GenderClassificationModelCNN
from ..utils.manager import MLFlowManager, ClearMLManager
from ..utils import utils
from ..utils.padding import PadToMaxDuration


MANAGER = None

def status_handler(func):
    def run_train(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except (Exception, KeyboardInterrupt) as error:
            # train is failed
            if MANAGER is not None:
                MANAGER.set_status("FAILED")
            raise error
        # train is successfull
        if MANAGER is not None:
            MANAGER.set_status("FINISHED")
    return run_train

def evaluate_step(model: nn.Module, 
                  dataloader: DataLoader, 
                  criterion: nn.Module, 
                  device: torch.device = torch.device("cpu")
                  ) -> Tuple[float, Dict[str, float]]:

    model.eval()
    running_loss = 0.0
    acc = f1_value = recall = precision = 0.0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            metrics = compute_metrics((outputs.to('cpu'), labels.to('cpu')))
            acc += metrics['acc']
            f1_value += metrics['f1']
            recall += metrics['recall']
            precision += metrics['prec']

    return running_loss / len(dataloader), {
        "acc": acc * 100 / len(dataloader), 
        "f1": f1_value * 100 / len(dataloader), 
        "recall": recall * 100 / len(dataloader), 
        "prec": precision * 100 / len(dataloader)
    }

def train_step(model: torch.nn.Module, 
               batch: torch.Tensor, 
               criterion: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device = torch.device("cpu")
               ) -> Tuple[float, Dict[str, float]]:

    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss = loss.item()
    metrics = compute_metrics((outputs.detach().to('cpu'), labels.to('cpu')))
    acc = metrics['acc']
    f1_value = metrics['f1']
    recall = metrics['recall']
    precision = metrics['prec']

    return running_loss, {
        "acc": acc * 100, 
        "f1": f1_value * 100, 
        "recall": recall * 100, 
        "prec": precision * 100
    }


@status_handler
def train(config: Union[str, dict] = 'config.yaml',
          experiment: str = 'experiment',
          use_mlflow: bool = False,
          use_clearml: bool = False,
          comment: Optional[str] = None):
    
    if isinstance(config, str):
        config = utils.config_from_yaml(config)
    
    config['task'] = 'train'
    manager_params = config["manager"]
    experiment = experiment.lower().replace(' ', '_')
    global MANAGER

    if use_clearml and use_mlflow:
        raise ValueError("Choose either mlflow or clearml for management")

    save_dir = config['train']['save_dir']
    manager_params = config["manager"]

    # init result foldets
    runs_dir = os.path.join(save_dir, experiment, 'train')
    if not exists(runs_dir):
        os.makedirs(runs_dir)
    run_name = "{:03d}".format(utils.get_new_run_id(runs_dir))
    if comment:
        run_name += '_' + comment

    # init dirs
    run_dir = join(runs_dir, run_name)
    os.makedirs(run_dir)
    os.makedirs(join(run_dir, 'weights/'))
    # save config to run_dir
    config_yaml = join(run_dir, 'config.yaml')
    utils.dict2yaml(config, config_yaml)
    # set path to best weights.pt
    best_weights_path = join(run_dir, "weights/best.pt")

    log_file = join(run_dir, config['train']['log_file'])
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Experiment storage: {run_dir}\n")

    # Init manager
    if use_mlflow or use_clearml:
        params = {
            "experiment": experiment,
            "run_name": 'train-' + run_name,
            "train": True,
        }
        if use_clearml:
            params.update(manager_params["clearml"])
            MANAGER = ClearMLManager(**params) # type: ignore
        else:
            params.update(manager_params["mlflow"])
            MANAGER = MLFlowManager(**params) # type: ignore

        MANAGER.set_iterations(config['train']['num_epochs'])  # type: ignore
        MANAGER.log_hyperparams(manager_params["hparams"])     # type: ignore
        MANAGER.log_config(config_yaml, 'config.yaml')         # type: ignore
        logger.info(f"Manager experiment run name: {'train-' + run_name}\n")

    logger.info(f"Configuration file: {config}\n")

    train_dataset = GenderDatasetDefault(config['dataset']['train'], 
                                         config['model']['sample_rate'])
    test_dataset = GenderDatasetDefault(config['dataset']['test'], 
                                        config['model']['sample_rate'])

    def collate_fn(batch):
        waveforms, labels = zip(*batch)
        pad_to_max = PadToMaxDuration(config['dataset']['max_len_sample'] *
                                      config['model']['sample_rate'])
        padded_waveforms = pad_to_max(waveforms)
        return padded_waveforms, torch.tensor(labels)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config['dataset']['batch_size'], 
                                  shuffle=True, 
                                  num_workers=config['dataset']['num_workers'],
                                  collate_fn=collate_fn
    )
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=config['dataset']['batch_size'], 
                                 shuffle=False, 
                                 num_workers=config['dataset']['num_workers'],
                                 collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenderClassificationModelCNN(config, task='train').to(device)
    logger.info(f"\nModel: {model}")

    # Log model
    if MANAGER:
        MANAGER.log_model(model)
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!\n")
        model = nn.DataParallel(model) # type: ignore

    criterion = getattr(torch.nn, config['train']['loss'])()

    optimizer_cls = getattr(torch.optim, config['optimizer']['name'])
    learning_rate = config['optimizer']['params']['lr']
    optimizer = optimizer_cls(model.parameters(), learning_rate)

    num_epochs = config['train']['num_epochs']
    best_val_f1 = float('-inf')

    for epoch in range(num_epochs):
        model.train()

        logger.info(f'Begin training epoch {epoch+1} on "train" subset')
        sum_train_loss = 0.0
        sum_train_metrics = {"acc": 0.0, "f1": 0.0, "recall": 0.0, "prec": 0.0}
        for i, batch in enumerate(train_dataloader):
            train_loss, metrics = train_step(model, batch, criterion, 
                                             optimizer, device)
            
            sum_train_loss += train_loss
            for key in sum_train_metrics:
                sum_train_metrics[key] += metrics[key]

            if i % config['train']['logging_steps'] == 0 and i != 0:
                batch_metrics = {key: value/(i+1) \
                                 for key, value in sum_train_metrics.items()}
                batch_train_loss = sum_train_loss / (i+1)
                logger.info("Step number {}: " \
                            "Train Loss = {:.4f}, " \
                            "Train Accuracy: {:.4f}, " \
                            "Train F1-score: {:.4f}, " \
                            "Train Recall-score: {:.4f}, " \
                            "Train Precision-score: {:.4f}".format(
                            i, 
                            batch_train_loss, 
                            batch_metrics['acc'], 
                            batch_metrics['f1'], 
                            batch_metrics['recall'], 
                            batch_metrics['prec']))

        avg_train_metrics = {
            k: val/len(train_dataloader)
            for k, val in sum_train_metrics.items()
        }
        avg_train_loss = sum_train_loss / len(train_dataloader)
        logger.info("Epoch [{}/{}], " \
                    "Train Loss: {:.4f}, " \
                    "Train Accuracy: {:.4f}, " \
                    "Train F1-score: {:.4f}, " \
                    "Val Recall-score: {:.4f}, " \
                    "Train Precision-score: {:.4f}\n".format(
                    epoch+1, 
                    num_epochs, 
                    avg_train_loss, 
                    avg_train_metrics['acc'], 
                    avg_train_metrics['f1'], 
                    avg_train_metrics['recall'], 
                    avg_train_metrics['prec']))                 
        
        logger.info('Begin validation on "valid" subset')
        val_loss, val_metrics = evaluate_step(model, \
                                      test_dataloader, \
                                      criterion, \
                                      device)

        logger.info("Epoch [{}/{}], " \
                    "Val Loss: {:.4f}, " \
                    "Val Accuracy: {:.4f}, " \
                    "Val F1-score: {:.4f}, " \
                    "Val Recall-score: {:.4f}, " \
                    "Val Precision-score: {:.4f}".format(
                    epoch+1,
                    num_epochs,
                    val_loss,
                    val_metrics['acc'],
                    val_metrics['f1'],
                    val_metrics['recall'],
                    val_metrics['prec']))
                
        if MANAGER:
            log_metrics = {
                'Train ' + config['train']['loss']: avg_train_loss,
                'Val ' + config['train']['loss']: val_loss
            }
            MANAGER.log_step_metrics(
                log_metrics, 
                (epoch + 1))
            MANAGER.log_step_metrics(
                avg_train_metrics, epoch + 1, 
                'Train metrics')
            MANAGER.log_step_metrics(
                avg_train_metrics, epoch + 1, 
                'Val metrics')                
        
        if best_val_f1 < val_metrics['f1']:
            best_val_f1 = val_metrics['f1']
            logger.info(f"Saving new best model {best_weights_path} "\
                        f"on epoch {epoch + 1} ...")
            model_state_dict = model.state_dict()
            new_state_dict = {
                k.replace("module.", ""): v
                for k, v in model_state_dict.items()
            }
            torch.save(new_state_dict, best_weights_path)
            if MANAGER:
                MANAGER.log_summary_metrics(val_metrics)

        logger.info(f'End of epoch {epoch+1}')
        logger.info('{:-^50}'.format(''))
    logger.info(f"Training completed. Best val f1-score: {best_val_f1:.4f}")
    if MANAGER:
        MANAGER.set_status("FINISHED")
        MANAGER.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Training', 
        description='Gender classification'
    )
    parser.add_argument('--config', type=str,
                    default='../config/config.yml',
                    help='config with differents parameters',
                    required=True)
    parser.add_argument('--experiment', '-exp', type=str, 
                    default='experiment', 
                    help='Name of existed MLFlow experiment')
    manager_group = parser.add_mutually_exclusive_group()
    manager_group.add_argument('--mlflow', action='store_true',
                        dest='use_mlflow', default=False,
                        help='whether to use MLFlow for experiment manager')
    manager_group.add_argument('--clearml', action='store_true',
                        dest='use_clearml', default=False,
                        help='whether to use ClearML for experiment manager')
    parser.add_argument('--comment', '-m', type=str, default=None, 
                        help='Postfix for experiment run name')

    args = parser.parse_args()
    args = vars(args)  # type: ignore
    train(**args)
