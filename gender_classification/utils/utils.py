import os
from collections import OrderedDict
import oyaml

def get_new_run_id(runs_dir: str) -> int:
    existed_runs = os.listdir(runs_dir)
    max_id = 0
    if existed_runs:
        max_id = max((int(run_name[:3]) \
                      for run_name in existed_runs))
    return max_id + 1

def config_from_yaml(yaml_path: str) -> dict:
    with open(yaml_path) as file:
        config = oyaml.load(file, Loader=oyaml.FullLoader)
    config = OrderedDict(config)
    return config

def dict2yaml(data: dict, yaml_path: str):
    with open(yaml_path, 'w') as file:
        file.write(oyaml.dump(data))
