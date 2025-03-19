import os
import json, pdb
import torch
import numpy as np
import random
import pandas as pd
from constants import get_result_dir
from omegaconf import DictConfig

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
def flatten_dict(d):
    new_d = {}
    for k1 in d:
        if isinstance(d[k1], DictConfig) or isinstance(d[k1], dict):
            d1 = flatten_dict(d[k1])
            for k2 in d1:
                new_d[f"{k1}.{k2}"] = d1[k2]
        else:
            new_d[k1] = d[k1]
    return new_d
                
    
def print_configs(args_dict):
    # print experiment configuration
    print("########"*3)
    print('## Experiment Setting:')
    print("########"*3)
    for key, value in args_dict.items():
        print(f"| {key}: {value}")
    print("########"*3)
    
def check_results(
    dataset_name,
    shot,
    model_name,
    template_type,
):
    local_dir = get_result_dir(dataset_name, model_name, shot, template_type)
    test_dataset = pd.read_parquet(os.path.join(local_dir, 'test.parquet'))
    for idx, row in test_dataset.iterrows():
        print(row['prompt'])
        print(row['ground_truth'])
        print(row['response'])
        print('-'*100)
        
