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
    response_length,
    num_samples,
    feature_noise,
    label_noise
):
    local_dir = get_result_dir(
        dataset_name=dataset_name,
        model_name=model_name,
        shot=shot,
        template_type=template_type,
        response_length=response_length,
        num_samples=num_samples,
        feature_noise=feature_noise,
        label_noise=label_noise
    )
    test_dataset = pd.read_parquet(os.path.join(local_dir, 'test.parquet'))
    while True:
        idx = input("Enter the index of the example to check: ")
        if idx == 'q':
            break
        idx = int(idx)
        print(f"prompt: {test_dataset.iloc[idx]['prompt'][0]['content']}")
        print(f"label: {test_dataset.iloc[idx]['label']}")
        print(f"response: {test_dataset.iloc[idx]['responses'][0]}")
        print('-'*100)
        
