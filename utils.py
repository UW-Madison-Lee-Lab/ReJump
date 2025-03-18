import os
import json
import torch
import numpy as np
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
def flatten_dict(config, prefix=''):
    new_config = {}
    for key in config:
        if isinstance(config[key], dict):
            new_config.update(flatten_dict(config[key], prefix=f'{prefix}.{key}'))
        else:
            new_config[f'{prefix}.{key}'] = config[key]
    return new_config
    
def print_configs(args_dict):
    # print experiment configuration
    print("########"*3)
    print('## Experiment Setting:')
    print("########"*3)
    for key, value in args_dict.items():
        print(f"| {key}: {value}")
    print("########"*3)
    