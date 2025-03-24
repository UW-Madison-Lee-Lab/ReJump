# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import hydra
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.reward_score import math, gsm8k
import os
import pandas as pd
import numpy as np
import pdb, wandb
from utils import flatten_dict, print_configs
from environment import WANDB_INFO
from constants import get_configs_via_result_dir

def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        return math.compute_score
    elif data_source == "blobs":
        from examples.data_preprocess.blobs import blobs_reward_fn
        return blobs_reward_fn
    elif data_source == "moons":
        from examples.data_preprocess.moons import moons_reward_fn
        return moons_reward_fn
    elif data_source == "linear":
        from examples.data_preprocess.linear import linear_reward_fn
        return linear_reward_fn
    else:
        raise NotImplementedError


@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    if config.trainer.wandb:
        wandb_configs = flatten_dict(config)
        wandb_configs.update(get_configs_via_result_dir(os.path.dirname(config.data.path)))
        wandb.init(
            project=f"{WANDB_INFO['project']}-evaluation",
            entity=WANDB_INFO['entity'],
            config=wandb_configs
        )
    

    print_configs(flatten_dict(config))
    
    local_path = copy_local_path_from_hdfs(config.data.path)
    dataset = pd.read_parquet(local_path)
    prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0

    total = len(dataset)

    k = None
    for i in range(total):
        response_lst = responses[i]
        if k is None: k = len(response_lst)
        data_source = data_sources[i]
        # select reward score based on data_source
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        score_lst = []
        for r in response_lst:
            score = reward_fn(r, ground_truth)
            score_lst.append(score)

    

        max_score = np.max(score_lst)

        if max_score == 1:
            passes += 1

    print(f'pass@{k}: {passes / total}')

    if config.trainer.wandb:
        wandb.log({
            f'pass@{k}': passes / total,
        })
        wandb.finish()

if __name__ == '__main__':
    main()
