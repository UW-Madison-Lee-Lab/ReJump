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
Generate responses given a dataset of prompts
"""
import ray
import numpy as np
import hydra
import os
import time

from datetime import datetime


os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import pandas as pd

from transformers import AutoTokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torch.utils.data import DataLoader
from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.trainer.ppo.helper import RewardManager

from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from utils import flatten_dict, print_configs
from constants import get_configs_via_result_dir
from verl.trainer.fsdp_sft_trainer import extract_model_name
from environment import WANDB_INFO
import wandb

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    config.model.path = extract_model_name(config.model.path)
    
    if config.trainer.wandb == 1:

        run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=f"{WANDB_INFO['project']}-{config.trainer.project_name}",
            entity=WANDB_INFO['entity'],

            name=run_name,
            config=flatten_dict(config)
        )
    elif config.trainer.wandb == 2:
        wandb_configs = flatten_dict(config)
        wandb_configs.update(get_configs_via_result_dir(os.path.dirname(config.data.output_path)))
        wandb.init(
            project=f"{WANDB_INFO['project']}-generation",
            entity=WANDB_INFO['entity'],
            config=wandb_configs
        )
    else:
        raise ValueError(f"Invalid wandb mode: {config.trainer.wandb}")
    
    print_configs(flatten_dict(config))
    
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    total_samples = len(dataset)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()
    

    rlhf_dataset = RLHFDataset(
        parquet_files=config.data.path,
        tokenizer=tokenizer,
        prompt_key=config.data.prompt_key,
        max_prompt_length=config.rollout.prompt_length,
        filter_prompts=True,
        return_raw_chat=config.data.get('return_raw_chat', False),
        truncation='error'
    )
    
    dataloader = DataLoader(
        rlhf_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    reward_tensor_lst = [[] for _ in range(config.data.n_samples)]
    output_lst = [[] for _ in range(config.data.n_samples)]
    reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=0, 
        return_dict=True,
    )
    for batch_idx, test_data in enumerate(dataloader):
        print(f"Start batch [{batch_idx}/{len(dataloader)}]")
        test_batch = DataProto.from_single_dict(test_data)
        test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
        
        test_gen_batch.meta_info = {
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': False,
            'validate': True,
        }
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, wg.world_size)
        
        for i in range(config.data.n_samples):
            test_output_gen_batch_padded = wg.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch_padded = test_output_gen_batch_padded[:test_batch.batch.batch_size[0]]
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            test_batch = test_batch.union(test_output_gen_batch)
            reward_dict = reward_fn(test_batch)
            reward_tensor = reward_dict['reward_tensor']
            reward_tensor_lst[i].extend(reward_tensor.sum(dim=1).tolist())
            output_lst[i].extend(reward_dict['sequences_lst'])
            
    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist() 
    
    dataset["responses"] = output_lst
    
    # convert reward_tensor_lst from (n_samples, n_data) to (n_data, n_samples)
    reward_tensor_lst = np.array(reward_tensor_lst, dtype=object)
    reward_tensor_lst = np.transpose(reward_tensor_lst, axes=(1, 0)).tolist() 
    

    # Log final summary statistics to wandb
    if config.trainer.wandb:
        all_response_lengths = []
        for responses in output_lst:
            for response in responses:
                all_response_lengths.append(len(response.split()))
        
        summary_metrics = {
            'final/total_samples': total_samples,
            'final/total_responses': total_samples * config.data.n_samples,
            'final/avg_response_length': np.mean(all_response_lengths),
            'final/response_length_std': np.std(all_response_lengths),
        }
        wandb.log(flatten_dict(summary_metrics))
    
    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)
    # save a json copy
    dataset.to_json(config.data.output_path.replace(".parquet", ".json"), orient="records", lines=True)
        
    
    # eval
    passes = 0


    k = None
    for i in range(total_samples):
        if k is None: k = len(reward_tensor_lst[i])
        max_score = np.max(reward_tensor_lst[i])

        if max_score == 1:
            passes += 1

    print(f'pass@{k}: {passes / total_samples}')

    if config.trainer.wandb:
        wandb.log({
            f'pass@{k}': passes / total_samples,
        })

    if config.trainer.wandb:
        wandb.finish()



if __name__ == '__main__':
    main()
