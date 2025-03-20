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
import requests
from typing import List, Dict, Any
from openai import OpenAI
import concurrent.futures
from functools import partial

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from utils import flatten_dict, print_configs
from constants import get_configs_via_result_dir
import wandb   
try:
    from environment import WANDB_INFO, HUGGINGFACE_API_KEY, DEEPSEEK_API_KEY
except ImportError:
    raise ImportError("""
Please create environment.py file in the project root directory.
Here is the expected format of WANDB_INFO, HUGGINGFACE_API_KEY, and DEEPSEEK_API_KEY:

WANDB_INFO = {"project": "your-project-id", "entity": "your-entity-name"}
HUGGINGFACE_API_KEY = "your-huggingface-api-key"
DEEPSEEK_API_KEY = "your-deepseek-api-key"
""")

from huggingface_hub import login
login(token=HUGGINGFACE_API_KEY)

class DeepseekAPI:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> str:
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stream=False
            )
            #print(response)
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in Deepseek API call: {str(e)}")
            raise

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    
    if config.trainer.wandb:
        wandb_configs = flatten_dict(config)
        wandb_configs.update(get_configs_via_result_dir(config.data.output_path))
        wandb.init(
            project=f"{WANDB_INFO['project']}-generation",
            entity=WANDB_INFO['entity'],
            config=wandb_configs
        )
        
    
    print_configs(flatten_dict(config))
    
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    if config.model.path == "deepseek-chat":
        use_api = True
    else:
        use_api = False

    # Initialize model based on config
    if use_api:
        model = DeepseekAPI(DEEPSEEK_API_KEY)
        tokenizer = None  # Deepseek API handles tokenization
    else:
        local_path = copy_local_path_from_hdfs(config.model.path)
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(local_path)
        model = None

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    # Convert chat list to proper format for Deepseek API
    if use_api:
        def convert_to_string(chat):
            if isinstance(chat, np.ndarray):
                chat = chat.tolist()
            if isinstance(chat, list):
                chat = " ".join(str(x) for x in chat)
            return str(chat)
            
        chat_lst = [{"role": "user", "content": convert_to_string(chat)} for chat in chat_lst]
    else:
        chat_lst = [chat.tolist() for chat in chat_lst]

    if not use_api:
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if not use_api:
        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    
    if not use_api:
        dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    else:
        dp_size = 1  # When using API, we don't need distributed processing
    
    num_batch = (total_samples // config_batch_size) + 1
    output_lst = [[] for _ in range(config.data.n_samples)]
    #breakpoint()
    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_start_time = time.time()
        batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]

        if use_api:
            # Process with Deepseek API in parallel
            def process_single_chat(chat, sample_idx):
                gen_start_time = time.time()
                response = model.generate(
                    messages=[chat],  # chat is already in the correct format
                    max_tokens=config.rollout.response_length,
                    temperature=config.rollout.temperature
                )
                response_length = len(response.split())
                generation_time = time.time() - gen_start_time
                
                metrics = {
                    f'sample_{sample_idx}/generation_time': generation_time,
                    f'sample_{sample_idx}/tokens_per_second': response_length / max(generation_time, 1e-6),
                    f'sample_{sample_idx}/response_length': response_length,
                    'batch': batch_idx,
                    'sample_idx': sample_idx  # Add sample_idx to metrics
                }
                return response, metrics

            # Create a thread pool for parallel processing
            max_workers = min(config_batch_size, len(batch_chat_lst) * config.data.n_samples)  # Limit max workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create all tasks
                futures = []
                for chat in batch_chat_lst:
                    for i in range(config.data.n_samples):
                        futures.append(executor.submit(process_single_chat, chat, i))
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    response, metrics = future.result()
                    # Find which sample this response belongs to
                    sample_idx = metrics['sample_idx']
                    output_lst[sample_idx].append(response)
                    
                    # Log metrics
                    if config.trainer.wandb:
                        wandb.log(metrics)
        else:
            # Process with HuggingFace model
            inputs = tokenizer.apply_chat_template(batch_chat_lst,
                                               add_generation_prompt=True,
                                               padding=True,
                                               truncation=True,
                                               max_length=config.rollout.prompt_length,
                                               return_tensors='pt',
                                               return_dict=True,
                                               tokenize=True)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            position_ids = compute_position_id_with_mask(attention_mask)

            batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
            data = DataProto.from_dict(batch_dict)
            real_batch_size = data.batch['input_ids'].shape[0]
            if real_batch_size % dp_size != 0:
                dummy_data_size = dp_size - real_batch_size % dp_size
                dummy_data = data[:dummy_data_size]
                data = DataProto.concat([data, dummy_data])
                print(
                    f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
                )

            batch_size = data.batch['input_ids'].shape[0]
            assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

            print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
            # START TO GENERATE FOR n_samples TIMES
            batch_metrics = {
                'batch_size': real_batch_size,
                'processing_time': time.time() - batch_start_time,
            }
            
            for i in range(config.data.n_samples):
                gen_start_time = time.time()
                output = wg.generate_sequences(data)
                # remove dummy data
                output = output[:real_batch_size]
                output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                     skip_special_tokens=False)

                # remove the padding
                pad_token = tokenizer.pad_token
                output_text_unpad = []
                for text in output_text:
                    output_text_unpad.append(text.replace(pad_token, ''))

                output_lst[i].extend(output_text_unpad)

                # Log generation metrics
                response_lengths = [len(text.split()) for text in output_text_unpad]
                generation_time = time.time() - gen_start_time
                
                # Compile metrics for this sample
                sample_metrics = {
                    f'sample_{i}/generation_time': generation_time,
                    f'sample_{i}/tokens_per_second': sum(response_lengths) / max(generation_time, 1e-6),
                    f'sample_{i}/avg_response_length': np.mean(response_lengths),
                    f'sample_{i}/max_response_length': np.max(response_lengths),
                    f'sample_{i}/min_response_length': np.min(response_lengths),
                }
                batch_metrics.update(sample_metrics)
                breakpoint()
            # Log batch metrics to wandb
            if config.trainer.wandb:
                batch_metrics['batch'] = batch_idx
                batch_metrics['completion_percentage'] = (batch_idx + 1) / num_batch * 100
                wandb.log(flatten_dict(batch_metrics))

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    # add to the data frame
    dataset[f'responses'] = output_lst

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

    if config.trainer.wandb:
        wandb.finish()

    # Convert output_lst to the same format as output_text for API path
    if use_api:
        # Flatten the output_lst to match the format of output_text
        flattened_output = []
        for responses in output_lst:
            flattened_output.extend(responses)
        return flattened_output
    else:
        return output_text  # For non-API path, return the output_text variable


if __name__ == '__main__':
    main()
