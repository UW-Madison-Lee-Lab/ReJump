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
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import pdb
import hydra
import os
import time
import re
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
from verl.utils.llm_api import LLMAPI

from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from utils import flatten_dict, print_configs
from constants import get_configs_via_result_dir, supported_llms, supported_datasets
from verl.trainer.fsdp_sft_trainer import extract_model_name
from examples.data_preprocess.helper import classification_extract_solution, regression_extract_solution
import wandb   
try:
    from environment import WANDB_INFO, HUGGINGFACE_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY
except ImportError:
    raise ImportError("""
Please create environment.py file in the project root directory.
Here is the expected format of WANDB_INFO, HUGGINGFACE_API_KEY, DEEPSEEK_API_KEY, and OPENAI_API_KEY:

WANDB_INFO = {"project": "your-project-id", "entity": "your-entity-name"}
HUGGINGFACE_API_KEY = "your-huggingface-api-key"
DEEPSEEK_API_KEY = "your-deepseek-api-key"
OPENAI_API_KEY = "your-openai-api-key"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
""")

from huggingface_hub import login

login(token=HUGGINGFACE_API_KEY)

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    wandb_configs = flatten_dict(config)
    wandb_configs.update(get_configs_via_result_dir(os.path.dirname(config.data.output_path)))
    if config.trainer.wandb == 1:
        config.model.path = extract_model_name(config.model.path)
        run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=f"{WANDB_INFO['project']}-{config.trainer.project_name}",
            entity=WANDB_INFO['entity'],

            name=run_name,
            config=wandb_configs
        )
    elif config.trainer.wandb == 2:
    
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

    # Initialize model based on config
    use_api = supported_llms[config.model.path]["type"] == "api"
    if use_api:
        api_key = supported_llms[config.model.path]["api_key"]
        model = LLMAPI(api_key=api_key, model_name=config.model.path, template_type=wandb_configs["template_type"])
        chat_lst_converter = LLMAPI.convert_chat_list
        # Use Qwen tokenizer for API mode
        local_path = "Qwen/Qwen2.5-3B-Instruct"
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(local_path)
    else:
        local_path = copy_local_path_from_hdfs(config.model.path)
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    total_samples = len(dataset)
    chat_lst = dataset[config.data.prompt_key].tolist()
    dataset_indices = list(range(total_samples))  # Add indices to track order

    # Debug: Print the structure of the first chat
    print("\n=== Debug: First chat structure ===")
    print("Type:", type(chat_lst[0]))
    print("Content:", chat_lst[0])
    print("=====================================\n")

    # Convert chat list to proper format for API
    if use_api:
        chat_lst = chat_lst_converter(chat_lst)
    else:
        chat_lst = [chat.tolist() for chat in chat_lst]

    if not use_api:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

    # Use RLHFDataset for both API and non-API modes
    if use_api:
        max_prompt_length = 100_000
    else:
        max_prompt_length = config.rollout.prompt_length
        
    rlhf_dataset = RLHFDataset(
        parquet_files=config.data.path,
        tokenizer=tokenizer,
        prompt_key=config.data.prompt_key,
        max_prompt_length=max_prompt_length,
        filter_prompts=True,
        return_raw_chat=config.data.get('return_raw_chat', False),
        truncation='error'
    )
    
    dataloader = DataLoader(
        rlhf_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=not use_api,  # Disable shuffle only when using API mode
        drop_last=False,
        collate_fn=collate_fn
    )
    
    reward_tensor_lst = [[] for _ in range(config.data.n_samples)]
    output_lst = [[] for _ in range(config.data.n_samples)]
    reasonings_lst = [[] for _ in range(config.data.n_samples)]
    answer_lst = [[] for _ in range(config.data.n_samples)]
    if not use_api:
        reward_fn = RewardManager(
            tokenizer=tokenizer, 
            num_examine=0, 
            return_dict=True,
        )

    for batch_idx, test_data in enumerate(dataloader):
        print(f"Start batch [{batch_idx}/{len(dataloader)}]")
        
        if use_api:
            # Process with API
            batch_chat_lst = chat_lst[batch_idx * config.data.batch_size:(batch_idx + 1) * config.data.batch_size]
            
            # Get ground truths and data sources from test_data
            test_batch = DataProto.from_single_dict(test_data)
            
            # Access ground truths and data sources from each data item
            batch_ground_truths = []
            for i in range(len(test_batch)):
                data_item = test_batch[i]
                batch_ground_truths.append(data_item.non_tensor_batch['reward_model']['ground_truth'])
            batch_data_sources = test_batch.non_tensor_batch['data_source']
            
            # Process batch and get rewards
            batch_output_lst, reward_dict, batch_reasoning_lst, batch_answer_lst = model.process_batch(
                batch_chat_lst=batch_chat_lst,
                n_samples=config.data.n_samples,
                config=config,
                batch_idx=batch_idx,
                wandb=wandb if config.trainer.wandb else None,
                ground_truths=batch_ground_truths,
                data_sources=batch_data_sources
            )
            
            # Debug: Print all samples' input, output, ground truth, and reward
            print(f"\n=== Batch {batch_idx} Debug Info (API Mode) ===")
            for i in range(len(batch_chat_lst)):
                print(f"\nSample {i}:")
                print("Input prompt:", batch_chat_lst[i])
                print("Ground truth:", batch_ground_truths[i])
                print("Data source:", batch_data_sources[i])
                for j in range(config.data.n_samples):
                    print(f"Output {j}:", batch_output_lst[j][i])
                    if batch_reasoning_lst[j][i] is not None:
                        print(f"Reasoning {j}:", batch_reasoning_lst[j][i])
                print("-" * 50)
            print("=" * 50)
            
            #breakpoint()
            # Add batch outputs and rewards to the main lists
            for i in range(config.data.n_samples):
                output_lst[i].extend(batch_output_lst[i])
                reward_tensor_lst[i].extend(reward_dict['reward_tensor'].tolist())
                reasonings_lst[i].extend(batch_reasoning_lst[i])
                answer_lst[i].extend(batch_answer_lst[i])
        else:
            # Process with HuggingFace model
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
                indexes = [item.non_tensor_batch["extra_info"]["index"] for item in test_batch]
                sort_idx = np.argsort(indexes)
                reward_tensor_lst[i].extend(reward_tensor.sum(dim=1)[sort_idx].tolist())
                output_lst[i].extend(np.array(reward_dict['sequences_lst'])[sort_idx].tolist())
                answer_lst[i].extend(np.array(reward_dict['answer_lst'])[sort_idx].tolist())
                reasonings_lst[i].extend(np.array(reward_dict['reasoning_lst'])[sort_idx].tolist())
                
                # Debug: Print first sample's output
                if batch_idx == 0 and i == 0:
                    print("Output:", reward_dict['sequences_lst'][0])
                    print("=====================================\n")
            
    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist() 
    
    dataset["responses"] = output_lst
    
    # convert answer_lst from (n_samples, n_data) to (n_data, n_samples)
    answer_lst = np.array(answer_lst, dtype=object)
    answer_lst = np.transpose(answer_lst, axes=(1, 0)).tolist()
    dataset["answers"] = answer_lst
    # convert reasonings_lst from (n_samples, n_data) to (n_data, n_samples)
    reasonings_lst = np.array(reasonings_lst, dtype=object)
    reasonings_lst = np.transpose(reasonings_lst, axes=(1, 0)).tolist()
    dataset["reasonings"] = reasonings_lst
    
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
    # Upload results to wandb
    if config.trainer.wandb:
        # Log the dataset as an artifact
        artifact = wandb.Artifact(
            name=f"generation_results_{wandb.run.id}",
            type="dataset"
        )
        artifact.add_file(config.data.output_path)
        wandb.log_artifact(artifact)
    
    config_dict = get_configs_via_result_dir(os.path.dirname(config.data.output_path))
    
    y_pred, y_true = [], []
    
    task_type = supported_datasets[config_dict['dataset_name']]['type']
    k = config.data.n_samples
    n_query = int(wandb_configs["n_query"])
    valid_prompts = np.zeros(total_samples)
    for i in range(total_samples):
        best_metric = -np.inf
        best_answer = [0 for _ in range(n_query)]
        true_labels = dataset["reward_model"].iloc[i]["ground_truth"]["label"]
        valid_flag = False
        for j in range(k):
            if task_type == 'classification':
                answers = np.array(classification_extract_solution(output_lst[i][j]))
                if len(answers) == n_query:
                    metric = (answers == true_labels).mean()
                    if metric > best_metric:
                        best_metric = metric
                        best_answer = answers
                        valid_flag = True
            elif task_type == 'regression':
                answers = np.array(regression_extract_solution(output_lst[i][j]))
                if len(answers) == n_query:
                    metric = -((answers - true_labels)**2).mean()
                    if metric > best_metric:
                        best_metric = metric
                        best_answer = answers
                        valid_flag = True
                        
        valid_prompts[i] = valid_flag
        y_pred.append(best_answer)
        y_true.append(true_labels)
            
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    valid_prompts = valid_prompts.astype(bool)
    if task_type == 'regression':
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        r2 = r2_score(y_true, y_pred)
        
        print(f'mse@{k}: {mse: .3f}')
        print(f'r2@{k}: {r2: .3f}')
        
        valid_mse = mean_squared_error(y_true[valid_prompts].flatten(), y_pred[valid_prompts].flatten())
        valid_r2 = r2_score(y_true[valid_prompts], y_pred[valid_prompts])
        print(f'valid_mse@{k}: {valid_mse: .3f}')
        print(f'valid_r2@{k}: {valid_r2: .3f}')
        
        if config.trainer.wandb:
            wandb.log({
                f'mse@{k}': mse,
                f'r2@{k}': r2,
                f'valid_mse@{k}': valid_mse,
                f'valid_r2@{k}': valid_r2,
            })
    elif task_type == 'classification':
        strict_accuracy = accuracy_score(np.ones(total_samples), np.all(y_pred == y_true, axis=1))
        y_true = y_true.flatten()
        accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
        print(f'pass@{k}: {accuracy: .3f}')
        print(f'strict_pass@{k}: {strict_accuracy: .3f}')
        
        valid_strict_accuracy = accuracy_score(np.ones(valid_prompts.sum()), np.all(y_pred[valid_prompts] == y_true[valid_prompts], axis=1))
        valid_accuracy = accuracy_score(y_true[valid_prompts].flatten(), y_pred[valid_prompts].flatten())
        print(f'valid_pass@{k}: {valid_accuracy: .3f}')
        print(f'valid_strict_pass@{k}: {valid_strict_accuracy: .3f}')
        if config.trainer.wandb:
            wandb.log({
                f'pass@{k}': accuracy,
                f'strict_pass@{k}': strict_accuracy,
                f'valid_pass@{k}': valid_accuracy,
                f'valid_strict_pass@{k}': valid_strict_accuracy,
            })
            
    else:
        reward_lst = []
        for i in range(total_samples):
            reward_lst.append(max(reward_tensor_lst[i]))
        reward_lst = np.array(reward_lst)
        passk = reward_lst.mean()
        print(f'pass@{k}: {passk: .3f}')
        if config.trainer.wandb:
            wandb.log({
                f'pass@{k}': passk,
            })
        valid_prompts = [1]
                
            
    valid_ratio = np.mean(valid_prompts)
    print(f'valid_ratio@{k}: {valid_ratio: .3f}')
    if config.trainer.wandb:
        wandb.log({
            f'valid_ratio@{k}': valid_ratio,
        })

    if config.trainer.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()