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
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets
import pdb
from verl.utils.hdfs_io import copy, makedirs
import argparse
from constants import get_dataset_dir
import numpy as np
from utils import set_seed

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return {
        "label": [int(final_solution)],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--feature_noise', type=float, default=None)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--n_shot', type=int, default=None)
    parser.add_argument('--n_query', type=int, default=1)
    parser.add_argument('--template_type', type=str, default="reasoning_api")
    parser.add_argument('--label_noise', type=float, default=None)
    parser.add_argument('--data_mode', type=str, default="default", choices=["default"])
    
    args = parser.parse_args()
    
    data_source = 'gsm8k'
    set_seed(42)
    
    if args.n_query != 1:
        raise ValueError("n_query must be 1 for GSM8k")
    if args.n_shot != 0:
        raise ValueError("n_shot must be 0 for GSM8k")
    if args.feature_noise is not None and args.feature_noise != 0:
        raise ValueError("feature_noise must be 0 or None for GSM8k")
    if args.label_noise is not None and args.label_noise != 0:
        raise ValueError("label_noise must be 0 or None for GSM8k")
    

    dataset = datasets.load_dataset(data_source, 'main')
    
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    n_test = int(args.num_samples * args.test_ratio)
    n_train = args.num_samples - n_test
    idx_train = np.random.choice(range(len(train_dataset)), size=n_train, replace=False)
    idx_test = np.random.choice(range(len(test_dataset)), size=n_test, replace=False)

    train_dataset = train_dataset.select(idx_train)
    test_dataset = test_dataset.select(idx_test)

    def make_instruction_following(question):
        if "reasoning_api" in args.template_type or "standard_api_no_reasoning" in args.template_type:
            answer_example = "0"
        else:
            answer_example = "<answer>0</answer>"
        if args.template_type == 'base':
            instruction_following = f"""
            A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
            User: {question} Please provide your thinking process in <think> </think> tags. Your final answer should be enclosed in <answer> and </answer> tags, containing only answer with no additional text—for example, {answer_example}
            Assistant: Let me solve this step by step.
            <think>
            """
        elif args.template_type == 'base_no_reasoning':
            instruction_following = """
            A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
            User: {question} Your final answer should be enclosed in <answer> and </answer> tags, containing only answer with no additional text—for example, {answer_example}
            Assistant: 
            """
        elif args.template_type == "qwen-instruct":
            instruction_following = """
            <|im_start|>system
            You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.
            <|im_end|>
            <|im_start|>user
            {question} Please provide your thinking process in <think> </think> tags. Your final answer should be enclosed in <answer> and </answer> tags, containing only answer with no additional text—for example, {answer_example}
            <|im_end|>
            <|im_start|>assistant
            Let me solve this step by step.
            <think>
            """
        elif args.template_type == "qwen-instruct_no_reasoning":
            instruction_following = """
            <|im_start|>system
            You are a helpful assistant. You always provide the user directly with the answer without any reasoning.
            <|im_end|>
            <|im_start|>user
            {question} Your response should contain only the final answer enclosed in <answer> and </answer> tags, with no additional text—specifically, just {label_str}, for example: {answer_example}
            <|im_end|>
            <|im_start|>assistant
            """
        elif args.template_type == "reasoning_api":
            instruction_following = f"""
            {question} Your response should just be the answer with no additional text—for example, {answer_example}
            """
        elif args.template_type == "standard_api_no_reasoning":
            instruction_following = f"""
            {question} Your response should just be the answer, containing only answer with no additional text—for example, {answer_example}
            """
        elif args.template_type == "standard_api":
            instruction_following = f"""
            {question} Please provide your thinking process in <think> </think> tags. Your final answer should be enclosed in <answer> and </answer> tags, containing only answer with no additional text—for example, {answer_example}
            """
        else:
            raise ValueError(f"Template type {args.template_type} is not supported for GSM8k")

        return instruction_following
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')

            question = make_instruction_following(question_raw)

            answer_raw = example.pop('answer')
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    

    local_dir = get_dataset_dir(
        dataset_name='gsm8k',
        shot=args.n_shot,
        template_type=args.template_type,
        num_samples=args.num_samples,
        feature_noise=None,
        label_noise=0.0,
        data_mode="default",
        n_query=1,
    )
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train_default.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_default.parquet'))
    
    print(f"Train dataset saved to {local_dir}/train_default.parquet")
    print(f"Test dataset saved to {local_dir}/test_default.parquet")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
