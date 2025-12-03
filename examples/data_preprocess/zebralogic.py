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
Preprocess the game24 dataset to parquet format
"""

import os, re
from environment import root_dir
import numpy as np
import json
import pdb
from verl.utils.hdfs_io import copy, makedirs
import argparse
from constants import get_dataset_dir
from utils import set_seed

from datasets import Dataset, load_dataset
from examples.data_preprocess.helper import make_other_prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--feature_noise', type=lambda x: float(x) if x.lower() != 'none' else None, default=None)
    parser.add_argument('--test_ratio', type=float, default=1)
    parser.add_argument('--n_shot', type=int, default=None)
    parser.add_argument('--n_query', type=int, default=1)
    parser.add_argument('--template_type', type=str, default="reasoning_api")
    parser.add_argument('--label_noise', type=float, default=None)
    parser.add_argument('--data_mode', type=str, default="default", choices=["default"])
    parser.add_argument('--additional_instruction_path', type=str, default=None)
    args = parser.parse_args()
    
    data_source = "zebralogic"
    set_seed(42)
    
    if args.n_query != 1:
        raise ValueError("n_query must be 1 for zebralogic dataset")
    if args.n_shot != 0:
        raise ValueError("n_shot must be 0 for zebralogic dataset")
    if args.feature_noise is not None and args.feature_noise != 0:
        raise ValueError("feature_noise must be 0 or None for zebralogic dataset")
    if args.label_noise is not None and args.label_noise != 0:
        raise ValueError("label_noise must be 0 or None for zebralogic dataset")
    if args.test_ratio != 1:
        raise ValueError("test_ratio must be 1 for zebralogic dataset")

    # Load the specific zebralogic subset
    # game24 datasets often only have a 'train' split containing all examples
    dataset = load_dataset(
        "WildEval/ZebraLogic", 
        'mc_mode',
        split='test' # Assuming the relevant data is in the train split
    )
    def extract_size(example):
        match = re.search(r'\d+x\d+', example["id"])
        return {"size": match.group() if match else None}
    dataset = dataset.map(extract_size)
    subsets, subsets_size = dict(), dict()

    # easier than 3x3
    subsets_size["easy"] = ["2x2", "2x3", "2x4", "2x5", "2x6", "3x2", "3x3"]
    # easier than 4x4
    subsets_size["medium"] = ["3x4", "3x5", "3x6", "4x2", "4x3", "4x4", "5x4"]
    # easier than 5x5
    subsets_size["hard"] = ["4x5", "4x6", "5x3", "5x4", "5x5", "6x2", "6x3"]
    # remaining
    subsets_size["challenge"] = ["5x6", "6x4", "6x5", "6x6"]

    dataset = dataset.filter(lambda x: x["size"] in subsets_size["challenge"])
    
    print(f"Loaded {len(dataset)} examples from zebralogic challenge split")

    # Shuffle and split the data
    dataset = dataset.shuffle(seed=42)
    
    n_total = len(dataset)
    if args.num_samples > n_total:
        print(f"Warning: Requested {args.num_samples} samples, but dataset only has {n_total}. Using all examples.")
        args.num_samples = n_total


    num_samples = args.num_samples if args.num_samples > 0 else n_total
    n_test = int(num_samples * args.test_ratio) 
    n_train = num_samples - n_test 
    
    # Select samples
    sampled_indices = np.random.choice(range(len(dataset)), size=num_samples, replace=False)
    train_indices = sampled_indices[:n_train]
    test_indices = sampled_indices[n_train:]

    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)

    print(f"Created {len(train_dataset)} training examples and {len(test_dataset)} test examples")

    # Define the processing function
    def make_map_fn(split):
        def process_fn(example, idx):
            # Construct the complete ZebraLogic problem with puzzle, question, and choices
            
            # The puzzle contains the problem setup with all constraints
            # The question is a separate field asking what to find
            puzzle_text = example["puzzle"]
            question_text = example["question"]
            
            # Format choices as a numbered list for better readability
            choices_formatted = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(example["choices"])])
            
            # Combine puzzle (constraints) + question + choices into complete prompt
            question_raw = f"""{puzzle_text}

{question_text}

Please choose the correct answer from the following options:
{choices_formatted}"""
            

            if args.additional_instruction_path is not None:
                with open(args.additional_instruction_path, 'r') as f:
                    additional_instructions = f.read()
                question_raw += "\n" + additional_instructions

            answer_raw = example['answer']

            question = make_other_prefix(
                question = question_raw, 
                template_type = args.template_type, 
                solution_example = "Alice", 
                answer_format = "tags",
                label_str = "answer"
            )
            
            solution = {"label": [answer_raw]}

            data = {
                "data_source": data_source,
                "additional_instruction_path": args.additional_instruction_path,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "logic_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'original_index': example.get('index', None),
                    'id': example.get('id', None),
                    'answer': answer_raw,
                    'question': example["question"],
                    'puzzle': puzzle_text,
                    'choices': example["choices"],
                }
            }

            return data
        return process_fn

    # Apply the processing function
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=test_dataset.column_names)

    # Define output directories
    local_dir = get_dataset_dir(
        dataset_name=data_source,
        shot=args.n_shot,
        template_type=args.template_type,
        num_samples=args.num_samples,
        feature_noise=None, # Pass None explicitly if function expects it
        label_noise=0.0,   # Pass 0.0 explicitly if function expects it
        data_mode="default",
        n_query=1,
    )
    hdfs_dir = args.hdfs_dir

    # Save datasets
    os.makedirs(local_dir, exist_ok=True) # Ensure local directory exists
    train_dataset.to_parquet(os.path.join(local_dir, 'train_default.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_default.parquet'))
    
    print(f"Train dataset saved to {os.path.join(local_dir, 'train_default.parquet')}")
    print(f"Test dataset saved to {os.path.join(local_dir, 'test_default.parquet')}")

    # Copy to HDFS if specified
    if hdfs_dir is not None:
        # Ensure verl.utils.hdfs_io handles directory creation correctly or create manually
        # makedirs(hdfs_dir) # This might need adjustment based on verl's implementation
        
        print(f"Copying data from {local_dir} to HDFS directory {hdfs_dir}")
        # Assuming verl.utils.hdfs_io.copy works like shutil.copytree or similar
        copy(src=local_dir, dst=hdfs_dir) 
        print("Copy to HDFS complete.")