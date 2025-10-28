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

import os
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
    
    print(f"Loaded {len(dataset)} examples from game24 train split")

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
            # Assuming 'Question' and 'Correct Answer' columns exist. Adjust if needed.

            question_raw = f"""
            The question is: {example["question"]}
            Please choose the correct answer from the following options:
            {','.join(example["choices"])}
            """
            answer_raw = example['answer']

            question = make_other_prefix(
                question = question_raw, 
                template_type = args.template_type, 
                solution_example = "Alice", 
                answer_format = "tags",
                label_str = "answer"
            )
            # game24 answers are typically direct strings, no complex extraction needed
            solution = {"label": [answer_raw]} # Ensure label is a list of strings

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "expert_level_qa", # More fitting ability for game24
                "reward_model": {
                    "style": "rule", # Assuming simple string matching or similar rule-based check
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx, # Use the provided index
                    'original_index': example.get('index', None), # If original dataset has an index
                    'answer': answer_raw,
                    'question': question_raw,
                    # Add other relevant fields from the original dataset if needed
                    'Incorrect Answers': example.get('Incorrect Answers', None),
                    'Explanation': example.get('Explanation', None),
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