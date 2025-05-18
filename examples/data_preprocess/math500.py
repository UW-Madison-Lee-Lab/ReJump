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
Preprocess the MATH dataset to parquet format
"""

import os
from environment import root_dir
import numpy as np
import urllib.request
import zipfile
import json
import glob
import pdb
from verl.utils.hdfs_io import copy, makedirs
import argparse
from constants import get_dataset_dir
from utils import set_seed

from verl.utils.reward_score.general import extract_solution
from datasets import Dataset
from examples.data_preprocess.helper import make_other_prefix
from datasets import load_dataset




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
    
    data_source = "math500"
    set_seed(42)
    
    if args.n_query != 1:
        raise ValueError("n_query must be 1 for MATH dataset")
    if args.n_shot != 0:
        raise ValueError("n_shot must be 0 for MATH dataset")
    if args.feature_noise is not None and args.feature_noise != 0:
        raise ValueError("feature_noise must be 0 or None for MATH dataset")
    if args.label_noise is not None and args.label_noise != 0:
        raise ValueError("label_noise must be 0 or None for MATH dataset")
    if args.test_ratio != 1:
        raise ValueError("test_ratio must be 1 for MATH500 dataset")

    dataset = load_dataset("HuggingFaceH4/MATH-500")
    
    # Load train data
    train_dataset = Dataset.from_list([])
    test_dataset = dataset["test"]
    
    n_total = len(test_dataset)
    if args.num_samples > n_total:
        print(f"Warning: Requested {args.num_samples} samples, but dataset only has {n_total}. Using all examples.")
        args.num_samples = n_total

    num_samples = args.num_samples if args.num_samples > 0 else n_total
    n_test = int(num_samples * args.test_ratio) 
    n_train = num_samples - n_test 
    
    print(f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples")
    
    idx_train = np.random.choice(range(len(train_dataset)), size=n_train, replace=False)
    idx_test = np.random.choice(range(len(test_dataset)), size=n_test, replace=False)

    train_dataset = train_dataset.select(idx_train)
    test_dataset = test_dataset.select(idx_test)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('problem')
            
            question = make_other_prefix(
                question = question_raw, 
                template_type = args.template_type, 
                solution_example = "0", 
                answer_format = "box",
                label_str = "answer"
            )
            # print(question)

            answer_raw = example.pop('solution')
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
                    'question': question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = get_dataset_dir(
        dataset_name=data_source,
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
