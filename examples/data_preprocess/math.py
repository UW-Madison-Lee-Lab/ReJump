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

from verl.utils.reward_score.general import remove_boxed, last_boxed_only_string
from datasets import Dataset
from examples.data_preprocess.helper import make_other_prefix

def extract_solution(solution_str):
    return {
        "label": [remove_boxed(last_boxed_only_string(solution_str))],
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
    
    data_source = "math"
    set_seed(42)
    
    if args.n_query != 1:
        raise ValueError("n_query must be 1 for MATH dataset")
    if args.n_shot != 0:
        raise ValueError("n_shot must be 0 for MATH dataset")
    if args.feature_noise is not None and args.feature_noise != 0:
        raise ValueError("feature_noise must be 0 or None for MATH dataset")
    if args.label_noise is not None and args.label_noise != 0:
        raise ValueError("label_noise must be 0 or None for MATH dataset")


    source_dir = f"{root_dir}/datasets/math"
    if not os.path.exists(f"{source_dir}/MATH"):
        # Create the directory for the math dataset
        makedirs(source_dir)
        
        # Download the dataset
        math_zip_path = os.path.join(source_dir, "MATH.zip")
        if not os.path.exists(math_zip_path):
            print(f"Downloading MATH dataset to {math_zip_path}...")
            urllib.request.urlretrieve(
                "https://www.modelscope.cn/datasets/opencompass/competition_math/resolve/master/data/MATH.zip",
                math_zip_path
            )
        
        # Extract the dataset
        with zipfile.ZipFile(math_zip_path, 'r') as zip_ref:
            zip_ref.extractall(source_dir)
        
    
    # Load the MATH dataset
    math_dir = os.path.join(source_dir, "MATH")
    train_data = []
    test_data = []
    
    # Load train data
    train_path = os.path.join(math_dir, "train")
    for category_dir in glob.glob(os.path.join(train_path, "*")):
        if os.path.isdir(category_dir):
            for problem_file in glob.glob(os.path.join(category_dir, "*.json")):
                with open(problem_file, 'r') as f:
                    problem_data = json.load(f)
                    train_data.append(problem_data)
    
    # Load test data
    test_path = os.path.join(math_dir, "test")
    for category_dir in glob.glob(os.path.join(test_path, "*")):
        if os.path.isdir(category_dir):
            for problem_file in glob.glob(os.path.join(category_dir, "*.json")):
                with open(problem_file, 'r') as f:
                    problem_data = json.load(f)
                    test_data.append(problem_data)
    
    # Convert to Dataset objects
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    print(f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples")
    
    n_test = int(args.num_samples * args.test_ratio)
    n_train = args.num_samples - n_test
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
