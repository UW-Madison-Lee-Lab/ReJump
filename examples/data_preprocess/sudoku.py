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
Preprocess the sudoku dataset to parquet format
"""

import os
from environment import root_dir
import numpy as np
import json
from verl.utils.hdfs_io import copy
import argparse
from constants import get_dataset_dir
from utils import set_seed

from datasets import Dataset
from examples.data_preprocess.helper import make_other_prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument(
        '--feature_noise',
        type=lambda x: float(x) if x.lower() != 'none' else None,
        default=None
    )
    parser.add_argument('--test_ratio', type=float, default=1)
    parser.add_argument('--n_shot', type=int, default=None)
    parser.add_argument('--n_query', type=int, default=1)
    parser.add_argument('--template_type', type=str, default="reasoning_api")
    parser.add_argument('--label_noise', type=float, default=None)
    parser.add_argument(
        '--data_mode',
        type=str,
        default="default",
        choices=["default"]
    )
    parser.add_argument('--additional_instruction_path', type=str, default=None)

    args = parser.parse_args()
    
    data_source = "sudoku"
    set_seed(42)
    
    if args.n_query != 1:
        raise ValueError("n_query must be 1 for sudoku dataset")
    if args.n_shot != 0:
        raise ValueError("n_shot must be 0 for sudoku dataset")
    if args.feature_noise is not None and args.feature_noise != 0:
        raise ValueError(
            "feature_noise must be 0 or None for sudoku dataset"
        )
    if args.label_noise is not None and args.label_noise != 0:
        raise ValueError(
            "label_noise must be 0 or None for sudoku dataset"
        )
    if args.test_ratio != 1:
        raise ValueError("test_ratio must be 1 for sudoku dataset")

    # Load the sudoku dataset from jsonl file
    sudoku_file = os.path.join(
        root_dir, "examples/data_preprocess/sudoku_6x6_500.jsonl"
    )

    with open(sudoku_file, 'r') as f:
        sudoku_data = [json.loads(line) for line in f]

    print(f"Loaded {len(sudoku_data)} examples from sudoku dataset")

    # Shuffle the data
    np.random.shuffle(sudoku_data)
    
    n_total = len(sudoku_data)
    if args.num_samples > n_total:
        print(
            f"Warning: Requested {args.num_samples} samples, "
            f"but dataset only has {n_total}. Using all examples."
        )
        args.num_samples = n_total

    num_samples = args.num_samples if args.num_samples > 0 else n_total
    n_test = int(num_samples * args.test_ratio)
    n_train = num_samples - n_test

    # Select samples
    train_data = sudoku_data[:n_train]
    test_data = sudoku_data[n_train:num_samples]

    print(
        f"Created {len(train_data)} training examples "
        f"and {len(test_data)} test examples"
    )

    # Define the processing function
    def make_map_fn(split):
        def process_fn(example, idx):
            # Format the sudoku puzzle
            puzzle = example['input']
            solution = example['output']

            # Detect grid size from the puzzle
            lines = puzzle.strip().split('\n')
            grid_size = len(lines)  # Number of rows = grid size
            
            # Generate example solution for the detected size
            if grid_size == 4:
                solution_example = "Answer: 1234\\n2143\\n3412\\n4321"
            elif grid_size == 5:
                solution_example = "Answer: 12345\\n23451\\n34512\\n45123\\n51234"
            elif grid_size == 6:
                solution_example = "Answer: 123456\\n234561\\n345612\\n456123\\n561234\\n612345"
            elif grid_size == 9:
                solution_example = "Answer: 123456789\\n234567891\\n345678912\\n456789123\\n567891234\\n678912345\\n789123456\\n891234567\\n912345678"
            else:
                # Generic example
                solution_example = f"Answer: <{grid_size}x{grid_size} grid>"

            # Create a formatted representation of the puzzle
            question_raw = f'''
Solve the following {grid_size}x{grid_size} Latin Square puzzle. Fill in the missing numbers \
(represented by 0) using only the numbers from 1 to {grid_size}, so that:
- Each row contains all the numbers from 1 to {grid_size} exactly once
- Each column contains all the numbers from 1 to {grid_size} exactly once

In other words, every cell must contain a number between 1 and {grid_size}, and no number can \
repeat in the same row or column.

Puzzle:
{puzzle}

Provide the complete solution grid in the same format as the input. \
Please format your final answer as: Answer: <your solution grid>
            '''.strip()

            if args.additional_instruction_path is not None:
                with open(args.additional_instruction_path, 'r') as f:
                    additional_instructions = f.read()
                question_raw += "\n" + additional_instructions

            answer_raw = solution

            question = make_other_prefix(
                question=question_raw,
                template_type=args.template_type,
                solution_example=solution_example,
                answer_format="tags",
                label_str="the complete solution grid"
            )

            # Store the solution
            solution_dict = {"label": [answer_raw]}

            data = {
                "data_source": data_source,
                "additional_instruction_path": args.additional_instruction_path,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "expert_level_qa",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution_dict
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    'question': question_raw,
                    'puzzle': puzzle,
                }
            }

            return data
        return process_fn

    # Convert to Dataset and apply the processing function
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    train_dataset = train_dataset.map(
        function=make_map_fn('train'),
        with_indices=True,
        remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        function=make_map_fn('test'),
        with_indices=True,
        remove_columns=test_dataset.column_names
    )

    # Define output directories
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

    # Save datasets
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(
        os.path.join(local_dir, 'train_default.parquet')
    )
    test_dataset.to_parquet(
        os.path.join(local_dir, 'test_default.parquet')
    )

    print(
        f"Train dataset saved to "
        f"{os.path.join(local_dir, 'train_default.parquet')}"
    )
    print(
        f"Test dataset saved to "
        f"{os.path.join(local_dir, 'test_default.parquet')}"
    )

    # Copy to HDFS if specified
    if hdfs_dir is not None:
        print(
            f"Copying data from {local_dir} to HDFS directory {hdfs_dir}"
        )
        copy(src=local_dir, dst=hdfs_dir)
        print("Copy to HDFS complete.")


