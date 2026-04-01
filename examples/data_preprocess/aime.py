"""
Preprocess the AIME 2026 dataset to parquet format
"""

import os
from environment import root_dir
import numpy as np
import argparse
from constants import get_dataset_dir
from utils import set_seed

from datasets import Dataset
from examples.data_preprocess.helper import make_other_prefix
from datasets import load_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--feature_noise', type=lambda x: float(x) if x.lower() != 'none' else None, default=None)
    parser.add_argument('--test_ratio', type=float, default=1)
    parser.add_argument('--n_shot', type=int, default=0)
    parser.add_argument('--n_query', type=int, default=1)
    parser.add_argument('--template_type', type=str, default="reasoning_api")
    parser.add_argument('--label_noise', type=float, default=None)
    parser.add_argument('--data_mode', type=str, default="default", choices=["default"])

    args = parser.parse_args()
    
    data_source = "aime"
    set_seed(42)
    
    if args.n_query != 1:
        raise ValueError("n_query must be 1 for AIME dataset")
    if args.n_shot != 0:
        raise ValueError("n_shot must be 0 for AIME dataset")

    dataset = load_dataset("MathArena/aime_2026", split="train")
    
    train_dataset = Dataset.from_list([])
    test_dataset = dataset
    
    n_total = len(test_dataset)
    if args.num_samples > n_total or args.num_samples <= 0:
        args.num_samples = n_total

    num_samples = args.num_samples
    n_test = int(num_samples * args.test_ratio) 
    n_train = num_samples - n_test 
    
    print(f"Loaded {n_total} AIME 2026 problems, using {num_samples}")
    
    if n_test < n_total:
        idx_test = np.random.choice(range(n_total), size=n_test, replace=False)
        test_dataset = test_dataset.select(idx_test)

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('problem')
            answer_raw = str(example.pop('answer'))
            
            question = make_other_prefix(
                question=question_raw, 
                template_type=args.template_type, 
                solution_example="0", 
                answer_format="tags",
                label_str="answer"
            )

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"label": [answer_raw]}
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

    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, 'train_default.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_default.parquet'))
    
    print(f"Saved to {local_dir}")
    print(f"Test dataset: {len(test_dataset)} problems")
