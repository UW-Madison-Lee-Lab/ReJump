"""
Preprocess dataset for blobs task - a synthetic classification task with n-dimensional data points
"""

import os
import numpy as np
from datasets import Dataset
from sklearn.datasets import make_blobs
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
from constants import data_dir


def gen_dataset(
    num_samples: int,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    seed_value: int = 42,
) -> List[Tuple]:
    """Generate synthetic blob dataset for classification task.
    
    Args:
        num_samples: Number of samples to generate
        n_features: Number of features for each sample
        centers: Number of classes/clusters
        cluster_std: Standard deviation of the clusters
        center_box: Bounding box for each cluster center
        seed_value: Random seed for reproducibility
        
    Returns:
        List of tuples containing (features, label)
    """
    np.random.seed(seed_value)
    
    # Generate synthetic data
    X, y = make_blobs(
        n_samples=num_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        center_box=center_box,
        random_state=seed_value
    )
    
    samples = []
    for i in tqdm(range(num_samples)):
        features = X[i].tolist()
        label = int(y[i])
        samples.append((features, label))
    
    return samples

def make_prefix(dp, template_type):
    features = dp['features']
    label = dp['label']
    
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Given the data point with features {features}, classify it into one of the possible classes. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> Class 2 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\nGiven the data point with features {features}, classify it into one of the possible classes. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> Class 2 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--n_features', type=int, default=2)
    parser.add_argument('--centers', type=int, default=3)
    parser.add_argument('--cluster_std', type=float, default=1.0)
    parser.add_argument('--train_size', type=int, default=80000)
    parser.add_argument('--test_size', type=int, default=20000)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'blobs'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    
    # Generate synthetic dataset
    samples = gen_dataset(
        num_samples=args.num_samples,
        n_features=args.n_features,
        centers=args.centers,
        cluster_std=args.cluster_std,
        seed_value=42
    )
    
    # Create dataset
    dataset_dict = {
        'features': [sample[0] for sample in samples],
        'label': [sample[1] for sample in samples]
    }
    
    raw_dataset = Dataset.from_dict(dataset_dict)
    
    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "features": example['features'],
                "label": example['label']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "classification",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = os.path.join(data_dir, "blobs")
    hdfs_dir = args.hdfs_dir

    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
