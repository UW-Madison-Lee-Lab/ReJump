"""
Preprocess dataset for multitask learning - combines multiple existing datasets
"""

import numpy as np
import argparse
import os
from utils import set_seed
from datasets import Dataset, concatenate_datasets
from examples.data_preprocess.helper import classification_reward_fn, store_data
from constants import get_mixed_configs, get_dataset_filename, get_dataset_dir
from typing import List
from environment import root_dir

def combine_datasets(
    dataset_paths: List[str],
    dataset_ratios: List[float],
    num_samples: int,
    seed_value: int = 42,
    data_mode: str = "default"
):
    """Combine existing datasets based on specified ratios.
    
    Args:
        dataset_paths: Paths to the existing datasets
        dataset_ratios: Ratios for each dataset (should sum to 1.0)
        num_samples: Total number of samples in the combined dataset
        seed_value: Random seed for reproducibility
        
    Returns:
        Tuple of (combined_train_dataset, combined_test_dataset)
    """
    np.random.seed(seed_value)
    
    # Convert ratio strings to floats if needed
    dataset_ratios = [float(ratio) for ratio in dataset_ratios]
    
    # Validate dataset ratios
    total_ratio = sum(dataset_ratios)
    if not np.isclose(total_ratio, 1.0):
        print(f"Warning: Dataset ratios sum to {total_ratio}, normalizing to 1.0")
        dataset_ratios = [ratio / total_ratio for ratio in dataset_ratios]
    
    # Calculate samples per dataset
    samples_per_dataset = [int(np.floor(ratio * num_samples)) for ratio in dataset_ratios]
    
    # Adjust to ensure we get exactly num_samples
    remaining = num_samples - sum(samples_per_dataset)
    if remaining > 0:
        # Add remaining samples to dataset with highest ratio
        max_idx = dataset_ratios.index(max(dataset_ratios))
        samples_per_dataset[max_idx] += remaining
    
    train_datasets = []
    test_datasets = []
    
    # Load and sample from each dataset
    for i, (dataset_path, samples_count) in enumerate(zip(dataset_paths, samples_per_dataset)):
        if samples_count <= 0:
            continue
            
        # Extract task name from the path (first directory)
        task_name = dataset_path.split('/')[0]
        
        # Construct full path to the dataset
        
        # Load train and test datasets
        train_path = os.path.join(dataset_path, get_dataset_filename(split="train", data_mode=data_mode))
        test_path = os.path.join(dataset_path, get_dataset_filename(split="test", data_mode=data_mode))
        
        train_dataset = Dataset.from_parquet(train_path)
        test_dataset = Dataset.from_parquet(test_path)
        
        # Calculate number of samples to take
        dataset_ratio = samples_count / num_samples
        train_size = min(int(dataset_ratio * len(train_dataset)), len(train_dataset))
        test_size = min(int(dataset_ratio * len(test_dataset)), len(test_dataset))
        
        # Sample from datasets
        train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
        test_indices = np.random.choice(len(test_dataset), test_size, replace=False)
        
        sampled_train = train_dataset.select(train_indices)
        sampled_test = test_dataset.select(test_indices)
        
        # Add task information
        def add_task_info(example):
            example['task'] = task_name
            return example
        
        sampled_train = sampled_train.map(add_task_info)
        sampled_test = sampled_test.map(add_task_info)
        
        train_datasets.append(sampled_train)
        test_datasets.append(sampled_test)
        
        print(f"Added {train_size} training and {test_size} testing samples from {dataset_path}")
        

    
    # Combine all datasets
    if train_datasets and test_datasets:
        combined_train = concatenate_datasets(train_datasets)
        combined_test = concatenate_datasets(test_datasets)
        
        # Shuffle the combined datasets
        train_indices = np.random.permutation(len(combined_train))
        test_indices = np.random.permutation(len(combined_test))
        
        combined_train = combined_train.select(train_indices)
        combined_test = combined_test.select(test_indices)
        
        return combined_train, combined_test
    else:
        raise ValueError("No valid datasets were loaded")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--dataset_path', type=str, nargs='+', default=[f'{root_dir}/datasets/blobs/50_shot/qwen-instruct/10000_samples_3.0_noise'])
    parser.add_argument('--dataset_ratio', type=str, nargs='+', default=['1.0'])
    parser.add_argument('--data_mode', type=str, default="default")

    args = parser.parse_args()
    set_seed(42)
    
    # Validate input
    if len(args.dataset_path) != len(args.dataset_ratio):
        raise ValueError(f"Number of dataset paths ({len(args.dataset_path)}) must match number of ratios ({len(args.dataset_ratio)})")
    
    # Combine existing datasets
    combined_train, combined_test = combine_datasets(
        dataset_paths=args.dataset_path,
        dataset_ratios=args.dataset_ratio,
        num_samples=args.num_samples,
        seed_value=42,
        data_mode=args.data_mode
    )
    
    # Create directory if it doesn't exist
    mixed_configs = get_mixed_configs(
        dataset_paths=args.dataset_path,
        dataset_ratios=args.dataset_ratio,
        num_samples=args.num_samples,
    )
    
    output_dir = get_dataset_dir(**mixed_configs)
    
    store_data(
        train_dataset=combined_train,
        test_dataset=combined_test,
        local_dir=output_dir,
        args=args,
        data_mode=args.data_mode
    )

def multitask_reward_fn(solution_str, ground_truth):
    """Reward function for multitask datasets that uses classification_reward_fn"""
    return classification_reward_fn(solution_str, ground_truth)
