"""
Preprocess dataset for linear classification task - a synthetic binary classification task based on linear boundary
"""

import os
import numpy as np
from datasets import Dataset
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
from constants import data_dir
import re
from utils import set_seed


def gen_dataset(
    num_samples: int,
    noise: float = 0.1,
    seed_value: int = 42,
    coefficients: List[float] = [3.55, -0.3],
    intercept: float = 2.0,
) -> List[Tuple]:
    """Generate synthetic linear binary classification dataset.
    
    Args:
        num_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise added to the data
        seed_value: Random seed for reproducibility
        coefficients: Coefficients for the linear boundary
        intercept: Intercept for the linear boundary
        
    Returns:
        List of tuples containing (features, label)
    """
    np.random.seed(seed_value)
    
    # To ensure a balanced dataset, we'll generate samples for each class separately
    samples_per_class = num_samples // 2
    
    samples = []
    
    # Generate positive class samples (y > 0)
    count_positive = 0
    while count_positive < samples_per_class:
        # Generate random features
        feature_vector = np.random.uniform(-5, 5, len(coefficients))
        
        # Calculate raw value using linear function
        raw_value = np.dot(feature_vector, coefficients) + intercept
        
        # Add noise
        noisy_value = raw_value + np.random.normal(0, noise)
        
        # Check if it belongs to the positive class
        if noisy_value > 0:
            samples.append((feature_vector.tolist(), 1))
            count_positive += 1
    
    # Generate negative class samples (y ≤ 0)
    count_negative = 0
    while count_negative < samples_per_class:
        # Generate random features
        feature_vector = np.random.uniform(-5, 5, len(coefficients))
        
        # Calculate raw value using linear function
        raw_value = np.dot(feature_vector, coefficients) + intercept
        
        # Add noise
        noisy_value = raw_value + np.random.normal(0, noise)
        
        # Check if it belongs to the negative class
        if noisy_value <= 0:
            samples.append((feature_vector.tolist(), 0))
            count_negative += 1
    
    # Shuffle the samples
    np.random.shuffle(samples)
    
    return samples


def format_features(features):
    return ", ".join([f"{x:.3f}" for x in features])

def make_prefix(dp, template_type, n_classes, n_shot=0, in_context_dataset=None):
    features = dp['features']
    label = dp['label']
    
    # Add in-context examples if requested
    in_context_examples = ""
    if n_shot > 0 and in_context_dataset is not None:
        in_context_examples = "We first provide you with some examples of how to classify data points.\n"
        # Randomly select indices for in-context examples
        random_indices = np.random.choice(len(in_context_dataset), n_shot, replace=False)
        for i in random_indices:
            example = in_context_dataset[i.item()]
            example_features = example['features']
            example_label = example['label']
            
            in_context_examples += f"Features: {format_features(example_features)}, Label: {example_label}\n"
    
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""
        A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.

        User: The dataset has {n_classes} classes: {list(range(n_classes))}. {in_context_examples} Given the data point with features {format_features(features)}, classify it into one of the possible classes. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer>1</answer>.
        Assistant: Let me solve this step by step.
        <think>
        """
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""
        <|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n
        <|im_start|>user\n The dataset has {n_classes} classes: {list(range(n_classes))}. {in_context_examples} Given the data point with features {format_features(features)}, classify it into one of the possible classes. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer>1</answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>
        """
    return prefix



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--n_shot', type=int, default=0)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'linear'
    TEST_SIZE = int(args.num_samples * args.test_ratio)
    TRAIN_SIZE = args.num_samples - TEST_SIZE
    n_classes = 2  # Binary classification task
    
    # Generate synthetic dataset
    samples = gen_dataset(
        num_samples=args.num_samples,
        noise=args.noise,
        seed_value=42
    )
    
    in_context_samples = gen_dataset(
        num_samples=args.num_samples,
        noise=args.noise,
        seed_value=42
    )
    
    dataset_dict = {
        'features': [sample[0] for sample in samples],
        'label': [sample[1] for sample in samples]
    }
    
    in_context_dataset_dict = {
        'features': [sample[0] for sample in in_context_samples],
        'label': [sample[1] for sample in in_context_samples]
    }
    
    raw_dataset = Dataset.from_dict(dataset_dict)
    raw_in_context_dataset = Dataset.from_dict(in_context_dataset_dict)
    
    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    in_context_dataset = {
        "train": raw_in_context_dataset.select(range(TRAIN_SIZE)),
        "test": raw_in_context_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    }

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(
                example, 
                template_type=args.template_type,
                n_classes=n_classes,
                n_shot=args.n_shot, 
                in_context_dataset=in_context_dataset[split]
            )
            
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

    local_dir = os.path.join(data_dir, "linear", f"{args.n_shot}_shot")
    hdfs_dir = args.hdfs_dir

    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


def linear_reward_fn(response, ground_truth):
    response_extract = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if response_extract is not None and response_extract.group(1).strip().isdigit():
        response_class = int(response_extract.group(1).strip())
    else:
        return 0
    return response_class == ground_truth['label']
