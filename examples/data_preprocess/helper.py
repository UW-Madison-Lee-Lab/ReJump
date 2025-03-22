from datasets import Dataset
import os
from verl.utils.hdfs_io import copy, makedirs
import numpy as np
import re
from constants import get_dataset_dir
import pandas as pd
from ray.util import pdb
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
    elif template_type == 'no_reasoning':
        """This do not allow any reasoning"""
        prefix = f"""
        <|im_start|>system
        You are a helpful assistant. You always provide the user directly with the answer without any reasoning.
        <|im_end|>
        <|im_start|>user
        The dataset has {n_classes} classes: {list(range(n_classes))}. {in_context_examples} Given the data point with features {format_features(features)}, classify it into one of the possible classes. Your response should be in <answer> </answer> tags without any other text, for example <answer>1</answer>.
        <|im_end|>
        <|im_start|>assistant
        <answer>
        """
    return prefix

def save_data(
    dataset_dict,
    in_context_dataset_dict,
    data_source,
    args,
    n_classes,
    TRAIN_SIZE,
    TEST_SIZE,
): 
    raw_dataset = Dataset.from_dict(dataset_dict)
    raw_in_context_dataset = Dataset.from_dict(in_context_dataset_dict)
    
    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    # Create non-overlapping train and test sets
    all_indices = np.arange(len(raw_dataset))
    train_indices = np.random.choice(all_indices, TRAIN_SIZE, replace=False)
    # Remove train indices from the pool before selecting test indices
    remaining_indices = np.setdiff1d(all_indices, train_indices)
    test_indices = np.random.choice(remaining_indices, TEST_SIZE, replace=False)
    
    train_dataset = raw_dataset.select(train_indices)
    test_dataset = raw_dataset.select(test_indices)
    in_context_dataset = {
        "train": raw_in_context_dataset.select(train_indices),
        "test": raw_in_context_dataset.select(test_indices)
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

    local_dir = get_dataset_dir(data_source, args.n_shot, args.template_type)
    hdfs_dir = args.hdfs_dir

    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

def classification_reward_fn(solution_str, ground_truth):
    all_matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str, re.DOTALL))
    if all_matches:
        response_extract = None
        for match in all_matches[::-1]:
            if match.group(1).strip().isdigit():
                response_extract = match
                break
        if response_extract is not None and response_extract.group(1).strip().isdigit():
            response_class = int(response_extract.group(1).strip())
            return response_class == ground_truth['label']
        else:
            return 0
    else:
        return 0
    
