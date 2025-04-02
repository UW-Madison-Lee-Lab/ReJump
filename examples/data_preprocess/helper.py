from datasets import Dataset
import os
from verl.utils.hdfs_io import copy, makedirs
import numpy as np
import re
from constants import get_dataset_dir
import pandas as pd
import pdb

def prepare_dataset(args, gen_dataset):
    if args.n_shot <= 0:
        raise ValueError("n_shot must be greater than 0")
    samples, in_context_samples = [], []
    if args.data_mode == "default":
        TEST_SIZE = int(args.num_samples * args.test_ratio)
        TRAIN_SIZE = args.num_samples - TEST_SIZE
        
        # Generate synthetic dataset
        samples = gen_dataset(
            num_samples=args.num_samples,
            noise_level=args.noise_level,
            random = False,
            seed_value=12
        )
        
        in_context_samples = gen_dataset(
            num_samples=args.num_samples,
            noise_level=args.noise_level,
            label_flip_rate=args.label_flip_rate,
            random = False,
            seed_value=34
        )
    elif args.data_mode == "grid":
        samples = gen_grid_dataset(grid_size = int(args.num_samples ** 0.5))
        TEST_SIZE = len(samples)
        TRAIN_SIZE = 0
        
        # Generate a small set of in-context examples
        in_context_samples = gen_dataset(
            num_samples=args.n_shot,
            noise_level=args.noise_level,
            seed_value=34,
            random = False
        )
    elif args.data_mode == "mixed":
        TEST_SIZE = int(args.num_samples * args.test_ratio)
        TRAIN_SIZE = args.num_samples - TEST_SIZE
        for i in range(args.num_samples):
            data = gen_dataset(
                num_samples=args.n_shot+1,
                noise_level=args.noise_level,
                label_flip_rate=args.label_flip_rate,
                random = True,
                seed_value=i
            )
            samples.append(data[0])
            in_context_samples.append(data[1:])
    else:
        raise ValueError(f"Invalid data mode: {args.data_mode}")
    
    dataset_dict = {
        'features': [sample[0] for sample in samples],
        'label': [sample[1] for sample in samples]
    }
    
    if args.data_mode == "mixed":
        in_context_dataset_dict = in_context_samples
    else:
        in_context_dataset_dict = {
            'features': [sample[0] for sample in in_context_samples],
            'label': [sample[1] for sample in in_context_samples]
        }

    return {
        "dataset_dict": dataset_dict,
        "in_context_dataset_dict": in_context_dataset_dict,
        "TEST_SIZE": TEST_SIZE,
        "TRAIN_SIZE": TRAIN_SIZE
    }

def gen_grid_dataset(grid_size=100, x_range=(-10, 10), y_range=(-10, 10)):
    """Generate a grid of points for visualization and testing.
    
    Args:
        grid_size: Number of points in each dimension
        x_range: Range of x values
        y_range: Range of y values
    
    Returns:
        List of tuples containing (features, dummy_label)
    """
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    
    samples = []
    for i in range(grid_size):
        for j in range(grid_size):
            features = [xx[i, j], yy[i, j]]
            # Dummy label, will be predicted by the model
            label = 0
            samples.append((features, label))
    
    return samples

def format_features(features):
    return "[" + ", ".join([f"{x:.3f}" for x in features]) + "]"

def flip_label(y, label_flip_rate, n_classes):
    if label_flip_rate > 0:
        num_flips = int(label_flip_rate * len(y))
        flip_indices = np.random.choice(len(y), num_flips, replace=False)
        for i in flip_indices:
            possible_labels = list(range(n_classes))
            possible_labels.remove(y[i])
            y[i] = np.random.choice(possible_labels)
    return y

def make_prefix(dp, template_type, n_classes, n_shot=0, in_context_dataset=None):
    features = dp['features']
    label = dp['label']
    
    # Add in-context examples if requested
    in_context_examples, in_context_samples = "", []
    if n_shot > 0 and in_context_dataset is not None:
        in_context_examples = "We first provide you with some examples of how to classify data points.\n"
        # Randomly select indices for in-context examples
        random_indices = np.random.choice(len(in_context_dataset), n_shot, replace=False)
        
        
        for i in random_indices:
            example = in_context_dataset[i.item()]
            example_features = example['features']
            example_label = example['label']
            
            in_context_examples += f"Features: {format_features(example_features)}, Label: {example_label}\n"
            in_context_samples.append({"features": example_features, "label": example_label})
    
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
        <|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.<|im_end|>\n
        <|im_start|>user\n The dataset has {len(features)} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples} Given the data point with features {format_features(features)}, classify it into one of the possible classes. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer>1</answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>
        """
    elif template_type == 'no_reasoning':
        """This does not allow any reasoning"""
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
    return prefix, in_context_samples


def make_map_fn(split, args, n_classes, in_context_dataset, data_source, data_mode):

    
    def process_fn(example, idx):
        if data_mode in ["grid", "default"]:
            in_context_dataset_ = in_context_dataset[split]
        else:
            in_context_dataset_ = Dataset.from_dict({
                "features": [in_context_dataset[split][idx][i][0] for i in range(len(in_context_dataset[split][idx]))],
                "label": [in_context_dataset[split][idx][i][1] for i in range(len(in_context_dataset[split][idx]))]
            })
        
        question, in_context_samples = make_prefix(
            example, 
            template_type=args.template_type, 
            n_classes=n_classes, 
            n_shot=args.n_shot, 
            in_context_dataset=in_context_dataset_
        )
        
        solution = {
            "features": example['features'],
            "label": example['label'],
            "in_context_samples": in_context_samples
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
    
def save_data(
    dataset_dict,
    in_context_dataset_dict,
    data_source,
    args,
    n_classes,
    TRAIN_SIZE,
    TEST_SIZE,
    data_mode = "default",
): 
    raw_dataset = Dataset.from_dict(dataset_dict)
    
    
    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    # Create non-overlapping train and test sets
    all_indices = np.arange(len(raw_dataset))
    train_indices = np.random.choice(all_indices, TRAIN_SIZE, replace=False)
    # Remove train indices from the pool before selecting test indices
    remaining_indices = np.setdiff1d(all_indices, train_indices)
    test_indices = np.random.choice(remaining_indices, TEST_SIZE, replace=False)
    
    train_dataset = raw_dataset.select(train_indices)
    test_dataset = raw_dataset.select(test_indices)
    

    
    if data_mode == "default":
        raw_in_context_dataset = Dataset.from_dict(in_context_dataset_dict)
        in_context_dataset = {
            "train": raw_in_context_dataset.select(train_indices),
            "test": raw_in_context_dataset.select(test_indices)
        }
    elif data_mode == "grid":
        raw_in_context_dataset = Dataset.from_dict(in_context_dataset_dict)
        in_context_dataset = {
            "train": raw_in_context_dataset.select(train_indices),
            "test": raw_in_context_dataset
        }
    elif data_mode == "mixed":
        in_context_dataset = {
            "train": [in_context_dataset_dict[i] for i in train_indices],
            "test": [in_context_dataset_dict[i] for i in test_indices]
        }
    else:
        raise ValueError(f"Invalid data mode: {data_mode}")

    train_dataset = train_dataset.map(function=make_map_fn('train', args, n_classes, in_context_dataset, data_source, data_mode), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test', args, n_classes, in_context_dataset, data_source, data_mode), with_indices=True)


    local_dir = get_dataset_dir(
        dataset_name=data_source,
        shot=args.n_shot,
        template_type=args.template_type,
        num_samples=args.num_samples,
        noise_level=args.noise_level,
        label_flip_rate=args.label_flip_rate
    )

    store_data(
        local_dir=local_dir,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        args=args,
        data_mode=data_mode
    )
    
        
def store_data(
    local_dir,
    train_dataset,
    test_dataset,
    args,
    data_mode
):
    if data_mode == "grid":
        test_dataset.to_parquet(os.path.join(local_dir, 'grid.parquet'))
    else:
        hdfs_dir = args.hdfs_dir

        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        train_dataset.to_parquet(os.path.join(local_dir, f'train_{data_mode}.parquet'))
        test_dataset.to_parquet(os.path.join(local_dir, f'test_{data_mode}.parquet'))

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
    
