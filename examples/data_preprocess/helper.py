from datasets import Dataset
import os
from verl.utils.hdfs_io import copy, makedirs
import numpy as np
import re
from constants import get_dataset_dir, get_dataset_filename, supported_datasets
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
        raw_samples = gen_dataset(
            num_samples=args.num_samples*args.n_query,
            feature_noise=args.feature_noise,
            random = False,
            seed_value=12
        )
        
        in_context_samples = gen_dataset(
            num_samples=args.num_samples*args.n_shot,
            feature_noise=args.feature_noise,
            label_noise=args.label_noise,
            random = False,
            seed_value=34
        )

        samples = []
        for i in range(args.num_samples):
            random_indices = np.random.choice(args.num_samples*args.n_query, args.n_query, replace= args.n_query>=args.num_samples)
            batchs = []
            for j in random_indices:
                batchs.append(raw_samples[j])
            samples.append(batchs)

        ##samples = [raw_samples[i:i+args.n_query] for i in range(0, len(samples), args.n_query)] if we don't want repeated test samples

    elif args.data_mode == "grid":
        samples = gen_grid_dataset(grid_size = int(args.num_samples ** 0.5))
        TEST_SIZE = len(samples)
        TRAIN_SIZE = 0
        
        # Generate a small set of in-context examples
        in_context_samples = gen_dataset(
            num_samples=args.n_shot,
            feature_noise=args.feature_noise,
            seed_value=34,
            random = False
        )
    elif args.data_mode == "mixed":
        TEST_SIZE = int(args.num_samples * args.test_ratio)
        TRAIN_SIZE = args.num_samples - TEST_SIZE
        for i in range(args.num_samples):
            data = gen_dataset(
                num_samples=args.n_shot+1,
                feature_noise=args.feature_noise,
                label_noise=args.label_noise,
                random = True,
                seed_value=i
            )
            samples.append(data[0])
            in_context_samples.append(data[1:])
    else:
        raise ValueError(f"Invalid data mode: {args.data_mode}")
    
    dataset_dict = {
        'features': [[sample_tuple[0] for sample_tuple in batch] for batch in samples],
        'label': [[sample_tuple[1] for sample_tuple in batch] for batch in samples]
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

def flip_label(y, label_noise, n_classes):
    if label_noise > 0:
        num_flips = int(label_noise * len(y))
        flip_indices = np.random.choice(len(y), num_flips, replace=False)
        for i in flip_indices:
            possible_labels = list(range(n_classes))
            possible_labels.remove(y[i])
            y[i] = np.random.choice(possible_labels)
    return y

def make_prefix(dp, template_type, n_classes, n_shot=0, in_context_dataset=None, customized_prompt=None):
    features = dp['features']
    label = dp['label']

    # Add in-context examples if requested
    in_context_examples, in_context_samples = "", []
    if n_shot > 0 and in_context_dataset is not None:
        in_context_examples = "We first provide you with some examples of how to classify data points.\n"
        random_indices = np.random.choice(len(in_context_dataset), n_shot, replace= len(in_context_dataset) < n_shot)
        for i in random_indices:
            example = in_context_dataset[i.item()]
            example_features = example['features']
            example_label = example['label']
            
            in_context_examples += f"Features: {format_features(example_features)}, Label: {example_label}\n"
            in_context_samples.append({"features": example_features, "label": example_label})
    # prompt construction
    if n_query > 1:
        query = "Given the following data points:\n"
        for i in range(n_query):
            query += f"{i+1}. Features: {format_features(features[i])}\n"
        query += "Classify each data point into one of the possible classes, and list the corresponding class labels separated by commas."
        label_str = "class labels"
    else:
        query = f"Given the data point with features {format_features(features[0])}, classify it into one of the possible classes. "
        label_str = "class label"
    answer_example_number = np.random.choice(range(n_classes), n_query, replace=True)
    if "reasoning_api" in template_type or "standard_api_no_reasoning" in template_type:
        answer_example = f"{', '.join([str(x) for x in answer_example_number])}"
    else:
        answer_example = f"<answer>{', '.join([str(x) for x in answer_example_number])}</answer>"
    

    if template_type == 'base':
        prefix = f"""
        A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.

        User: The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples} {query} Please provide your thinking process in <think> </think> tags. Your final answer should be enclosed in <answer> and </answer> tags, containing only {label_str} with no additional text—for example, {answer_example}
        Assistant: Let me solve this step by step.
        <think>
        """
    elif template_type == 'qwen-instruct':
        prefix = f"""
        <|im_start|>system
        You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.
        <|im_end|>
        <|im_start|>user
        The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples} {query} Please provide your thinking process in <think> </think> tags. Your final answer should be enclosed in <answer> and </answer> tags, containing only {label_str} with no additional text—for example, {answer_example}
        <|im_end|>
        <|im_start|>assistant
        Let me solve this step by step.
        <think>
        """
    elif template_type == 'base_no_reasoning':
        prefix = f"""
        A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.

        User: The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples} {query} Your final answer should be enclosed in <answer> and </answer> tags, containing only {label_str} with no additional text—for example, {answer_example}
        Assistant: 
        """
    elif template_type == 'reasoning_api':
        prefix = f"""
        The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples} {query} Your final answer should just be the {label_str}, without any other text, e.g., {answer_example}
        """
    elif template_type == "reasoning_api_customized":
        prefix = f"""
        The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples} {query} {customized_prompt} Your answer should just be {label_str}, without any other text, e.g., {answer_example}
        """
    elif template_type == "standard_api_no_reasoning":
        prefix = f"""
        The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples} {query} Your answer should be just the {label_str}, without any other text, e.g., {answer_example}
        """
    elif template_type == "standard_api":
        prefix = f"""
        The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples} {query}
        Let's think step by step. Please provide your thinking process in <think> </think> tags. Your final answer should be enclosed in <answer> and </answer> tags, containing only {label_str} with no additional text—for example, {answer_example}
        """
    else:
        raise ValueError(f"Invalid template type: {template_type}")
    
    return prefix, in_context_samples

def make_regression_prefix(
    dp, 
    template_type, 
    n_classes = None, 
    n_shot=0, 
    n_query=1,
    in_context_dataset=None, 
    customized_prompt=None,
):
    features = dp['features']
    label = dp['label']

    
    # Add in-context examples if requested
    in_context_examples, in_context_samples = "", []
    if n_shot > 0 and in_context_dataset is not None:
        in_context_examples = "We first provide you with some examples of how to predict values for data points.\n"
        # randomly select n_shot examples
        random_indices = np.random.choice(len(in_context_dataset), n_shot, replace= len(in_context_dataset) < n_shot)
        
        for i in random_indices:
            example = in_context_dataset[i.item()]
            example_features = example['features']
            example_target = example['label']
            
            in_context_examples += f"Features: {format_features(example_features)}, target: {example_target:.3f}\n"
            in_context_samples.append({"features": example_features, "target": example_target})
    
    # prompt construction
    if n_query > 1:
        query = "Given the following data points with features:\n"
        for i in range(n_query):
            query += f"{i+1}. Features: {format_features(features[i])}\n"
        query += "predict target values for each data point, separated by commas. "
        target_str = "predicted values for all samples"

    else:
        query = f"Given the data point with features {format_features(features[0])}, predict the target value. "
        target_str = "predicted value"

    random_targets = [np.round(np.random.uniform(0, 10), 3) for _ in range(n_query)]
    if "reasoning_api" in template_type or "standard_api_no_reasoning" in template_type:
        answer_example = f"{', '.join(str(x) for x in random_targets)}"
    else:
        answer_example = f"<answer>{', '.join(str(x) for x in random_targets)}</answer>"

    if template_type == 'base':
        prefix = f"""
        A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.

        User: The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples} {query} Please provide your thinking process in <think> </think> tags. Your final answer should be enclosed in <answer> and </answer> tags, containing only {target_str} with no additional text—for example, {answer_example}
        Assistant: Let me solve this step by step.
        <think>
        """
    elif template_type == 'qwen-instruct':
        prefix = f"""
        <|im_start|>system
        You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.
        <|im_end|>
        <|im_start|>user
        The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples} {query} Please provide your thinking process in <think> </think> tags. Your final answer should be enclosed in <answer> and </answer> tags, containing only {target_str} with no additional text—for example, {answer_example}
        <|im_end|>
        <|im_start|>assistant
        Let me solve this step by step.
        <think>
        """
    elif template_type == 'qwen-instruct_no_reasoning':
        prefix = f"""
        <|im_start|>system
        You are a helpful assistant. You always provide the user directly with the answer without any reasoning.
        <|im_end|>
        <|im_start|>user
        The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples} {query} Your final answer should be enclosed in <answer> and </answer> tags, containing only {target_str} with no additional text—for example, {answer_example}
        <|im_end|>
        <|im_start|>assistant
        <answer>
        """
    elif template_type == 'base_no_reasoning':
        prefix = f"""
        A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.

        User: The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples} {query} Your final answer should be enclosed in <answer> and </answer> tags, containing only {target_str} with no additional text—for example, {answer_example}
        Assistant: 
        """
    elif template_type == 'reasoning_api':
        prefix = f"""
        The dataset has {len(features)} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples} Given the data point with features {format_features(features)}, classify it into one of the possible classes. Your answer should be just the class label, without any other text or punctuation.
        """
    elif template_type == "reasoning_api_customized":
        prefix = f"""
        The dataset has {len(features)} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples} Given the data point with features {format_features(features)}, classify it into one of the possible classes. \n{customized_prompt}\n Your answer should be just the class label, without any other text or punctuation.
        """
    elif template_type == "standard_api_no_reasoning":
        prefix = f"""
        The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples} {query} Your response should contain only {target_str} with no additional text—for example, {answer_example}
        """
    elif template_type == "standard_api":
        prefix = f"""
        The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples} {query} Let's think step by step. 
        Please provide your thinking process in <think> </think> tags. Your final answer should be enclosed in <answer> and </answer> tags, containing only {target_str} with no additional text—for example, {answer_example}
        """
    else:
        raise ValueError(f"Invalid template type: {template_type}")
    
    return prefix, in_context_samples


def make_map_fn(split, args, n_classes, in_context_dataset, data_source, data_mode, customized_prompt=None):

    
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
            task_type = supported_datasets[data_source]['type'],
            n_shot=args.n_shot, 
            in_context_dataset=in_context_dataset_,
            customized_prompt=customized_prompt
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
    
    #assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE

    # Create non-overlapping train and test sets
    all_indices = np.arange(len(raw_dataset))
    train_indices = np.random.choice(all_indices, TRAIN_SIZE, replace=False)
    # Remove train indices from the pool before selecting test indices
    remaining_indices = np.setdiff1d(all_indices, train_indices)
    test_indices = np.random.choice(remaining_indices, TEST_SIZE, replace=False)
    
    train_dataset = raw_dataset.select(train_indices)
    test_dataset = raw_dataset.select(test_indices)

    in_context_dataset_length = len(in_context_dataset_dict[list(in_context_dataset_dict.keys())[0]])
    all_indices = np.arange(in_context_dataset_length)
    train_indices = np.random.choice(all_indices, int((1-args.test_ratio)*in_context_dataset_length), replace=False)
    remaining_indices = np.setdiff1d(all_indices, train_indices)
    test_indices = np.random.choice(remaining_indices, int(args.test_ratio*in_context_dataset_length), replace=False)
    
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
    
    if "customized" in args.template_type:
        while True:
            customized_prompt = input("Enter a customized prompt for the reasoning API (use '123456' as line break): ")
            customized_prompt = customized_prompt.replace("123456", "\n")
            print("The prompt for the reasoning API is: \n\n", customized_prompt, "\n\nare you sure you want to use this prompt? (y/n/q)")
            if input() == "y":
                break
            elif input() == "n":
                continue
            elif input() == "q":
                exit()
            else:
                print("Please enter a valid prompt")
    else:
        customized_prompt = None

    train_dataset = train_dataset.map(function=make_map_fn(
        split='train', 
        args=args, 
        n_classes=n_classes, 
        in_context_dataset=in_context_dataset, 
        data_source=data_source, 
        data_mode=data_mode, 
        customized_prompt=customized_prompt
    ), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn(
        split='test', 
        args=args, 
        n_classes=n_classes, 
        in_context_dataset=in_context_dataset, 
        data_source=data_source, 
        data_mode=data_mode, 
        customized_prompt=customized_prompt
    ), with_indices=True)


    local_dir = get_dataset_dir(
        dataset_name=data_source,
        shot=args.n_shot,
        n_query=args.n_query,
        template_type=args.template_type,
        num_samples=args.num_samples,
        feature_noise=args.feature_noise,
        label_noise=args.label_noise,
        data_mode=data_mode,
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
        grid_path = os.path.join(local_dir, get_dataset_filename(split="grid", data_mode=data_mode))
        test_dataset.to_parquet(grid_path)
        print(f"Test dataset saved to {grid_path}")
    else:
        hdfs_dir = args.hdfs_dir

        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        train_path = os.path.join(local_dir, get_dataset_filename(split="train", data_mode=data_mode))
        test_path = os.path.join(local_dir, get_dataset_filename(split="test", data_mode=data_mode))
        train_dataset.to_parquet(train_path)
        test_dataset.to_parquet(test_path)

        print(f"Train dataset saved to {train_path}")
        print(f"Test dataset saved to {test_path}")

        if hdfs_dir is not None:
            makedirs(hdfs_dir)
            copy(src=local_dir, dst=hdfs_dir)
            
def classification_extract_solution(solution_str):
    # Extract the answer from the solution string
    direct_match = re.search(r'<answer>((?:[-+]?\d+(?:\s*,\s*[-+]?\d+)*))</answer>', solution_str, re.DOTALL)
    if direct_match:
        answer_content = direct_match.group(1).strip()
        try:
            answers = [int(val.strip()) for val in answer_content.split(',') if val.strip().isdigit()]
        except ValueError:
            answers = []
        return answers

    cleaned_solution_str = solution_str
    cleaned_solution_str = re.sub(r'</answer>(\s*</answer>)+', '</answer>', cleaned_solution_str)
    cleaned_solution_str = re.sub(r'</answer>(\s*</answer>)+', '</answer>', cleaned_solution_str)
    
    clean_match = re.search(r'<answer>(.*?)</answer>', cleaned_solution_str, re.DOTALL)
    if clean_match:
        answer_content = clean_match.group(1).strip()
        try:
            answers = [int(val.strip()) for val in answer_content.split(',') if val.strip().isdigit()]
        except ValueError:
            answers = []
        return answers
    
    lenient_matches = re.findall(r'<answer[^>]*>(\d+)', cleaned_solution_str)
    if lenient_matches:
        try:
            answers = [int(val.strip()) for val in answer_content.split(',') if val.strip().isdigit()]
        except ValueError:
            answers = []
        return answers
    return []

def classification_reward_fn(solution_str, ground_truth):
    # Extract the answer from the solution string
    extracted_answers = classification_extract_solution(solution_str)
    gt_labels = ground_truth['label'] if isinstance(ground_truth['label'], list) else list(ground_truth['label'])

    # Check if the extracted answers match the ground truth
    if extracted_answers and len(extracted_answers) == len(gt_labels):
        return sum(a == b for a, b in zip(extracted_answers, gt_labels))/ len(gt_labels)
    else:
        # Fallback to random scores if no answers are found
        import random
        answers = [random.uniform(0, 10) for _ in range(len(ground_truth['label']))]
        return sum(a == b for a, b in zip(extracted_answers, gt_labels))/ len(gt_labels)
    
def regression_extract_solution(solution_str):
    # Extract the answer from the solution string

    direct_match = re.search(r'<answer>((?:[-+]?\d+\.\d+)(?:,\s*[-+]?\d+\.\d+)*)</answer>', solution_str, re.DOTALL)
    if direct_match:
        answer_content = direct_match.group(1).strip()
        try:
            answers = [float(val.strip()) for val in answer_content.split(',') if val.strip() != '']
            return answers
        except ValueError:
            answers = []


    cleaned_solution_str = solution_str
    cleaned_solution_str = re.sub(r'<answer>(\s*<answer>)+', '</answer>', cleaned_solution_str)
    cleaned_solution_str = re.sub(r'</answer>(\s*</answer>)+', '</answer>', cleaned_solution_str)
    
    clean_match = re.search(r'<answer>((?:[-+]?\d+\.\d+)(?:,\s*[-+]?\d+\.\d+)*)</answer>', cleaned_solution_str, re.DOTALL)
    if clean_match:
        answer_content = clean_match.group(1).strip()
        try:
            answers = [float(val.strip()) for val in answer_content.split(',') if val.strip() != '']
            return answers
        except ValueError:
            answers = []

    clean_match_lenient = re.search(r'<answer>(.*?)</answer>', cleaned_solution_str, re.DOTALL)
    if clean_match_lenient:
        answer_content = clean_match_lenient.group(1).strip()
        try:
            if ',' in answer_content:
                answers = [float(val.strip()) for val in answer_content.split(',') if val.strip() != '']
            else:
                answers = [float(val.strip()) for val in answer_content.split() if val.strip() != '']
            return answers
        except ValueError:
            num_match = re.match(r'^((?:[-+]?\d+\.\d+\s*)+)', answer_content)
            if num_match:
                numbers_block = num_match.group(1)
                try:
                    answers = [float(x) for x in numbers_block.split() if x.strip() != '']
                    return answers
                except ValueError:
                    pass
            else: 
                multiline_matches = re.findall(r'((?:^\s*[-+]?\d+\.\d+\s*$\n?)+)', answer_content, re.M)
                if multiline_matches:
                    last_block = multiline_matches[-1]
                    try:
                        answers = [float(x) for x in last_block.split() if x.strip() != '']
                        return answers
                    except ValueError:
                        pass

    # # Use a more lenient pattern for other cases
    # # This handles partial or malformed tags
    # lenient_match = re.search(r'<answer[^>]*>(\d+)[^<]*(?:</answer>|<\\/answer>|<?/?answer>)', cleaned_solution_str)
    # if lenient_match:
    #     response_class = int(lenient_match.group(1).strip())
    #     return response_class 
    
    # # Last resort - just look for digits between any answer-like tags
    # fallback_matches = re.findall(r'<answer.*?>(\d+).*?(?:</answer>|<\\/answer>|answer>)', solution_str, re.DOTALL)
    # if fallback_matches:
    #     for answer in fallback_matches[::-1]:
    #         if answer.strip().isdigit():
    #             response_class 
    #             return response_class
            
    # last last resort - look for digits in the last line of the solution string
    # no_answer_lines = [line for line in cleaned_solution_str.splitlines() if line.strip() != '']
    # while no_answer_lines and re.search(r'\d', no_answer_lines[-1]) is None:
    #     no_answer_lines.pop()

    # if no_answer_lines:
    #     last_line = no_answer_lines[-1].strip()
    #     last_line = re.sub(r'(?<=\.)\s+', '', last_line)
    #     pattern = r'^([-+]?\d+\.\d+(?:\s*,\s*[-+]?\d+\.\d+)*)$'
    #     if re.match(pattern, last_line):
    #         try:
    #             answers = [float(val.strip()) for val in last_line.split(',')]
    #             return answers
    #         except ValueError:
    #             return []
    return []

def regression_reward_fn(solution_str, ground_truth):
    def criterion(y_pred, y_true):
        return -(y_true - y_pred) ** 2

    answers = regression_extract_solution(solution_str)
    gt_labels = ground_truth['label']
    # Convert numpy array to list if needed.
    if hasattr(gt_labels, 'tolist'):
        gt_labels = gt_labels.tolist()
    elif not isinstance(gt_labels, list):
        gt_labels = [gt_labels]
    if answers and len(answers) == len(gt_labels):
        scores = [criterion(y_pred, y_true) for y_pred, y_true in zip(answers, gt_labels)]
        return sum(scores) / len(scores)
    else:
        # Fallback to random scores if no answers are found
        import random
        answers = [random.uniform(0, 10) for _ in range(len(ground_truth['label']))]
        scores = [criterion(y_pred, y_true) for y_pred, y_true in zip(answers, gt_labels)]
        return sum(scores) / len(scores)
    
    
def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        from verl.utils.reward_score import gsm8k
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        from verl.utils.reward_score import multiply
        return multiply.compute_score
    elif "countdown" in data_source:
        from verl.utils.reward_score import countdown
        return countdown.compute_score
    elif data_source in supported_datasets:
        task_type = supported_datasets[data_source]['type']
        if task_type == 'classification':
            return classification_reward_fn
        elif task_type == 'regression':
            return regression_reward_fn
    else:
        raise NotImplementedError
    
def _select_parse_fn(data_source):
    if data_source == 'openai/gsm8k':
        from verl.utils.reward_score import gsm8k
        return gsm8k.extract_solution
    elif data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.extract_solution
    elif "multiply" in data_source or "arithmetic" in data_source:
        from verl.utils.reward_score import multiply
        return multiply.extract_solution
    elif "countdown" in data_source:
        from verl.utils.reward_score import countdown
        return countdown.extract_solution
    elif data_source in supported_datasets:
        task_type = supported_datasets[data_source]['type']
        if task_type == 'classification':
            return classification_extract_solution
        elif task_type == 'regression':
            return regression_extract_solution
    else:
        raise NotImplementedError