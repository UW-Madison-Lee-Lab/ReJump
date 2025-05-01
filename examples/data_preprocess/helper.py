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
    

def make_classification_prefix(
    dp, 
    template_type, 
    n_classes, 
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
        if 'inductive' in template_type:
            query = "Now, please apply your rule to the following data points:\n"
        else:
            query = "Given the following data points:\n"
        for i in range(n_query):
            query += f"{i+1}. Features: {format_features(features[i])}\n"
        query += "Classify each data point into one of the possible classes, and list the corresponding class labels separated by commas."
        label_str = "class labels"
    else:
        if 'inductive' in template_type:
            query = f"Now, please apply your rule to the data point {format_features(features[0])}, classify it into one of the possible classes. "
        else:
            query = f"Given the data point with features {format_features(features[0])}, classify it into one of the possible classes. "
        label_str = "class label"

    answer_example_number = np.random.choice(range(n_classes), n_query, replace=True)
    if "reasoning_api" in template_type or "standard_api_no_reasoning" in template_type:
        rule_example = " the class label is 1 if the first feature is greater than the second, otherwise 0 "
        answer_example = f"{', '.join([str(x) for x in answer_example_number])}"
    else:
        answer_example = f"<answer>{', '.join([str(x) for x in answer_example_number])}</answer>"
        rule_example = " <answer>the class label is 1 if the first feature is greater than the second, otherwise 0</answer> "

    if 'inductive' in template_type:
        # 1. base_inductive
        if template_type == 'base_inductive':
            datasample = f"""
            A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
            User: The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples}
            Please infer a rule that maps the features to the class labels.
            Your final answer should be enclosed in <answer> and </answer> tags. For example, {rule_example}.
            """
            question = f"""
            Good job! 
            {query}
            Your final answer should be enclosed in <answer> and </answer> tags, containing only the {label_str}, for example {answer_example}
            """
        # 2. qwen-instruct_inductive
        elif template_type == 'qwen-instruct_inductive':
            datasample = f"""
            <|im_start|>system
            You are a helpful assistant. You first infer a classification rule from the examples in your mind and then provide the answer.
            <|im_end|>
            <|im_start|>user
            The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples}
            Please infer a rule that maps the features to the class labels.
            Your final answer should be enclosed in <answer> and </answer> tags. For example, {rule_example}.
            <|im_end|>
            <|im_start|>assistant
            """
            question = f"""
            Good job!
            {query}
            Your final answer should be enclosed in <answer> and </answer> tags, containing only the {label_str}, for example {answer_example}
            """
        # 3. qwen-instruct_no_reasoning_inductive
        elif template_type == 'qwen-instruct_no_reasoning_inductive':
            datasample = f"""
            <|im_start|>system
            You are a helpful assistant. You infer the classification rule yourself but do not show your reasoning, only give the final label.
            <|im_end|>
            <|im_start|>user
            The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples}
            Please infer a rule that maps the features to the class labels.
            Your final answer should be enclosed in <answer> and </answer> tags. For example, {rule_example}.
            <|im_end|>
            <|im_start|>assistant
            <answer>
            """
            question = f"""
            <|im_start|>system
            You are a helpful assistant. You infer the classification rule yourself but do not show your reasoning, only give the final label.
            <|im_end|>
            <|im_start|>user
            {query} Your response should contain only the final answer enclosed in <answer> and </answer> tags, with no additional text—specifically, just {label_str}, for example: {answer_example}
            <|im_end|>
            <|im_start|>assistant
            """
        elif template_type == 'reasoning_api_inductive':
            datasample = f"""
            The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. Here are some examples:
            {in_context_examples}
            Please infer a rule that maps the features to the class labels.
            Return your rule in a few sentences—for example, {rule_example}.
            """
            question = f"""
            Good job!
            {query}
            Your final answer should just be the {label_str}, without any other text, e.g., {answer_example}
            """
        elif template_type == "reasoning_api_customized":
            datasample = f"""
            The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. Here are some examples:
            {in_context_examples}
            Please infer a rule that maps the features to the class labels.
            Return your rule in a few sentences—for example, {rule_example}.
            """
            question = f"""
            Good job!
            {query} {customized_prompt} Your answer should just be {label_str}, without any other text, e.g., {answer_example}
        """
        # 4. base_no_reasoning_inductive
        elif template_type == 'base_no_reasoning_inductive':
            datasample = f"""
            A conversation between User and Assistant. The user gives examples but the assistant does not show its inner reasoning.
            User: The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples}
            Please infer a rule that maps the features to the class labels.
            Your final answer should be enclosed in <answer> and </answer> tags. For example, {rule_example}.
            Assistant:
            """
            question = f"""
            Good job!
            {query}
            Your final answer should be enclosed in <answer> and </answer> tags, containing only the {label_str}, for example {answer_example}
            """
        # 5. standard_api_inductive
        elif template_type == 'standard_api_inductive':
            datasample = f"""
            The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. Here are some examples:
            {in_context_examples}
            Please infer a rule that maps the features to the class labels.
            Return your rule in a few sentences—for example, {rule_example}.
            if you think KNN is needed to classify the points, please specify the K value.
            """
            question = f"""
            Good job!
            {query} You can use KNN to classify the points if you think the rule is not enough.
            Your response should contain only the {label_str} in <answer> tags, with no extra text—for example, {answer_example}
            """
        # 6. standard_api_no_reasoning_inductive
        elif template_type == 'standard_api_no_reasoning_inductive':
            datasample = f"""
            The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples}
            Please infer a rule that maps the features to the class labels.
            Your final answer should be enclosed in <answer> and </answer> tags. For example, {rule_example}.
            """
            question = f"""
            Good job!
            {query} 
            Your answer should be just the {label_str}, without any other text, e.g., {answer_example}
            """
        else:
            raise ValueError(f"Invalid template type: {template_type}")

        return datasample, question, in_context_samples

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
    elif template_type == 'qwen-instruct_no_reasoning':
        prefix = f"""
        <|im_start|>system
        You are a helpful assistant. You always provide the user directly with the answer without any reasoning.
        <|im_end|>
        <|im_start|>user
        The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples} {query} Your response should contain only the final answer enclosed in <answer> and </answer> tags, with no additional text—specifically, just {label_str}, for example: {answer_example}
        <|im_end|>
        <|im_start|>assistant
        """
    elif template_type == 'base_no_reasoning':
        prefix = f"""
        A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.

        User: The dataset has {len(features[0])} features and {n_classes} classes: {list(range(n_classes))}. {in_context_examples} {query} Your response should contain only the final answer enclosed in <answer> and </answer> tags, with no additional text—specifically, just {label_str}, for example: {answer_example}
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
        if 'inductive' in template_type:
            query = "Given the following data points with features:\n"
        else:
            query = "Now, please apply your rule to the following data points::\n"
        for i in range(n_query):
            query += f"{i+1}. Features: {format_features(features[i])}\n"
        query += "predict target values for each data point, separated by commas. "
        target_str = "predicted values for all samples"
    else:
        if 'inductive' in template_type:
            query = f"Now, please apply your rule to the data point {format_features(features[0])}, and predict its target values. "
        else:
            query = f"Given the data point with features {format_features(features[0])} and predict its target values. "
        target_str = "predicted value"
    
    # Generate random target values for the example
    random_targets = [np.round(np.random.uniform(0, 10), 3) for _ in range(n_query)]
    if "reasoning_api" in template_type or "standard_api_no_reasoning" in template_type:
        answer_example = f"{', '.join(str(x) for x in random_targets)}"
        rule_example = f" the target is the average of the two features "

    else:
        answer_example = f"<answer>{', '.join(str(x) for x in random_targets)}</answer>"
        rule_example = f" <answer>the target is the average of the two features</answer> "



    if 'inductive' in template_type:   
        if template_type == 'base_inductive':
            datasample = f"""
            A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
            The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
            User: The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples}. 
            Please infer a rule that maps the features to the target values. 
            Your final answer should be enclosed in <answer> and </answer> tags. for example, {rule_example}.
            """
            question = f"""
            Good job! {query} Your final answer should be enclosed in <answer> and </answer> tags, containing only {target_str} with no additional text—for example, {answer_example}"
            """
        elif template_type == 'qwen-instruct_inductive':
            datasample = f"""
        <|im_start|>system
        You are a helpful assistant. You first infer a mapping rule from the examples in your mind and then provide the answer.
        <|im_end|>
        <|im_start|>user
        The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples}
        Please infer a rule that maps the features to the target values.
        Your final answer should be enclosed in <answer> and </answer> tags—for example, {rule_example}.
        <|im_end|>
        <|im_start|>assistant
        """
            question = f"""
        Good job!
        {query}
        Your final answer should be enclosed in <answer> and </answer> tags, containing only {target_str} with no additional text—for example, {answer_example}.
        """

        # 2. qwen-instruct_no_reasoning_inductive
        elif template_type == 'qwen-instruct_no_reasoning_inductive':
            datasample = f"""
        <|im_start|>system
        You are a helpful assistant. You always infer the rule yourself but do not show your reasoning, only give the final mapped value.
        <|im_end|>
        <|im_start|>user
        The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples}
        Please infer a rule that maps the features to the target values.
        Your final answer should be enclosed in <answer> and </answer> tags—for example, {rule_example}.
        <|im_end|>
        <|im_start|>assistant
        <answer>
        """
            question = f"""
        <|im_start|>system
        You are a helpful assistant. You always infer the rule yourself but do not show your reasoning, only give the final mapped value.   
        <|im_end|>
        <|im_start|>user
        Good job!
        {query}
        Your final answer should be enclosed in <answer> and </answer> tags—for example, {answer_example}.
        <|im_end|>
        <|im_start|>assistant
        <answer>
        """

        # 3. base_no_reasoning_inductive
        elif template_type == 'base_no_reasoning_inductive':
            datasample = f"""
            A conversation between User and Assistant. The user gives examples but the assistant does not show its inner reasoning.

            User: The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples}
            Please infer a rule that maps the features to the target values.
            Your final answer should be enclosed in <answer> and </answer> tags—for example, {rule_example}.
            Assistant:
            """
            question = f"""
            Good job!
            {query}
            Your final answer should be enclosed in <answer> and </answer> tags, containing only {target_str} with no additional text—for example, {answer_example}.
            """
        elif template_type == 'reasoning_api_inductive':
            datasample = f"""
            The dataset has {len(features[0])} features and 1 target attribute. Here are some examples:
            {in_context_examples}
            Please infer a rule that maps the features to the target values.
            Return your rule in a few sentences—for example, {rule_example}.
            """
            question = f"""
            Good job!
            {query} Your response should contain only {target_str} with no additional text—for example, {answer_example}
        """

        elif template_type == "reasoning_api_customized_inductive":
            datasample = f"""
            The dataset has {len(features[0])} features and 1 target attribute. Here are some examples:
            {in_context_examples}
            Please infer a rule that maps the features to the target values.
            Return your rule in a few sentences—for example, {rule_example}.
            """
            question = f"""
            Good job!
            {query} {customized_prompt} Your response should contain only {target_str} with no additional text—for example, {answer_example}
                """
        # 4. standard_api_inductive
        elif template_type == 'standard_api_inductive':
            datasample = f"""
            The dataset has {len(features[0])} features and 1 target attribute. Here are some examples:
            {in_context_examples}
            Please infer a rule that maps the features to the target values.
            Return your rule in a few sentences—for example, {rule_example}.
            """
            question = f"""
            Good job!
            {query}
            Your response should contain only the predicted {target_str} in <answer> tags, with no extra text—for example, {answer_example}.
            """

        # 5. standard_api_no_reasoning_inductive
        elif template_type == 'standard_api_no_reasoning_inductive':
            datasample = f"""
            The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples}
            Please infer a rule that maps the features to the target values.
            Your final answer should be enclosed in <answer> and </answer> tags—for example, {rule_example}.
            """
            question = f"""
            Good job!
            {query}
            Your response should contain only {target_str} with no additional text—for example, {answer_example}
        """
        else:
            raise ValueError(f"Invalid template type: {template_type}")
        return datasample, question, in_context_samples
    else:
        if template_type == 'base':
            prefix = f"""
            A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
            The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
            User: The dataset has {len(features[0])} features and 1 target attribute. 
            {in_context_examples} {query} Please provide your thinking process in <think> </think> tags. 
            Your final answer should be enclosed in <answer> and </answer> tags, containing only {target_str} with no additional text—for example, {answer_example}
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
            The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples} {query} Your response should contain only {target_str} with no additional text—for example, {answer_example}
            """
        elif template_type == "reasoning_api_customized":
            prefix = f"""
            The dataset has {len(features[0])} features and 1 target attribute. {in_context_examples} {query} {customized_prompt} Your response should contain only {target_str} with no additional text—for example, {answer_example}
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

def get_answer_format(answer_format, solution_str):
    if answer_format == "tags":
        return {
            "example": f"<answer>{solution_str}</answer>",
            "left": "<answer>",
            "right": "</answer>",
            "mention": "<answer> and </answer> tags"
        }
    elif answer_format == "box":
        return {
            "example": f"\\boxed{{{solution_str}}}",
            "left": "\\boxed{{",
            "right": "}}",
            "mention": "\\boxed{}",
        }
    elif answer_format == "none":
        return {
            "example": solution_str,
            "left": "",
            "right": "",
            "mention": ""
        }
    else:
        raise ValueError(f"Invalid answer format: {answer_format}")

def make_other_prefix(
    question, 
    template_type, 
    solution_example="0", 
    answer_format="tags",
    label_str="answer"
):
    if "reasoning_api" in template_type or "standard_api_no_reasoning" in template_type:
        answer_example = get_answer_format("none", solution_example)
    else:
        answer_example = get_answer_format(answer_format, solution_example)
    
    if template_type == 'base':
        instruction_following = f"""
        A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
        User: {question} Please provide your thinking process in <think> </think> tags. Your final answer should be enclosed in {answer_example['mention']}, containing only {label_str} with no additional text—for example, {answer_example['example']}
        Assistant: Let me solve this step by step.
        """
    elif template_type == 'base_no_reasoning':
        instruction_following = f"""
        A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
        User: {question} Your final answer should be enclosed in {answer_example['mention']}, containing only {label_str} with no additional text—for example, {answer_example['example']}
        Assistant: 
        """
    elif template_type == "qwen-instruct":
        instruction_following = f"""
        <|im_start|>system
        You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.
        <|im_end|>
        <|im_start|>user
        {question} Please provide your thinking process in <think> </think> tags. Your final answer should be enclosed in {answer_example['mention']}, containing only {label_str} with no additional text—for example, {answer_example['example']}
        <|im_end|>
        <|im_start|>assistant
        Let me solve this step by step.
        """
    elif template_type == "qwen-instruct_no_reasoning":
        instruction_following = f"""
        <|im_start|>system
        You are a helpful assistant. You always provide the user directly with the answer without any reasoning.
        <|im_end|>
        <|im_start|>user
        {question} Your response should contain only the final answer enclosed in {answer_example['mention']}, with no additional text—specifically, just {label_str}, for example: {answer_example['example']}
        <|im_end|>
        <|im_start|>assistant
        """
    elif template_type == "reasoning_api":
        instruction_following = f"""
        {question} Your response should just be the answer containing only {label_str} with no additional text—for example, {answer_example['example']}
        """
    elif template_type == "standard_api_no_reasoning":
        instruction_following = f"""
        {question} Your response should just be the answer containing only {label_str} with no additional text—for example, {answer_example['example']}
        """
    elif template_type == "standard_api":
        instruction_following = f"""
        {question} Please provide your thinking process in <think> </think> tags. Your final answer should be enclosed in {answer_example['mention']}, containing only {label_str} with no additional text—for example, {answer_example['example']}
        """
    else:
        raise ValueError(f"Template type {template_type} is not supported for GSM8k")

    return instruction_following

def make_prefix(
    dp, 
    template_type, 
    n_classes, 
    task_type,
    n_shot=0, 
    n_query=1,
    in_context_dataset=None, 
    customized_prompt=None,
):
    if task_type == "classification":
        return make_classification_prefix(
            dp = dp, 
            template_type = template_type, 
            n_classes = n_classes, 
            n_shot = n_shot, 
            n_query = n_query,
            in_context_dataset = in_context_dataset, 
            customized_prompt = customized_prompt
        )
    elif task_type == "regression":
        return make_regression_prefix(
            dp = dp, 
            template_type = template_type, 
            n_classes = n_classes, 
            n_shot = n_shot, 
            n_query = n_query,
            in_context_dataset = in_context_dataset, 
            customized_prompt = customized_prompt)
    else:
        raise ValueError(f"Invalid task type: {task_type}")

def make_map_fn(split, args, n_classes, in_context_dataset, data_source, data_mode, customized_prompt=None):

    
    def process_fn(example, idx):
        if data_mode in ["grid", "default"]:
            in_context_dataset_ = in_context_dataset[split]
        else:
            in_context_dataset_ = Dataset.from_dict({
                "features": [in_context_dataset[split][idx][i][0] for i in range(len(in_context_dataset[split][idx]))],
                "label": [in_context_dataset[split][idx][i][1] for i in range(len(in_context_dataset[split][idx]))]
            })
        datasample=None
        if "inductive" in args.template_type:   
            datasample, question, in_context_samples = make_prefix(
                example, 
                template_type=args.template_type, 
                n_classes=n_classes, 
                task_type = supported_datasets[data_source]['type'],
                n_shot=args.n_shot, 
                n_query=args.n_query,
                in_context_dataset=in_context_dataset_,
                customized_prompt=customized_prompt
            )
        else:
            question, in_context_samples = make_prefix(
                example, 
                template_type=args.template_type, 
                n_classes=n_classes, 
                task_type = supported_datasets[data_source]['type'],
                n_shot=args.n_shot, 
                n_query=args.n_query,
                in_context_dataset=in_context_dataset_,
                customized_prompt=customized_prompt
            )
            
        solution = {
            "features": example['features'],
            "label": example['label'],
            "in_context_samples": in_context_samples
        }
        if datasample is None:
            prompt={
                "role": "user",
                "content": question
            }
        else:
            prompt={
                "role": "user",
                "datasample": datasample,
                "content": question
            }
        data = {
            "data_source": data_source,
            "prompt": [prompt],
            "ability": "classification",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'inductive': "inductive" in args.template_type,
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
    if data_source == 'gsm8k':
        from verl.utils.reward_score import gsm8k
        return gsm8k.compute_score
    elif data_source in ["gpqa-diamond"]:
        from verl.utils.reward_score import general
        return lambda solution_str, ground_truth: general.compute_score(solution_str, ground_truth, answer_format = supported_datasets[data_source]['answer_format'])
    elif data_source in ["math", "math500"]:
        from verl.utils.reward_score import math500
        return lambda solution_str, ground_truth: math500.compute_score(solution_str, ground_truth, answer_format = supported_datasets[data_source]['answer_format'])
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
    if data_source in ['math', 'math500', "gpqa-diamond", "gsm8k"]:
        from verl.utils.reward_score import general
        return lambda solution_str: general.last_answer_string(solution_str, answer_format = supported_datasets[data_source]['answer_format'])
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